import os, re, json, time, math, random, hashlib, unicodedata, gc, platform
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import rankdata, spearmanr
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import model_info
from datasketch import MinHash, MinHashLSH
import transformers, sklearn, scipy, datasketch

PIPELINE_VERSION = 'applsci-4641089-v1'
FEATURE_NAMES = ['mean_valence', 'mean_arousal', 'mean_dominance',
                 'std_valence', 'std_arousal', 'std_dominance', 'coverage', 'log1p_matches']
WORD_RE = re.compile(r"[A-Za-z']+")

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)

def save_weights(path, state):
    tmp = Path(str(path) + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def normalize_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text)).casefold()).strip()

def duplicate_groups(texts, near=True):
    """Exact normalization + verified Jaccard pairs retrieved by approximate LSH."""
    keys = [normalize_text(t) for t in texts]
    codes, unique = pd.factorize(np.asarray(keys, dtype=object), sort=False)
    parent = np.arange(len(unique))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    def union(i, j):
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    @lru_cache(maxsize=2048)
    def shingles(i):
        words = re.findall(r'\w+', unique[i])
        # Short statements: exact normalized equality only.
        if len(words) < 5:
            return frozenset()
        return frozenset(' '.join(words[j:j+3]) for j in range(len(words)-2))
    edges = []
    if near:
        lsh = MinHashLSH(threshold=0.80, num_perm=128)
        for i in tqdm(range(len(unique)), desc='Duplicados cercanos (LSH aproximado)'):
            s = shingles(i)
            if not s:
                continue
            mh = MinHash(num_perm=128, seed=42)
            words = [x.encode() for x in sorted(s)]
            for k in range(0, len(words), 256):
                mh.update_batch(words[k:k+256])
            for j in sorted(lsh.query(mh)):
                t = shingles(j)
                score = len(s & t) / len(s | t)
                if score >= 0.90:
                    union(i, j)
                    edges.append((j, i, score))
            lsh.insert(i, mh)
    roots = np.array([root(int(i)) for i in codes])
    # Canonical group identifiers do not depend on union-find traversal order.
    groups = np.array([hashlib.sha256(unique[i].encode()).hexdigest() for i in roots])
    representatives = np.full(len(unique), -1, dtype=int)
    for row, code in enumerate(codes):
        if representatives[code] < 0:
            representatives[code] = row
    pairs = pd.DataFrame([(int(representatives[a]), int(representatives[b]), s)
                          for a, b, s in edges], columns=['position_a', 'position_b', 'jaccard'])
    return groups, pairs

def overlap_report(df, keys, split):
    tr = set(keys[split == 'train'])
    va = set(keys[split == 'validation'])
    test = split == 'test'
    table = pd.DataFrame({'key': keys, 'label': df.status, 'split': split})
    g = table.groupby('key').agg(n=('label', 'size'), labels=('label', 'nunique'),
                                 splits=('split', 'nunique'))
    return {'duplicate_groups': int((g.n > 1).sum()),
            'duplicate_excess_rows': int((g.n - 1).sum()),
            'conflicting_label_groups': int((g.labels > 1).sum()),
            'cross_split_groups': int((g.splits > 1).sum()),
            'test_rows_matching_train': int(np.isin(keys[test], list(tr)).sum()),
            'test_rows_matching_train_or_validation': int(np.isin(keys[test], list(tr | va)).sum())}

def prepare_data(cfg):
    rootdir = Path(cfg['output_dir']); rootdir.mkdir(parents=True, exist_ok=True)
    data_hash, lex_hash = sha_file(cfg['data_path']), sha_file(cfg['lexicon_path'])
    signature = {'pipeline': PIPELINE_VERSION, 'dataset_sha256': data_hash,
                 'lexicon_sha256': lex_hash, 'split_seed': cfg['split_seed'],
                 'near_duplicates': cfg['near_duplicates'], 'sklearn': sklearn.__version__,
                 'datasketch': datasketch.__version__, 'normalization': 'NFKC-casefold-whitespace',
                 'group_split': 'StratifiedGroupKFold20: test0-2, validation3-5, train6-19'}
    folder = rootdir / ('data_' + digest(signature)[:12]); folder.mkdir(exist_ok=True)
    raw = pd.read_csv(cfg['data_path'])
    df = raw.dropna(subset=['statement', 'status']).copy()
    df['row_id'] = df.index.astype(int)  # Original CSV row position, before removing missing rows.
    df = df.reset_index(drop=True)
    df['statement'] = df.statement.astype(str)
    assert df.statement.str.strip().ne('').all(), 'Hay textos vacíos; revisar antes de continuar.'
    labels = sorted(df.status.unique().tolist())
    assert len(labels) == 7, f'Se esperaban 7 clases, se encontraron {labels}'
    df['y'] = df.status.map({v: i for i, v in enumerate(labels)}).astype(int)
    allids = np.arange(len(df))
    train, tmp = train_test_split(allids, test_size=.30, random_state=42, stratify=df.y)
    valid, test = train_test_split(tmp, test_size=.50, random_state=42, stratify=df.y.iloc[tmp])
    original = np.full(len(df), 'train', dtype=object)
    original[valid], original[test] = 'validation', 'test'
    df['original_split'] = original
    if (folder / 'split_manifest.csv').exists():
        manifest = pd.read_csv(folder / 'split_manifest.csv')
        assert np.array_equal(manifest.row_id, df.row_id)
        df['group_id'], df['split'] = manifest.group_id, manifest.split
    else:
        groups, pairs = duplicate_groups(df.statement.tolist(), cfg['near_duplicates'])
        df['group_id'] = groups
        fold = np.full(len(df), -1)
        cv = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=cfg['split_seed'])
        for k, (_, idx) in enumerate(cv.split(allids, df.y, groups)):
            fold[idx] = k
        df['split'] = np.where(fold < 3, 'test', np.where(fold < 6, 'validation', 'train'))
        for col in ['position_a', 'position_b']:
            pairs[col.replace('position', 'row_id')] = df.row_id.to_numpy()[pairs[col].to_numpy(dtype=int)]
        pairs.to_csv(folder / 'near_duplicate_pairs.csv', index=False)
        df[['row_id', 'status', 'y', 'group_id', 'original_split', 'split']].to_csv(
            folder / 'split_manifest.csv', index=False)
    assert df.groupby('group_id').split.nunique().max() == 1
    for name in ['train', 'validation', 'test']:
        assert df.loc[df.split == name, 'y'].nunique() == len(labels)
    report = {'raw_rows': len(raw), 'valid_rows': len(df), 'labels': labels,
              'original_exact': overlap_report(df, df.statement.to_numpy(), original),
              'original_normalized': overlap_report(df, df.statement.map(normalize_text).to_numpy(), original),
              'original_detected_groups': overlap_report(df, df.group_id.to_numpy(), original),
              'new_detected_groups': overlap_report(df, df.group_id.to_numpy(), df.split.to_numpy()),
              'near_duplicate_limitation': 'LSH approximate retrieval; exact Jaccard>=0.90 on word trigrams; <5-word texts checked for normalized equality only. Source/user overlap not measured without source/user IDs.',
              'label_conflicts': 'All rows and supplied labels retained; conflicting duplicates stay in one split.'}
    write_json(folder / 'audit.json', report)
    write_json(folder / 'data_config.json', signature)
    pd.crosstab(df.status, df.split).to_csv(folder / 'class_counts.csv')
    print(pd.crosstab(df.status, df.split))
    print('Solapamiento exacto original:', report['original_exact'])
    print('Grupos detectados compartidos en la nueva partición:', report['new_detected_groups']['cross_split_groups'])
    lex = pd.read_csv(cfg['lexicon_path'], sep='\t', header=0,
                      names=['word', 'valence', 'arousal', 'dominance'])
    lex['word'] = lex.word.astype(str).str.lower().str.strip()
    for c in ['valence', 'arousal', 'dominance']:
        lex[c] = pd.to_numeric(lex[c], errors='coerce')
    lex = lex.dropna().drop_duplicates('word')
    values = lex[['valence', 'arousal', 'dominance']].to_numpy(np.float32)
    lookup = dict(zip(lex.word, values)); prior = values.mean(axis=0)
    revision = cfg.get('model_revision') or model_info(cfg['model_name']).sha
    assert revision, 'No se pudo fijar la revisión del modelo de Hugging Face.'
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], revision=revision, use_fast=True)
    assert tokenizer.is_fast
    lens_path = folder / ('lengths_' + revision[:12] + '.npy')
    if lens_path.exists():
        lengths = np.load(lens_path)
    else:
        lengths = []
        for start in tqdm(range(0, len(df), 512), desc='Longitudes completas'):
            enc = tokenizer(df.statement.iloc[start:start+512].tolist(), truncation=False,
                            padding=False, return_attention_mask=False, verbose=False)
            lengths.extend(map(len, enc['input_ids']))
        lengths = np.array(lengths); np.save(lens_path, lengths)
    df['n_subwords_full'] = lengths
    df['n_words_full'] = df.statement.map(lambda x: len(WORD_RE.findall(x)))
    audit = []
    for (split, label), part in df.groupby(['split', 'status']):
        for length in [32, 128]:
            audit.append({'split': split, 'label': label, 'max_length': length, 'n': len(part),
                          'truncated_n': int((part.n_subwords_full > length).sum()),
                          'truncated_rate': float((part.n_subwords_full > length).mean())})
    pd.DataFrame(audit).to_csv(folder / 'truncation_by_class.csv', index=False)
    fingerprint = {**signature, 'split_sha256': sha_file(folder / 'split_manifest.csv'),
                   'model_name': cfg['model_name'], 'model_revision': revision}
    write_json(folder / 'fingerprint.json', fingerprint)
    return dict(df=df, labels=labels, folder=folder, lookup=lookup, prior=prior,
                tokenizer=tokenizer, fingerprint=fingerprint, cache={})

def vad_vector(text, lookup, prior, cutoff=None):
    # Use original character offsets; exclude a word if only a prefix is visible.
    toks = [m.group().lower() for m in WORD_RE.finditer(text) if cutoff is None or m.end() <= cutoff]
    matched = [lookup[t] for t in toks if t in lookup]
    if not matched:
        return np.r_[prior, np.zeros(5)].astype(np.float32)
    a = np.stack(matched)
    return np.r_[a.mean(0), a.std(0), len(matched)/len(toks), np.log1p(len(matched))].astype(np.float32)

def features(data, length, scope):
    cache = data['cache']; df = data['df']; folder = data['folder']
    if ('tokens', length) not in cache:
        path = folder / f'tokens_{length}_{data["fingerprint"]["model_revision"][:12]}.npz'
        if path.exists():
            with np.load(path) as f:
                enc = {k: f[k] for k in f.files}
        else:
            out = data['tokenizer'](df.statement.tolist(), max_length=length, truncation=True,
                                   padding='max_length', return_offsets_mapping=True, return_tensors='np')
            offsets = out.pop('offset_mapping')
            enc = {'input_ids': out['input_ids'], 'attention_mask': out['attention_mask'],
                   'visible_end': offsets[:, :, 1].max(axis=1)}
            np.savez_compressed(path, **enc)
        cache['tokens', length] = enc
    enc = cache['tokens', length]
    key = ('vad', length if scope == 'visible' else 0, scope)
    if key not in cache:
        revision_tag = '_' + data['fingerprint']['model_revision'][:12] if scope == 'visible' else ''
        path = folder / f'vad_{scope}_{key[1]}{revision_tag}.npy'
        if path.exists():
            raw = np.load(path)
        else:
            cutoffs = enc['visible_end'] if scope == 'visible' else [None] * len(df)
            raw = np.stack([vad_vector(t, data['lookup'], data['prior'], c)
                            for t, c in tqdm(zip(df.statement, cutoffs), total=len(df), desc=f'VAD {scope}')])
            np.save(path, raw)
        train = df.split.eq('train').to_numpy()
        mean, std = raw[train].mean(0), raw[train].std(0) + 1e-8
        z = (raw - mean) / std
        assert np.isfinite(z).all()
        cache[key] = raw, z.astype(np.float32), mean, std
    return enc, cache[key]

class Classifier(nn.Module):
    def __init__(self, variant, num_labels, encoder):
        super().__init__()
        self.encoder, self.variant = encoder, variant
        self.hidden = encoder.config.hidden_size
        d = self.hidden
        self.dropout = nn.Dropout(.1)
        if variant == 'gate':
            self.gate = nn.Sequential(nn.Linear(8, d//2), nn.GELU(), nn.Dropout(.1), nn.Linear(d//2, 1))
        self.classifier = nn.Linear(d if variant == 'text' else d+8 if variant == 'concat' else 2*d+9, num_labels)
    def head(self, h, v, alpha=None):
        if self.variant == 'text':
            return self.classifier(self.dropout(h)), None
        if self.variant == 'concat':
            return self.classifier(self.dropout(torch.cat([h, v], 1))), None
        alpha = torch.sigmoid(self.gate(v)) if alpha is None else torch.full_like(v[:, :1], float(alpha))
        return self.classifier(self.dropout(torch.cat([h, alpha*h, v, alpha], 1))), alpha
    def forward(self, ids, mask, v):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        z, a = self.head(h, v)
        return z, a
    def complemented_logits(self, h, v, alpha):
        d = self.hidden; w = self.classifier.weight; b = self.classifier.bias
        A, B, C, q = w[:, :d], w[:, d:2*d], w[:, 2*d:2*d+8], w[:, -1:]
        a = 1-alpha
        return F.linear(h, A+B) + F.linear(a*h, -B) + F.linear(v, C) + F.linear(a, -q) + b + q[:, 0]

def scores(y, pred, classes):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=np.arange(classes), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, pred, labels=np.arange(classes), average='weighted', zero_division=0))}

def tensors(enc, z):
    return (torch.as_tensor(enc['input_ids'], dtype=torch.long),
            torch.as_tensor(enc['attention_mask'], dtype=torch.long), torch.from_numpy(z))

def batch_to_device(t, idx, device):
    return [x[idx].to(device) for x in t]

@torch.no_grad()
def predict(model, t, indices, device, batch_size=64, diagnostics=False):
    model.eval(); logits, extra = [], []
    for start in range(0, len(indices), batch_size):
        b = batch_to_device(t, indices[start:start+batch_size], device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            h = model.encoder(input_ids=b[0], attention_mask=b[1]).last_hidden_state[:, 0]
        # Evaluate heads in float32 for stable probabilities and complement checks.
        h = h.float(); z, a = model.head(h, b[2])
        logits.append(z.cpu().numpy())
        if diagnostics and model.variant == 'gate':
            fixed, _ = model.head(h, b[2], alpha=.5)
            complement = model.complemented_logits(h, b[2], a)
            residual = (z-complement).abs().amax(1)
            assert float(residual.max()) < 1e-4, 'Falló la comprobación de simetría.'
            extra.append(torch.stack([a[:, 0], fixed.argmax(1).float(),
                                     (z-fixed).abs().mean(1), (z-fixed).norm(dim=1), residual], 1).cpu().numpy())
    return np.concatenate(logits), np.concatenate(extra) if extra else None

def run_one(data, cfg, length, variant, scope, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = {'python': platform.python_version(), 'torch': torch.__version__,
           'transformers': transformers.__version__, 'numpy': np.__version__,
           'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__,
           'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
           'cuda': torch.version.cuda}
    spec = {**data['fingerprint'], 'length': length, 'variant': variant, 'scope': scope,
            'seed': seed, 'epochs': cfg['epochs'], 'lr': cfg['lr'], 'weight_decay': .01,
            'micro_batch': cfg['micro_batch'], 'effective_batch': cfg['effective_batch'],
            'class_weighted_ce': True, 'dropout': .1, 'environment': env}
    folder = Path(cfg['output_dir']) / 'runs' / f'{length}_{variant}_{scope}_s{seed}_{digest(spec)[:10]}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'done.json').exists():
        assert json.loads((folder/'config.json').read_text()) == spec
        print('Ya terminada:', folder.name)
        return folder
    write_json(folder / 'config.json', spec)
    enc, (raw, z, mean, std) = features(data, length, 'full' if scope == 'none' else scope)
    np.savez(folder / 'vad_scaler.npz', mean=mean, std=std)
    t = tensors(enc, z); df = data['df']; C = len(data['labels'])
    y = torch.tensor(df.y.to_numpy(), dtype=torch.long)
    train = np.flatnonzero(df.split.eq('train')); valid = np.flatnonzero(df.split.eq('validation'))
    test = np.flatnonzero(df.split.eq('test'))
    counts = np.bincount(df.y.iloc[train], minlength=C)
    weights = torch.tensor(len(train)/(C*counts), dtype=torch.float32, device=device)
    set_seed(seed)
    encoder = AutoModel.from_pretrained(cfg['model_name'], revision=data['fingerprint']['model_revision'],
                                       use_safetensors=True, attn_implementation='eager')
    model = Classifier(variant, C, encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best, best_epoch, history = -1., None, []
    bestpath = folder / 'best.pt'
    start_run = time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        set_seed(seed + 10000*(epoch+1))
        order = np.random.default_rng(seed + 10000*(epoch+1)).permutation(train)
        if device.type == 'cuda': torch.cuda.synchronize()
        tic = time.perf_counter(); total_loss = 0.; total_weight = 0.
        for offset in tqdm(range(0, len(order), cfg['effective_batch']), desc=f'{folder.name} época {epoch+1}'):
            group = order[offset:offset+cfg['effective_batch']]
            optimizer.zero_grad(set_to_none=True)
            denom = weights[y[group].to(device)].sum()
            for a in range(0, len(group), cfg['micro_batch']):
                idx = group[a:a+cfg['micro_batch']]
                b = batch_to_device(t, idx, device); target = y[idx].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    out, _ = model(*b)
                    # The denominator is for the entire effective batch, not each microbatch.
                    numerator = F.cross_entropy(out, target, weight=weights, reduction='sum')
                    loss = numerator / denom
                scaler.scale(loss).backward()
                total_loss += float(numerator.detach())
            scaler.step(optimizer); scaler.update()
            total_weight += float(denom)
        if device.type == 'cuda': torch.cuda.synchronize()
        train_seconds = time.perf_counter()-tic
        val_logits, _ = predict(model, t, valid, device, cfg['eval_batch'])
        val = scores(y[valid].numpy(), val_logits.argmax(1), C)
        row = {'epoch': epoch+1, 'train_seconds': train_seconds, 'train_loss': total_loss/total_weight,
               **{'val_'+k: v for k, v in val.items()}}
        history.append(row); pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
        print(row)
        if val['macro_f1'] > best:
            best, best_epoch = val['macro_f1'], epoch+1
            save_weights(bestpath, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(torch.load(bestpath, map_location=device, weights_only=True))
    out, diag = predict(model, t, test, device, cfg['eval_batch'], diagnostics=True)
    pred = out.argmax(1); truth = y[test].numpy()
    result = {'seed': seed, 'length': length, 'variant': variant, 'scope': scope,
              'best_epoch': best_epoch, 'best_val_macro_f1': best,
              'train_seconds': sum(x['train_seconds'] for x in history),
              'elapsed_seconds': time.perf_counter()-start_run, **scores(truth, pred, C)}
    prob = torch.softmax(torch.from_numpy(out), -1).numpy()
    ordered = np.sort(prob, axis=1)
    table = df.iloc[test][['row_id', 'group_id', 'status', 'n_subwords_full', 'n_words_full']].copy()
    table['y_true'], table['y_pred'] = truth, pred
    table['true_label'], table['pred_label'] = np.array(data['labels'])[truth], np.array(data['labels'])[pred]
    table['correct'] = truth == pred
    table['confidence'] = prob.max(1)
    table['entropy'] = -(prob*np.log(np.clip(prob, 1e-12, 1))).sum(1)
    table['margin'] = ordered[:, -1]-ordered[:, -2]
    table['seed'], table['length'], table['scope'] = seed, length, scope
    table['visible_end_char'] = enc['visible_end'][test]
    for j, name in enumerate(FEATURE_NAMES):
        table[name] = raw[test, j]; table['z_'+name] = z[test, j]
    if diag is not None:
        table['alpha'], table['fixed_gate_pred'] = diag[:, 0], diag[:, 1].astype(int)
        table['delta_logit_mae'], table['delta_logit_l2'] = diag[:, 2], diag[:, 3]
        table['complement_max_error'] = diag[:, 4]
        table['gate_changes_prediction'] = pred != diag[:, 1]
        result['fixed_gate_macro_f1'] = scores(truth, diag[:, 1].astype(int), C)['macro_f1']
        result['gate_prediction_change_rate'] = float(table.gate_changes_prediction.mean())
        result['mean_delta_logit_mae'] = float(table.delta_logit_mae.mean())
        table['text'] = df.statement.iloc[test].to_numpy()
        table.to_csv(folder/'mentalhealth_scalar_gate_instance_analysis.csv', index=False)
    table.drop(columns=['text'], errors='ignore').to_csv(folder/'predictions.csv.gz', index=False)
    np.savez_compressed(folder/'logits.npz', row_id=df.row_id.iloc[test].to_numpy(), logits=out,
                        y_true=truth, y_pred=pred)
    write_json(folder/'metrics.json', result)
    write_json(folder/'labels.json', data['labels'])
    if cfg['keep_weights'] == 'gates' and variant != 'gate': bestpath.unlink(missing_ok=True)
    write_json(folder/'done.json', {'completed': True})  # Written last: interrupted runs are rerun.
    del model, encoder, optimizer, scaler, t
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    print('Guardado:', folder, result)
    return folder

def experiment_grid(mode):
    if mode == 'pilot':
        return [(32, 'text', 'none', 42), (32, 'concat', 'full', 42), (32, 'gate', 'full', 42)]
    assert mode == 'full'
    return [(L, model, scope, seed) for L in [32, 128]
            for model, scope in [('text', 'none'), ('concat', 'full'), ('gate', 'full'),
                                 ('concat', 'visible'), ('gate', 'visible')]
            for seed in [7, 13, 21, 42, 73]]

def paired_bootstrap(a, b, C, n_boot=3000):
    a, b = a.sort_values('row_id'), b.sort_values('row_id')
    assert np.array_equal(a.row_id, b.row_id) and np.array_equal(a.y_true, b.y_true)
    assert np.array_equal(a.group_id, b.group_id)
    y, pa, pb = a.y_true.to_numpy(), a.y_pred.to_numpy(), b.y_pred.to_numpy()
    codes, unique = pd.factorize(a.group_id, sort=True)
    def macro(pred, weights=None):
        cm = np.bincount(C*y+pred, weights=weights, minlength=C*C).reshape(C, C)
        den = cm.sum(0)+cm.sum(1)
        return np.divide(2*np.diag(cm), den, out=np.zeros(C), where=den>0).mean()
    rng = np.random.default_rng(20261031)
    differences = np.empty(n_boot)
    for i in range(n_boot):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[codes]
        differences[i] = macro(pb, weights)-macro(pa, weights)
    lo, hi = np.quantile(differences, [.025, .975])
    return {'delta_macro_f1': float(macro(pb)-macro(pa)), 'ci95_low': float(lo), 'ci95_high': float(hi),
            'test_groups': len(unique), 'bootstrap_samples': n_boot}

def gate_diagnostics(table, destination):
    destination = Path(destination)
    table.groupby('true_label').alpha.agg(['count', 'mean', 'std', 'median', 'min', 'max']).to_csv(
        destination/'gate_by_class.csv')
    table.groupby('correct').alpha.agg(['count', 'mean', 'std']).to_csv(destination/'gate_by_correctness.csv')
    # Partial Spearman via rank residualization: length + one-hot source labels.
    X = np.column_stack([np.ones(len(table)), rankdata(table.n_subwords_full),
                         pd.get_dummies(table.true_label, drop_first=True).to_numpy(float)])
    def residual(v):
        r = rankdata(v)
        return r-X @ np.linalg.lstsq(X, r, rcond=None)[0]
    ra = residual(table.alpha); rows = []
    for name in FEATURE_NAMES + ['entropy', 'margin', 'confidence', 'correct']:
        v = table[name].to_numpy(float); rv = residual(v)
        raw = float(spearmanr(table.alpha, v).statistic) if len(np.unique(v))>1 and table.alpha.nunique()>1 else np.nan
        partial = float(np.corrcoef(ra, rv)[0, 1]) if np.std(rv)>1e-8 and np.std(ra)>1e-8 else np.nan
        rows.append({'feature': name, 'spearman': raw, 'partial_spearman_length_and_label': partial})
    pd.DataFrame(rows).to_csv(destination/'gate_correlations.csv', index=False)
    # Descriptive class comparisons after coarsened length matching; no significance claim.
    matched = table.copy()
    matched['length_bin'] = pd.qcut(matched.n_subwords_full, 10, labels=False, duplicates='drop')
    parts = []
    for _, part in matched.groupby('length_bin'):
        if part.true_label.nunique() == table.true_label.nunique():
            size = int(part.true_label.value_counts().min())
            for _, label_part in part.groupby('true_label'):
                parts.append(label_part.sample(size, random_state=42))
    matched = pd.concat(parts) if parts else matched.iloc[:0]
    matched[['row_id', 'true_label', 'length_bin', 'n_subwords_full', 'alpha']].to_csv(
        destination/'length_matched_instances.csv', index=False)
    matched.groupby('true_label').alpha.agg(['count', 'mean', 'std']).to_csv(
        destination/'gate_by_class_length_matched.csv')
    # Explicit purposive selection, entirely from this run and retaining row IDs.
    selected = []
    for label, part in table.groupby('true_label'):
        for idx, rule in [(part.alpha.idxmin(), 'minimum alpha within true label'),
                          (part.alpha.idxmax(), 'maximum alpha within true label')]:
            if idx not in [x[0] for x in selected]: selected.append((idx, rule))
    for idx in table.entropy.sort_values(ascending=False, kind='stable').index:
        if len(selected) >= 20: break
        if idx not in [x[0] for x in selected]: selected.append((idx, 'highest remaining entropy'))
    cases = table.loc[[i for i, _ in selected]].copy()
    cases['selection_rule'] = [rule for _, rule in selected]
    cases.to_csv(destination/'qualitative_cases_purposive.csv', index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for correct, part in table.groupby('correct'):
        axes[0].hist(part.alpha, bins=35, alpha=.55, label='Correct' if correct else 'Incorrect')
    axes[0].set(xlabel='Gate alpha', ylabel='Count'); axes[0].legend()
    axes[1].scatter(table.log1p_matches, table.alpha, s=4, alpha=.12)
    axes[1].set(xlabel='log(1 + VAD matches), unstandardized', ylabel='Gate alpha')
    fig.tight_layout(); fig.savefig(destination/'gate_diagnostics.png', dpi=200); plt.close(fig)

def summarize(data, cfg, folders):
    dest = Path(cfg['output_dir'])/'summaries'/digest([str(p) for p in folders])[:12]
    dest.mkdir(parents=True, exist_ok=True)
    rows, predictions, lookup = [], {}, {}
    for folder in folders:
        r = json.loads((folder/'metrics.json').read_text()); r['run_directory'] = str(folder); rows.append(r)
        key = (r['length'], r['variant'], r['scope'], r['seed'])
        predictions[key] = pd.read_csv(folder/'predictions.csv.gz')
        lookup[key] = folder
    frame = pd.DataFrame(rows); frame.to_csv(dest/'metrics_per_seed.csv', index=False)
    agg = frame.groupby(['length', 'variant', 'scope']).agg(
        n_seeds=('seed', 'nunique'), macro_f1_mean=('macro_f1', 'mean'),
        macro_f1_std=('macro_f1', 'std'), accuracy_mean=('accuracy', 'mean'),
        train_seconds_mean=('train_seconds', 'mean'))
    agg.to_csv(dest/'metrics_summary.csv')  # Sample SD; NaN when only one seed.
    comparisons = []
    for key in predictions:
        L, model, scope, seed = key
        if model == 'concat':
            other = (L, 'text', 'none', seed)
        elif model == 'gate':
            other = (L, 'concat', scope, seed)
        else: other = None
        if other in predictions: comparisons.append((other, key))
        if model == 'gate' and (L, 'text', 'none', seed) in predictions:
            comparisons.append(((L, 'text', 'none', seed), key))
        if scope == 'visible' and (L, model, 'full', seed) in predictions:
            comparisons.append(((L, model, 'full', seed), key))
        if L == 128 and (32, model, scope, seed) in predictions:
            comparisons.append(((32, model, scope, seed), key))
    paired = []
    for ka, kb in tqdm(comparisons, desc='Bootstrap pareado por grupos'):
        stats = paired_bootstrap(predictions[ka], predictions[kb], len(data['labels']), cfg['bootstrap_samples'])
        paired.append({'model_a': str(ka), 'model_b': str(kb), 'seed': kb[-1], **stats})
    pd.DataFrame(paired).to_csv(dest/'paired_bootstrap_per_seed.csv', index=False)
    for key, folder in lookup.items():
        if key[1] == 'gate':
            table = pd.read_csv(folder/'mentalhealth_scalar_gate_instance_analysis.csv')
            gate_diagnostics(table, folder)
    write_json(dest/'reporting_notes.json', {
        'new_experiments': 'These are new grouped-split experiments, not recovery of the historical seed-42 outputs.',
        'bootstrap': 'Paired test-group resampling within each seed; intervals conditional on fitted models. Seed variability is reported separately; do not average interval bounds or p-values.',
        'qualitative_cases': 'Purposive alpha extremes within true classes, then highest remaining entropy. Not representative.',
        'gate_intervention': 'Fixed weights and features; replace alpha by 0.5 in both alpha*h and the alpha feature.',
        'invariant_delta': 'z(alpha)-z(0.5) is invariant under the stated complement transformation; not a measure of total affective reliance.',
        'length_matching': 'Within pooled token-length quantile bins, equal sample count per class. Bins without all classes omitted. Report retained n and incomplete overlap.',
        'source_overlap': 'No author/source IDs in input; duplicate screening does not establish source-disjoint or user-disjoint evaluation.',
        'training': '3 epochs; AdamW lr=2e-5, weight_decay=.01; weighted CE; effective batch512 by accumulation. Dropout streams differ from the historical batch512 run.'})
    print(agg)
    print('Tablas y notas:', dest)
    return dest
