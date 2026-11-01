"""Recompute numerical results from the compact, text-free supplementary records.

No GPU or original corpus is needed. Use --bootstrap to repeat all 105 intervals
(3,000 resamples each); without that flag, recompute scores and gate diagnostics.
"""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

KEY = ['length', 'variant', 'scope', 'seed']
FEATURES = ['mean_valence', 'mean_arousal', 'mean_dominance', 'std_valence',
            'std_arousal', 'std_dominance', 'coverage', 'log1p_matches',
            'entropy', 'margin', 'confidence', 'correct']

def score(y, pred, weights=None):
    m = np.bincount(7*np.asarray(y, int)+np.asarray(pred, int),
                    weights=weights, minlength=49).reshape(7, 7)
    denominator = m.sum(0)+m.sum(1)
    f1 = np.divide(2*np.diag(m), denominator, out=np.zeros(7), where=denominator>0)
    return f1.mean(), np.trace(m)/m.sum(), f1

def paired(a, b):
    a, b = a.sort_values('row_id'), b.sort_values('row_id')
    np.testing.assert_array_equal(a[['row_id', 'y_true', 'group_id']],
                                  b[['row_id', 'y_true', 'group_id']])
    codes, groups = pd.factorize(a.group_id, sort=True)
    rng = np.random.default_rng(20261031)
    delta = np.empty(3000)
    for i in range(len(delta)):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        w = counts[codes]
        delta[i] = score(b.y_true, b.y_pred, w)[0]-score(a.y_true, a.y_pred, w)[0]
    lo, hi = np.quantile(delta, [.025, .975])
    return dict(delta_macro_f1=score(b.y_true,b.y_pred)[0]-score(a.y_true,a.y_pred)[0],
                ci95_low=lo, ci95_high=hi, test_groups=len(groups), bootstrap_samples=3000)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1]/'supplementary_data')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--bootstrap', action='store_true')
    p.add_argument('--bootstrap-limit', type=int, default=None,
                   help='Optional limit for checking a subset; omit to reproduce all intervals.')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    read = lambda n: pd.read_csv(args.data/n)
    manifest = read('split_manifest.csv')
    assert manifest.groupby('group_id').split.nunique().max()==1
    predictions = read('predictions_all_runs.csv.gz').merge(
        manifest[['row_id', 'group_id']], on='row_id', validate='many_to_one')
    lookup = dict(tuple(predictions.groupby(KEY)))
    assert len(lookup)==50
    labels = json.loads((args.data/'audit.json').read_text())['labels']
    scores, perclass = [], []
    for k, t in lookup.items():
        assert len(t)==7897 and t.row_id.is_unique
        f, a, fs = score(t.y_true, t.y_pred)
        scores.append(dict(zip(KEY,k), macro_f1=f, accuracy=a))
        perclass.extend(dict(zip(KEY,k), label=lab, f1=value) for lab,value in zip(labels,fs))
    metrics = pd.DataFrame(scores)
    metrics.to_csv(args.output/'metrics_recomputed.csv', index=False)
    pd.DataFrame(perclass).to_csv(args.output/'per_class_f1_recomputed.csv', index=False)
    metrics.groupby(KEY[:-1]).agg(n_seeds=('seed','nunique'),
        macro_f1_mean=('macro_f1','mean'), macro_f1_std=('macro_f1','std'),
        accuracy_mean=('accuracy','mean')).to_csv(args.output/'summary_recomputed.csv')
    reference = read('metrics_per_seed.csv').set_index(KEY).sort_index()
    np.testing.assert_allclose(metrics.set_index(KEY).sort_index()[['macro_f1','accuracy']],
                               reference[['macro_f1','accuracy']], atol=1e-12, rtol=1e-12)
    gate = read('gate_instances_all_runs.csv.gz')
    interventions, correlations = [], []
    for k,g in gate.groupby(KEY):
        t = g.merge(lookup[k][['row_id','y_true','y_pred']], on='row_id', validate='one_to_one')
        f,_,_ = score(t.y_true,t.y_pred); fixed,_,_ = score(t.y_true,t.fixed_gate_pred)
        change = t.y_pred.ne(t.fixed_gate_pred)
        interventions.append(dict(zip(KEY,k), changed_n=int(change.sum()),
            changed_pct=100*change.mean(), macro_f1=f, fixed_macro_f1=fixed,
            delta_f1=f-fixed, mean_delta_logit_mae=t.delta_logit_mae.mean(),
            max_complement_error=t.complement_max_error.max()))
        X=np.column_stack([np.ones(len(t)),rankdata(t.n_subwords_full),
                           pd.get_dummies(t.y_true,drop_first=True).to_numpy(float)])
        ranks=np.column_stack([rankdata(t[c]) for c in ['alpha']+FEATURES])
        residual=ranks-X@np.linalg.lstsq(X,ranks,rcond=None)[0]
        raw=np.corrcoef(ranks,rowvar=False)[0,1:]
        adj=np.corrcoef(residual,rowvar=False)[0,1:]
        correlations.extend(dict(zip(KEY,k),feature=c,spearman=r,
            partial_spearman_length_and_label=a) for c,r,a in zip(FEATURES,raw,adj))
    pd.DataFrame(interventions).to_csv(args.output/'gate_interventions_recomputed.csv',index=False)
    corr=pd.DataFrame(correlations)
    corr.to_csv(args.output/'gate_correlations_recomputed.csv',index=False)
    ck=['length','scope','seed','feature']; cv=['spearman','partial_spearman_length_and_label']
    np.testing.assert_allclose(corr.set_index(ck).sort_index()[cv],
        read('gate_correlations_per_seed.csv').set_index(ck).sort_index()[cv], atol=2e-6,rtol=2e-6)
    if args.bootstrap:
        pairs=read('paired_bootstrap_per_seed.csv')
        if args.bootstrap_limit is not None:pairs=pairs.head(args.bootstrap_limit)
        rows=[]
        for i,row in enumerate(pairs.itertuples(),1):
            a,b=ast.literal_eval(row.model_a),ast.literal_eval(row.model_b)
            r=paired(lookup[a],lookup[b])
            np.testing.assert_allclose([r['ci95_low'],r['ci95_high']],
                                       [row.ci95_low,row.ci95_high],atol=1e-12,rtol=1e-12)
            rows.append(dict(model_a=str(a),model_b=str(b),seed=b[-1],**r))
            print(f'Bootstrap {i}/{len(pairs)} verified',flush=True)
        pd.DataFrame(rows).to_csv(args.output/'paired_bootstrap_recomputed.csv',index=False)
    print('50 score sets and 20 gate diagnostic sets recomputed and checked.')

if __name__ == '__main__':
    main()
