"""Audit the supplied full experiment archive without retraining any model.

Usage: python audit_original_archive.py INPUT_ZIP OUTPUT_DIRECTORY [ORIGINAL_CORPUS_CSV]
Requires numpy, pandas and scipy. Does not require torch or transformers.
"""
import ast
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

source = Path(sys.argv[1])
dest = Path(sys.argv[2]); dest.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(source)
names = z.namelist()
assert len(names) == len(set(names))
def js(p): return json.loads(z.read(p))
def csv(p): return pd.read_csv(io.BytesIO(z.read(p)), compression='gzip' if p.endswith('.gz') else None)
def near(a,b,tol=1e-10): np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)
def cm(y,p,w=None): return np.bincount(7*np.asarray(y,int)+np.asarray(p,int),weights=w,minlength=49).reshape(7,7)
def f1s(m):
    den=m.sum(0)+m.sum(1)
    return np.divide(2*np.diag(m),den,out=np.zeros(7),where=den>0)
def score(y,p):
    m=cm(y,p); f=f1s(m)
    return dict(accuracy=np.trace(m)/m.sum(),macro_f1=f.mean(),weighted_f1=np.sum(f*m.sum(1))/m.sum())

dp=next(n.rsplit('/',1)[0] for n in names if n.endswith('/split_manifest.csv'))
sp=next(n.rsplit('/',1)[0] for n in names if n.endswith('/metrics_summary.csv'))
manifest=csv(dp+'/split_manifest.csv'); fp=js(dp+'/fingerprint.json'); audit=js(dp+'/audit.json')
assert hashlib.sha256(z.read(dp+'/split_manifest.csv')).hexdigest()==fp['split_sha256']
assert manifest.row_id.is_unique and len(manifest)==52681
assert manifest.groupby('group_id').split.nunique().max()==1
links=csv(dp+'/near_duplicate_pairs.csv'); indexed=manifest.set_index('row_id')
np.testing.assert_array_equal(indexed.loc[links.row_id_a,'group_id'],indexed.loc[links.row_id_b,'group_id'])
corpus=None
if len(sys.argv)>3:
    corpus_path=Path(sys.argv[3])
    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest()==fp['dataset_sha256']
    corpus=pd.read_csv(corpus_path)
    np.testing.assert_array_equal(corpus.loc[manifest.row_id,'status'],manifest.status)
test=manifest.loc[manifest.split=='test'].reset_index(drop=True)
labels=audit['labels']; N=len(test)
assert N==7897
dirs=sorted(n.rsplit('/',1)[0] for n in names if n.startswith('runs/') and n.endswith('/metrics.json'))
features=['mean_valence','mean_arousal','mean_dominance','std_valence','std_arousal','std_dominance','coverage','log1p_matches']
models=[('text','none'),('concat','full'),('gate','full'),('concat','visible'),('gate','visible')]
expected={(L,v,s,seed) for L in [32,128] for v,s in models for seed in [7,13,21,42,73]}
predictions={}; configs=[]; metrics=[]; gates=[]; correlations=[]; perclass=[]; pair_selected=[]
reference_features={}; numeric_errors=[]
for d in dirs:
    c=js(d+'/config.json'); m=js(d+'/metrics.json'); h=csv(d+'/history.csv')
    k=(c['length'],c['variant'],c['scope'],c['seed'])
    assert k not in predictions
    assert js(d+'/done.json')['completed'] is True
    assert js(d+'/labels.json')==labels
    for key,value in fp.items(): assert c[key]==value,(d,key)
    assert d.endswith(hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:10])
    assert h.epoch.tolist()==[1,2,3]
    best=h.loc[h.val_macro_f1.idxmax()]
    assert m['best_epoch']==int(best.epoch)
    near(m['best_val_macro_f1'],best.val_macro_f1)
    near(m['train_seconds'],h.train_seconds.sum())
    p=csv(d+'/predictions.csv.gz'); a=np.load(io.BytesIO(z.read(d+'/logits.npz')))
    assert len(p)==N and p.row_id.is_unique
    np.testing.assert_array_equal(p.row_id,test.row_id)
    np.testing.assert_array_equal(p.group_id,test.group_id)
    np.testing.assert_array_equal(p.y_true,test.y)
    assert np.isfinite(a['logits']).all()
    assert a['logits'].shape==(N,7)
    for key in ['row_id','y_true','y_pred']: np.testing.assert_array_equal(a[key],p[key])
    np.testing.assert_array_equal(a['logits'].argmax(1),p.y_pred)
    np.testing.assert_array_equal(p.correct,p.y_true==p.y_pred)
    np.testing.assert_array_equal(p.true_label,np.array(labels)[p.y_true])
    np.testing.assert_array_equal(p.pred_label,np.array(labels)[p.y_pred])
    q=a['logits'].astype(float); q=np.exp(q-q.max(1,keepdims=True)); q/=q.sum(1,keepdims=True)
    near(p.confidence,q.max(1),2e-6)
    near(p.entropy,-(q*np.log(np.maximum(q,1e-12))).sum(1),2e-6)
    ordered=np.sort(q,axis=1);near(p.margin,ordered[:,-1]-ordered[:,-2],2e-6)
    for metric,value in score(p.y_true,p.y_pred).items(): near(m[metric],value)
    sc=np.load(io.BytesIO(z.read(d+'/vad_scaler.npz')))
    near(p[['z_'+f for f in features]],(p[features]-sc['mean'])/sc['std'],3e-5)
    rawkey=(0,'full') if k[2] in ['full','none'] else (k[0],'visible')
    if rawkey in reference_features: near(p[features],reference_features[rawkey],1e-7)
    else: reference_features[rawkey]=p[features].to_numpy()
    for label,f in zip(labels,f1s(cm(p.y_true,p.y_pred))):
        perclass.append(dict(length=k[0],variant=k[1],scope=k[2],seed=k[3],label=label,f1=f))
    if k[1]=='gate':
        t=csv(d+'/mentalhealth_scalar_gate_instance_analysis.csv')
        pd.testing.assert_frame_equal(t.drop(columns='text'),p)
        if corpus is not None: np.testing.assert_array_equal(corpus.loc[t.row_id,'statement'],t.text)
        assert t.alpha.between(0,1).all()
        assert t.complement_max_error.ge(0).all() and t.complement_max_error.max()<1e-4
        assert t.delta_logit_mae.ge(0).all() and t.delta_logit_l2.ge(0).all()
        assert ((t.delta_logit_mae*np.sqrt(7))<=t.delta_logit_l2+1e-6).all()
        fixed=score(t.y_true,t.fixed_gate_pred)
        near(m['fixed_gate_macro_f1'],fixed['macro_f1'])
        changes=t.y_pred!=t.fixed_gate_pred
        np.testing.assert_array_equal(changes,t.gate_changes_prediction)
        near(m['gate_prediction_change_rate'],changes.mean())
        near(m['mean_delta_logit_mae'],t.delta_logit_mae.mean(),1e-7)
        qual=csv(d+'/qualitative_cases_purposive.csv'); chosen=[]
        for _,part in t.groupby('true_label'):
            for ix,rule in [(part.alpha.idxmin(),'minimum alpha within true label'),(part.alpha.idxmax(),'maximum alpha within true label')]:
                if ix not in [v[0] for v in chosen]: chosen.append((ix,rule))
        for ix in t.entropy.sort_values(ascending=False,kind='stable').index:
            if len(chosen)>=20: break
            if ix not in [v[0] for v in chosen]: chosen.append((ix,'highest remaining entropy'))
        assert qual.row_id.tolist()==t.loc[[v[0] for v in chosen],'row_id'].tolist()
        assert qual.selection_rule.tolist()==[v[1] for v in chosen]
        pd.testing.assert_frame_equal(qual.drop(columns='selection_rule'),t.loc[[v[0] for v in chosen]].reset_index(drop=True))
        X=np.column_stack([np.ones(N),rankdata(t.n_subwords_full),pd.get_dummies(t.true_label,drop_first=True).to_numpy(float)])
        vals=features+['entropy','margin','confidence','correct']
        ranks=np.column_stack([rankdata(t[f]) for f in ['alpha']+vals])
        resid=ranks-X@np.linalg.lstsq(X,ranks,rcond=None)[0]
        raw=np.corrcoef(ranks,rowvar=False)[0,1:]; partial=np.corrcoef(resid,rowvar=False)[0,1:]
        corr=csv(d+'/gate_correlations.csv')
        assert corr.feature.tolist()==vals
        near(corr.spearman,raw,2e-6);near(corr.partial_spearman_length_and_label,partial,2e-6)
        for row in corr.to_dict('records'): correlations.append(dict(length=k[0],scope=k[2],seed=k[3],**row))
        matched=csv(d+'/length_matched_instances.csv')
        assert matched.row_id.is_unique
        counts=matched.groupby(['length_bin','true_label']).size().unstack(fill_value=0)
        assert (counts.nunique(axis=1)==1).all()
        near(matched.alpha,t.set_index('row_id').loc[matched.row_id,'alpha'],1e-8)
        orig_class=csv(d+'/gate_by_class.csv').set_index('true_label')
        near(orig_class['mean'],t.groupby('true_label').alpha.mean())
        near(orig_class['std'],t.groupby('true_label').alpha.std())
        gates.append(dict(length=k[0],scope=k[2],seed=k[3],n=N,changed_n=int(changes.sum()),changed_pct=100*changes.mean(),
            macro_f1=m['macro_f1'],fixed_macro_f1=fixed['macro_f1'],delta_f1=m['macro_f1']-fixed['macro_f1'],
            beneficial_n=int(((t.y_pred==t.y_true)&(t.fixed_gate_pred!=t.y_true)).sum()),
            harmful_n=int(((t.y_pred!=t.y_true)&(t.fixed_gate_pred==t.y_true)).sum()),
            alpha_mean=t.alpha.mean(),alpha_std=t.alpha.std(),alpha_min=t.alpha.min(),alpha_max=t.alpha.max(),
            mean_delta_logit_mae=t.delta_logit_mae.mean(),max_complement_error=t.complement_max_error.max(),
            matched_n=len(matched),matched_pct=100*len(matched)/N))
    predictions[k]=p;configs.append(c);metrics.append(m)
    print('Verified',k,flush=True)
assert set(predictions)==expected
base_configs=[{k:v for k,v in c.items() if k not in ['seed','length','variant','scope']} for c in configs]
assert all(c==base_configs[0] for c in base_configs)
md=pd.DataFrame(metrics).sort_values(['length','variant','scope','seed'])
summary=md.groupby(['length','variant','scope']).agg(n_seeds=('seed','nunique'),macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),accuracy_mean=('accuracy','mean'),train_seconds_mean=('train_seconds','mean'))
supplied=csv(sp+'/metrics_summary.csv').set_index(['length','variant','scope'])
near(summary,supplied.loc[summary.index])
perseed=csv(sp+'/metrics_per_seed.csv').set_index(['length','variant','scope','seed'])
for r in metrics:
    for field in ['macro_f1','accuracy','weighted_f1','train_seconds','best_epoch']:
        near(perseed.loc[(r['length'],r['variant'],r['scope'],r['seed']),field],r[field])
paired=csv(sp+'/paired_bootstrap_per_seed.csv'); assert len(paired)==105
pairrows=[]
for r in paired.to_dict('records'):
    ka,kb=ast.literal_eval(r['model_a']),ast.literal_eval(r['model_b'])
    assert ka[-1]==kb[-1]==r['seed']
    a,b=predictions[ka],predictions[kb]
    near(r['delta_macro_f1'],score(b.y_true,b.y_pred)['macro_f1']-score(a.y_true,a.y_pred)['macro_f1'])
    assert r['test_groups']==a.group_id.nunique()==7628 and r['bootstrap_samples']==3000
    assert r['ci95_low']<r['ci95_high']
    pairrows.append({**r,'contrast':str(ka[:-1])+' -> '+str(kb[:-1]),'positive':int(r['delta_macro_f1']>0),
        'ci_positive':int(r['ci95_low']>0),'ci_negative':int(r['ci95_high']<0),'ci_crosses_zero':int(r['ci95_low']<=0<=r['ci95_high'])})
pairdf=pd.DataFrame(pairrows)
pa=pairdf.groupby('contrast').agg(mean_delta=('delta_macro_f1','mean'),sd_delta=('delta_macro_f1','std'),min_delta=('delta_macro_f1','min'),max_delta=('delta_macro_f1','max'),positive_seeds=('positive','sum'),ci_positive_seeds=('ci_positive','sum'),ci_negative_seeds=('ci_negative','sum'),ci_crosses_zero_seeds=('ci_crosses_zero','sum'))
# Independently repeat three representative conditional, paired group bootstraps.
chosen=[((32,'text','none',42),(32,'gate','full',42)),((128,'concat','visible',42),(128,'gate','visible',42)),((32,'text','none',42),(128,'text','none',42))]
bootchecks=[]
for ka,kb in chosen:
    a,b=predictions[ka].sort_values('row_id'),predictions[kb].sort_values('row_id')
    codes,groups=pd.factorize(a.group_id,sort=True);rng=np.random.default_rng(20261031);deltas=[]
    for _ in range(3000):
        counts=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));w=counts[codes]
        deltas.append(f1s(cm(b.y_true,b.y_pred,w)).mean()-f1s(cm(a.y_true,a.y_pred,w)).mean())
    lo,hi=np.quantile(deltas,[.025,.975])
    row=paired.loc[(paired.model_a==str(ka))&(paired.model_b==str(kb))].iloc[0]
    near([lo,hi],[row.ci95_low,row.ci95_high])
    bootchecks.append(dict(model_a=str(ka),model_b=str(kb),ci95_low=lo,ci95_high=hi))
gd=pd.DataFrame(gates);cd=pd.DataFrame(correlations);pc=pd.DataFrame(perclass)
gs=gd.groupby(['length','scope']).agg(changed_n_mean=('changed_n','mean'),changed_n_min=('changed_n','min'),changed_n_max=('changed_n','max'),changed_pct_mean=('changed_pct','mean'),delta_f1_mean=('delta_f1','mean'),delta_f1_min=('delta_f1','min'),delta_f1_max=('delta_f1','max'),mae_mean=('mean_delta_logit_mae','mean'),complement_max=('max_complement_error','max'),alpha_mean_min=('alpha_mean','min'),alpha_mean_max=('alpha_mean','max'))
cs=cd.groupby(['length','scope','feature']).agg(raw_min=('spearman','min'),raw_max=('spearman','max'),partial_min=('partial_spearman_length_and_label','min'),partial_max=('partial_spearman_length_and_label','max'))
trunc=csv(dp+'/truncation_by_class.csv');ta=trunc.groupby(['split','max_length'])[['n','truncated_n']].sum();ta['truncated_rate']=ta.truncated_n/ta.n
result=dict(archive=source.name,archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),files=len(names),verified_runs=len(metrics),test_n=N,test_groups=int(test.group_id.nunique()),split_sizes=manifest.split.value_counts().to_dict(),
    source_texts_and_labels_verified=corpus is not None,verified_near_duplicate_links=len(links),
    environment=base_configs[0]['environment'],shared_config=base_configs[0],best_epoch_counts=md.best_epoch.value_counts().to_dict(),
    duplicate_audit=audit,bootstrap_checks=bootchecks,training_hours=float(md.train_seconds.sum()/3600),elapsed_run_hours=float(md.elapsed_seconds.sum()/3600),
    max_exported_complement_error=float(gd.max_complement_error.max()),gate_prediction_change_range=[int(gd.changed_n.min()),int(gd.changed_n.max())],
    length_matching_n=sorted(gd.matched_n.unique().tolist()),limits='No retraining or checkpoint inference performed. Metrics, correlations, sample provenance and selected bootstrap intervals recalculated from archived predictions; complement errors checked as exported diagnostics. No fixed-gate logits or weights in ZIP.')
outputs={'metrics_verified.csv':md,'metrics_summary_verified.csv':summary,'paired_contrasts_summary.csv':pa,'paired_bootstrap_verified.csv':pairdf,'gate_intervention_per_seed.csv':gd,'gate_intervention_summary.csv':gs,'gate_correlations_per_seed.csv':cd,'gate_correlations_summary.csv':cs,'per_class_f1.csv':pc,'truncation_summary.csv':ta}
for name,frame in outputs.items():frame.to_csv(dest/name,index=not isinstance(frame.index,pd.RangeIndex))
(dest/'audit_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('\nCHECKS',json.dumps(result,ensure_ascii=False,indent=2))
print('\nPAIRED\n',pa.to_string())
print('\nGATES\n',gs.to_string())
print('\nGATE PER SEED\n',gd[['length','scope','seed','changed_n','delta_f1','alpha_mean','alpha_std','mean_delta_logit_mae']].to_string(index=False))
print('\nMATCH CORRELATIONS\n',cs.loc[(slice(None),slice(None),'log1p_matches'),:].to_string())
print('\nTRUNCATION\n',ta.to_string())
