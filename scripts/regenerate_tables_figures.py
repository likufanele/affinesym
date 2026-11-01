"""Regenerate the manuscript tables and figures from the included text-free data.
Run from any directory: python scripts/regenerate_tables_figures.py
Requires numpy, pandas, matplotlib. Does not retrain models or need a GPU.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
DATA=P/'supplementary_data'
def read(name): return pd.read_csv(DATA/name)
def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#').replace('$',r'\$')
def table(name,caption,label,cols,heads,rows,small=True):
    t=[r'\begin{table}[H]',r'\caption{'+caption+r'\label{'+label+'}}',r'\centering',r'\begingroup',r'\ifmarkchanges\rowcolors{1}{white}{white}\fi']
    if small:t+=[r'\small']
    t+=[r'\begin{tabularx}{\textwidth}{'+cols+'}',r'\toprule',' & '.join(r'\textbf{'+h+'}' for h in heads)+r' \\',r'\midrule']
    t+=[' & '.join(map(str,row))+r' \\' for row in rows]
    t += [r'\bottomrule',r'\end{tabularx}',r'\endgroup',r'\end{table}']
    (P/'tables'/name).write_text('\n'.join(t)+'\n')

metrics=read('metrics_per_seed.csv'); summary=read('metrics_summary.csv')
manifest=read('split_manifest.csv');trunc=read('truncation_by_class.csv')
labels=json.loads((DATA/'audit.json').read_text())['labels']
counts=manifest.groupby(['status','split']).size().unstack()
rows=[[lab,f'{r.train:,}',f'{r.validation:,}',f'{r.test:,}',f'{r.sum():,}'] for lab,r in counts.iterrows()]
rows += [['Total','36,885','7,899','7,897','52,681']]
table('corpus.tex','Corpus distribution after assigning detected duplicate groups to a single partition.','tab:corpus','Xrrrr',['Source label','Train','Validation','Test','Total'],rows)
rows=[]
for lab in labels+['All classes']:
    part=trunc if lab=='All classes' else trunc[trunc.label==lab]
    rates=[]
    for split,L in [('all',32),('all',128),('test',32),('test',128)]:
        v=part[part.max_length==L]
        if split!='all':v=v[v.split==split]
        rates.append(f'{100*v.truncated_n.sum()/v.n.sum():.2f}')
    rows.append([lab]+rates)
table('lengths.tex','Percentage of sequences exceeding each subword limit, including special tokens. The full corpus contains 52,681 records and the grouped test split contains 7,897.','tab:lengths','Xrrrr',['Source label','Full: 32','Full: 128','Test: 32','Test: 128'],rows)
rows=[]
order=[('text','none','Text'),('concat','full','Concatenation'),('concat','visible','Concatenation'),('gate','full','Modulation'),('gate','visible','Modulation')]
for L in [32,128]:
 for v,s,n in order:
    r=summary[(summary.length==L)&(summary.variant==v)&(summary.scope==s)].iloc[0]
    rows.append([L,n,'--' if s=='none' else s,f'${r.macro_f1_mean:.4f}\\pm{r.macro_f1_std:.4f}$',f'{r.accuracy_mean:.4f}',f'{r.train_seconds_mean:.1f}'])
table('performance.tex','Test performance over 5 seeds. Macro-F1 is mean $\\pm$ sample standard deviation; accuracy and training seconds (3 epochs) are means. All runs used a Tesla T4.','tab:mainresults','rXlrrr',['Tokens','Model','VAD','Macro-F1','Accuracy','Train (s)'],rows)
rows=[]
for L in [32,128]:
 for v,s,n in order:
    rr=metrics[(metrics.length==L)&(metrics.variant==v)&(metrics.scope==s)].set_index('seed')
    rows.append([L,n+(' / '+s if s!='none' else '')]+[f'{rr.loc[seed,"macro_f1"]:.4f}' for seed in [7,13,21,42,73]])
table('per_seed.tex','Macro-F1 for every training seed. Each column uses the same test records; seeds are repeated training realizations, not independent test sets.','tab:perseed','rXrrrrr',['Tokens','Model / VAD','7','13','21','42','73'],rows)

auditdir=DATA
gd=pd.read_csv(auditdir/'gate_intervention_per_seed.csv');gs=pd.read_csv(auditdir/'gate_intervention_summary.csv')
rows=[]
for r in gs.itertuples():
 rows.append([r.length,r.scope,f'{r.changed_n_min}--{r.changed_n_max}',f'{r.changed_pct_mean:.3f}',f'{r.delta_f1_mean:+.6f}',f'{r.mae_mean:.4f}'])
table('intervention.tex','Intervention with fixed trained parameters: replace $\\alpha$ by 0.5 in both $\\alpha\\mathbf{h}$ and the scalar feature. Counts are ranges across seeds out of 7,897 test records; the other columns are seed means. $\\Delta F_1=F_1(\\mathrm{learned})-F_1(0.5)$.','tab:intervention','rlrrrr',['Tokens','VAD','Changed $n$','Changed (\\%)','$\\Delta F_1$','Logit MAE'],rows)
c=pd.read_csv(auditdir/'gate_correlations_per_seed.csv');cm=c[c.feature=='log1p_matches']
rows=[]
for seed in [7,13,21,42,73]:
 vals=[]
 for L,scope in [(32,'full'),(32,'visible'),(128,'full'),(128,'visible')]:
    r=cm[(cm.seed==seed)&(cm.length==L)&(cm.scope==scope)].iloc[0]
    vals.append(f'{r.spearman:+.3f} / {r.partial_spearman_length_and_label:+.3f}')
 rows.append([seed]+vals)
table('match_correlations.tex','Spearman correlation of $\\alpha$ with $\\log(1+m)$ for every seed. Each cell gives raw / partial correlation; partial correlations residualize ranks on full subword length and source-label indicators.','tab:matchcorrelations','rXXXX',['Seed','32 / full','32 / visible','128 / full','128 / visible'],rows)
g=read('gate_instances_all_runs.csv.gz')
pred=read('predictions_all_runs.csv.gz')
g=g[(g.length==128)&(g.scope=='visible')&(g.seed==42)]
pred=pred[(pred.length==128)&(pred.variant=='gate')&(pred.scope=='visible')&(pred.seed==42)]
t=g.merge(pred[['row_id','y_true','y_pred']],on='row_id',validate='one_to_one')
t['true_label']=np.array(labels)[t.y_true]
t['pred_label']=np.array(labels)[t.y_pred]
q=read('illustrative_cases.csv')
rows=[]
for lab,part in t.groupby('true_label'):
 rows.append([lab,len(part),f'${part.alpha.mean():.4f}\\pm{part.alpha.std():.4f}$',f'{part.correct.mean():.4f}'])
table('class_gate.tex','Class summaries for the declared illustrative run: 128 tokens, visible-span VAD, seed 42. The spread of $\\alpha$ is the sample standard deviation across instances, not across seeds. The last column is recall.','tab:classgate','Xrrr',['Source label','$n$','$\\alpha$','Recall'],rows)
rows=[]
for correct,part in t.groupby('correct'):
 rows.append(['Correct' if correct else 'Incorrect',len(part),f'${part.alpha.mean():.4f}\\pm{part.alpha.std():.4f}$',f'{part.confidence.mean():.4f}',f'{part.entropy.mean():.4f}'])
table('outcome.tex','Outcome summaries for the same illustrative run as Table~\\ref{tab:classgate}. Entropy is in natural-log units.','tab:correct','Xrrrr',['Outcome','$n$','$\\alpha$','Confidence','Entropy'],rows)
cr=c[(c.length==128)&(c.scope=='visible')&(c.seed==42)]
mapping={'mean_valence':'Mean valence','mean_arousal':'Mean arousal','mean_dominance':'Mean dominance','std_valence':'Valence SD','std_arousal':'Arousal SD','std_dominance':'Dominance SD','coverage':'Lexical coverage','log1p_matches':r'$\log(1+m)$','entropy':'Entropy','margin':'Probability margin','confidence':'Confidence','correct':'Correctness'}
rows=[[mapping[r.feature],f'{r.spearman:+.4f}',f'{r.partial_spearman_length_and_label:+.4f}'] for r in cr.itertuples()]
table('correlations.tex','Raw and adjusted associations for 128 tokens, visible-span VAD, seed 42. VAD features are direct inputs to the gate. All seeds and settings are available in the supplement.','tab:correlations','Xrr',['Variable','Spearman','Partial Spearman'],rows)
abbr={'Anxiety':'A','Bipolar':'B','Depression':'D','Normal':'N','Personality disorder':'P','Stress':'S','Suicidal':'U'}
rows=[]
for r in q.itertuples():
 rule='Min' if r.selection_rule.startswith('minimum') else 'Max' if r.selection_rule.startswith('maximum') else 'Entropy'
 rows.append([r.row_id,abbr[r.true_label],abbr[r.pred_label],f'{r.alpha:.4f}',f'{r.entropy:.4f}',rule])
table('cases.tex','The 20 purposively selected records from the same run as Tables~\\ref{tab:classgate}--\\ref{tab:correlations}. IDs are zero-based row positions in the original CSV, before removing missing texts. Min/Max select within source label; Entropy selects the highest remaining entropy. A: Anxiety; B: Bipolar; D: Depression; N: Normal; P: Personality disorder; S: Stress; U: Suicidal.','tab:examples','rccrrX',['Row ID','True','Pred.','$\\alpha$','Entropy','Rule'],rows)

pc=pd.read_csv(auditdir/'per_class_f1.csv');rows=[]
for lab in labels:
 vals=[]
 for v,s in [('text','none'),('concat','visible'),('gate','visible')]:
    a=pc[(pc.length==128)&(pc.variant==v)&(pc.scope==s)&(pc.label==lab)].f1
    vals.append(f'${a.mean():.4f}\\pm{a.std():.4f}$')
 rows.append([lab]+vals)
table('class_f1.tex','Per-class F1 at 128 tokens with equal visible text access; mean $\\pm$ sample standard deviation over 5 seeds.','tab:classf1','Xrrr',['Source label','Text','Concat. visible','Modul. visible'],rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
for correct,color in [(True,'#24779d'),(False,'#d28b37')]:
 a=t[t.correct==correct].alpha
 ax[0].hist(a,bins=np.linspace(.20,.73,45),alpha=.65,label='Correct' if correct else 'Incorrect',color=color)
ax[0].set(xlabel=r'Gate $\alpha$',ylabel='Count');ax[0].legend(frameon=False)
ax[1].scatter(t.log1p_matches,t.alpha,s=5,alpha=.12,color='#24779d',rasterized=True)
ax[1].set(xlabel='log(1 + VAD matches), unstandardized',ylabel=r'Gate $\alpha$')
fig.tight_layout();fig.savefig(P/'figures/diagnostic_run.pdf',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,L in zip(axes,[32,128]):
 for scope,color in [('full','#24779d'),('visible','#d28b37')]:
    d=cm[(cm.length==L)&(cm.scope==scope)].set_index('seed').loc[[7,13,21,42,73]]
    ax.plot(range(5),d.spearman,'o-',color=color,label=scope+', raw')
    ax.plot(range(5),d.partial_spearman_length_and_label,'s--',color=color,label=scope+', adjusted')
 ax.axhline(0,color='gray',lw=.7);ax.set(xticks=range(5),xticklabels=[7,13,21,42,73],xlabel='Training seed',title=f'{L} subword tokens',ylim=(-.4,.75))
axes[0].set_ylabel('Spearman correlation');axes[1].legend(frameon=False,fontsize=8,loc='upper right')
fig.tight_layout();fig.savefig(P/'figures/match_correlations_seeds.pdf',bbox_inches='tight');plt.close(fig)


print('Tables and figures regenerated from supporting data.')
