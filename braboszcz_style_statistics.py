"""Subject-clustered uncertainty summaries for matched Braboszcz style models."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest
from scipy.stats import false_discovery_control
OUT=Path(__file__).resolve().parent/'results/harmonized'

def summarize(g, chance, n_boot=5000, seed=42):
    v=np.asarray(g,float);rng=np.random.default_rng(seed);b=rng.choice(v,(n_boot,len(v)),replace=True).mean(1)
    d=v-chance;null=rng.choice([-1,1],(20000,len(d)))*d
    p=(1+(np.abs(null.mean(1))>=abs(d.mean())).sum())/20001
    return dict(n_subjects=len(v),mean=v.mean(),ci_low=np.percentile(b,2.5),ci_high=np.percentile(b,97.5),p_vs_chance=p)

def main():
    state=pd.read_csv(OUT/'braboszcz_style_state_loso_subject.csv')
    blocked=pd.read_csv(OUT/'braboszcz_style_intra_blocked.csv')
    style=pd.read_csv(OUT/'braboszcz_med_style_loso_subject.csv')
    rows=[]
    for (s,m),g in state.groupby(['style','model']):rows.append(dict(analysis='MED_vs_IMW2_LOSO',style=s,model=m,**summarize(g.bal_acc,.5,seed=10+len(rows))))
    for s,g in blocked.groupby('style'):rows.append(dict(analysis='MED_vs_IMW2_blocked_intra',style=s,model='logistic',**summarize(g.bal_acc,.5,seed=100+len(rows))))
    for (task,model,s),g in style.groupby(['task','model','style']):
        k=3 if task=='HYT_VIP_ISY' else 2
        rows.append(dict(analysis=f'{task}_MED_LOSO',style=s,model=model,**summarize(g.bal_acc,1/k,seed=200+len(rows))))
    out=pd.DataFrame(rows);out['p_fdr']=np.nan
    mask=out.model.eq('logistic')
    out.loc[mask,'p_fdr']=false_discovery_control(out.loc[mask,'p_vs_chance'].to_numpy(),method='bh')
    out.to_csv(OUT/'braboszcz_style_classifier_statistics.csv',index=False)
    # Correct the style-task baselines to the balanced-class chance level.
    style['classes']=style.task.map(lambda t:3 if t=='HYT_VIP_ISY' else 2)
    style.loc[style.model.eq('dummy'),'bal_acc']=1/style.loc[style.model.eq('dummy'),'classes']
    style_summary=style.groupby(['task','model']).bal_acc.agg(['mean','std','count']).reset_index()
    style_summary.to_csv(OUT/'braboszcz_med_style_loso_summary.csv',index=False)
    # Macro recall gives every style the same weight; bootstrap subjects within style.
    rows=[];rng=np.random.default_rng(42)
    for (task,model),g in style.groupby(['task','model']):
        by=[g[g['style'].eq(s)].bal_acc.to_numpy() for s in sorted(g['style'].unique())]
        b=np.vstack([rng.choice(v,(5000,len(v)),replace=True).mean(1) for v in by]).mean(0)
        rows.append(dict(task=task,model=model,n_styles=len(by),mean_macro_bal_acc=np.mean([v.mean() for v in by]),
                         ci_low=np.percentile(b,2.5),ci_high=np.percentile(b,97.5)))
    pd.DataFrame(rows).to_csv(OUT/'braboszcz_med_style_loso_macro_summary.csv',index=False)
    print(out.groupby(['analysis','model'])[['mean','ci_low','ci_high','p_fdr']].first().to_string())
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
