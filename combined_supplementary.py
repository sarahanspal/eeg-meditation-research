"""Subject-level uncertainty, learning curve, and feature-direction summaries."""
import numpy as np, pandas as pd
from scipy.stats import binomtest, false_discovery_control
from sklearn.metrics import balanced_accuracy_score
from combined_analysis import ROOT,OUT,load_data,clf,cell_weights

def bootstrap_ci(v,n=5000,seed=42):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    z=rng.choice(v,(n,len(v)),replace=True).mean(axis=1)
    return np.percentile(z,[2.5,97.5])
def signflip(v,n=20000,seed=42):
    d=np.asarray(v,float)-.5; observed=d.mean(); rng=np.random.default_rng(seed)
    null=(rng.choice([-1,1],size=(n,len(d)))*d).mean(axis=1)
    return (1+np.sum(np.abs(null)>=abs(observed)))/(n+1)
def stats():
    loso_path=OUT/'combined_loso_subject_results.csv'
    if loso_path.exists():
        ld=pd.read_csv(loso_path);rows=[]
        for (ds,scheme,model),g in ld.groupby(['dataset','scheme','model']):
            v=g.groupby('subject').bal_acc.mean().to_numpy();ci=bootstrap_ci(v)
            rows.append(dict(dataset=ds,scheme=scheme,model=model,n_subjects=len(v),mean=v.mean(),ci_low=ci[0],ci_high=ci[1],p_vs_chance=signflip(v) if model=='logistic' else np.nan))
        ls=pd.DataFrame(rows)
        if len(ls):
            ok=ls.p_vs_chance.notna();ls['p_fdr']=np.nan
            if ok.any():ls.loc[ok,'p_fdr']=false_discovery_control(ls.loc[ok,'p_vs_chance'].to_numpy(),method='bh')
        ls.to_csv(OUT/'combined_loso_statistics.csv',index=False)
        paired=[]
        for (ds,model),g in ld.groupby(['dataset','model']):
            wide=g.pivot_table(index='subject',columns='scheme',values='bal_acc',aggfunc='mean')
            for a,b in [('pooled','within'),('cross','pooled')]:
                if a not in wide or b not in wide:continue
                z=wide[[a,b]].dropna();delta=z[a].to_numpy()-z[b].to_numpy();ci=bootstrap_ci(delta)
                paired.append(dict(dataset=ds,model=model,comparison=f'{a}_minus_{b}',n_subjects=len(delta),mean_difference=delta.mean(),ci_low=ci[0],ci_high=ci[1],p_signflip=signflip(delta+.5)))
        ps=pd.DataFrame(paired)
        if len(ps):ps['p_fdr']=false_discovery_control(ps.p_signflip.to_numpy(),method='bh')
        ps.to_csv(OUT/'combined_loso_paired_statistics.csv',index=False)
    d=pd.read_csv(OUT/'combined_blocked_controls.csv')
    rows=[]
    for task,g in d.groupby('task'):
        v=g.groupby('subject').bal_acc.mean().to_numpy(); ci=bootstrap_ci(v)
        rows.append(dict(task=task,n_subjects=len(v),mean=v.mean(),ci_low=ci[0],ci_high=ci[1],p_vs_chance=signflip(v)))
    pd.DataFrame(rows).to_csv(OUT/'combined_control_statistics.csv',index=False)
    # Paired real-vs-file control on subjects with both values.
    p=d.pivot_table(index='subject',columns='task',values='bal_acc',aggfunc='mean').dropna()
    if {'MED_vs_IMW','think2_vs_think1'}.issubset(p.columns):
        delta=p.MED_vs_IMW-p.think2_vs_think1;ci=bootstrap_ci(delta.to_numpy())
        pd.DataFrame([dict(n_subjects=len(delta),mean_paired_difference=delta.mean(),ci_low=ci[0],ci_high=ci[1],p_signflip=signflip(delta.to_numpy()+.5))]).to_csv(OUT/'combined_control_paired_statistics.csv',index=False)
def learning_curve(ns=(3,6,12,24,36),draws=2,seed=42):
    d,feats=load_data();X=d[feats].to_numpy(float);y=d.y.to_numpy();s=d.subj.to_numpy();ds=d.dataset.to_numpy();subs=np.unique(s);rng=np.random.default_rng(seed)
    subj_meta=d[['subj','dataset','style']].drop_duplicates('subj').set_index('subj');rows=[]
    for held in subs:
        te=s==held
        if np.unique(y[te]).size<2:continue
        held_ds=ds[te][0]
        pool=np.array([x for x in subs if x!=held and subj_meta.loc[x,'dataset']==held_ds])
        for n in ns:
            if n>len(pool):continue
            if held_ds=='braboszcz':
                if n%3:continue
                n_each=n//3
                available={style:np.array([x for x in pool if subj_meta.loc[x,'style']==style]) for style in ('HYT','VIP','ISY')}
                if any(len(v)<n_each for v in available.values()):continue
            for rep in range(draws):
                if held_ds=='braboszcz':
                    chosen=np.concatenate([rng.choice(available[style],n_each,replace=False) for style in ('HYT','VIP','ISY')])
                else:chosen=rng.choice(pool,n,replace=False)
                tr=np.isin(s,chosen)
                if np.unique(y[tr]).size<2:continue
                train=d.iloc[np.flatnonzero(tr)]
                w=cell_weights(train);model=clf().fit(X[tr],y[tr],sc__sample_weight=w,clf__sample_weight=w)
                rows.append(dict(subject=held,n_train_subjects=n,draw=rep,dataset=held_ds,style_balanced=(held_ds=='braboszcz'),bal_acc=balanced_accuracy_score(y[te],model.predict(X[te]))))
    r=pd.DataFrame(rows);r.to_csv(OUT/'combined_learning_curve_subject.csv',index=False)
    per=r.groupby(['dataset','n_train_subjects','subject']).bal_acc.mean().reset_index()
    rng=np.random.default_rng(seed);summ=[]
    for (dataset,n),g in per.groupby(['dataset','n_train_subjects']):
        v=g.bal_acc.to_numpy();boot=rng.choice(v,(2000,len(v)),replace=True).mean(axis=1)
        summ.append(dict(dataset=dataset,n_train_subjects=n,mean=v.mean(),std=v.std(ddof=1),n_subjects=len(v),
                         ci_low=np.percentile(boot,2.5),ci_high=np.percentile(boot,97.5)))
    pd.DataFrame(summ).to_csv(OUT/'combined_learning_curve_summary.csv',index=False)
def sign_consistency():
    d,feats=load_data();rows=[]
    for f in feats:
        effects=[]
        for subj,g in d.groupby('subj'):
            a=pd.to_numeric(g.loc[g.y==1,f],errors='coerce').dropna();b=pd.to_numeric(g.loc[g.y==0,f],errors='coerce').dropna()
            if len(a)<5 or len(b)<5:continue
            pooled=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2+1e-12)
            if np.isfinite(pooled) and pooled>0:effects.append((a.mean()-b.mean())/pooled)
        if effects:
            pos=int(np.sum(np.asarray(effects)>0));n=len(effects)
            rows.append(dict(feature=f,n=n,frac_positive=pos/n,median_standardized_difference=np.median(effects),p_binomial=binomtest(pos,n,.5).pvalue))
    r=pd.DataFrame(rows).sort_values('p_binomial');r['p_fdr_bh']=false_discovery_control(r.p_binomial.to_numpy(),method='bh');r.to_csv(OUT/'combined_feature_sign_consistency.csv',index=False)
if __name__=='__main__':
    stats();sign_consistency();learning_curve()
    print('Wrote supplementary statistics, feature effects, and learning curves to',OUT)
