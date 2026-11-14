"""Preprocessing-independent feature/model specification curve on fixed LOSO splits.
The raw re-extraction fixes notch/reference/ICA/epoch length; this explores
feature families and regularization without changing the held-out subjects.
"""
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score
from combined_analysis import ROOT,OUT,load_data,cell_weights,clf
OUT=ROOT/'results/harmonized'
def main():
 d,allf=load_data();families={
  'all':allf,
  'no_gamma':[f for f in allf if 'gamma' not in f.lower()],
  'no_asymmetry':[f for f in allf if not f.startswith('asym_')],
  'no_IAF':[f for f in allf if not f.endswith('IAF')],
  'power_only':[f for f in allf if 'ratio_' not in f and not f.startswith('asym_') and not f.endswith('IAF')],
  'power_plus_IAF':[f for f in allf if 'ratio_' not in f and not f.startswith('asym_')],
  'no_gamma_no_ratio':[f for f in allf if 'gamma' not in f.lower() and 'ratio_' not in f],
 }
 Xall=d[allf].to_numpy(float);y=d.y.to_numpy();s=d.subj.to_numpy();datasets=d.dataset.to_numpy();rows=[]
 for subject in np.unique(s):
  te=s==subject;ds=datasets[te][0];tr=~te
  if np.unique(y[te]).size<2 or np.unique(y[tr]).size<2:continue
  w=cell_weights(d.iloc[np.flatnonzero(tr)])
  for family,features in families.items():
   ids=np.array([allf.index(f) for f in features]);X=Xall[:,ids]
   for C in (.01,.1,1.0):
    p=clf();p.named_steps['clf'].set_params(C=C)
    p.fit(X[tr],y[tr],sc__sample_weight=w,clf__sample_weight=w)
    rows.append(dict(subject=subject,dataset=ds,features=family,n_features=len(features),C=C,
                     bal_acc=balanced_accuracy_score(y[te],p.predict(X[te]))))
 r=pd.DataFrame(rows);r.to_csv(OUT/'feature_specification_subject.csv',index=False)
 per=r.groupby(['dataset','features','C','subject']).bal_acc.mean().reset_index();rng=np.random.default_rng(42);out=[]
 for (ds,f,C),g in per.groupby(['dataset','features','C']):
  v=g.bal_acc.to_numpy();b=rng.choice(v,(3000,len(v)),replace=True).mean(1)
  out.append(dict(dataset=ds,features=f,C=C,n_subjects=len(v),mean=v.mean(),ci_low=np.percentile(b,2.5),ci_high=np.percentile(b,97.5)))
 pd.DataFrame(out).to_csv(OUT/'feature_specification_summary.csv',index=False)
 print(pd.DataFrame(out).groupby(['dataset','features']).mean(numeric_only=True).to_string())
if __name__=='__main__':main()
