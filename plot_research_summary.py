"""Plot the accuracy ladder, negative controls, calibration, and learning curve."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
OUT=Path('results/harmonized')
RESULTS=Path('results')
def boot(v,n=3000,seed=42):
 v=np.asarray(v,float);r=np.random.default_rng(seed);b=r.choice(v,(n,len(v)),replace=True).mean(1);return v.mean(),np.percentile(b,[2.5,97.5])
def main():
 fig,ax=plt.subplots(2,2,figsize=(13,9));chance=.5
 p=RESULTS/'combined_loso_subject_results.csv'
 if p.exists():
  d=pd.read_csv(p);d=d[d.model=='logistic'];a=ax[0,0];schemes=['within','pooled','cross'];datasets=['braboszcz','brandmeyer'];x=np.arange(3);width=.34
  for j,ds in enumerate(datasets):
   vals=[];lo=[];hi=[]
   for sc in schemes:
    v=d[(d.dataset==ds)&(d.scheme==sc)].groupby('subject').bal_acc.mean().to_numpy();m,ci=boot(v,seed=42+j)
    vals.append(m);lo.append(m-ci[0]);hi.append(ci[1]-m)
   a.errorbar(x+(j-.5)*width,vals,yerr=[lo,hi],fmt='o',capsize=3,label=ds)
  a.axhline(chance,color='gray',ls='--');a.set_xticks(x,schemes);a.set_ylim(.25,1.02);a.set_ylabel('Balanced accuracy');a.set_title('A. Subject-held-out MED vs mind wandering');a.legend(frameon=False)
 p=RESULTS/'combined_blocked_controls.csv'
 if p.exists():
  d=pd.read_csv(p);a=ax[0,1];order=['MED_vs_IMW','think2_vs_think1','think2_first_vs_second_half'];labels=['MED vs IMW2','Think2 vs think1','Think2 first vs second half']
  for i,t in enumerate(order):
   v=d[d.task==t].groupby('subject').bal_acc.mean().to_numpy()
   if len(v):m,ci=boot(v);a.errorbar(m,i,xerr=[[m-ci[0]],[ci[1]-m]],fmt='o',capsize=3,color='tab:orange' if i else 'tab:blue')
  a.axvline(chance,color='gray',ls='--');a.set_yticks(range(3),labels);a.set_xlim(.35,1.03);a.set_xlabel('Balanced accuracy');a.set_title('B. Blocked within-subject controls')
 p=OUT/'calibration_decomposition_subject_reps.csv'
 if p.exists():
  d=pd.read_csv(p);a=ax[1,0];metrics=['base','bias_only','full_update'];names=['Population','Threshold only','Full update']
  for j,task in enumerate(d.task.unique()):
   g=d[d.task==task].groupby('subject')[metrics].mean();color=f'C{j}'
   for k,col in enumerate(metrics):
    v=g[col].dropna().to_numpy();m,ci=boot(v,seed=j*10+k);a.errorbar(k+(j-.5)*.15,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt='o',capsize=3,color=color,label=task if k==0 else None)
  a.axhline(chance,color='gray',ls='--');a.set_xticks(range(3),names);a.set_ylim(.25,1.02);a.set_ylabel('Balanced accuracy');a.set_title('C. Calibration decomposition');a.legend(frameon=False)
 p=RESULTS/'combined_learning_curve_subject.csv'
 if p.exists():
  d=pd.read_csv(p);a=ax[1,1]
  for ds,g in d.groupby('dataset'):
   per=g.groupby(['n_train_subjects','subject']).bal_acc.mean().reset_index()
   xs=[];ys=[];los=[];his=[]
   for n,z in per.groupby('n_train_subjects'):
    m,ci=boot(z.bal_acc);xs.append(n);ys.append(m);los.append(m-ci[0]);his.append(ci[1]-m)
   a.errorbar(xs,ys,yerr=[los,his],marker='o',capsize=3,label=ds)
  a.axhline(chance,color='gray',ls='--');a.set_ylim(.25,1.02);a.set_xlabel('Training subjects');a.set_ylabel('Balanced accuracy');a.set_title('D. LOSO learning curve');a.legend(frameon=False)
 fig.suptitle('What survives subject, dataset, and calibration controls',fontsize=15);fig.tight_layout();p=OUT/'research_summary.png';fig.savefig(p,dpi=180,bbox_inches='tight');print(p)
if __name__=='__main__':main()
