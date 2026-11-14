"""Population, threshold-only, and full-model calibration decomposition."""
import numpy as np,pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from combined_analysis import ROOT,OUT,load_data,clf,cell_weights,estimate_buffer_epochs
from braboszcz_classifier_cohort import selected_subjects
OUT=ROOT/'results/harmonized';OUT.mkdir(exist_ok=True)

def calibration_split(units_by_class,frac,rep,n_rep,buffer):
    """Choose contiguous per-class calibration blocks and guard intervals."""
    cal,test={},{}
    for c,u in units_by_class.items():
        n=len(u);k=max(2,int(round(frac*n)))
        start=0 if n_rep==1 else int(round(rep/(n_rep-1)*(n-k)))
        excluded=np.arange(max(0,start-buffer),min(n,start+k+buffer))
        cal[c]=set(u[np.arange(start,start+k)])
        test[c]=set(np.delete(u,excluded))
    return cal,test

def run_task(d,feats,pos,neg,task,n_rep=5,frac=.05,cal_weight=20):
    d=d[d.condition.isin([pos,neg])].copy();d['y']=(d.condition==pos).astype(int)
    rows=[]
    for subj,g in d.groupby('subj'):
        g=g.sort_values(['condition','epoch_idx']).reset_index(drop=True)
        unit_col='probe_id' if 'probe_id' in g and g.probe_id.notna().any() else 'epoch_idx'
        # Calibration units remain together; order by within-class time.
        units={};order={}
        for c in (0,1):
            z=g[g.y==c]
            if unit_col=='cal_unit':
                u=z.groupby(unit_col)['_unit_order'].min().sort_values().index.to_numpy()
            elif unit_col=='probe_id':
                u=z.groupby(unit_col).t_start.min().sort_values().index.to_numpy()
            else:u=np.sort(z[unit_col].dropna().unique())
            units[c]=u
        if min(map(len,units.values()))<6:continue
        buffer=estimate_buffer_epochs(g,feats)
        if unit_col=='probe_id':buffer=max(1,int(np.ceil(buffer/2)))
        go=g
        other=d[d.subj.ne(subj)].copy()
        Xo=other[feats].to_numpy(float);yo=other.y.to_numpy();Xt=go[feats].to_numpy(float);yt=go.y.to_numpy()
        if np.unique(yo).size<2:continue
        wpop=cell_weights(other);pop=clf().fit(Xo,yo,sc__sample_weight=wpop,clf__sample_weight=wpop)
        for rep in range(n_rep):
            cal,test=calibration_split(units,frac,rep,n_rep,buffer)
            u=go[unit_col].to_numpy()
            cm=np.isin(u,list(cal[0]))&(yt==0) | (np.isin(u,list(cal[1]))&(yt==1))
            tm=np.isin(u,list(test[0]))&(yt==0) | (np.isin(u,list(test[1]))&(yt==1))
            if cm.sum()<2 or tm.sum()<2 or np.unique(yt[cm]).size<2 or np.unique(yt[tm]).size<2:continue
            Xc,yc=Xt[cm],yt[cm];Xte,yte=Xt[tm],yt[tm]
            base=balanced_accuracy_score(yte,pop.predict(Xte))
            sc=pop.decision_function(Xc);st=pop.decision_function(Xte)
            candidates=np.r_[-np.inf,np.unique(sc),np.inf]
            scores=[balanced_accuracy_score(yc,sc>t) for t in candidates]
            thr=candidates[int(np.argmax(scores))]
            bias=balanced_accuracy_score(yte,st>thr)
            training=pd.concat([other,go.iloc[np.flatnonzero(cm)]],ignore_index=True)
            weights=cell_weights(training);weights[-cm.sum():]*=cal_weight
            full=clf().fit(training[feats].to_numpy(float),training.y.to_numpy(),sc__sample_weight=weights,clf__sample_weight=weights)
            updated=balanced_accuracy_score(yte,full.predict(Xte))
            rows.append(dict(subject=subj,task=task,rep=rep,frac=frac,n_cal=int(cm.sum()),calibration_seconds=int(cm.sum()*4),n_test=int(tm.sum()),buffer_units=buffer,
                             base=base,bias_only=bias,full_update=updated,gain_bias=bias-base,gain_full=updated-base))
    return pd.DataFrame(rows)

def main():
    d,feats=load_data();parts=[]
    parts.append(run_task(d,feats,'MED','IMW2','MED_vs_IMW2'))
    # Brandmeyer lacks the think1/think2 negative label control; control is Braboszcz only.
    p=OUT/'braboszcz_all_subjects_features.csv'
    if p.exists():
        c=pd.read_csv(p);c=c[c.condition.isin(['IMW1','IMW2']) & c.subject.astype(str).isin(selected_subjects())].copy();c['dataset']='braboszcz';c['subj']='braboszcz_'+c.subject.astype(str)
        cf=sorted((set(c.columns)&set(feats)))
        parts.append(run_task(c,cf,'IMW2','IMW1','think2_vs_think1'))
    res=pd.concat(parts,ignore_index=True);res.to_csv(OUT/'calibration_decomposition_subject_reps.csv',index=False)
    per=res.groupby(['task','subject'])[['base','bias_only','full_update','gain_bias','gain_full']].mean().reset_index()
    per.to_csv(OUT/'calibration_decomposition_subject.csv',index=False)
    print(per.groupby('task')[['base','bias_only','full_update','gain_bias','gain_full']].agg(['mean','std','count']).round(3).to_string())
    # Null-adjusted gain among subjects with both task types.
    wide=per.pivot(index='subject',columns='task',values='gain_full').dropna()
    if not wide.empty:
        wide['null_adjusted_gain']=wide['MED_vs_IMW2']-wide['think2_vs_think1'];wide.reset_index().to_csv(OUT/'calibration_null_adjusted_gain.csv',index=False)
if __name__=='__main__':main()
