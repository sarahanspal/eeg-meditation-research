"""Combined Braboszcz/Brandmeyer MED vs mind-wandering analysis.
Run from repository root: python3 combined_analysis.py
Writes subject-level results and summaries under results/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from braboszcz_classifier_cohort import select_classifier_cohort, selected_subjects

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
META={"subject","style","group","condition","epoch_idx","probe_id","sub_epoch","t_start","q1","q2","q3","label","dataset","subj","y","experience_group","years_of_practice","age","session","bad_channels","ica_excluded","artifact_detection"}

def load_data():
    bh=ROOT/'results/harmonized/braboszcz_all_subjects_features.csv'
    mh=ROOT/'results/harmonized/brandmeyer_all_subjects_features.csv'
    b=pd.read_csv(bh if bh.exists() else ROOT/'braboszcz_all_subjects_features.csv')
    m=pd.read_csv(mh if mh.exists() else ROOT/'brandmeyer_all_subjects_features.csv')
    b=b[b.condition.isin(['MED','IMW2','IMW'])].copy()
    b,_=select_classifier_cohort(b)
    m=m[m.condition.isin(['MED','MW'])].copy()
    feats=sorted((set(b.columns)&set(m.columns))-META)
    if not feats: raise ValueError('No common features found')
    b['dataset']='braboszcz'; m['dataset']='brandmeyer'
    d=pd.concat([b,m],ignore_index=True,sort=False)
    d['y']=(d.condition=='MED').astype(int)
    d['subj']=d.dataset+'_'+d.subject.astype(str)
    # Apply the established project transform, but imputation is fitted inside each training fold.
    logcols=[f for f in feats if not f.startswith('asym') and not f.endswith('IAF')]
    for f in logcols:
        vals=pd.to_numeric(d[f],errors='coerce')
        d[f]=np.log(vals.clip(lower=1e-12))
    d[feats]=d[feats].replace([np.inf,-np.inf],np.nan)
    return d,feats

def clf():
    return Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),
                     ('vt',VarianceThreshold(0.0)),('sc',StandardScaler()),
                     ('clf',LogisticRegression(C=.1,max_iter=2000,random_state=42,solver="liblinear"))])

def cell_weights(d):
    w=1.0/d.groupby(['dataset','y']).y.transform('size').to_numpy()
    return w/w.mean()

def evaluate(d,feats):
    X=d[feats].to_numpy(float); y=d.y.to_numpy(); s=d.subj.to_numpy(); ds=d.dataset.to_numpy()
    rows=[]
    for subj in np.unique(s):
        te=s==subj; dataset=ds[te][0]
        if np.unique(y[te]).size<2: continue
        schemes={'within':(~te)&(ds==dataset),'pooled':~te,'cross':ds!=dataset}
        for scheme,tr in schemes.items():
            if np.unique(y[tr]).size<2: continue
            w=cell_weights(d.iloc[np.flatnonzero(tr)]);model=clf().fit(X[tr],y[tr],sc__sample_weight=w,clf__sample_weight=w)
            pred=model.predict(X[te])
            rows.append(dict(subject=subj,dataset=dataset,scheme=scheme,model='logistic',n_test=int(te.sum()),
                             bal_acc=balanced_accuracy_score(y[te],pred)))
            w=cell_weights(d.iloc[np.flatnonzero(tr)])
            dummy=DummyClassifier(strategy='most_frequent').fit(X[tr],y[tr],sample_weight=w)
            rows.append(dict(subject=subj,dataset=dataset,scheme=scheme,model='dummy',n_test=int(te.sum()),
                             bal_acc=balanced_accuracy_score(y[te],dummy.predict(X[te]))))
    return pd.DataFrame(rows)

def dataset_probe(d,feats):
    # Subject-grouped CV; weight the two dataset cells equally in every fit.
    y=(d.dataset=='brandmeyer').astype(int).to_numpy(); groups=d.subj.to_numpy();X=d[feats].to_numpy(float)
    n=min(5,len(np.unique(groups)))
    if n<2: raise ValueError('Need at least two subjects for dataset probe')
    pred=np.full(len(y),-1,dtype=int)
    cv=GroupKFold(n_splits=n)
    for tr,te in cv.split(X,y,groups):
        dtr=d.iloc[tr]; w=1.0/dtr.groupby('dataset').dataset.transform('size').to_numpy();w=w/w.mean()
        model=clf().fit(X[tr],y[tr],sc__sample_weight=w,clf__sample_weight=w);pred[te]=model.predict(X[te])
    return pd.DataFrame([dict(metric='dataset_identity_bal_acc',value=balanced_accuracy_score(y,pred),
                              n_subjects=len(np.unique(groups)),n_epochs=len(y))])

def blocked_folds_by_class(y,k=5,buffer=3):
    idx={c:np.flatnonzero(y==c) for c in np.unique(y)}
    k=min(k,min(map(len,idx.values())))
    if k<2:return
    for i in range(k):
        te=[]; excl=[]
        for c,ix in idx.items():
            e=np.linspace(0,len(ix),k+1).astype(int)
            te.extend(ix[e[i]:e[i+1]])
            excl.extend(ix[max(0,e[i]-buffer):min(len(ix),e[i+1]+buffer)])
        te=np.asarray(te,dtype=int)
        tr=np.setdiff1d(np.arange(len(y)),np.asarray(excl,dtype=int))
        yield tr,te

def estimate_buffer_epochs(g,feats,max_lag=50):
    # Pick the subject-specific guard interval from autocorrelation in the
    # strongest features; require three successive lags inside the 95% band.
    lags=[]
    for _,block in g.groupby('condition'):
        order=['session','t_start'] if {'session','t_start'}.issubset(block.columns) else ['epoch_idx']
        block=block.sort_values(order);n=len(block)
        if n<12:continue
        var=block[feats].var(numeric_only=True).sort_values(ascending=False)
        for f in var.head(min(10,len(var))).index:
            x=pd.to_numeric(block[f],errors='coerce').to_numpy(float);x=x[np.isfinite(x)]
            if len(x)<12 or np.std(x)==0:continue
            limit=min(max_lag,len(x)//4);thresholds=[];ac=[]
            for lag in range(1,limit+1):
                a,b=x[:-lag],x[lag:]
                ac.append(np.corrcoef(a,b)[0,1] if np.std(a)>0 and np.std(b)>0 else 0.)
                thresholds.append(1.96/np.sqrt(len(a)))
            decor=limit
            for j in range(max(0,len(ac)-2)):
                if all(np.isfinite(ac[t]) and abs(ac[t])<thresholds[t] for t in range(j,j+3)):
                    decor=j+1;break
            lags.append(decor)
    return max(1,int(np.percentile(lags,95))) if lags else 3

def intra_blocked(d,feats,pos,neg,col='condition',k=5,buffer=None,min_n=20):
    rows=[]
    for subj,g in d[d[col].isin([pos,neg])].groupby('subj'):
        g=g.sort_values([col,'epoch_idx']).reset_index(drop=True)
        y=(g[col]==pos).astype(int).to_numpy()
        if min((y==0).sum(),(y==1).sum())<min_n:continue
        guard=estimate_buffer_epochs(g,feats) if buffer is None else int(buffer)
        X=g[feats].to_numpy(float); scores=[]
        for tr,te in blocked_folds_by_class(y,k,guard):
            if np.unique(y[tr]).size<2 or np.unique(y[te]).size<2:continue
            model=clf().fit(X[tr],y[tr]); scores.append(balanced_accuracy_score(y[te],model.predict(X[te])))
        if scores:rows.append(dict(subject=subj,bal_acc=float(np.mean(scores)),folds=len(scores),buffer_epochs=guard))
    return pd.DataFrame(rows)

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    d,feats=load_data()
    # Cohort/label flow and data compatibility diagnostics
    flow=d.groupby(['dataset','subj','condition']).size().rename('epochs').reset_index()
    flow.to_csv(OUT/'combined_cohort_flow.csv',index=False)
    res=evaluate(d,feats)
    res.to_csv(OUT/'combined_loso_subject_results.csv',index=False)
    summary=res.groupby(['dataset','scheme','model']).bal_acc.agg(['mean','std','count']).reset_index()
    summary.to_csv(OUT/'combined_loso_summary.csv',index=False)
    probe=dataset_probe(d,feats); probe.to_csv(OUT/'combined_dataset_probe.csv',index=False)
    # Braboszcz within-person blocked MED/IMW task and file identity control.
    bz=d[d.dataset=='braboszcz'].copy()
    real=intra_blocked(bz,feats,'MED','IMW2' if 'IMW2' in set(bz.condition) else 'IMW')
    real['task']='MED_vs_IMW'
    harmonized_b=ROOT/'results/harmonized/braboszcz_all_subjects_features.csv'
    if harmonized_b.exists():
        bc=pd.read_csv(harmonized_b);bc=bc[bc.condition.isin(['IMW1','IMW2'])].copy()
        bc=bc[bc.subject.astype(str).isin(selected_subjects())]
        bc['dataset']='braboszcz';bc['subj']='braboszcz_'+bc.subject.astype(str)
        fc=intra_blocked(bc,feats,'IMW2','IMW1');fc['task']='think2_vs_think1'
        drift=bc[bc.condition=='IMW2'].copy();med=drift.groupby('subj').epoch_idx.transform('median')
        drift['half']=np.where(drift.epoch_idx<med,'H1','H2')
        dr=intra_blocked(drift,feats,'H2','H1',col='half');dr['task']='think2_first_vs_second_half'
        real=pd.concat([real,fc,dr],ignore_index=True)
    real.to_csv(OUT/'combined_blocked_controls.csv',index=False)
    print(f'Common features: {len(feats)}; subjects: {d.subj.nunique()}; rows: {len(d)}')
    print('\nLOSO summary\n',summary.to_string(index=False))
    print('\nDataset probe\n',probe.to_string(index=False))
    print('\nBlocked within-subject controls\n',real.groupby('task').bal_acc.agg(['mean','std','count']).to_string())
    print(f'\nWrote outputs to {OUT}')
if __name__=='__main__':main()
