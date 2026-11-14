"""Style-matched Braboszcz classifiers on the harmonized feature table."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from sklearn.dummy import DummyClassifier
from combined_analysis import load_data,clf,cell_weights,intra_blocked
from braboszcz_classifier_cohort import select_classifier_cohort

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/harmonized';OUT.mkdir(exist_ok=True)
STYLES=('HYT','VIP','ISY')

def fit_score(train,test,features,label):
    ytr=train[label].to_numpy();yte=test[label].to_numpy()
    if np.unique(ytr).size<2:return None
    w=1.0/train.groupby(label)[label].transform('size').to_numpy();w=w/w.mean()
    model=clf().fit(train[features].to_numpy(float),ytr,sc__sample_weight=w,clf__sample_weight=w)
    pred=model.predict(test[features].to_numpy(float))
    dummy=DummyClassifier(strategy='most_frequent').fit(train[features],ytr,sample_weight=w)
    metric=balanced_accuracy_score if np.unique(yte).size>1 else accuracy_score
    return metric(yte,pred),metric(yte,dummy.predict(test[features]))

def main():
    d,features=load_data();b=d[d.dataset.eq('braboszcz')].copy()
    # load_data already applies the equal-count cohort policy. Assert it here
    # so future changes cannot silently bring CTR/TM or unequal style counts back.
    counts=b[['subject','style']].drop_duplicates().groupby('style').subject.nunique().to_dict()
    if counts!={s:18 for s in STYLES}:raise RuntimeError(f'Expected 18 matched subjects/style; got {counts}')
    if b['style'].isna().any() or not set(b['style'].unique()).issubset(STYLES):raise RuntimeError('Non-meditation/control style in classifier data')

    rows=[]
    # Per-style held-subject-out meditation-state decoding.
    for style in STYLES:
        z=b[b['style'].eq(style)].copy()
        for subject in sorted(z.subject.unique()):
            te=z.subject.eq(subject);tr=~te
            out=fit_score(z.loc[tr],z.loc[te],features,'y')
            if out is not None:
                rows.append(dict(task='MED_vs_IMW2',style=style,subject=subject,model='logistic',bal_acc=out[0]))
                rows.append(dict(task='MED_vs_IMW2',style=style,subject=subject,model='dummy',bal_acc=out[1]))
    pd.DataFrame(rows).to_csv(OUT/'braboszcz_style_state_loso_subject.csv',index=False)
    s=pd.DataFrame(rows).groupby(['style','task','model']).bal_acc.agg(['mean','std','count']).reset_index()
    s.to_csv(OUT/'braboszcz_style_state_loso_summary.csv',index=False)

    # Cross-subject style decoding on MED-only epochs, equal style totals.
    med=b[b.condition.eq('MED')].copy();style_rows=[]
    tasks=[('HYT_vs_VIP',('HYT','VIP')),('HYT_vs_ISY',('HYT','ISY')),('VIP_vs_ISY',('VIP','ISY')),('HYT_VIP_ISY',STYLES)]
    for task,styles in tasks:
        z=med[med['style'].isin(styles)].copy();z['style_label']=pd.Categorical(z['style'],categories=styles).codes
        for subject in sorted(z.subject.unique()):
            te=z.subject.eq(subject);tr=~te
            out=fit_score(z.loc[tr],z.loc[te],features,'style_label')
            if out is not None:
                style_rows.append(dict(task=task,subject=subject,style=z.loc[te,'style'].iloc[0],model='logistic',bal_acc=out[0]))
                # Balanced-class dummy baseline; per-subject accuracy cannot
                # itself be balanced because each held-out participant has one style.
                style_rows.append(dict(task=task,subject=subject,style=z.loc[te,'style'].iloc[0],model='dummy',bal_acc=1.0/len(styles)))
    sr=pd.DataFrame(style_rows);sr.to_csv(OUT/'braboszcz_med_style_loso_subject.csv',index=False)
    style_summary=sr.groupby(['task','model','style']).bal_acc.agg(['mean','std','count']).reset_index()
    style_summary.to_csv(OUT/'braboszcz_med_style_loso_by_style.csv',index=False)
    style_summary.groupby(['task','model']).mean(numeric_only=True).reset_index().rename(columns={'mean':'balanced_acc'}).to_csv(OUT/'braboszcz_med_style_loso_summary.csv',index=False)

    # Within-subject buffered state decoding, separately by equally sized style.
    blocked=[]
    for style in STYLES:
        z=b[b['style'].eq(style)].copy();r=intra_blocked(z,features,'MED','IMW2')
        r['style']=style;blocked.append(r)
    pd.concat(blocked,ignore_index=True).to_csv(OUT/'braboszcz_style_intra_blocked.csv',index=False)
    print('selected subjects/style',counts)
    print('state LOSO\n',s.to_string(index=False))
    print('MED-only style LOSO\n',sr.groupby(['task','model']).bal_acc.agg(['mean','std','count']).to_string())
    print('blocked within-subject\n',pd.concat(blocked).groupby('style').bal_acc.agg(['mean','std','count']).to_string())
if __name__=='__main__':main()
