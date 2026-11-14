"""Deterministic equal-count Braboszcz meditation-style classifier cohort."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
FEATURES=ROOT/'results/harmonized/braboszcz_all_subjects_features.csv'
OUT=ROOT/'results/harmonized'
SEED=20261111
STYLE_MAP={'htr':'HYT','himalayan yoga tradition':'HYT','hyt':'HYT',
           'vip':'VIP','vipassana':'VIP',
           'sny':'ISY','isha shoonya':'ISY','isy':'ISY'}

def style_subject_map(df):
    if 'style' in df:
        raw=df['style']
    elif 'group' in df:
        raw=df['group']
    else:
        raise ValueError('Braboszcz feature table must include group/style metadata')
    mapped=raw.astype('string').str.strip().str.lower().map(STYLE_MAP)
    return pd.DataFrame({'subject':df.subject.astype(str),'style':mapped}).drop_duplicates()

def select_classifier_cohort(df,save=True):
    """Keep the same minimum eligible subject count in each meditation style.

    Eligibility requires both MED and IMW2 epochs. CTR/TM/unknown groups are
    excluded. If styles differ in size, sample the larger style with a fixed
    seed; currently only ISY needs downsampling (20 to 18).
    """
    sm=style_subject_map(df)
    d=df.copy();d['style']=d.subject.astype(str).map(sm.set_index('subject')['style'])
    pairs=d[d.condition.isin(['MED','IMW2'])].groupby(['style','subject']).condition.nunique()
    eligible=pairs[pairs.eq(2)].rename('has_both').reset_index()
    counts=eligible.groupby('style')['subject'].nunique().reindex(['HYT','VIP','ISY'],fill_value=0)
    if (counts==0).any():raise ValueError(f'Missing eligible meditation style: {counts.to_dict()}')
    n=int(counts.min());rng=np.random.default_rng(SEED);chosen={}
    for style in ('HYT','VIP','ISY'):
        subs=np.array(sorted(eligible.loc[eligible['style'].eq(style),'subject'].astype(str)))
        chosen[style]=sorted(rng.choice(subs,size=n,replace=False).tolist()) if len(subs)>n else subs.tolist()
    rows=[]
    for style in ('HYT','VIP','ISY'):
        keep=set(chosen[style])
        for subject in sorted(eligible.loc[eligible['style'].eq(style),'subject'].astype(str)):
            rows.append(dict(subject=subject,style=style,eligible=True,in_classifier_cohort=subject in keep,
                             reason='selected' if subject in keep else 'equal-count downsample',seed=SEED,target_subjects_per_style=n))
    cohort=pd.DataFrame(rows)
    if save:
        OUT.mkdir(parents=True,exist_ok=True)
        cohort.to_csv(OUT/'braboszcz_classifier_cohort.csv',index=False)
    selected=set(cohort.loc[cohort.in_classifier_cohort,'subject'])
    result=d[d.subject.astype(str).isin(selected)].copy()
    result=result[result.condition.isin(['MED','IMW2','IMW1'])]
    if result.group.astype(str).str.lower().eq('ctr').any():raise AssertionError('CTR leaked into classifier cohort')
    return result,cohort

def selected_subjects():
    cohort_path=OUT/'braboszcz_classifier_cohort.csv'
    if cohort_path.exists():cohort=pd.read_csv(cohort_path)
    else:
        df=pd.read_csv(FEATURES)
        _,cohort=select_classifier_cohort(df)
    return set(cohort.loc[cohort.in_classifier_cohort,'subject'].astype(str))

if __name__=='__main__':
    data=pd.read_csv(FEATURES)
    selected,cohort=select_classifier_cohort(data)
    print('eligible style counts:',cohort.groupby('style').eligible.sum().to_dict())
    print('selected style counts:',cohort[cohort.in_classifier_cohort].groupby('style').subject.nunique().to_dict())
    print('classifier epochs:',len(selected),'subjects:',selected.subject.nunique())
