"""Cohort reconciliation: source metadata, available raw runs, and prior feature-table cohorts."""
from pathlib import Path
import pandas as pd
from config import RAW_PATHS
from decode_brandmeyer_probes import resolve_dir,brandmeyer_sessions
import scipy.io
OUT=Path('results/harmonized');OUT.mkdir(parents=True,exist_ok=True)
rows=[]
# Braboszcz: current study cohort vs paired MED/IMW and file-control availability.
br=Path(RAW_PATHS['braboszcz']);old=pd.read_csv('braboszcz_all_subjects_features.csv',usecols=['subject']).subject.astype(str).unique();pt=pd.read_csv(br/'participants.tsv',sep='\t');
matched_path=OUT/'braboszcz_classifier_cohort.csv'
matched=set(pd.read_csv(matched_path).query('in_classifier_cohort').subject.astype(str)) if matched_path.exists() else set()
for s in old:
 eeg=br/s/'eeg';available={t:(eeg/f'{s}_task-{t}_eeg.bdf').exists() for t in ('med2','think2','think1')}
 rows.append(dict(dataset='braboszcz',subject=s,in_existing_feature_cohort=True,in_classifier_cohort=s in matched,in_participants=s in set(pt.participant_id),n_sessions=int(sum(available.values())),has_primary_pair=available['med2'] and available['think2'],has_file_control=available['think1'],experience_metadata=bool(s in set(pt.participant_id))))
# Brandmeyer: source participants, raw/events, prior feature table.
bm=resolve_dir(Path(RAW_PATHS['brandmeyer']).parent,Path(RAW_PATHS['brandmeyer']).name.strip());bp=pd.read_csv(bm/'participants.tsv',sep='\t');oldbm=set(pd.read_csv('brandmeyer_all_subjects_features.csv',usecols=['subject']).subject.astype(str).unique());sessions=brandmeyer_sessions(bm,bp.participant_id);counts={}
for r in sessions:
 if r['eeg'] is not None:counts.setdefault(r['subject'],[0,0])[0]+=1
 if r['events'] is not None:counts.setdefault(r['subject'],[0,0])[1]+=1
for s in bp.participant_id.astype(str):
 n,e=counts.get(s,[0,0]);rows.append(dict(dataset='brandmeyer',subject=s,in_existing_feature_cohort=s in oldbm,in_classifier_cohort=False,in_participants=True,n_sessions=n,has_primary_pair=n>0 and e>0,has_file_control=False,experience_metadata=True))
# SKY: 40 main EDF files with timing metadata.
sk=Path(RAW_PATHS['sky']);mat=scipy.io.loadmat(sk/'subjects_epochs.mat');tim=mat['timings'];meta=pd.read_csv(sk/'SubjectMetaData.csv');
for i in range(tim.shape[0]):
 s=f'sky-{i+1:02d}';rows.append(dict(dataset='sky',subject=s,in_existing_feature_cohort=False,in_classifier_cohort=False,in_participants=(i+1 in set(meta['S.No.'])),n_sessions=int((sk/f'{i+1:02d}.edf').exists()),has_primary_pair=True,has_file_control=False,experience_metadata=(i+1 in set(meta['S.No.']))))
out=pd.DataFrame(rows);out.to_csv(OUT/'cohort_reconciliation.csv',index=False)
print(out.groupby('dataset').agg(n_raw_or_metadata=('subject','nunique'),n_prior_feature_cohort=('in_existing_feature_cohort','sum'),n_classifier_cohort=('in_classifier_cohort','sum'),n_primary_pair=('has_primary_pair','sum'),n_file_control=('has_file_control','sum'),n_experience_metadata=('experience_metadata','sum')).to_string())
