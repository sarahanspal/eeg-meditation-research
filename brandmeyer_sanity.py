"""B6: Brandmeyer sanity checks. No models are tuned here; this only audits the existing harmonized results.
(1) every probe's epochs share a fold in every CV used on Brandmeyer; (2) ratings per expertise group and per decoded label
vs the Table 1 values recorded in decode_brandmeyer_probes.py; (3) the LOSO null-result summary.
Outputs: results/harmonized/brandmeyer_probe_fold_check.csv, brandmeyer_ratings_by_group.csv, brandmeyer_ratings_by_decoded_label.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from config import RAW_PATHS
from decode_brandmeyer_probes import decode_probes, resolve_dir, brandmeyer_sessions, TABLE1_MEANS
H = Path('results/harmonized')
f = pd.read_csv(H/'brandmeyer_all_subjects_features.csv', usecols=['subject', 'condition', 'session', 'probe_id', 'sub_epoch', 'group'])
f = f[f.condition.isin(['MED', 'MW'])].copy()
f['probe_key'] = f.subject + '|' + f.session.astype(str) + '|' + f.probe_id.astype(str)   # probe_id restarts per session
# (1) probe -> fold integrity
per_probe = f.groupby('probe_key').agg(n_epochs=('sub_epoch', 'size'), n_subjects=('subject', 'nunique'), n_labels=('condition', 'nunique'))
assert (per_probe.n_subjects == 1).all() and (per_probe.n_labels == 1).all()
rows = []
for name, cv in [('LOSO (combined_analysis.evaluate: within/pooled/cross)', LeaveOneGroupOut()), ('GroupKFold-5 by subject (dataset identity probe)', GroupKFold(n_splits=5))]:
    leaks = 0
    for tr, te in cv.split(f, groups=f.subject):
        leaks += len(set(f.probe_key.iloc[tr]) & set(f.probe_key.iloc[te]))
    rows.append(dict(cv=name, n_probes=len(per_probe), probes_with_2_epochs=int((per_probe.n_epochs == 2).sum()), probes_with_1_epoch=int((per_probe.n_epochs == 1).sum()), probes_split_across_train_test=leaks))
pd.DataFrame(rows).to_csv(H/'brandmeyer_probe_fold_check.csv', index=False); print(pd.DataFrame(rows).to_string(index=False))
print('Note: no CV in the harmonized pipeline splits Brandmeyer epochs within a subject; both CVs group by subject, so probes cannot straddle folds.')
# (2) ratings
root = resolve_dir(Path(RAW_PATHS['brandmeyer']).parent, Path(RAW_PATHS['brandmeyer']).name.strip())
grp = dict(zip(*[pd.read_csv(root/'participants.tsv', sep='\t')[c] for c in ('participant_id', 'group')]))
subs = sorted(f.subject.unique()); P = []
for s in brandmeyer_sessions(root, subs):
    if s['events'] is None: continue
    d = decode_probes(s['events']); d['subject'] = s['subject']; d['session'] = s['session']; d['group'] = grp[s['subject']]; P.append(d)
P = pd.concat(P); ok = P[~P.ambiguous]
sub = ok.groupby(['subject', 'group'])[['q1', 'q2', 'q3']].mean().reset_index().rename(columns={'q1': 'Med', 'q2': 'MW', 'q3': 'Sleep'})
g = sub.groupby('group').agg(n_subjects=('subject', 'size'), Med=('Med', 'mean'), MW=('MW', 'mean'), Sleep=('Sleep', 'mean')).round(2)
paper = pd.DataFrame(TABLE1_MEANS, index=['Med_paper', 'MW_paper', 'Sleep_paper']).T
out = g.join(paper); out.to_csv(H/'brandmeyer_ratings_by_group.csv'); print('\nPer-subject mean ratings by expertise group (cohort of', len(subs), 'subjects) vs Table 1 values as recorded in repo:\n', out.to_string())
lab = P[P.label.isin(['MED', 'MW'])].groupby('label').agg(n_probes=('probe_id', 'size'), depth_Q1=('q1', 'mean'), mind_wandering_Q2=('q2', 'mean')).round(2)
lab.to_csv(H/'brandmeyer_ratings_by_decoded_label.csv'); print('\nBy decoded label (labels are defined from Q1 vs Q2, so the contrast is by construction):\n', lab.to_string())
print('\nexpertise in the 15-subject cohort:', sub.group.value_counts().to_dict(), '| classifier cohort (14, excl sub-017):', sub[sub.subject != 'sub-017'].group.value_counts().to_dict())
st = pd.read_csv('results/combined_loso_statistics.csv'); print('\nLOSO (logistic):\n', st[(st.dataset == 'brandmeyer') & (st.model == 'logistic')].round(3).to_string(index=False))
