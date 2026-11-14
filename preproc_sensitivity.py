"""B4 analysis: does the real-vs-control gap and the theta direction survive preprocessing changes? Braboszcz matched 54-subject cohort only.
Variants: main (harmonized table), ep2, ep8 (2 s / 8 s epochs), noica, ref64 (average reference over all 64 channels before the 14-channel pick).
Per variant: blocked within-subject MED/IMW2 (real) and IMW2/IMW1 (file control) balanced accuracy, 5 blocked folds with the autocorrelation-based
guard, subjects need >=20 epochs/class (as in combined_analysis.intra_blocked); the paired real-minus-control difference over subjects with both;
within-dataset LOSO (logistic, class-balanced training weights, MED/IMW2); theta direction = fraction of subjects with positive MED-minus-IMW2
standardized frontal-theta effect (F3,F4,F7,F8 mean) and the median effect size. Feature schema, log transform and classifier = combined_analysis.
Output: results/harmonized/preproc_sensitivity.csv (+ preproc_sensitivity_subject.csv)
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import combined_analysis as ca
from combined_supplementary import bootstrap_ci, signflip
from braboszcz_classifier_cohort import selected_subjects

H = Path('results/harmonized'); VD = H/'preproc_variants'; FR = ['F3', 'F4', 'F7', 'F8']
_, FEATS = ca.load_data(); FEATS = list(FEATS)
LOGC = [f for f in FEATS if not f.startswith('asym') and not f.endswith('IAF')]

def load_variant(v):
    keep = set(selected_subjects())
    if v == 'main': T = pd.read_csv(H/'braboszcz_all_subjects_features.csv')
    else: T = pd.concat([pd.read_csv(p) for p in sorted((VD/v).glob('sub-*.csv'))], ignore_index=True)
    T = T[T.subject.isin(keep) & T.condition.isin(['MED', 'IMW2', 'IMW1'])].copy()
    for f in LOGC: T[f] = np.log(pd.to_numeric(T[f], errors='coerce').clip(lower=1e-12))
    T[FEATS] = T[FEATS].replace([np.inf, -np.inf], np.nan); T['dataset'] = 'braboszcz'; T['subj'] = 'braboszcz_' + T.subject; return T

def loso_within(T):
    D = T[T.condition.isin(['MED', 'IMW2'])].copy(); D['y'] = (D.condition == 'MED').astype(int); X = D[FEATS].to_numpy(float); y = D.y.to_numpy(); s = D.subj.to_numpy(); rows = []
    for h in np.unique(s):
        te = s == h
        if np.unique(y[te]).size < 2: continue
        w = ca.cell_weights(D.iloc[np.flatnonzero(~te)]); m = ca.clf().fit(X[~te], y[~te], sc__sample_weight=w, clf__sample_weight=w)
        rows.append(dict(subject=h, bal_acc=balanced_accuracy_score(y[te], m.predict(X[te]))))
    return pd.DataFrame(rows)

def theta_direction(T):
    eff = []
    for s, g in T[T.condition.isin(['MED', 'IMW2'])].groupby('subj'):
        a, b = g[g.condition == 'MED'], g[g.condition == 'IMW2']
        if len(a) < 5 or len(b) < 5: continue
        eff.append(np.mean([(a[f'{c}_theta'].mean()-b[f'{c}_theta'].mean())/np.sqrt((a[f'{c}_theta'].var(ddof=1)+b[f'{c}_theta'].var(ddof=1))/2) for c in FR]))
    e = np.asarray(eff); return len(e), float((e > 0).mean()), float(np.median(e))

rows, subj_rows = [], []
for v in ['main', 'ep2', 'ep8', 'noica', 'ref64']:
    if v != 'main' and not (VD/v).exists(): continue
    T = load_variant(v); n_sub = T.subject.nunique()
    real = ca.intra_blocked(T, FEATS, 'MED', 'IMW2').assign(task='real'); ctrl = ca.intra_blocked(T, FEATS, 'IMW2', 'IMW1').assign(task='control')
    pair = real.merge(ctrl, on='subject', suffixes=('_r', '_c')); dlt = (pair.bal_acc_r-pair.bal_acc_c).to_numpy(); lo, hi = bootstrap_ci(dlt)
    lo_s = loso_within(T); n_t, frac, med = theta_direction(T); ci_lo, ci_hi = bootstrap_ci(lo_s.bal_acc.to_numpy())
    rows.append(dict(variant=v, blocked_real=real.bal_acc.mean(), blocked_control=ctrl.bal_acc.mean(), difference=dlt.mean(), LOSO=lo_s.bal_acc.mean(), N=len(lo_s),
                     difference_ci_low=lo, difference_ci_high=hi, p_difference_signflip=signflip(dlt), LOSO_ci_low=ci_lo, LOSO_ci_high=ci_hi, p_LOSO_vs_chance=signflip(lo_s.bal_acc.to_numpy()),
                     N_blocked_real=len(real), N_blocked_control=len(ctrl), N_blocked_paired=len(pair), mean_guard_epochs=real.buffer_epochs.mean(),
                     frontal_theta_frac_subjects_positive=frac, frontal_theta_median_d=med, N_theta=n_t, n_subjects_in_table=n_sub))
    subj_rows += [real.assign(variant=v), ctrl.assign(variant=v), lo_s.assign(variant=v, task='LOSO')]
    print(v, {k: round(x, 3) if isinstance(x, float) else x for k, x in rows[-1].items() if k in ('blocked_real', 'blocked_control', 'difference', 'LOSO', 'N', 'N_blocked_paired', 'frontal_theta_frac_subjects_positive')}, flush=True)
pd.DataFrame(rows).to_csv(H/'preproc_sensitivity.csv', index=False); pd.concat(subj_rows).to_csv(H/'preproc_sensitivity_subject.csv', index=False)
