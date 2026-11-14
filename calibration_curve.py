"""C2 support: calibration gain as a function of calibration data in SECONDS (4 s per epoch), MED/IMW2 vs the think2/think1 control.
Reuses calibration_decomposition.run_task unchanged (same contiguous calibration blocks, guard intervals, 5 repetitions, calibration weight 20,
same logistic model). The existing run used one amount (frac=0.05, ~56 s). Here frac in {0.02,0.05,0.10,0.20,0.30}.
Unlike calibration_decomposition.main(), BOTH tasks use the log-transformed features (main() feeds the control the raw table).
Output: results/harmonized/calibration_curve_subject_reps.csv (new file) and calibration_curve_summary.csv
Usage: venv/bin/python calibration_curve.py [--limit-fracs ...] [--jobs J]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from combined_analysis import load_data
from calibration_decomposition import run_task
from braboszcz_classifier_cohort import selected_subjects
from combined_supplementary import bootstrap_ci, signflip
H = Path('results/harmonized'); FRACS = [0.02, 0.05, 0.10, 0.20, 0.30]

def job(task, frac):
    d, feats = load_data()
    if task == 'MED_vs_IMW2': return run_task(d, feats, 'MED', 'IMW2', task, frac=frac)
    _, f = load_data(); c = pd.read_csv(H/'braboszcz_all_subjects_features.csv'); c = c[c.condition.isin(['IMW1', 'IMW2']) & c.subject.astype(str).isin(selected_subjects())].copy()
    c['dataset'] = 'braboszcz'; c['subj'] = 'braboszcz_' + c.subject.astype(str)
    for col in [x for x in f if not x.startswith('asym') and not x.endswith('IAF')]: c[col] = np.log(pd.to_numeric(c[col], errors='coerce').clip(lower=1e-12))
    c[list(f)] = c[list(f)].replace([np.inf, -np.inf], np.nan)
    return run_task(c, sorted(set(c.columns) & set(f)), 'IMW2', 'IMW1', task, frac=frac)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=2); ap.add_argument('--fracs', nargs='*', type=float, default=FRACS); a = ap.parse_args()
    parts = Parallel(n_jobs=a.jobs)(delayed(job)(t, f) for f in a.fracs for t in ('MED_vs_IMW2', 'think2_vs_think1'))
    R = pd.concat(parts, ignore_index=True); R.to_csv(H/'calibration_curve_subject_reps.csv', index=False)
    per = R.groupby(['task', 'frac', 'subject']).agg(calibration_seconds=('calibration_seconds', 'mean'), gain_full=('gain_full', 'mean'), gain_bias=('gain_bias', 'mean'), full_update=('full_update', 'mean'), base=('base', 'mean')).reset_index()
    per.to_csv(H/'calibration_curve_subject.csv', index=False); rows = []
    for (task, frac), g in per.groupby(['task', 'frac']):
        v = g.gain_full.to_numpy(); lo, hi = bootstrap_ci(v); rows.append(dict(task=task, frac=frac, calibration_seconds=g.calibration_seconds.mean(), N=len(v), mean_gain_full=v.mean(), ci_low=lo, ci_high=hi, p_signflip=signflip(v+.5)))
    w = per.pivot_table(index=['frac', 'subject'], columns='task', values='gain_full').dropna(); w['null_adj'] = w.MED_vs_IMW2-w.think2_vs_think1
    for frac, g in w.groupby('frac'):
        v = g.null_adj.to_numpy(); lo, hi = bootstrap_ci(v); rows.append(dict(task='null_adjusted (MED_vs_IMW2 minus think2_vs_think1)', frac=frac, calibration_seconds=per[per.frac == frac].calibration_seconds.mean(), N=len(v), mean_gain_full=v.mean(), ci_low=lo, ci_high=hi, p_signflip=signflip(v+.5)))
    S = pd.DataFrame(rows); S.to_csv(H/'calibration_curve_summary.csv', index=False); print(S.round(4).to_string(index=False))
