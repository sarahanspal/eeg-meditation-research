"""B5: is the MED>IMW2 theta increase an oscillation or a 1/f shift? (matched 54-subject Braboszcz cohort, harmonized epochs)
Per epoch and frontal channel (F3,F4,F7,F8): multitaper PSD 2-40 Hz (same MNE multitaper defaults as the feature pipeline),
FOOOF fit (fixed aperiodic, peak width 1-8 Hz, <=6 peaks, min peak height 0.05 log10 units) -> exponent, offset,
theta_peak_power (height of the largest fitted peak with centre 4-8 Hz above the aperiodic fit; NaN if none) and
theta_flat_power (mean of log10 PSD minus aperiodic fit over 4-8 Hz; defined for every epoch).
Per-subject standardized MED-minus-IMW2 difference, then fraction of subjects positive, binomial test, BH-FDR (same method as
combined_supplementary.sign_consistency). The existing log relative theta is included for the same subjects as the reference,
plus relative theta after removing the epoch-level exponent (per-subject linear residual).
Outputs: results/harmonized/aperiodic_epoch_params.csv, aperiodic_sign_consistency.csv
Usage: venv/bin/python aperiodic_theta.py [--limit N] [--jobs J]
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest, false_discovery_control
from braboszcz_feature_extraction import COMMON_CHANNELS_14, band_power
from braboszcz_classifier_cohort import selected_subjects

CH = ['F3', 'F4', 'F7', 'F8']; IDX = [COMMON_CHANNELS_14.index(c) for c in CH]; H = Path('results/harmonized'); SF = 256.
from mne.time_frequency import psd_array_multitaper

def fit_subject(s):
    from fooof import FOOOF
    z = np.load(H/'epoch_cache'/f'{s}.npz'); rows = []
    for cond in ('MED', 'IMW2'):
        X = z[cond].astype(np.float64)
        psd, fr = psd_array_multitaper(X[:, IDX, :], SF, fmin=2., fmax=40., verbose=False)
        for e in range(len(X)):
            for k, ch in enumerate(CH):
                fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, min_peak_height=0.05, aperiodic_mode='fixed', verbose=False)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore'); fm.fit(fr, psd[e, k], [2, 40])
                    off, expo = fm.aperiodic_params_
                    pk = [p for p in fm.peak_params_ if 4 <= p[0] < 8] if fm.peak_params_.size else []
                    flat = np.log10(psd[e, k]) - (off - np.log10(fr**expo)); th = (fr >= 4) & (fr < 8)
                    rows.append(dict(subject=s, condition=cond, epoch_pos=e, channel=ch, offset=off, exponent=expo,
                                     theta_peak_power=max(p[1] for p in pk) if len(pk) else np.nan, theta_flat_power=float(flat[th].mean()), r2=fm.r_squared_))
                except Exception:
                    rows.append(dict(subject=s, condition=cond, epoch_pos=e, channel=ch, offset=np.nan, exponent=np.nan, theta_peak_power=np.nan, theta_flat_power=np.nan, r2=np.nan))
    return rows

def alignment_check(subs):
    """cached epoch order == feature-table order? compare my own relative theta (1-70 Hz total) with the table's F3_theta."""
    out = []
    for s in subs:
        z = np.load(H/'epoch_cache'/f'{s}.npz'); T = pd.read_csv(H/'braboszcz_all_subjects_features.csv', usecols=['subject', 'condition', 'epoch_idx', 'F3_theta'])
        for cond in ('MED', 'IMW2'):
            psd, fr = psd_array_multitaper(z[cond][:, [COMMON_CHANNELS_14.index('F3')], :].astype(np.float64), SF, fmin=1., fmax=70., verbose=False)
            mine = band_power(fr, psd, (4., 8.))[:, 0]; ref = T[(T.subject == s) & (T.condition == cond)].sort_values('epoch_idx').F3_theta.to_numpy()
            out.append((s, cond, len(mine) == len(ref), float(np.corrcoef(mine, ref)[0, 1]) if len(mine) == len(ref) else np.nan))
    return out

def sign_rows(S, value_cols, label):
    rows = []
    for f in value_cols:
        eff = []
        for s, g in S.groupby('subject'):
            a, b = g.loc[g.condition == 'MED', f].dropna(), g.loc[g.condition == 'IMW2', f].dropna()
            if len(a) < 5 or len(b) < 5: continue
            sd = np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
            if np.isfinite(sd) and sd > 0: eff.append((a.mean()-b.mean())/sd)
        if eff:
            pos = int((np.asarray(eff) > 0).sum()); n = len(eff)
            rows.append(dict(measure=label, feature=f, n=n, frac_positive=pos/n, median_standardized_difference=float(np.median(eff)), p_binomial=binomtest(pos, n, .5).pvalue))
    return rows

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); ap.add_argument('--jobs', type=int, default=2); a = ap.parse_args()
    subs = sorted(selected_subjects()); subs = subs[:a.limit] if a.limit else subs
    print('alignment check (n epochs equal, corr of my relative theta with table F3_theta):', [(s, c, ok, round(r, 4)) for s, c, ok, r in alignment_check(subs[:2])], flush=True)
    res = Parallel(n_jobs=a.jobs)(delayed(fit_subject)(s) for s in subs)
    P = pd.DataFrame([r for rs in res for r in rs]); path = H/('aperiodic_epoch_params.csv' if not a.limit else 'aperiodic_epoch_params_smoke.csv'); P.to_csv(path, index=False)
    print('fits:', len(P), '| failed:', int(P.exponent.isna().sum()), '| mean r2 %.3f' % P.r2.mean(), '| epochs with a theta peak: MED %.2f IMW2 %.2f' % tuple(P.groupby('condition').theta_peak_power.apply(lambda x: x.notna().mean()).reindex(['MED', 'IMW2'])))
    # wide: one row per epoch, columns per channel x measure
    W = P.pivot_table(index=['subject', 'condition', 'epoch_pos'], columns='channel', values=['exponent', 'offset', 'theta_peak_power', 'theta_flat_power'])
    W.columns = [f'{ch}_{m}' for m, ch in W.columns]; W = W.reset_index()
    for m in ('exponent', 'offset', 'theta_peak_power', 'theta_flat_power'): W[f'frontal_{m}'] = W[[f'{c}_{m}' for c in CH]].mean(axis=1, skipna=True)
    # reference: harmonized log relative theta (same transform as combined_analysis) and exponent-residualised version
    T = pd.read_csv(H/'braboszcz_all_subjects_features.csv', usecols=['subject', 'condition', 'epoch_idx'] + [f'{c}_theta' for c in CH]); T = T[T.subject.isin(subs) & T.condition.isin(['MED', 'IMW2'])]
    T = T.sort_values(['subject', 'condition', 'epoch_idx']); T['epoch_pos'] = T.groupby(['subject', 'condition']).cumcount()
    for c in CH: T[f'{c}_logtheta'] = np.log(T[f'{c}_theta'].clip(lower=1e-12))
    W = W.merge(T.drop(columns=[f'{c}_theta' for c in CH] + ['epoch_idx']), on=['subject', 'condition', 'epoch_pos'], how='left')
    W['frontal_logtheta'] = W[[f'{c}_logtheta' for c in CH]].mean(axis=1)
    for c in CH + ['frontal']:
        resid = W.groupby('subject', group_keys=False).apply(lambda g: g[f'{c}_logtheta'] - np.polyval(np.polyfit(g[f'{c}_exponent'].fillna(g[f'{c}_exponent'].mean()), g[f'{c}_logtheta'], 1), g[f'{c}_exponent'].fillna(g[f'{c}_exponent'].mean())))
        W[f'{c}_logtheta_resid_exponent'] = resid
    rows = []
    for label, suffix in [('aperiodic exponent', '_exponent'), ('aperiodic offset', '_offset'), ('periodic theta peak power (NaN if no peak)', '_theta_peak_power'),
                          ('periodic theta flattened power 4-8 Hz', '_theta_flat_power'), ('reference: log relative theta (existing feature)', '_logtheta'),
                          ('relative theta after removing exponent', '_logtheta_resid_exponent')]:
        rows += sign_rows(W, [f'{c}{suffix}' for c in CH + ['frontal']], label)
    R = pd.DataFrame(rows); R['p_fdr_bh'] = false_discovery_control(R.p_binomial.to_numpy(), method='bh')
    R.to_csv(H/('aperiodic_sign_consistency.csv' if not a.limit else 'aperiodic_sign_consistency_smoke.csv'), index=False)
    print(R.round(4).to_string(index=False))
