#script 5: is the med > imw2 theta increase an oscillation or a 1/f shift? (64 channels)
#
#same method as aperiodic_theta.py (task b5) but on F3, F4, Fz, FCz, Cz.
#the b5 epoch cache only has the 14 shared channels, so this script redoes the
#extract_64.py preprocessing (notch, bad channels, average reference, filter,
#shared ica over med2 / think1 / think2) for each matched braboszcz subject and
#keeps the med and imw2 epochs.
#
#per epoch and channel: multitaper psd 2-40 hz, fooof fit (fixed aperiodic,
#peak width 1-8 hz, at most 6 peaks, min peak height 0.05) giving
#exponent, offset
#theta_peak_power: height of the biggest fitted peak centred 4-8 hz (nan if none)
#theta_flat_power: mean of log10 psd minus the aperiodic fit over 4-8 hz
#plus the existing log relative theta from braboszcz_features.csv as the reference,
#and that theta after removing the exponent (per-subject linear residual).
#per-subject standardized med minus imw2 difference, fraction positive, binomial
#test, bh-fdr (aperiodic_theta.sign_rows).
#
#fits are cached per subject in results/ch64/aperiodic_cache, so it is safe to stop
#and restart. each worker loads three raw recordings, so keep --jobs low (1-2).
#
#run from the repo root:
#.venv/bin/python ch64/aperiodic_64.py --jobs 2
#.venv/bin/python ch64/aperiodic_64.py --n-brab 3   (test run)
import argparse
import time
import traceback
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.time_frequency import psd_array_multitaper

#importing extract_64 also swaps the 64-channel list into prep_paired
import extract_64
from extract_64 import SHARED_64
from braboszcz_feature_extraction import (EPOCH_LEN_SEC, apply_channel_types_from_bids, band_power,
                                          compute_psd, load_raw_bdf, rename_to_standard_1020)
from reextract_harmonized import braboszcz_jobs
from aperiodic_theta import sign_rows

from common_64 import add_fdr, add_subset_args, load_braboszcz, output_dir, print_table

CH = ['F3', 'F4', 'Fz', 'FCz', 'Cz']
REGIONS = {'frontal': ['F3', 'F4', 'Fz'], 'midline': ['Fz', 'FCz', 'Cz']}
CONDITIONS = ('MED', 'IMW2')
MEASURES = ['exponent', 'offset', 'theta_peak_power', 'theta_flat_power']


def preprocessed_epochs(sub, paths):
    #same steps as extract_64.run_braboszcz, returns the med and imw2 epochs
    raws = []
    for cond, bdf, tsv in paths:
        r = load_raw_bdf(str(bdf))
        r = apply_channel_types_from_bids(r, str(tsv))
        r = rename_to_standard_1020(r)
        r.pick_types(eeg=True, eog=False, misc=False, exclude=[])
        r.pick_channels(SHARED_64, ordered=True)
        raws.append((cond, r))
    pre, bads, ics = extract_64.prep_paired(raws)

    epochs = {}
    for cond, r in pre:
        if cond not in CONDITIONS:
            continue
        ev = mne.make_fixed_length_events(r, duration=EPOCH_LEN_SEC)
        epochs[cond] = mne.Epochs(r, ev, tmin=0, tmax=EPOCH_LEN_SEC - 1 / r.info['sfreq'],
                                  baseline=None, preload=True, reject_by_annotation=True, verbose=False)
    return epochs


def fit_spectrum(fr, spectrum):
    #one fooof fit, same settings as aperiodic_theta.py
    from fooof import FOOOF
    fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, min_peak_height=0.05,
               aperiodic_mode='fixed', verbose=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fm.fit(fr, spectrum, [2, 40])
        offset, exponent = fm.aperiodic_params_
        peaks = [p for p in fm.peak_params_ if 4 <= p[0] < 8] if fm.peak_params_.size else []
        flat = np.log10(spectrum) - (offset - np.log10(fr ** exponent))
        theta = (fr >= 4) & (fr < 8)
        return dict(offset=offset, exponent=exponent,
                    theta_peak_power=max(p[1] for p in peaks) if peaks else np.nan,
                    theta_flat_power=float(flat[theta].mean()), r2=fm.r_squared_)
    except Exception:
        return dict(offset=np.nan, exponent=np.nan, theta_peak_power=np.nan,
                    theta_flat_power=np.nan, r2=np.nan)


def fit_subject(sub, paths, table, cache_dir):
    #preprocess, fit every epoch x channel, save to the cache. returns the alignment check.
    cache_file = cache_dir / f'{sub}.csv'
    check_file = cache_dir / f'{sub}_check.csv'
    if cache_file.exists() and check_file.exists():
        return pd.read_csv(check_file)

    start = time.time()
    epochs = preprocessed_epochs(sub, paths)
    rows, checks = [], []
    for cond, ep in epochs.items():
        #alignment check: same number of epochs as the feature table, and the same
        #log relative theta (if this is ~1 the epoch order matches epoch_idx)
        ref = table[(table.subject == sub) & (table.condition == cond)].sort_values('epoch_idx')
        freqs, psd_full = compute_psd(ep.copy().pick(CH))
        mine = np.log(np.clip(band_power(freqs, psd_full, (4., 8.)), 1e-12, None))
        same_n = len(ep) == len(ref)
        corr = float(np.corrcoef(mine[:, 0], ref['F3_theta'])[0, 1]) if same_n else np.nan
        checks.append(dict(subject=sub, condition=cond, n_epochs=len(ep), n_table=len(ref),
                           same_n=same_n, corr_F3_theta=corr))

        #fooof on 2-40 hz, as in aperiodic_theta.py
        data = ep.get_data(picks=CH).astype(np.float64)
        psd, fr = psd_array_multitaper(data, ep.info['sfreq'], fmin=2., fmax=40., verbose=False)
        for e in range(len(data)):
            for k, ch in enumerate(CH):
                rows.append(dict(subject=sub, condition=cond, epoch_pos=e, channel=ch,
                                 **fit_spectrum(fr, psd[e, k])))

    pd.DataFrame(rows).to_csv(cache_file, index=False)
    check = pd.DataFrame(checks)
    check.to_csv(check_file, index=False)
    print(f'{sub} done {time.time() - start:.0f}s', flush=True)
    return check


def safe_fit_subject(sub, paths, table, cache_dir):
    #one bad subject should not stop the whole run
    try:
        return fit_subject(sub, paths, table, cache_dir)
    except Exception as e:
        print(f'{sub} FAILED {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        return pd.DataFrame([dict(subject=sub, failed=str(e))])


def residual_after_exponent(g, theta_col, exp_col):
    #per subject: theta minus the linear fit of theta on the exponent (both conditions together)
    x = g[exp_col].fillna(g[exp_col].mean())
    slope, intercept = np.polyfit(x, g[theta_col], 1)
    return g[theta_col] - (slope * x + intercept)


def main():
    parser = argparse.ArgumentParser()
    add_subset_args(parser)
    parser.add_argument('--jobs', type=int, default=1, help='parallel subjects (each loads 3 raw files)')
    args = parser.parse_args()
    out = output_dir(args)
    cache_dir = out / 'aperiodic_cache'
    cache_dir.mkdir(exist_ok=True)

    #matched cohort and its reference log relative theta from the 64-channel table
    table = load_braboszcz(n_brab=args.n_brab)
    table = table[table.condition.isin(CONDITIONS)]
    subjects = sorted(table.subject.unique())
    table = table[['subject', 'condition', 'epoch_idx'] + [f'{c}_theta' for c in CH]]
    jobs = [(sub, paths) for sub, paths, _ in braboszcz_jobs(all_available=True) if sub in subjects]
    print(f'subjects: {len(jobs)}', flush=True)

    checks = Parallel(n_jobs=args.jobs)(delayed(safe_fit_subject)(sub, paths, table, cache_dir)
                                        for sub, paths in jobs)
    check = pd.concat(checks, ignore_index=True)
    check.to_csv(out / 'aperiodic_alignment_check.csv', index=False)
    print_table('alignment check (epoch counts and corr with the feature table)', check)

    #all cached fits, long format: one row per subject x condition x epoch x channel
    files = [cache_dir / f'{sub}.csv' for sub, _ in jobs if (cache_dir / f'{sub}.csv').exists()]
    P = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    P.to_csv(out / 'aperiodic_epoch_params.csv', index=False)
    print(f'\nfits: {len(P)}  failed: {int(P.exponent.isna().sum())}  mean r2: {P.r2.mean():.3f}')
    print('share of epochs with a theta peak:', P.groupby('condition').theta_peak_power
          .apply(lambda x: round(x.notna().mean(), 3)).to_dict())

    #wide: one row per epoch, a column per channel x measure
    W = P.pivot_table(index=['subject', 'condition', 'epoch_pos'], columns='channel', values=MEASURES)
    W.columns = [f'{ch}_{m}' for m, ch in W.columns]
    W = W.reset_index()

    #reference log relative theta (already logged in the 64-channel table), matched on epoch order
    ref = table.rename(columns={'epoch_idx': 'epoch_pos'})
    ref = ref.rename(columns={f'{c}_theta': f'{c}_logtheta' for c in CH})
    W = W.merge(ref, on=['subject', 'condition', 'epoch_pos'], how='left')

    #region means, then theta with the exponent regressed out
    for region, chans in REGIONS.items():
        for m in MEASURES + ['logtheta']:
            W[f'{region}_{m}'] = W[[f'{c}_{m}' for c in chans]].mean(axis=1, skipna=True)
    for c in CH + list(REGIONS):
        W[f'{c}_logtheta_resid_exponent'] = np.nan
        for sub, g in W.groupby('subject'):
            W.loc[g.index, f'{c}_logtheta_resid_exponent'] = residual_after_exponent(
                g, f'{c}_logtheta', f'{c}_exponent')

    #sign consistency, same labels as aperiodic_theta.py
    rows = []
    for label, suffix in [('aperiodic exponent', '_exponent'), ('aperiodic offset', '_offset'),
                          ('periodic theta peak power (NaN if no peak)', '_theta_peak_power'),
                          ('periodic theta flattened power 4-8 Hz', '_theta_flat_power'),
                          ('reference: log relative theta (existing feature)', '_logtheta'),
                          ('relative theta after removing exponent', '_logtheta_resid_exponent')]:
        rows += sign_rows(W, [f'{c}{suffix}' for c in CH + list(REGIONS)], label)
    R = add_fdr(pd.DataFrame(rows), 'p_binomial')
    R.to_csv(out / 'aperiodic_sign_consistency.csv', index=False)
    print_table('sign consistency (med minus imw2)', R)
    print(f'\nwrote results to {out}')


if __name__ == '__main__':
    main()
