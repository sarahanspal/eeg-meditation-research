"""Save harmonized Braboszcz MED/IMW2 epochs (same preprocessing as reextract_harmonized.py)
so CSP can be run on exactly the epochs that produced the harmonized features.
Usage: venv/bin/python csp_harmonized_epochs.py [--limit N]
Writes results/harmonized/epoch_cache/<subject>.npz (float32, volts) and
epoch_cache_check.csv (epoch counts vs harmonized feature table).
"""
import argparse, gc
from pathlib import Path
import numpy as np, pandas as pd, mne
from reextract_harmonized import braboszcz_jobs, prep_paired
from braboszcz_feature_extraction import (COMMON_CHANNELS_14, EPOCH_LEN_SEC, load_raw_bdf,
    apply_channel_types_from_bids, rename_to_standard_1020)
from braboszcz_classifier_cohort import selected_subjects

CACHE = Path('results/harmonized/epoch_cache'); CACHE.mkdir(parents=True, exist_ok=True)

def epoch_array(raw):
    ev = mne.make_fixed_length_events(raw, duration=EPOCH_LEN_SEC)
    ep = mne.Epochs(raw, ev, tmin=0, tmax=EPOCH_LEN_SEC-1/raw.info['sfreq'], baseline=None,
                    preload=True, reject_by_annotation=True, verbose=False)
    ep.pick_channels(COMMON_CHANNELS_14, ordered=True)
    return ep.get_data().astype(np.float32)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); a = ap.parse_args()
    keep = set(selected_subjects())
    feats = pd.read_csv('results/harmonized/braboszcz_all_subjects_features.csv', usecols=['subject', 'condition'])
    nfeat = feats.groupby(['subject', 'condition']).size()
    rows = []; done = 0
    for s, paths, _ in braboszcz_jobs(all_available=True):
        if s not in keep: continue
        if a.limit and done >= a.limit: break
        if (CACHE/f'{s}.npz').exists(): done += 1; continue
        raws = []
        for cond, bdf, tsv in paths:
            r = load_raw_bdf(str(bdf)); r = apply_channel_types_from_bids(r, str(tsv)); r = rename_to_standard_1020(r)
            r.pick_types(eeg=True, eog=False, misc=False, exclude=[]); r.pick_channels(COMMON_CHANNELS_14, ordered=True)
            raws.append((cond, r))
        pre, _, _ = prep_paired(raws)
        out = {c: epoch_array(r) for c, r in pre if c in ('MED', 'IMW2')}
        np.savez(CACHE/f'{s}.npz', MED=out['MED'], IMW2=out['IMW2'])
        for c in out:
            rows.append(dict(subject=s, condition=c, n_epochs_cache=len(out[c]), n_epochs_features=int(nfeat.get((s, c), -1))))
        print(s, {c: v.shape for c, v in out.items()}, flush=True); done += 1; gc.collect()
    if rows:
        df = pd.DataFrame(rows); df['match'] = df.n_epochs_cache == df.n_epochs_features
        df.to_csv(CACHE/f'epoch_cache_check_{"smoke" if a.limit else "full"}.csv', index=False); print(df)
