"""B3 + B4 extraction. One pass per subject over the Braboszcz matched cohort, loading each raw run once:
  main prep (ICA, 14ch average ref)  -> think1 4s epochs for the Riemannian control (epoch_cache_think1/) and 2s / 8s feature tables
  noica prep                         -> 4s features (ICA skipped)
  ref64 prep                         -> 4s features (average reference over all 64 EEG channels BEFORE picking the 14 channels)
prep_variant() is a copy of reextract_harmonized.prep_paired with the two switches; with ica=True, ref_all=False it must reproduce the
main pipeline, which is checked against epoch_cache/ (epoch counts and values) for every subject.
Outputs: results/harmonized/preproc_variants/<variant>/<subject>.csv, results/harmonized/epoch_cache_think1/<subject>.npz,
         results/harmonized/preproc_variants/main_reproduction_check.csv
Usage: venv/bin/python preproc_variants.py [--limit N] [--shard i n]
"""
import argparse, gc, warnings
from pathlib import Path
import numpy as np, pandas as pd, mne
from threadpoolctl import threadpool_limits
from reextract_harmonized import braboszcz_jobs, standardize_and_clean, feature_rows
from braboszcz_feature_extraction import (COMMON_CHANNELS_14, TARGET_SFREQ, load_raw_bdf, apply_channel_types_from_bids, rename_to_standard_1020)
from braboszcz_classifier_cohort import selected_subjects

HP = 1.0; H = Path('results/harmonized'); VD = H/'preproc_variants'; T1 = H/'epoch_cache_think1'; CACHE = H/'epoch_cache'

def prep_variant(raws, ica=True, ref_all=False):
    """Copy of prep_paired. ica=False skips ICA; ref_all=True average-references every EEG channel before picking the 14."""
    bad_union = set()
    for _, r in raws:
        _, b = standardize_and_clean(r, '', annotate=True); bad_union.update(b)
    processed = []
    for name, r in raws:
        common = set(r.ch_names) & set.intersection(*(set(x.ch_names) for _, x in raws))
        r.pick_channels([c for c in r.ch_names if c in common], ordered=True)
        active_bad = sorted(bad_union & set(r.ch_names)); r.info['bads'] = active_bad
        if active_bad:
            pos = (r.get_montage().get_positions().get('ch_pos', {}) if r.get_montage() else {})
            located = [ch for ch in active_bad if ch in pos and np.isfinite(pos[ch]).all()]
            unlocated = [ch for ch in active_bad if ch not in located]
            r.info['bads'] = located
            if located:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning); r.interpolate_bads(reset_bads=True, verbose=False)
            r.info['bads'] = unlocated
        missing = [ch for ch in COMMON_CHANNELS_14 if ch not in r.ch_names]
        if missing: raise RuntimeError(f'missing common EEG channels before reference: {missing}')
        if ref_all:
            r.set_eeg_reference('average', projection=False, verbose=False)      # all remaining EEG channels (64)
            r.pick_channels(COMMON_CHANNELS_14, ordered=True)
        else:
            r.pick_channels(COMMON_CHANNELS_14, ordered=True)
            r.set_eeg_reference('average', projection=False, verbose=False)      # 14-channel reference (main pipeline)
        r.filter(l_freq=HP, h_freq=None, method='iir', iir_params=dict(order=6, ftype='butter'), phase='zero', picks='eeg', verbose=False)
        if not np.isclose(r.info['sfreq'], TARGET_SFREQ): r.resample(TARGET_SFREQ, npad='auto', verbose=False)
        processed.append((name, r))
    n_removed = 0
    if ica:
        cat = mne.concatenate_raws([r.copy() for _, r in processed], preload=True, verbose=False)
        cat_f = cat.copy().filter(l_freq=1., h_freq=None, verbose=False)
        model = mne.preprocessing.ICA(n_components=.99, method='infomax', fit_params=dict(extended=True), random_state=97, max_iter='auto')
        model.fit(cat_f, decim=2, verbose=False)
        exclude = set()
        for fn in (lambda: model.find_bads_eog(cat, ch_name=[c for c in ('Fp1', 'Fp2') if c in cat.ch_names], verbose=False)[0],
                   lambda: model.find_bads_muscle(cat, verbose=False)[0]):
            try: exclude.update(fn())
            except Exception as e: warnings.warn(f'ICA auto-labeling failed: {e}')
        model.exclude = sorted(exclude); n_removed = len(model.exclude)
        for _, r in processed: model.apply(r, verbose=False)
    return processed, sorted(bad_union), n_removed

def epochs_of(raw, epoch_len):
    ev = mne.make_fixed_length_events(raw, duration=epoch_len)
    meta = pd.DataFrame({'epoch_idx': np.arange(len(ev))})
    ep = mne.Epochs(raw, ev, tmin=0, tmax=epoch_len-1/raw.info['sfreq'], baseline=None, preload=True, reject_by_annotation=True, metadata=meta, verbose=False)
    ep.pick_channels(COMMON_CHANNELS_14, ordered=True); return ep

def load_runs(paths, all_eeg):
    raws = []
    for cond, bdf, tsv in paths:
        r = load_raw_bdf(str(bdf)); r = apply_channel_types_from_bids(r, str(tsv)); r = rename_to_standard_1020(r)
        r.pick_types(eeg=True, eog=False, misc=False, exclude=[])
        if not all_eeg: r.pick_channels(COMMON_CHANNELS_14, ordered=True)
        raws.append((cond, r))
    return raws

def save_feats(variant, s, processed, epoch_len, bads, n_ic):
    frames = [feature_rows(epochs_of(r, epoch_len), 'braboszcz', s, c, None, {'bad_channels': ','.join(bads), 'n_ica_excluded': n_ic}) for c, r in processed]
    (VD/variant).mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(VD/variant/f'{s}.csv', index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); ap.add_argument('--shard', nargs=2, type=int, default=[0, 1]); a = ap.parse_args()
    T1.mkdir(parents=True, exist_ok=True); VD.mkdir(parents=True, exist_ok=True)
    keep = sorted(selected_subjects()); mine = [s for i, s in enumerate(keep) if i % a.shard[1] == a.shard[0]]
    jobs = {s: p for s, p, _ in braboszcz_jobs(all_available=True) if s in mine}
    todo = [s for s in mine if s in jobs and not all((VD/v/f'{s}.csv').exists() for v in ('ep2', 'ep8', 'noica', 'ref64'))]
    if a.limit: todo = todo[:a.limit]
    print(f'shard {a.shard}: {len(todo)} subjects to do', flush=True)
    with threadpool_limits(limits=1):
        for s in todo:
            paths = jobs[s]
            main, bads, n_ic = prep_variant(load_runs(paths, False))
            check = []
            z = np.load(CACHE/f'{s}.npz')
            for c, r in main:
                arr = epochs_of(r, 4.0).get_data().astype(np.float32)
                if c == 'IMW1': np.savez(T1/f'{s}.npz', IMW1=arr)
                else:
                    ref = z[c]; same_n = arr.shape == ref.shape
                    check.append(dict(subject=s, condition=c, n_new=len(arr), n_cached=len(ref), shape_equal=same_n,
                                      max_abs_diff_rel=float(np.abs(arr-ref).max()/np.abs(ref).max()) if same_n else np.nan))
            pd.DataFrame(check).to_csv(VD/'main_reproduction_check.csv', mode='a', header=not (VD/'main_reproduction_check.csv').exists(), index=False)
            save_feats('ep2', s, main, 2.0, bads, n_ic); save_feats('ep8', s, main, 8.0, bads, n_ic)
            del main; gc.collect()
            nz, bads_n, _ = prep_variant(load_runs(paths, False), ica=False); save_feats('noica', s, nz, 4.0, bads_n, 0); del nz; gc.collect()
            r64, bads_r, n_r = prep_variant(load_runs(paths, True), ref_all=True); save_feats('ref64', s, r64, 4.0, bads_r, n_r); del r64; gc.collect()
            print(s, 'done', flush=True)
