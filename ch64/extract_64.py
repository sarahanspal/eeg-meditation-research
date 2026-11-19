#64-channel feature extraction for braboszcz and brandmeyer (no sky)
#same pipeline as reextract_harmonized.py, but keeps all 64 shared channels
#and only saves relative band power (log) and iaf per channel.
#
#run from the repo root:
#  .venv/bin/python ch64/extract_64.py --limit 2
#
#saves after every subject and skips subjects that are already in the csv,
#so it is safe to stop and restart.
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

#so the existing modules in the repo root can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import reextract_harmonized as rh
from reextract_harmonized import braboszcz_jobs, prep_paired
from braboszcz_feature_extraction import (
    CANONICAL_BANDS, EPOCH_LEN_SEC, load_raw_bdf, apply_channel_types_from_bids,
    rename_to_standard_1020, compute_psd, band_power, individual_alpha_frequency,
)
from decode_brandmeyer_probes import decode_probes, resolve_dir, brandmeyer_sessions
from config import RAW_PATHS

OUT = Path('results/ch64')
OUT.mkdir(parents=True, exist_ok=True)

#the 64 channels both datasets share (from ch64/shared_channels.py)
SHARED_64 = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5',
    'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz',
    'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]

#prep_paired in reextract_harmonized.py uses a module level channel list, so we
#swap in the 64 channel list instead of copying the whole function
rh.COMMON_CHANNELS_14 = SHARED_64

#channels in the brandmeyer files that are not eeg
BRAND_NON_EEG = ('Status', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp')

FAIL_FILE = OUT / 'extraction_failures.csv'
QC_FILE = OUT / 'subject_qc.csv'


def append_csv(df, path):
    #add rows to a csv, write the header only the first time
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


def subjects_done(path):
    #subjects already saved in a feature csv
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=['subject']).subject.astype(str))


def save_qc(dataset, subject, bads, ics):
    #bad channels and removed ica components for this subject
    #skip if this subject is already in the file (e.g. after a crash and restart)
    if QC_FILE.exists():
        old = pd.read_csv(QC_FILE)
        if ((old.dataset == dataset) & (old.subject == subject)).any():
            return
    row = pd.DataFrame([{
        'dataset': dataset, 'subject': subject,
        'bad_channels': ','.join(bads), 'n_bad_channels': len(bads),
        'ica_excluded': ','.join(map(str, ics)), 'n_ica_excluded': len(ics),
    }])
    append_csv(row, QC_FILE)


def save_failure(dataset, subject, err):
    row = pd.DataFrame([{
        'dataset': dataset, 'subject': subject,
        'reason': f'{type(err).__name__}: {err}',
        'traceback': traceback.format_exc(),
    }])
    append_csv(row, FAIL_FILE)


def make_features(epochs, dataset, subject, condition, extra=None):
    #relative band power (log) and iaf for every channel
    freqs, psd = compute_psd(epochs)
    cols = {}
    for band, rng in CANONICAL_BANDS.items():
        bp = band_power(freqs, psd, rng)
        #same log transform for every condition (clip so log never sees 0)
        bp = np.log(np.clip(bp, 1e-12, None))
        for i, ch in enumerate(epochs.ch_names):
            cols[f'{ch}_{band}'] = bp[:, i]
    iaf = individual_alpha_frequency(freqs, psd)
    for i, ch in enumerate(epochs.ch_names):
        cols[f'{ch}_IAF'] = iaf[:, i]
    feats = pd.DataFrame(cols)

    #id columns go first
    n = len(feats)
    ids = pd.DataFrame({
        'dataset': dataset, 'subject': subject,
        'condition': condition, 'epoch_idx': np.arange(n),
    })
    out = pd.concat([ids, feats], axis=1)
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


# ---------------- braboszcz ----------------

def run_braboszcz(limit=None):
    out_file = OUT / 'braboszcz_features.csv'
    done = subjects_done(out_file)
    part = pd.read_csv(Path(RAW_PATHS['braboszcz']) / 'participants.tsv', sep='\t').set_index('participant_id')

    n = 0
    #all_available=True keeps every subject with med2 + think2, including ctr
    for sub, paths, exp in braboszcz_jobs(all_available=True):
        if sub in done:
            continue
        if limit and n >= limit:
            break
        start = time.time()
        try:
            raws = []
            for cond, bdf, tsv in paths:
                r = load_raw_bdf(str(bdf))
                r = apply_channel_types_from_bids(r, str(tsv))
                r = rename_to_standard_1020(r)
                r.pick_types(eeg=True, eog=False, misc=False, exclude=[])
                missing = [c for c in SHARED_64 if c not in r.ch_names]
                if missing:
                    raise RuntimeError(f'{sub}/{cond} missing channels: {missing}')
                r.pick_channels(SHARED_64, ordered=True)
                raws.append((cond, r))

            #notch, bad channels, average reference, filter, shared ica
            pre, bads, ics = prep_paired(raws)

            frames = []
            for cond, r in pre:
                ev = mne.make_fixed_length_events(r, duration=EPOCH_LEN_SEC)
                ep = mne.Epochs(r, ev, tmin=0, tmax=EPOCH_LEN_SEC - 1 / r.info['sfreq'],
                                baseline=None, preload=True, reject_by_annotation=True, verbose=False)
                extra = {
                    'group': part.loc[sub, 'group'] if sub in part.index else None,
                    'experience_group': exp,
                }
                frames.append(make_features(ep, 'braboszcz', sub, cond, extra))

            #qc first, then features, so the feature csv is the "done" marker
            save_qc('braboszcz', sub, bads, ics)
            append_csv(pd.concat(frames, ignore_index=True), out_file)
            n += 1
            print(f'braboszcz {sub} ok  {time.time() - start:.0f}s', flush=True)
        except Exception as e:
            save_failure('braboszcz', sub, e)
            print(f'braboszcz {sub} FAILED {e}', flush=True)
        finally:
            gc.collect()
    return n


# ---------------- brandmeyer ----------------

def read_brandmeyer_raw(path):
    #pick the right mne reader from the file extension
    readers = {'.bdf': mne.io.read_raw_bdf, '.edf': mne.io.read_raw_edf,
               '.set': mne.io.read_raw_eeglab, '.vhdr': mne.io.read_raw_brainvision}
    ext = Path(path).suffix.lower()
    if ext not in readers:
        raise ValueError(f'unsupported file format: {ext}')
    return readers[ext](str(path), preload=True, verbose=False)


def run_brandmeyer(limit=None):
    out_file = OUT / 'brandmeyer_features.csv'
    done = subjects_done(out_file)
    root = resolve_dir(Path(RAW_PATHS['brandmeyer']).parent, Path(RAW_PATHS['brandmeyer']).name.strip())
    pt = pd.read_csv(root / 'participants.tsv', sep='\t')
    group = dict(zip(pt.participant_id, pt.group))
    age = dict(zip(pt.participant_id, pt.age))
    subjects = sorted(pt.participant_id.astype(str).unique())

    #group sessions by subject (only sessions that have both eeg and events)
    by_sub = {}
    for row in brandmeyer_sessions(root, subjects):
        if row['eeg'] is not None and row['events'] is not None:
            by_sub.setdefault(row['subject'], []).append(row)

    n = 0
    for sub, sessions in sorted(by_sub.items()):
        if sub in done:
            continue
        if limit and n >= limit:
            break
        start = time.time()
        try:
            raws = []
            probes_by_session = {}
            for j, row in enumerate(sessions):
                r = read_brandmeyer_raw(row['eeg'])
                #drop aux channels, everything left is eeg
                drop = [c for c in r.ch_names if c.upper().startswith('EXG') or c in BRAND_NON_EEG]
                if drop:
                    r.drop_channels(drop)
                r.set_channel_types({c: 'eeg' for c in r.ch_names}, verbose=False)
                r = rename_to_standard_1020(r)
                missing = [c for c in SHARED_64 if c not in r.ch_names]
                if missing:
                    raise RuntimeError(f'{sub} missing channels: {missing}')
                r.pick_channels(SHARED_64, ordered=True)
                raws.append((str(j), r))
                probes_by_session[str(j)] = decode_probes(row['events'])

            pre, bads, ics = prep_paired(raws)

            frames = []
            for key, r in pre:
                probes = probes_by_session[key]
                probes = probes[probes.label.isin(['MED', 'MW'])]
                sf = r.info['sfreq']
                events, meta = [], []
                #two 4s epochs before each probe (offsets match reextract_harmonized.py)
                for _, p in probes.iterrows():
                    for sub_epoch, off in enumerate((-8.05, -4.05)):
                        s = int(round((p.t_probe + off) * sf))
                        if s < 0 or s + int(EPOCH_LEN_SEC * sf) > r.n_times:
                            continue
                        events.append([s, 0, 1])
                        meta.append({'condition': p.label, 'probe_id': int(p.probe_id),
                                     'sub_epoch': sub_epoch, 't_start': s / sf})
                if not events:
                    continue
                ep = mne.Epochs(r, np.asarray(events), event_id=1, tmin=0,
                                tmax=EPOCH_LEN_SEC - 1 / sf, baseline=None, preload=True,
                                reject_by_annotation=True, metadata=pd.DataFrame(meta), verbose=False)
                #a session can lose every epoch to bad segment rejection, skip it
                if len(ep) == 0:
                    print(f'  {sub} session {key}: all epochs rejected, skipping', flush=True)
                    continue
                m = ep.metadata
                f = make_features(ep, 'brandmeyer', sub, m.condition.to_numpy(),
                                  {'group': group.get(sub), 'age': age.get(sub), 'session': key})
                f['probe_id'] = m.probe_id.to_numpy()
                f['sub_epoch'] = m.sub_epoch.to_numpy()
                f['t_start'] = m.t_start.to_numpy()
                frames.append(f)

            if not frames:
                raise RuntimeError(f'{sub} has no usable epochs')
            save_qc('brandmeyer', sub, bads, ics)
            append_csv(pd.concat(frames, ignore_index=True), out_file)
            n += 1
            print(f'brandmeyer {sub} ok  {time.time() - start:.0f}s', flush=True)
        except Exception as e:
            save_failure('brandmeyer', sub, e)
            print(f'brandmeyer {sub} FAILED {e}', flush=True)
        finally:
            gc.collect()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='max new subjects per dataset')
    ap.add_argument('--datasets', nargs='+', default=['braboszcz', 'brandmeyer'],
                    choices=['braboszcz', 'brandmeyer'])
    args = ap.parse_args()

    #1 worker: everything runs one subject at a time with 1 thread
    with threadpool_limits(limits=1):
        if 'braboszcz' in args.datasets:
            print('start braboszcz', flush=True)
            print('done braboszcz, new subjects:', run_braboszcz(args.limit), flush=True)
        if 'brandmeyer' in args.datasets:
            print('start brandmeyer', flush=True)
            print('done brandmeyer, new subjects:', run_brandmeyer(args.limit), flush=True)


if __name__ == '__main__':
    main()
