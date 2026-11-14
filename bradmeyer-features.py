from pathlib import Path
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from decode_brandmeyer_probes import decode_probes, resolve_dir, brandmeyer_sessions
from braboszcz_feature_extraction import (CANONICAL_BANDS, COMMON_CHANNELS_14, TARGET_SFREQ, EPOCH_LEN_SEC,
    rename_to_standard_1020, detect_bad_segments, detect_bad_channels_by_variance, clean_bad_channels,
    highpass_filter, run_ica_and_clean, compute_psd, band_power, individual_alpha_frequency,
    asymmetry_index, cross_band_ratios)

EPOCH_STARTS = (-8.05, -4.05)   # two 4 s epochs ending at -0.05 s (the paper's 10 s window minus its first 2 s)

# Dataset setup, matching decode_brandmeyer_probes.py.
DATA_ROOT = Path("/Users/sarshanspal/Library/CloudStorage/GoogleDrive-sarahanspal@gmail.com/My Drive/meditation-eeg")

def load_any_raw(path):
    ext = Path(path).suffix.lower(); path = str(path)
    if ext == ".set":  return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    if ext == ".bdf":  return mne.io.read_raw_bdf(path, preload=True, verbose=False)
    if ext == ".edf":  return mne.io.read_raw_edf(path, preload=True, verbose=False)
    if ext == ".vhdr": return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    raise ValueError(f"unsupported EEG format {ext}")

def epochs_to_features(epochs, meta_cols):
    freqs, psd = compute_psd(epochs); chs = epochs.ch_names
    band_df = pd.DataFrame(index=range(len(epochs)))
    for b, rng in CANONICAL_BANDS.items():
        bp = band_power(freqs, psd, rng)
        for ci, ch in enumerate(chs): band_df[f"{ch}_{b}"] = bp[:, ci]
    iaf = individual_alpha_frequency(freqs, psd)
    for ci, ch in enumerate(chs): band_df[f"{ch}_IAF"] = iaf[:, ci]
    feat = pd.concat([band_df, asymmetry_index(band_df), cross_band_ratios(band_df, chs)], axis=1)
    return pd.concat([epochs.metadata.reset_index(drop=True)[meta_cols], feat], axis=1)

def brandmeyer_subject_features(subject, group, raw, events_tsv, run_ica=True):
    probes = decode_probes(events_tsv)
    probes = probes[probes.label.isin(["MED", "MW"])].reset_index(drop=True)      # ties / ambiguous dropped
    raw = rename_to_standard_1020(raw)
    keep = [c for c in raw.ch_names if c in mne.channels.make_standard_montage("biosemi64").ch_names]
    raw.pick(keep)
    raw.set_annotations(detect_bad_segments(raw))
    bads = detect_bad_channels_by_variance(raw)
    if bads: raw = clean_bad_channels(raw, bads)
    raw.set_eeg_reference("average", projection=False, verbose=False)            # paper: average reference
    raw = highpass_filter(raw)
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-6: raw.resample(TARGET_SFREQ, npad="auto")
    if run_ica: raw, _, _ = run_ica_and_clean(raw)
    raw.pick([c for c in COMMON_CHANNELS_14 if c in raw.ch_names])
    sf = raw.info["sfreq"]; rows, ev = [], []
    for _, p in probes.iterrows():
        for k, off in enumerate(EPOCH_STARTS):
            s = int(round((p.t_probe + off) * sf))
            if s < 0 or s + int(EPOCH_LEN_SEC * sf) > raw.n_times: continue
            ev.append([s, 0, 1])
            rows.append(dict(subject=subject, group=group, condition=p.label, probe_id=int(p.probe_id),
                             sub_epoch=k, t_start=s / sf, q1=p.q1, q2=p.q2, q3=p.q3))
    meta = pd.DataFrame(rows)
    epochs = mne.Epochs(raw, np.array(ev), event_id=1, tmin=0, tmax=EPOCH_LEN_SEC - 1 / sf, baseline=None,
                        preload=True, reject_by_annotation=True, metadata=meta, verbose=False)
    out = epochs_to_features(epochs, list(meta.columns))
    return pd.concat([out.iloc[:, :3], pd.Series(range(len(out)), name="epoch_idx"), out.iloc[:, 3:]], axis=1)

def bm_worker(subject, group, raw_path, events_tsv, run_ica=True):
    from threadpoolctl import threadpool_limits          # kernel already imported numpy, so cap BLAS threads here
    try:
        with threadpool_limits(limits=1):
            raw = load_any_raw(raw_path)
            return dict(subject=subject, status="ok", df=brandmeyer_subject_features(subject, group, raw, events_tsv, run_ica))
    except Exception as e:
        import traceback
        return dict(subject=subject, status="failed", reason=f"{type(e).__name__}: {e}", tb=traceback.format_exc(), df=None)

def main():
    bm_root = resolve_dir(DATA_ROOT, "openneuro ")
    participants = pd.read_csv(bm_root / "participants.tsv", sep="\t")
    group_of = dict(zip(participants.participant_id, participants.group))
    sessions = brandmeyer_sessions(bm_root, participants.participant_id)
    jobs = []
    for session in sessions:
        subject, eeg, events = session["subject"], session["eeg"], session["events"]
        print(subject, session["session"], eeg.name if eeg else "NO EEG",
              events.name if events else "NO MATCHING EVENTS")
        if eeg is not None and events is not None:
            jobs.append((subject, group_of[subject], eeg, events))

    if not jobs:
        raise RuntimeError(f"No matched EEG/events session pairs found under {bm_root}")
    results = Parallel(n_jobs=7, backend="loky", verbose=10)(
        delayed(bm_worker)(*job) for job in jobs
    )
    for result in results:
        if result["status"] != "ok":
            print("FAILED", result["subject"], result["reason"])
    successful = [result["df"] for result in results if result["status"] == "ok"]
    if not successful:
        raise RuntimeError("All matched Brandmeyer EEG/events extractions failed")
    bm_df = pd.concat(successful, ignore_index=True)
    bm_df.to_csv("brandmeyer_all_subjects_features.csv", index=False)
    print(bm_df.groupby(["subject", "condition"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
