"""
run_batch_extraction_braboszcz.py
==================================
Batch-runs braboszcz_feature_extraction.process_subject_condition() over a
list of subjects for the two runs needed for classification:
    task-med2   (last 10 min of the tradition-specific MED block)
    task-think2 (last 10 min of the IMW block)
No task-med1breath, no sliding windows - matches what was requested for the
classification pass.

DEBUG NOTES
-----------
- Each subject/condition extraction is wrapped in try/except so ONE missing
  or corrupted file (e.g. sub-005 was missing task-think2 entirely, found
  earlier) does not kill the whole batch - failures are logged and printed
  in a summary at the end, and saved to braboszcz_extraction_failures.csv.
- PARALLELIZED across subject-condition jobs via joblib (see N_JOBS below).
  Each job (one subject, one condition) is independent, so this parallelizes
  cleanly with no shared state between workers. BLAS thread counts are
  pinned to 1 per worker (see the os.environ block below) so joblib's
  process-level parallelism doesn't get oversubscribed by each worker also
  trying to multithread its own linear algebra - skipping that step is the
  single most common way to make "parallel" code slower than sequential.
- To debug a single subject/condition without spinning up the whole pool,
  call _extract_one(subject_id, condition_label, task, DATA_ROOT) directly
  in a REPL/notebook - it's a plain function, no joblib involved until
  batch_extract() wraps it.
"""

import os

# CRITICAL - must happen before numpy/mne get imported in each worker process
# (workers are separate OS processes that import fresh; they inherit this
# environment). Without this, every worker ALSO multithreads its own BLAS
# calls on top of joblib's own process-level parallelism, oversubscribing
# your cores and making this run SLOWER than sequential, not faster.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path
import traceback
import warnings

import pandas as pd
from joblib import Parallel, delayed

from braboszcz_feature_extraction import process_subject_condition, COMMON_CHANNELS_14

DATA_ROOT = ("/Users/sarshanspal/Library/CloudStorage/"
             "GoogleDrive-sarahanspal@gmail.com/My Drive/meditation-eeg/delorme2")

# EDIT: full subject list for the batch. Casts a broad net (sub-005..sub-098)
# since not every number in that range necessarily exists/has complete data -
# missing ones are skipped and logged, not fatal.
SUBJECT_IDS = [f"sub-{i:03d}" for i in range(5, 99)]

RUNS = {
    "MED": "task-med2",
    "IMW": "task-think2",
}

# -1 = use every core; -2 = every core but one (default - leaves you a core
# to keep using the machine while this runs).
N_JOBS = 7

# ============================================================================
# GROUP FILTERING - exclude CTR, cap each tradition at 16 subjects (paper's
# matched-group design). UNCONFIRMED: exact column/label names in
# participants.tsv. This prints what it finds so you can correct
# GROUP_COLUMN / GROUP_LABEL_MAP below on the first run rather than silently
# filtering on a guess.
# ============================================================================
PARTICIPANTS_TSV = Path(DATA_ROOT) / "participants.tsv"
GROUP_COLUMN = None  # None = auto-detect from a candidate list; set explicitly
                      # once you see the printed column list if auto-detect misses
GROUP_COLUMN_CANDIDATES = ["group", "Group", "tradition", "Tradition",
                           "condition", "meditation_group", "Group_x"]
# Maps raw values found in the group column to our canonical labels.
# VERIFIED against the actual participants.tsv (not a guess): the real column
# uses ctr/htr/sny/vip/tm. There is a 5th group, "tm" (4 subjects: sub-056,
# sub-057, sub-058, sub-059; 20-35 years of practice each), that appears
# NOWHERE in the paper's text - possibly Transcendental Meditation, recruited
# but excluded from the published analysis, but that's a guess. Deliberately
# left OUT of this map so it's excluded by default rather than silently
# mapped to something wrong - if you confirm what it is, add it explicitly.
GROUP_LABEL_MAP = {
    "htr": "HYT",   # Himalayan Yoga Tradition
    "vip": "VIP",   # Vipassana
    "sny": "ISY",   # Isha Shoonya
    "ctr": "CTR",   # Control (excluded via INCLUDE_GROUPS below regardless)
}
# ============================================================================
# GROUP FILTERING
# ============================================================================
N_PER_GROUP_CAP = None  # None = no cap, use every available subject per style.
                         # (was 16, matching the paper's matched-group design;
                         # you've now decided to use everyone available instead)
INCLUDE_GROUPS = ("HYT", "VIP", "ISY")  # CTR intentionally excluded


def load_group_filtered_subjects(subject_ids_pool):
    """Reads participants.tsv, maps to canonical group labels, drops CTR,
    and optionally caps each remaining group at N_PER_GROUP_CAP subjects.
    Returns (kept_subject_ids, subject_to_style_dict) - the second return
    value is new, needed so main() can attach a 'style' column to the
    output CSV for later per-style classification.
    """
    if not PARTICIPANTS_TSV.exists():
        raise FileNotFoundError(
            f"participants.tsv not found at {PARTICIPANTS_TSV} - check DATA_ROOT "
            f"points at the dataset root (should contain participants.tsv alongside "
            f"the sub-XXX folders), not a subfolder."
        )
    pdf = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    print(f"[groups] participants.tsv columns: {list(pdf.columns)}")

    group_col = GROUP_COLUMN
    if group_col is None:
        for cand in GROUP_COLUMN_CANDIDATES:
            if cand in pdf.columns:
                group_col = cand
                break
    if group_col is None:
        raise ValueError(
            f"Could not auto-detect a group column among {GROUP_COLUMN_CANDIDATES}. "
            f"Actual columns are {list(pdf.columns)} - set GROUP_COLUMN explicitly."
        )
    print(f"[groups] using column '{group_col}'; unique raw values: "
          f"{sorted(pdf[group_col].dropna().unique().tolist())}")

    pdf["_canonical_group"] = pdf[group_col].map(GROUP_LABEL_MAP)
    unmapped = pdf[pdf["_canonical_group"].isna() & pdf[group_col].notna()][group_col].unique()
    if len(unmapped):
        warnings.warn(f"Unmapped raw group value(s) found: {unmapped.tolist()} - "
                       f"these subjects will be excluded. Add them to GROUP_LABEL_MAP "
                       f"if they should map to HYT/VIP/ISY/CTR.")

    id_col = "participant_id" if "participant_id" in pdf.columns else pdf.columns[0]
    pdf["_subject_id"] = pdf[id_col].astype(str).apply(
        lambda s: s if s.startswith("sub-") else f"sub-{s}"
    )

    kept = []
    subject_to_style = {}
    for group in INCLUDE_GROUPS:
        group_subjects = sorted(pdf.loc[pdf["_canonical_group"] == group, "_subject_id"].tolist())
        group_subjects = [s for s in group_subjects if s in subject_ids_pool]
        print(f"[groups] {group}: {len(group_subjects)} subject(s) available "
              f"in both participants.tsv and your data pool")
        if N_PER_GROUP_CAP is not None and len(group_subjects) > N_PER_GROUP_CAP:
            print(f"[groups] {group}: capping {len(group_subjects)} -> {N_PER_GROUP_CAP} "
                  f"(first {N_PER_GROUP_CAP} by subject ID)")
            group_subjects = group_subjects[:N_PER_GROUP_CAP]
        elif N_PER_GROUP_CAP is not None and len(group_subjects) < N_PER_GROUP_CAP:
            warnings.warn(f"{group}: only {len(group_subjects)} subjects available, "
                           f"fewer than the cap of {N_PER_GROUP_CAP}.")
        kept.extend(group_subjects)
        for s in group_subjects:
            subject_to_style[s] = group

    print(f"[groups] final filtered subject list: {len(kept)} subjects "
          f"({', '.join(kept)})")
    return kept, subject_to_style





def _extract_one(subject_id: str, condition_label: str, task: str, data_root: str) -> dict:
    """Runs in its own worker process - must stay a top-level function (not a
    nested closure/lambda) so joblib/multiprocessing can pickle it. Returns a
    plain dict rather than raising, so one bad subject can't take down the
    whole parallel batch (an uncaught exception in a worker DOES propagate
    and kill the pool with the default joblib settings, which is why every
    failure mode here is caught explicitly and returned instead)."""
    base = Path(data_root) / subject_id / "eeg"
    bdf_path = base / f"{subject_id}_{task}_eeg.bdf"
    channels_tsv = base / f"{subject_id}_{task}_channels.tsv"

    if not bdf_path.exists() or not channels_tsv.exists():
        return {"subject": subject_id, "condition": condition_label,
                "status": "skipped", "reason": "missing_file", "feat_df": None}

    try:
        feat_df, freqs, mean_psd, ch_names = process_subject_condition(
            bdf_path=str(bdf_path),
            channels_tsv_path=str(channels_tsv),
            subject_id=subject_id,
            condition_label=condition_label,
            keep_channels=COMMON_CHANNELS_14,
        )
        return {"subject": subject_id, "condition": condition_label, "status": "ok",
                "n_epochs": len(feat_df), "n_channels": len(ch_names), "feat_df": feat_df}
    except Exception as e:
        return {"subject": subject_id, "condition": condition_label, "status": "failed",
                "reason": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "feat_df": None}


def batch_extract(subject_ids=SUBJECT_IDS, data_root=DATA_ROOT, n_jobs=N_JOBS):
    jobs = [(s, cond, task) for s in subject_ids for cond, task in RUNS.items()]
    print(f"[batch] dispatching {len(jobs)} subject-condition jobs, n_jobs={n_jobs}")
    print("[batch] NOTE: with parallel workers, per-subject print statements "
          "from inside process_subject_condition may print out of order or in "
          "bursts as workers finish - that's expected, not a bug. joblib's own "
          "verbose progress line below is the reliable indicator of how far "
          "along the batch is.")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_extract_one)(subject_id, condition_label, task, data_root)
        for subject_id, condition_label, task in jobs
    )

    all_feats, failures = [], []
    for r in results:
        if r["status"] == "ok":
            print(f"[OK] {r['subject']} {r['condition']}: {r['n_epochs']} epochs, "
                  f"{r['n_channels']}/{len(COMMON_CHANNELS_14)} channels kept")
            all_feats.append(r["feat_df"])
        else:
            print(f"[{r['status'].upper()}] {r['subject']} {r['condition']}: {r['reason']}")
            if r["status"] == "failed":
                print(r["traceback"])
            failures.append((r["subject"], r["condition"], r["reason"]))

    if not all_feats:
        raise RuntimeError(
            "No subjects/conditions extracted successfully - check DATA_ROOT and "
            "SUBJECT_IDS before looking anywhere else."
        )

    master_df = pd.concat(all_feats, ignore_index=True)

    print("\n=== BATCH SUMMARY ===")
    print(f"Successful subject-condition runs: {len(all_feats)}")
    print(f"Failed/skipped: {len(failures)}")
    for f in failures:
        print(f"  {f}")
    print(f"Master table: {master_df.shape[0]} epochs x {master_df.shape[1]} columns")
    print(master_df.groupby(["subject", "condition"]).size())

    return master_df, failures


if __name__ == "__main__":
    filtered_subject_ids, subject_to_style = load_group_filtered_subjects(SUBJECT_IDS)
    master_df, failures = batch_extract(subject_ids=filtered_subject_ids)

    # NEW: attach style label per epoch, right after 'subject', so downstream
    # classification can filter/group by tradition without a re-merge.
    master_df.insert(1, "style", master_df["subject"].map(subject_to_style))
    print("\n[main] epochs per style:")
    print(master_df.groupby("style").size())

    out_path = "braboszcz_all_subjects_features.csv"
    master_df.to_csv(out_path, index=False)
    print(f"\nSaved master feature table to {out_path}")

    failures_df = pd.DataFrame(failures, columns=["subject", "condition", "reason"])
    failures_df.to_csv("braboszcz_extraction_failures2.csv", index=False)
    print("Saved failure log to braboszcz_extraction_failures.csv")