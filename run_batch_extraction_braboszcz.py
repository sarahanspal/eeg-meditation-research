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
- Per-subject epoch counts print as it goes, so a subject with suspiciously
  few/zero clean epochs is visible immediately rather than only surfacing
  once the classifier trains on garbage.
"""

from pathlib import Path
import traceback

import pandas as pd

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


def batch_extract(subject_ids=SUBJECT_IDS, data_root=DATA_ROOT):
    all_feats = []
    failures = []

    for si, subject_id in enumerate(subject_ids):
        base = Path(data_root) / subject_id / "eeg"
        for condition_label, task in RUNS.items():
            bdf_path = base / f"{subject_id}_{task}_eeg.bdf"
            channels_tsv = base / f"{subject_id}_{task}_channels.tsv"
            if not bdf_path.exists() or not channels_tsv.exists():
                print(f"[SKIP] {subject_id} {condition_label}: file(s) missing at {bdf_path}")
                failures.append((subject_id, condition_label, "missing_file"))
                continue
            try:
                feat_df, freqs, mean_psd, ch_names = process_subject_condition(
                    bdf_path=str(bdf_path),
                    channels_tsv_path=str(channels_tsv),
                    subject_id=subject_id,
                    condition_label=condition_label,
                    keep_channels=COMMON_CHANNELS_14,
                )
                print(f"[OK] ({si+1}/{len(subject_ids)}) {subject_id} {condition_label}: "
                      f"{len(feat_df)} epochs, {feat_df.shape[1]} columns, "
                      f"{len(ch_names)}/{len(COMMON_CHANNELS_14)} channels kept")
                all_feats.append(feat_df)
            except Exception as e:
                print(f"[FAIL] {subject_id} {condition_label}: {e}")
                traceback.print_exc()
                failures.append((subject_id, condition_label, str(e)))

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
    master_df, failures = batch_extract()

    out_path = "braboszcz_all_subjects_features.csv"
    master_df.to_csv(out_path, index=False)
    print(f"\nSaved master feature table to {out_path}")

    failures_df = pd.DataFrame(failures, columns=["subject", "condition", "reason"])
    failures_df.to_csv("braboszcz_extraction_failures.csv", index=False)
    print("Saved failure log to braboszcz_extraction_failures.csv")