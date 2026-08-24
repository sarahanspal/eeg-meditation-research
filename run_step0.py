"""
Step 0 driver script.

Loops over all subjects in 3 datasets (Brandmeyer/OpenNeuro, SKY, ds003969/Braboszcz),
runs each dataset's loader (raw file -> preprocessed -> labeled 4s epochs), and saves
the result to a standardized file layout that every downstream script (FOOOF, feature
extraction, classifiers) reads from instead of touching raw data directly.

Mantra is intentionally excluded (all subjects experienced -> bias, see project notes).

Output:
  data/processed/<dataset>/<subject_id>_epochs.npy   (n_epochs, 14, 1024)
  data/processed/<dataset>/<subject_id>_labels.npy   (n_epochs,)
  data/subjects_metadata.csv                          one row per subject
"""

import os
import gc
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

from config import RAW_PATHS, PROCESSED_ROOT
from loaders import (
    load_openneuro_subject_session,
    load_sky_subject,
    load_ds3969_subject,
)
from experience_mapping import build_sky_experience_map, build_ds3969_experience_map

OUT_DIR = Path(PROCESSED_ROOT)


def save_subject(dataset_name, subject_id, epochs_array, labels_array):
    """
    Writes one subject's epochs + labels to disk in the shared layout.
    Asserts shape correctness first so a bad loader fails loudly here,
    not silently three scripts downstream.
    """
    assert epochs_array.shape[1] == 14, f"expected 14 channels, got {epochs_array.shape[1]}"
    assert epochs_array.shape[2] == 1024, f"expected 1024 samples, got {epochs_array.shape[2]}"
    assert epochs_array.shape[0] == labels_array.shape[0], "epoch count != label count"

    out_path = OUT_DIR / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / f"{subject_id}_epochs.npy", epochs_array)
    np.save(out_path / f"{subject_id}_labels.npy", labels_array)
    print(f"  {dataset_name}/{subject_id}: saved {epochs_array.shape[0]} epochs")


# accumulates one row per subject; written to subjects_metadata.csv at the very end
metadata_rows = []

# ============ Brandmeyer ============
# 19 subjects x up to 2 sessions each = up to 38 subject-sessions.
# Experience label comes directly from participants.tsv's 'group' column
# (group == 'expert'), NOT a hardcoded subject-ID range -- this is the
# ground-truth source, safer than assuming a fixed ID cutoff.
print("=== Brandmeyer ===")
on_meta = pd.read_csv(os.path.join(RAW_PATHS["brandmeyer"], "participants.tsv"), sep="\t")
on_exp_map = {}
for _, row in on_meta.iterrows():
    subj_num = int(row["participant_id"].split("-")[1])
    on_exp_map[subj_num] = 1 if row["group"] == "expert" else 0

for i in range(1, 20):
    subj_str = f"sub-{i:03d}"
    for ses_num in [1, 2]:
        # loader handles missing files internally, returns (None, None) if skipped
        epochs, labels = load_openneuro_subject_session(RAW_PATHS["brandmeyer"], subj_str, ses_num)
        if epochs is None:
            continue

        subject_id = f"{subj_str}_ses-{ses_num:02d}"
        save_subject("brandmeyer", subject_id, epochs, labels)

        exp_val = on_exp_map.get(i, -1)
        metadata_rows.append({
            "dataset": "brandmeyer",
            "subject_id": subject_id,
            "n_epochs": epochs.shape[0],
            "label_type": "binary",  # 1=focused meditation, 0=mind-wandering
            "experience_group": "experienced" if exp_val == 1 else ("novice" if exp_val == 0 else None),
        })

        # free memory immediately -- don't let 38 subject-sessions of epoch arrays
        # pile up in RAM at once, we only need them saved to disk
        del epochs, labels
        gc.collect()

# ============ SKY ============
# 40 subjects, pre-SKY baseline (label 0) vs post-SKY rest (label 1).
# 'timings' array tells the loader exactly where each epoch starts in the raw signal.
# Experience map includes manual corrections for two subject-ID edge cases
# found during the original analysis (see experience_mapping.py docstring).
print("=== SKY ===")
mat = scipy.io.loadmat(os.path.join(RAW_PATHS["sky"], "subjects_epochs.mat"))
timings = mat["timings"]
sky_exp_map = build_sky_experience_map(RAW_PATHS["sky"])

for subj_idx in range(40):  # 0-indexed subject IDs, per SKY's file naming convention
    epochs, labels = load_sky_subject(RAW_PATHS["sky"], subj_idx, timings)
    if epochs is None:
        continue

    subject_id = f"sky-{subj_idx+1:02d}"
    save_subject("sky", subject_id, epochs, labels)

    exp_val = sky_exp_map.get(subj_idx, -1)
    metadata_rows.append({
        "dataset": "sky",
        "subject_id": subject_id,
        "n_epochs": epochs.shape[0],
        "label_type": "binary",  # 1=post-SKY rest, 0=pre-SKY baseline
        "experience_group": "experienced" if exp_val == 1 else ("novice" if exp_val == 0 else None),
    })
    del epochs, labels
    gc.collect()

# ============Braboszcz ============
# 64 subjects, 4 task files each (med1breath/med2 = meditation, think1/think2 = IMW control).
# This is the biggest dataset (~57K epochs total) -- processed one subject at a time
# and saved immediately, so we never hold more than one subject's epochs in memory.
# Experience map includes manual corrections for two NaN-years subjects confirmed
# experienced via their group/notes columns (see experience_mapping.py docstring).
print("=== Braboszcz  ===")
subs_3969 = sorted([f for f in os.listdir(RAW_PATHS["braboszcz"]) if f.startswith("sub-")])
ds3969_exp_map = build_ds3969_experience_map(RAW_PATHS["braboszcz"])

for subj_str in subs_3969:
    epochs, labels = load_ds3969_subject(RAW_PATHS["braboszcz"], subj_str)
    if epochs is None:
        continue

    save_subject("braboszcz", subj_str, epochs, labels)

    subj_num = int(subj_str.split("-")[1])
    exp_val = ds3969_exp_map.get(subj_num, -1)
    metadata_rows.append({
        "dataset": "braboszcz",
        "subject_id": subj_str,
        "n_epochs": epochs.shape[0],
        "label_type": "binary",  # 1=meditation, 0=instructed mind-wandering
        "experience_group": "experienced" if exp_val == 1 else ("novice" if exp_val == 0 else None),
    })
    del epochs, labels
    gc.collect()

# ============ Save metadata + summary ============
# One row per subject across all 3 datasets -- this is what everyone (A/B/C) reads
# instead of touching raw files or re-deriving subject counts themselves.
df = pd.DataFrame(metadata_rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/subjects_metadata.csv", index=False)

print("\n=== DONE ===")
print(df.groupby("dataset").agg(subjects=("subject_id", "count"), total_epochs=("n_epochs", "sum")))
print("\n=== Experience breakdown ===")
print(df.groupby(["dataset", "experience_group"]).size())