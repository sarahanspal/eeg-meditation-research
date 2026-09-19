"""
run_losocv_by_tradition.py
Re-runs Braboszcz LOSOCV as 3 separate single-tradition tasks
(HTR/MW, VIP/MW, SNY/MW), matching the Frontiers paper's SAM/VIP/MET
structure -- instead of one pooled "any meditation" vs MW task.
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_PATHS, PROCESSED_ROOT, FEATURES_HANDCRAFTED_ROOT
from harness import filter_multiclass_subjects, standardize_per_subject, CLASSIFIERS
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

meta = pd.read_csv(f"{RAW_PATHS['braboszcz']}/participants.tsv", sep="\t")
group_map = dict(zip(meta["participant_id"], meta["group"]))

def load_tradition_subset(tradition_code):
    """Loads only subjects belonging to one tradition (e.g. 'htr', 'vip', 'sny')."""
    feature_dir = Path(FEATURES_HANDCRAFTED_ROOT) / "braboszcz"
    processed_dir = Path(PROCESSED_ROOT) / "braboszcz"

    X_list, y_list, groups_list = [], [], []
    for feat_file in sorted(feature_dir.glob("*_features.npy")):
        subject_id = feat_file.stem.replace("_features", "")
        if group_map.get(subject_id) != tradition_code:
            continue
        X = np.load(feat_file)
        y = np.load(processed_dir / f"{subject_id}_labels.npy")
        X_list.append(X); y_list.append(y)
        groups_list.append(np.full(y.shape[0], subject_id))

    if not X_list:
        return None, None, None
    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)

for tradition in ["htr", "vip", "sny"]:
    X, y, groups = load_tradition_subset(tradition)
    if X is None:
        print(f"{tradition}: no subjects found"); continue

    X, y, groups, excluded = filter_multiclass_subjects(X, y, groups)
    X = standardize_per_subject(X, groups)

    logo = LeaveOneGroupOut()
    results = {name: [] for name in CLASSIFIERS}
    for train_idx, test_idx in logo.split(X, y, groups):
        for name, clf in CLASSIFIERS.items():
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            results[name].append(balanced_accuracy_score(y[test_idx], preds) * 100)

    print(f"\n=== {tradition} vs MW (N={len(np.unique(groups))} subjects) ===")
    for name, scores in results.items():
        print(f"  {name}: {np.mean(scores):.1f}% ± {np.std(scores):.1f}%")