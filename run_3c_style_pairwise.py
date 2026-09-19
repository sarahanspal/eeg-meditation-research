"""
run_3c_style_pairwise.py
Braboszcz-only pairwise tradition classification: using ONLY each subject's
meditation-condition epochs (y==1), can a classifier tell which tradition
(ctr/htr/sny/vip) a held-out subject belongs to?

Heeds prior findings:
- LOSO across subjects (not intra-subject k-fold) -- avoids the epoch-order
  leak found earlier, since same-subject epochs never split train/test.
- Runs BOTH the 406 handcrafted features AND CSP on raw epochs, since these
  showed very different behavior in the binary meditation/non-meditation task.
- Parallelized per fold, incremental save per comparison pair.
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from mne.decoding import CSP
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from config import RAW_PATHS, PROCESSED_ROOT, FEATURES_HANDCRAFTED_ROOT

print("Script started, loading data...", flush=True)

TRADITIONS = ["ctr", "htr", "sny", "vip"]

meta = pd.read_csv(f"{RAW_PATHS['braboszcz']}/participants.tsv", sep="\t")
group_map = dict(zip(meta["participant_id"], meta["group"]))


def load_meditation_only(dataset_name, feature_root, kind="handcrafted"):
    """
    Loads only y==1 (meditation) epochs per subject, tagged with subject_id
    and tradition group. kind='handcrafted' loads the 406-feature files;
    kind='raw' loads raw epoch arrays for CSP.
    """
    processed_dir = Path(PROCESSED_ROOT) / dataset_name
    X_list, subj_list, tradition_list = [], [], []

    for label_file in sorted(processed_dir.glob("*_labels.npy")):
        subject_id = label_file.stem.replace("_labels", "")
        tradition = group_map.get(subject_id)
        if tradition not in TRADITIONS:
            continue  # skip 'tm' or unmapped subjects

        y = np.load(label_file)
        med_mask = y == 1
        if med_mask.sum() == 0:
            continue

        if kind == "handcrafted":
            X = np.load(Path(feature_root) / dataset_name / f"{subject_id}_features.npy")
        else:
            X = np.load(processed_dir / f"{subject_id}_epochs.npy")

        X_list.append(X[med_mask])
        subj_list.append(np.full(med_mask.sum(), subject_id))
        tradition_list.append(np.full(med_mask.sum(), tradition, dtype=object))

    return (np.vstack(X_list) if kind == "handcrafted" else np.concatenate(X_list, axis=0),
            np.concatenate(subj_list), np.concatenate(tradition_list))


def standardize_per_subject(X, groups):
    X_scaled = np.zeros_like(X, dtype=float)
    for subj in np.unique(groups):
        mask = groups == subj
        X_scaled[mask] = StandardScaler().fit_transform(X[mask])
    return X_scaled


def run_one_fold_handcrafted(X, y, groups, train_idx, test_idx):
    clf = SVC(kernel="linear", C=1.0, random_state=42)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    return balanced_accuracy_score(y[test_idx], preds)


def run_one_fold_csp(X, y, groups, train_idx, test_idx, n_components=6):
    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X[train_idx], y[train_idx])
    X_test_csp = csp.transform(X[test_idx])
    clf = SVC(kernel="linear", C=1.0, random_state=42)
    clf.fit(X_train_csp, y[train_idx])
    preds = clf.predict(X_test_csp)
    return balanced_accuracy_score(y[test_idx], preds)


def run_pairwise(tradition_a, tradition_b, n_jobs=5):
    print(f"\n=== {tradition_a} vs {tradition_b} ===", flush=True)

    results_row = {"pair": f"{tradition_a}_vs_{tradition_b}"}

    for kind in ["handcrafted", "csp"]:
        feature_root = FEATURES_HANDCRAFTED_ROOT if kind == "handcrafted" else None
        X_all, subj_all, trad_all = load_meditation_only("braboszcz", feature_root, kind=kind)

        mask = np.isin(trad_all, [tradition_a, tradition_b])
        X, subj, trad = X_all[mask], subj_all[mask], trad_all[mask]
        y = (trad == tradition_b).astype(int)

        n_subj_a = len(np.unique(subj[trad == tradition_a]))
        n_subj_b = len(np.unique(subj[trad == tradition_b]))
        print(f"  [{kind}] {tradition_a}: {n_subj_a} subjects, {tradition_b}: {n_subj_b} subjects", flush=True)

        if kind == "handcrafted":
            X = standardize_per_subject(X, subj)

        logo = LeaveOneGroupOut()
        splits = list(logo.split(X, y, subj))
        run_fold = run_one_fold_handcrafted if kind == "handcrafted" else run_one_fold_csp

        start = time.time()
        accs = Parallel(n_jobs=n_jobs)(
            delayed(run_fold)(X, y, subj, train_idx, test_idx)
            for train_idx, test_idx in splits
        )
        elapsed = time.time() - start

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print(f"  [{kind}] {mean_acc:.1f}% ± {std_acc:.1f}% ({len(splits)} folds, {elapsed:.1f}s)", flush=True)

        results_row[f"{kind}_mean_acc"] = mean_acc
        results_row[f"{kind}_std_acc"] = std_acc
        results_row[f"{kind}_n_folds"] = len(splits)

    return results_row


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    all_rows = []

    for tradition_a, tradition_b in itertools.combinations(TRADITIONS, 2):
        row = run_pairwise(tradition_a, tradition_b)
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv("results/braboszcz_3c_pairwise.csv", index=False)
        print(f"  [saved progress to results/braboszcz_3c_pairwise.csv]", flush=True)

    print("\n=== ALL PAIRWISE COMPARISONS DONE ===")
    print(pd.DataFrame(all_rows))