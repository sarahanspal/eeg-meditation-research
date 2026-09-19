"""
csp_harness.py
LOSOCV using CSP (Common Spatial Patterns) instead of the 406 handcrafted
features. CSP is fit fresh inside each fold on the raw epoch arrays --
never precomputed, since fitting it on all data first would leak the
held-out subject's class structure into the spatial filters.
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from mne.decoding import CSP
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

from config import PROCESSED_ROOT
from harness import filter_multiclass_subjects


def load_raw_epochs(dataset_name):
    processed_dir = Path(PROCESSED_ROOT) / dataset_name
    X_list, y_list, groups_list = [], [], []
    for label_file in sorted(processed_dir.glob("*_labels.npy")):
        subject_id = label_file.stem.replace("_labels", "")
        X = np.load(processed_dir / f"{subject_id}_epochs.npy").astype(np.float64)
        y = np.load(label_file)
        X_list.append(X)
        y_list.append(y)
        groups_list.append(np.full(y.shape[0], subject_id))
    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)


def run_one_fold(X, y, groups, train_idx, test_idx, n_components, dataset_name, fold_num, n_folds):
    test_subject = groups[test_idx][0]
    start = time.time()
    print(f"  [{dataset_name}] START fold {fold_num}/{n_folds} (subject {test_subject})", flush=True)

    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False, rank="full")
    X_train_csp = csp.fit_transform(X[train_idx], y[train_idx])
    X_test_csp = csp.transform(X[test_idx])

    clf = SVC(kernel="linear", C=1.0, random_state=42)
    clf.fit(X_train_csp, y[train_idx])
    preds = clf.predict(X_test_csp)
    acc = balanced_accuracy_score(y[test_idx], preds)

    elapsed = time.time() - start
    print(f"  [{dataset_name}] DONE  fold {fold_num}/{n_folds} (subject {test_subject}) "
          f"-> {acc*100:.1f}% in {elapsed:.1f}s", flush=True)

    return {"dataset": dataset_name, "subject": test_subject, "accuracy": acc * 100}


def run_losocv_csp_parallel(dataset_name, n_components=6, n_jobs=5):
    X, y, groups = load_raw_epochs(dataset_name)
    X, y, groups, excluded = filter_multiclass_subjects(X, y, groups)

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups))
    n_folds = len(splits)
    print(f"\n=== {dataset_name}: {n_folds} folds, {n_jobs} at a time (started {time.strftime('%H:%M:%S')}) ===", flush=True)

    dataset_start = time.time()
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(run_one_fold)(X, y, groups, train_idx, test_idx, n_components, dataset_name, i+1, n_folds)
        for i, (train_idx, test_idx) in enumerate(splits)
    )
    dataset_elapsed = time.time() - dataset_start

    accs = [r["accuracy"] for r in results]
    print(f"=== {dataset_name} FINISHED in {dataset_elapsed/60:.1f} min: "
          f"{np.mean(accs):.2f}% ± {np.std(accs):.2f}% ===\n", flush=True)
    return results, excluded


def save_results(new_results, csv_path="results/csp_losocv_persubject.csv"):
    """Append new fold results to the CSV, replacing any existing rows
    for the same dataset(s) so reruns don't create duplicates."""
    new_df = pd.DataFrame(new_results)
    csv_path = Path(csv_path)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        datasets_in_new = new_df["dataset"].unique()
        existing_df = existing_df[~existing_df["dataset"].isin(datasets_in_new)]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(csv_path, index=False)
    return combined_df


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    all_results = []
    all_excluded = []
    overall_start = time.time()

    for dataset in ["braboszcz"]:
        results, excluded = run_losocv_csp_parallel(dataset, n_jobs=4)
        all_results.extend(results)
        for subj, n_epochs, classes in excluded:
            all_excluded.append({"dataset": dataset, "subject": subj,
                                  "n_epochs": n_epochs, "classes_present": classes})
        save_results(all_results)
        print(f"  [saved progress to results/csp_losocv_persubject.csv]\n", flush=True)

    if all_excluded:
        pd.DataFrame(all_excluded).to_csv("results/csp_losocv_excluded.csv", index=False)

    total_elapsed = time.time() - overall_start
    print(f"=== ALL DONE in {total_elapsed/60:.1f} min ===")
    full_df = pd.read_csv("results/csp_losocv_persubject.csv")
    print(full_df.groupby("dataset")["accuracy"].agg(["mean", "std", "count"]))