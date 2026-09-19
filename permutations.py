"""
permutation_test.py
Rigorous test: is real LOSOCV accuracy significantly above what you'd get
from randomly shuffled labels, given this exact data/fold structure?
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from config import FEATURES_HANDCRAFTED_ROOT
from harness import load_dataset, filter_multiclass_subjects, standardize_per_subject

N_PERMUTATIONS = 200
SEED = 42


def run_losocv_single_clf(X, y, groups, clf):
    logo = LeaveOneGroupOut()
    accs = []
    for train_idx, test_idx in logo.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(balanced_accuracy_score(y[test_idx], preds))
    return np.mean(accs)


def run_one_permutation(X, y, groups, unique_subjects, seed):
    rng = np.random.RandomState(seed)
    y_shuffled = y.copy()
    for s in unique_subjects:
        mask = groups == s
        if rng.choice([0, 1]) == 1:
            y_shuffled[mask] = 1 - y[mask]
    clf = RandomForestClassifier(random_state=SEED)
    return run_losocv_single_clf(X, y_shuffled, groups, clf)


def permutation_test(dataset_name, feature_root, n_perm=N_PERMUTATIONS, n_jobs=5):
    X, y, groups = load_dataset(dataset_name, feature_root)
    X, y, groups, _ = filter_multiclass_subjects(X, y, groups)
    X = standardize_per_subject(X, groups)
    unique_subjects = np.unique(groups)

    clf = RandomForestClassifier(random_state=SEED)
    real_acc = run_losocv_single_clf(X, y, groups, clf)
    print(f"{dataset_name}: real LOSOCV accuracy = {real_acc*100:.2f}%  ({len(unique_subjects)} subjects)")
    print(f"  running {n_perm} permutations, {n_jobs} at a time...")

    null_accs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_permutation)(X, y, groups, unique_subjects, seed)
        for seed in range(n_perm)
    )
    null_accs = np.array(null_accs)
    p_value = np.mean(null_accs >= real_acc)

    print(f"  null distribution: mean={null_accs.mean()*100:.2f}%, std={null_accs.std()*100:.2f}%")
    print(f"  p-value: {p_value:.4f}")
    print()
    return real_acc, null_accs, p_value


if __name__ == "__main__":
    results = {}
    for dataset in ["brandmeyer", "sky", "braboszcz"]:
        results[dataset] = permutation_test(dataset, FEATURES_HANDCRAFTED_ROOT)

    print("=== SUMMARY ===")
    for dataset, (real_acc, null_accs, p_value) in results.items():
        sig = "significant" if p_value < 0.05 else "not significant"
        print(f"{dataset}: real={real_acc*100:.2f}%, p={p_value:.4f} ({sig})")