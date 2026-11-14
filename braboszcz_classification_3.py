"""
braboszcz_style_classification.py
==================================
Classifies meditation STYLE (HYT vs VIP vs ISY) from MED-condition epochs
only - pairwise (HYT-VIP, HYT-ISY, VIP-ISY) and 3-class (all three).

WHY INTER-SUBJECT ONLY
-----------------------
Style is a between-subject variable: every epoch from a given subject has
the same style label (whatever tradition that person practices). There is
no intra-subject version of "which style is this" - a subject can't be
compared against themselves. So this script only runs LOSO (leave-one-
subject-out), which is also the ONLY safe way to split here: style is
fully confounded with subject identity, so any non-subject-level split
would let the classifier learn individual quirks instead of style signal.

WHY MED-CONDITION EPOCHS ONLY
-------------------------------
This mirrors the paper's Figure 4 panels B/C (pairwise and 3-class
comparisons BETWEEN meditation styles, as opposed to panel A which is
each style vs mind-wandering, already done in braboszcz_classification.py).
IMW epochs carry no style information by construction (mind-wandering
isn't a "style"), so they're excluded here.

EPOCH FLOOR - REAPPLIED, NOT INHERITED
-----------------------------------------
braboszcz_classification.py's MIN_EPOCHS_PER_CONDITION filter lived in
memory in that script and was never written back to
braboszcz_all_subjects_features.csv. This script reads that same raw CSV,
so subjects with too few MED epochs (sub-097: 1 epoch, sub-093: 19, etc.)
are still in there and get re-filtered here independently.

NaN HANDLING
------------
Same global-median-fill approach as braboszcz_classification.py: computed
once across ALL MED epochs (all styles, all subjects) before any split.
This is safe against STYLE leakage for the same reason it was safe
against CONDITION leakage before - it's a per-feature-column constant, not
correlated with the label we're predicting. Worth knowing this assumption
exists; if you want to stress-test it, compare per-style missingness
rates before trusting borderline results.

USAGE
-----
    python braboszcz_style_classification.py
"""

import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

RANDOM_SEED = 42

# ============================================================================
# CONFIG
# ============================================================================
FEATURES_CSV = "braboszcz_all_subjects_features.csv"
META_COLS = ["subject", "style", "condition", "epoch_idx"]

APPLY_SCALING = True
MIN_EPOCHS_MED = 20   # see module docstring - reapplied independently here
N_JOBS = -2

PAIRWISE_TASKS = [("HYT", "VIP"), ("HYT", "ISY"), ("VIP", "ISY")]
THREE_CLASS_STYLES = ["HYT", "VIP", "ISY"]

OUT_PAIRWISE_TEMPLATE = "braboszcz_style_pairwise_{a}_vs_{b}.csv"
OUT_THREECLASS_CSV = "braboszcz_style_threeclass.csv"


# ============================================================================
# CLASSIFIERS - same spec as braboszcz_classification.py
# ============================================================================
def build_classifiers():
    return {
        "kNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED),
        "SVM-linear": SVC(kernel="linear", C=1.0, random_state=RANDOM_SEED),
        "SVM-RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED),
        "GaussianNB": GaussianNB(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
            random_state=RANDOM_SEED, max_iter=1000,
        ),
    }


class DropZeroVarianceColumns:
    """Same fix as braboszcz_classification.py - fits on training fold only,
    drops columns with zero variance there before scaling. Prevents the
    divide-by-zero-in-matmul warnings from a subject-constant feature."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_mask_ = np.nanstd(X, axis=0) > 0
        if not self.keep_mask_.any():
            raise ValueError("All columns are zero-variance in this training fold.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_mask_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def make_pipeline(clf, apply_scaling=APPLY_SCALING):
    steps = [("drop_zero_var", DropZeroVarianceColumns())]
    if apply_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ============================================================================
# BLOCK 1 - LOAD MED-ONLY FEATURE TABLE
# ============================================================================
def load_med_style_table(csv_path=FEATURES_CSV):
    df = pd.read_csv(csv_path)
    print(f"[load] {df.shape[0]} rows x {df.shape[1]} columns from {csv_path}")

    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing expected metadata columns: {missing_meta}")

    df = df[df["condition"] == "MED"].reset_index(drop=True)
    print(f"[load] {len(df)} MED-condition epochs after dropping IMW")

    feature_cols = [c for c in df.columns if c not in META_COLS]

    all_nan_mask = df[feature_cols].isna().all(axis=0)
    if all_nan_mask.any():
        bad_cols = all_nan_mask[all_nan_mask].index.tolist()
        warnings.warn(f"{len(bad_cols)} feature column(s) are ALL NaN across every "
                       f"MED epoch - dropping: {bad_cols}")
        feature_cols = [c for c in feature_cols if c not in bad_cols]

    # Global median fill across all MED epochs, all styles, before any split -
    # see module docstring "NaN HANDLING" for why this is safe here.
    global_medians = df[feature_cols].median()
    still_all_nan = global_medians[global_medians.isna()].index.tolist()
    if still_all_nan:
        warnings.warn(f"{len(still_all_nan)} feature column(s) NaN across the "
                       f"entire MED population - dropping: {still_all_nan}")
        feature_cols = [c for c in feature_cols if c not in still_all_nan]
        global_medians = global_medians.drop(index=still_all_nan)
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(global_medians)
    print(f"[load] filled remaining NaNs with global per-feature medians; "
          f"{df[feature_cols].isna().sum().sum()} NaN cells remain (should be 0)")

    per_subject_counts = df.groupby("subject").size()
    too_few = per_subject_counts[per_subject_counts < MIN_EPOCHS_MED]
    if len(too_few):
        print(f"[load] dropping {len(too_few)} subject(s) below "
              f"MIN_EPOCHS_MED={MIN_EPOCHS_MED}:")
        print(too_few)
        df = df[~df["subject"].isin(too_few.index.tolist())]

    per_style_subjects = df.groupby("style")["subject"].nunique()
    print(f"[load] final subjects per style:\n{per_style_subjects}")
    print(f"[load] final: {df['subject'].nunique()} subjects, {len(df)} epochs")

    return df, feature_cols


# ============================================================================
# BLOCK 2 - LOSO STYLE CLASSIFICATION (pairwise or multi-class, same code)
# ============================================================================
def _run_style_loso_one(held_out, train_df, test_df, feature_cols, label_map):
    classifiers = build_classifiers()
    X_train = train_df[feature_cols].values
    y_train = train_df["style"].map(label_map).values
    X_test = test_df[feature_cols].values
    y_test = test_df["style"].map(label_map).values

    log = [f"[style-loso] held-out={held_out}: train n={len(y_train)} "
           f"(from {train_df['subject'].nunique()} subjects), test n={len(y_test)}"]

    assert held_out not in train_df["subject"].unique(), \
        f"LEAKAGE: {held_out} present in its own training set"

    rows = []
    for clf_name, clf in classifiers.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        rows.append({"held_out_subject": held_out, "classifier": clf_name,
                     "n_train": len(y_train), "n_test": len(y_test),
                     "accuracy": acc, "balanced_accuracy": bal_acc})
        log.append(f"    {clf_name}: accuracy = {acc:.3f}, balanced accuracy = {bal_acc:.3f}")
    return rows, log


def run_style_loso(df, feature_cols, label_map, n_jobs=N_JOBS):
    """df must already be filtered to exactly the subjects/styles wanted
    (pairwise: 2 styles; 3-class: 3 styles). label_map e.g. {"HYT": 0, "VIP": 1}."""
    subjects = sorted(df["subject"].unique())
    print(f"[style-loso] dispatching {len(subjects)} held-out-subject job(s), "
          f"n_jobs={n_jobs}, label_map={label_map}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_run_style_loso_one)(
            held_out, df[df["subject"] != held_out], df[df["subject"] == held_out],
            feature_cols, label_map
        )
        for held_out in subjects
    )

    all_rows = []
    for rows, log in results:
        for line in log:
            print(line)
        all_rows.extend(rows)

    result_df = pd.DataFrame(all_rows)
    if result_df.empty:
        warnings.warn("run_style_loso produced zero rows.")
    return result_df


# ============================================================================
# MAIN
# ============================================================================
def main():
    df, feature_cols = load_med_style_table()

    # --- Pairwise: HYT-VIP, HYT-ISY, VIP-ISY ---
    for style_a, style_b in PAIRWISE_TASKS:
        print("\n" + "#" * 70)
        print(f"PAIRWISE: {style_a} vs {style_b}")
        print("#" * 70)
        subset = df[df["style"].isin([style_a, style_b])].reset_index(drop=True)
        label_map = {style_a: 0, style_b: 1}
        result_df = run_style_loso(subset, feature_cols, label_map)
        if not result_df.empty:
            out_path = OUT_PAIRWISE_TEMPLATE.format(a=style_a, b=style_b)
            result_df.to_csv(out_path, index=False)
            print(f"Saved {out_path}")
            print(result_df.groupby("classifier")[["accuracy", "balanced_accuracy"]]
                  .agg(["mean", "std", "count"]))

    # --- 3-class: HYT vs VIP vs ISY ---
    print("\n" + "#" * 70)
    print(f"3-CLASS: {' vs '.join(THREE_CLASS_STYLES)}")
    print("#" * 70)
    subset = df[df["style"].isin(THREE_CLASS_STYLES)].reset_index(drop=True)
    label_map = {s: i for i, s in enumerate(THREE_CLASS_STYLES)}
    result_df = run_style_loso(subset, feature_cols, label_map)
    if not result_df.empty:
        result_df.to_csv(OUT_THREECLASS_CSV, index=False)
        print(f"Saved {OUT_THREECLASS_CSV}")
        print(result_df.groupby("classifier")[["accuracy", "balanced_accuracy"]]
              .agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()