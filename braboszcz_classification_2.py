"""
braboszcz_classification.py
=============================
[... module docstring unchanged from before, plus:]

NEW (per-style runs): STYLE_LABELS below lists which style subsets to run
separately (HYT-only, VIP-only, ISY-only). For each style, intra-subject
and inter-subject classification run independently, using only that
style's subjects - this is what you asked for now, before the pairwise/
3-class step. A subject appears in exactly one style (see 'style' column,
added by the updated run_batch_extraction_braboszcz.py), so there's no
cross-contamination between style runs.

NEW (epoch floor): MIN_EPOCHS_PER_CONDITION drops a subject ENTIRELY
(not just from intra-subject folds) if either MED or IMW has fewer than
this many clean epochs. Rationale: a subject with 1-3 epochs contributes
nothing to intra (already skipped by the n_folds check) but still gets
used as a LOSO test subject, where 1-3 test samples means the reported
accuracy for that fold is close to a coin flip - noise inflating your
LOSO variance for no real signal. Default 20 is a judgment call, not
derived from anything - change it if you have a principled reason to.
"""

import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # CHANGED: added balanced_accuracy_score

RANDOM_SEED = 42

# ============================================================================
# CONFIG
# ============================================================================
FEATURES_CSV = "braboszcz_all_subjects_features.csv"
META_COLS = ["subject", "style", "condition", "epoch_idx"]  # CHANGED: added 'style'
LABEL_MAP = {"IMW": 0, "MED": 1}

APPLY_SCALING = True
N_INTRA_FOLDS = 10
DEBUG_SUBJECT_LIMIT = None
RUN_INTRA = True
RUN_INTER = True
N_JOBS = -2

# NEW: minimum clean epochs required in EACH condition for a subject to be
# used at all (intra or inter). Subjects below this are dropped entirely
# and logged - see module docstring above for why.
MIN_EPOCHS_PER_CONDITION = 20

# NEW: which style subsets to run separately. Each entry runs its own
# full intra+inter pass, filtered to only that style's subjects.
STYLE_LABELS = ["HYT", "VIP", "ISY"]

OUT_INTRA_CSV_TEMPLATE = "braboszcz_intra_subject_accuracies_{style}.csv"
OUT_INTER_CSV_TEMPLATE = "braboszcz_inter_subject_accuracies_{style}.csv"


# ============================================================================
# CLASSIFIERS - unchanged
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


# NEW: drops zero-variance columns before scaling. A feature nulled across
# a subject's whole column (see feature_extraction's leak-fix comment) is
# exactly constant for that subject, which makes StandardScaler divide by
# zero internally - harmless numerically (the warnings were noise, not
# wrong output) but noisy and worth doing properly.
class DropZeroVarianceColumns:
    """Fits on training data only; drops columns with zero variance there.
    Not a sklearn TransformerMixin subclass to keep this dependency-free -
    just enough interface for Pipeline to call fit/transform."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_mask_ = np.nanstd(X, axis=0) > 0
        if not self.keep_mask_.any():
            raise ValueError("All columns are zero-variance in this training fold - "
                              "cannot proceed. This should not happen in practice; "
                              "if it does, something upstream is wrong.")
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
    steps = [("drop_zero_var", DropZeroVarianceColumns())]  # CHANGED: added this step
    if apply_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ============================================================================
# BLOCK 1 - LOAD + SANITY CHECK
# ============================================================================
def load_feature_table(csv_path=FEATURES_CSV):
    df = pd.read_csv(csv_path)
    print(f"[load] {df.shape[0]} rows x {df.shape[1]} columns from {csv_path}")

    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing expected metadata columns: {missing_meta}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"[load] {len(feature_cols)} feature columns")

    all_nan_mask = df[feature_cols].isna().all(axis=0)
    if all_nan_mask.any():
        bad_cols = all_nan_mask[all_nan_mask].index.tolist()
        warnings.warn(f"{len(bad_cols)} feature column(s) are ALL NaN across every "
                       f"epoch - dropping: {bad_cols}")
        feature_cols = [c for c in feature_cols if c not in bad_cols]

    nan_frac_before = df[feature_cols].isna().mean().mean()
    print(f"[load] mean fraction of NaN across feature cells before global fill: "
          f"{nan_frac_before:.4f}")

    # global median fill - unchanged from before
    subject_feature_any_nan = df.groupby("subject")[feature_cols].transform(
        lambda col: col.isna().any()
    )
    df[feature_cols] = df[feature_cols].mask(subject_feature_any_nan)
    global_medians = df[feature_cols].median()
    still_all_nan = global_medians[global_medians.isna()].index.tolist()
    if still_all_nan:
        warnings.warn(f"{len(still_all_nan)} feature column(s) are NaN for the ENTIRE "
                       f"population - dropping them: {still_all_nan}")
        feature_cols = [c for c in feature_cols if c not in still_all_nan]
        global_medians = global_medians.drop(index=still_all_nan)
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(global_medians)
    print(f"[load] filled remaining NaNs with global per-feature medians; "
          f"{df[feature_cols].isna().sum().sum()} NaN cells remain (should be 0)")

    unexpected = set(df["condition"].unique()) - set(LABEL_MAP.keys())
    if unexpected:
        raise ValueError(f"Unexpected condition label(s) {unexpected}")
    df = df.copy()
    df["label"] = df["condition"].map(LABEL_MAP)

    per_subject_counts = df.groupby(["subject", "condition"]).size().unstack(fill_value=0)
    print("[load] per-subject epoch counts by condition:")
    print(per_subject_counts)

    incomplete_mask = (per_subject_counts == 0).any(axis=1)
    if incomplete_mask.any():
        incomplete_subjects = per_subject_counts[incomplete_mask].index.tolist()
        warnings.warn(f"Dropping subject(s) missing one condition entirely: "
                       f"{incomplete_subjects}")
        df = df[~df["subject"].isin(incomplete_subjects)]

    # NEW: epoch-floor drop. Applied AFTER the missing-condition drop above,
    # on whatever's left.
    per_subject_counts2 = df.groupby(["subject", "condition"]).size().unstack(fill_value=0)
    too_few_mask = (per_subject_counts2 < MIN_EPOCHS_PER_CONDITION).any(axis=1)
    if too_few_mask.any():
        too_few_subjects = per_subject_counts2[too_few_mask]
        print(f"[load] dropping {too_few_mask.sum()} subject(s) below "
              f"MIN_EPOCHS_PER_CONDITION={MIN_EPOCHS_PER_CONDITION}:")
        print(too_few_subjects)
        df = df[~df["subject"].isin(too_few_subjects.index.tolist())]

    print(f"[load] final: {df['subject'].nunique()} subjects, {len(df)} epochs")
    return df, feature_cols


# ============================================================================
# BLOCK 2 - INTRA-SUBJECT (unchanged except accuracy -> accuracy + balanced_accuracy)
# ============================================================================
def _run_intra_subject_one(subject, sub_df, feature_cols, n_folds):
    classifiers = build_classifiers()
    X = sub_df[feature_cols].values
    y = sub_df["label"].values
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    log = [f"[intra] {subject}: n_epochs={len(y)} (MED={n_pos}, IMW={n_neg})"]

    if min(n_pos, n_neg) < n_folds:
        log.append(f"    SKIPPED: smaller class has only {min(n_pos, n_neg)} epochs, "
                    f"fewer than n_folds={n_folds}")
        return [], log

    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    rows = []
    for clf_name, clf in classifiers.items():
        fold_accs, fold_bal_accs = [], []
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            pipe = make_pipeline(clf)
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], y_pred)
            bal_acc = balanced_accuracy_score(y[test_idx], y_pred)  # NEW
            fold_accs.append(acc)
            fold_bal_accs.append(bal_acc)
            rows.append({"subject": subject, "classifier": clf_name, "fold": fold_i,
                         "n_train": len(train_idx), "n_test": len(test_idx),
                         "accuracy": acc, "balanced_accuracy": bal_acc})  # NEW column
        log.append(f"    {clf_name}: mean accuracy = {np.mean(fold_accs):.3f}, "
                   f"mean balanced accuracy = {np.mean(fold_bal_accs):.3f} "
                   f"over {len(fold_accs)} folds")
    return rows, log


def run_intra_subject(df, feature_cols, n_folds=N_INTRA_FOLDS,
                       subject_limit=DEBUG_SUBJECT_LIMIT, n_jobs=N_JOBS):
    subjects = sorted(df["subject"].unique())
    if subject_limit:
        subjects = subjects[:subject_limit]
        print(f"[intra][DEBUG] limited to {len(subjects)} subject(s): {subjects}")

    print(f"[intra] dispatching {len(subjects)} subject(s) across n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_run_intra_subject_one)(
            s, df[df["subject"] == s].reset_index(drop=True), feature_cols, n_folds
        )
        for s in subjects
    )

    all_rows = []
    for rows, log in results:
        for line in log:
            print(line)
        all_rows.extend(rows)

    fold_df = pd.DataFrame(all_rows)
    if fold_df.empty:
        warnings.warn("run_intra_subject produced zero rows.")
    return fold_df


# ============================================================================
# BLOCK 3 - INTER-SUBJECT (unchanged except accuracy -> accuracy + balanced_accuracy)
# ============================================================================
def _run_inter_subject_one(held_out, train_df, test_df, feature_cols):
    classifiers = build_classifiers()
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label"].values

    log = [f"[inter] held-out={held_out}: train n={len(y_train)} "
           f"(from {train_df['subject'].nunique()} subjects), test n={len(y_test)} "
           f"(MED={int((y_test == 1).sum())}, IMW={int((y_test == 0).sum())})"]

    assert held_out not in train_df["subject"].unique(), \
        f"LEAKAGE: {held_out} present in its own training set"

    rows = []
    for clf_name, clf in classifiers.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)  # NEW
        rows.append({"held_out_subject": held_out, "classifier": clf_name,
                     "n_train": len(y_train), "n_test": len(y_test),
                     "accuracy": acc, "balanced_accuracy": bal_acc})  # NEW column
        log.append(f"    {clf_name}: accuracy = {acc:.3f}, balanced accuracy = {bal_acc:.3f}")
    return rows, log


def run_inter_subject(df, feature_cols, subject_limit=DEBUG_SUBJECT_LIMIT, n_jobs=N_JOBS):
    subjects = sorted(df["subject"].unique())
    if subject_limit:
        subjects = subjects[:subject_limit]
        print(f"[inter][DEBUG] limited to {len(subjects)} subject(s): {subjects}")

    print(f"[inter] dispatching {len(subjects)} held-out-subject job(s) across n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_run_inter_subject_one)(
            held_out, df[df["subject"] != held_out], df[df["subject"] == held_out], feature_cols
        )
        for held_out in subjects
    )

    all_rows = []
    for rows, log in results:
        for line in log:
            print(line)
        all_rows.extend(rows)

    inter_df = pd.DataFrame(all_rows)
    if inter_df.empty:
        warnings.warn("run_inter_subject produced zero rows.")
    return inter_df


# ============================================================================
# NEW: per-style runner
# ============================================================================
def run_for_style(df, feature_cols, style):
    """Filters df to one style's subjects and runs the full intra+inter pass.
    Returns nothing - writes its own CSVs, mirroring the pooled version."""
    style_df = df[df["style"] == style].reset_index(drop=True)
    n_subjects = style_df["subject"].nunique()
    print(f"\n{'#' * 70}")
    print(f"STYLE: {style} ({n_subjects} subjects, {len(style_df)} epochs)")
    print(f"{'#' * 70}")

    if n_subjects < 2:
        warnings.warn(f"Style {style} has fewer than 2 subjects after filtering - "
                       f"skipping entirely.")
        return

    if RUN_INTRA:
        print(f"\n--- {style}: INTRA-SUBJECT ---")
        intra_fold_df = run_intra_subject(style_df, feature_cols)
        if not intra_fold_df.empty:
            out_by_fold = OUT_INTRA_CSV_TEMPLATE.format(style=style).replace(".csv", "_by_fold.csv")
            intra_fold_df.to_csv(out_by_fold, index=False)
            intra_summary = (
                intra_fold_df.groupby(["subject", "classifier"])[["accuracy", "balanced_accuracy"]]
                .mean().reset_index()
            )
            out_summary = OUT_INTRA_CSV_TEMPLATE.format(style=style)
            intra_summary.to_csv(out_summary, index=False)
            print(f"Saved {out_summary}")
            print(intra_summary.groupby("classifier")[["accuracy", "balanced_accuracy"]]
                  .agg(["mean", "std", "count"]))

    if RUN_INTER:
        print(f"\n--- {style}: INTER-SUBJECT (LOSO) ---")
        inter_df = run_inter_subject(style_df, feature_cols)
        if not inter_df.empty:
            out_inter = OUT_INTER_CSV_TEMPLATE.format(style=style)
            inter_df.to_csv(out_inter, index=False)
            print(f"Saved {out_inter}")
            print(inter_df.groupby("classifier")[["accuracy", "balanced_accuracy"]]
                  .agg(["mean", "std", "count"]))


# ============================================================================
# MAIN
# ============================================================================
def main():
    df, feature_cols = load_feature_table()

    print("\n[main] subjects and epochs per style (after all filtering):")
    print(df.groupby("style")["subject"].nunique())
    print(df.groupby("style").size())

    for style in STYLE_LABELS:
        run_for_style(df, feature_cols, style)


if __name__ == "__main__":
    main()