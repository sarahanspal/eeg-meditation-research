"""
braboszcz_classification.py
=============================
Intra-subject and inter-subject (LOSO) classification of MED vs IMW using
the full feature set from braboszcz_feature_extraction.py, on task-med2
(last 10 min MED) and task-think2 (last 10 min IMW) only. No sliding-window
analysis.

CLASSIFIERS (matching the AI Frontiers paper's spec)
------------------------------------------------------
kNN (k=5, Euclidean) | Random Forest (default hyperparams) |
SVM-linear (C=1.0) | SVM-RBF (C=1.0, gamma='scale') |
Gaussian Naive Bayes (default) | MLP (hidden layers (100,50), ReLU, Adam)
random_state=42 for every stochastic component.

ONE DELIBERATE DEVIATION FROM A LITERAL READING OF THE PAPER
----------------------------------------------------------------
A median imputer + StandardScaler is inserted ahead of every classifier,
fit ONLY on the training partition each fold/subject (never on test data -
this does not introduce leakage). The paper doesn't mention scaling, but
without it, kNN/SVM/MLP would be dominated by whichever raw feature has
the largest numeric range (e.g. band power ~0-1 vs. IAF ~7-13 Hz) - not a
meaningful test of feature discriminability. RF and GaussianNB are
insensitive to monotonic per-feature scaling, so the comparison across all
six classifiers stays fair. Set APPLY_SCALING=False below to switch to a
literal no-preprocessing replication if you want to compare the two.

LEAK FOUND AND FIXED (from diagnose_intra_subject_leakage.py results)
------------------------------------------------------------------------
Per-fold `SimpleImputer(strategy="median")` was the source of the earlier
near-perfect intra-subject accuracy. Mechanism: task-med2 and task-think2
are SEPARATE recording files, and clean_bad_channels() can drop/interpolate
DIFFERENT channels per condition per subject (confirmed earlier: sub-006
had 11 bad channels in MED vs 6 in IMW). When a channel is missing in only
ONE condition for a subject, a per-fold imputer fit on that subject's own
training data fills every epoch of the missing condition with a constant
derived from the OTHER condition's data - handing the classifier a
trivial, perfect, non-neural "this value is suspiciously constant -> must
be the condition with the dropped channel" signal. The label-shuffle
control confirmed this: shuffled labels -> ~50% (correct), real labels ->
100% for every subject tested, with no single global feature responsible
(ruling out one broken feature; the responsible channel differs per
subject, which is exactly the fingerprint of this mechanism).

FIX: missing values are now filled ONCE, at load time, with a GLOBAL
per-feature median computed across the ENTIRE population (all subjects,
both conditions) - not per-fold, not per-subject. The fill value no longer
depends on which condition happened to have data available in a given
fold/subject, which removes the leak mechanism entirely. This is standard
practice for population-level imputation (comparable to using a fixed
clinical reference value) and is computed before any subject is held out,
so it isn't "test-set-specific" leakage in the harmful sense - it's a
fixed constant, identical regardless of the training/test split.

PARALLELIZATION
----------------
Both intra-subject and inter-subject loops now run across subjects via
joblib (see N_JOBS below), same pattern as run_batch_extraction_braboszcz.py.
Individual classifiers are NOT given n_jobs=-1 internally (RF/kNN support
it) specifically to avoid nested parallelism fighting the outer joblib pool.

DEBUG CONTROLS
--------------
DEBUG_SUBJECT_LIMIT: set to e.g. 3 to smoke-test the whole pipeline on a
    handful of subjects before committing to a full run.
RUN_INTRA / RUN_INTER: toggle independently.
Every block prints shapes/checks before proceeding so a bad input surfaces
immediately rather than three stages downstream.
"""

import os

# See run_batch_extraction_braboszcz.py for why this must happen before
# numpy/sklearn import in worker processes - prevents each worker from also
# multithreading its own BLAS calls on top of joblib's process parallelism.
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
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42

# ============================================================================
# CONFIG
# ============================================================================
FEATURES_CSV = "braboszcz_all_subjects_features.csv"  # output of run_batch_extraction_braboszcz.py
META_COLS = ["subject", "condition", "epoch_idx"]
LABEL_MAP = {"IMW": 0, "MED": 1}

APPLY_SCALING = True         # see module docstring
N_INTRA_FOLDS = 10           # paper: SKF k=10 per subject
DEBUG_SUBJECT_LIMIT = None   # e.g. 3 for a fast smoke test
RUN_INTRA = True
RUN_INTER = True
N_JOBS = -2                  # -1 = every core, -2 = every core but one

OUT_INTRA_CSV = "braboszcz_intra_subject_accuracies.csv"
OUT_INTER_CSV = "braboszcz_inter_subject_accuracies.csv"


# ============================================================================
# CLASSIFIERS - exact spec from the paper
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
            random_state=RANDOM_SEED,
            max_iter=1000,  # bumped up from sklearn's low default so a real
                             # convergence-warning bug isn't silently masked
                             # as "it just didn't need more iterations";
                             # paper doesn't specify max_iter.
        ),
    }


def make_pipeline(clf, apply_scaling=APPLY_SCALING):
    # NOTE: no imputer here anymore - see module docstring "LEAK FOUND AND
    # FIXED". Missing values are filled once, globally, in load_feature_table()
    # before any subject/fold splitting happens, specifically to avoid a
    # per-fold imputer picking up condition-correlated missingness.
    steps = []
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
    per_feature_nan = df[feature_cols].isna().mean()
    heavily_missing = per_feature_nan[per_feature_nan > 0].sort_values(ascending=False)
    print(f"[load] mean fraction of NaN across feature cells before global fill: "
          f"{nan_frac_before:.4f}")
    if len(heavily_missing):
        print(f"[load] {len(heavily_missing)} feature column(s) have some missingness "
              f"(top 10 shown) - these are exactly the columns where the OLD per-fold "
              f"imputer could leak condition identity; now safe since fill is global:")
        print(heavily_missing.head(10).to_string())

    # GLOBAL median fill, computed ONCE across the whole population/both
    # conditions, BEFORE any subject is held out - see module docstring
    # "LEAK FOUND AND FIXED". This is a fixed constant per feature, not
    # fit per-fold, so it can't encode which condition a fold's data came from.
    # CRITICAL: a feature missing in only ONE condition for a subject is still
    # a leak even with a GLOBAL constant fill - that condition gets an
    # artificially IDENTICAL value across every epoch while the other
    # condition varies naturally, and "suspiciously constant vs. naturally
    # varying" is itself detectable regardless of which constant is used
    # (confirmed by testing: GaussianNB/RF still hit ~1.0 on a subject with a
    # single planted single-condition-missing feature, even after switching
    # to a global median). Real fix: if a feature is missing in ANY condition
    # for a subject, null it for that subject's ENTIRE feature column (both
    # conditions) BEFORE filling - so the constant is equally uninformative
    # for MED and IMW for that subject, not a giveaway for one side.
    subject_feature_any_nan = df.groupby("subject")[feature_cols].transform(
        lambda col: col.isna().any()
    )
    n_cells_nulled_extra = int((subject_feature_any_nan & df[feature_cols].notna().values).values.sum())
    if n_cells_nulled_extra:
        print(f"[load] nulling {n_cells_nulled_extra} additional cell(s) where a feature "
              f"was clean for a subject in one condition but missing in the other - "
              f"prevents a per-subject single-condition constant leak (see comment above)")
    df[feature_cols] = df[feature_cols].mask(subject_feature_any_nan)

    global_medians = df[feature_cols].median()
    still_all_nan = global_medians[global_medians.isna()].index.tolist()
    if still_all_nan:
        warnings.warn(f"{len(still_all_nan)} feature column(s) are NaN for the ENTIRE "
                       f"population (not just some subjects) - dropping them, no value "
                       f"to impute from: {still_all_nan}")
        feature_cols = [c for c in feature_cols if c not in still_all_nan]
        global_medians = global_medians.drop(index=still_all_nan)
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(global_medians)
    print(f"[load] filled remaining NaNs with global per-feature medians; "
          f"{df[feature_cols].isna().sum().sum()} NaN cells remain (should be 0)")

    unexpected = set(df["condition"].unique()) - set(LABEL_MAP.keys())
    if unexpected:
        raise ValueError(f"Unexpected condition label(s) {unexpected}; "
                          f"expected only {list(LABEL_MAP.keys())}")
    df = df.copy()
    df["label"] = df["condition"].map(LABEL_MAP)

    per_subject_counts = df.groupby(["subject", "condition"]).size().unstack(fill_value=0)
    print("[load] per-subject epoch counts by condition:")
    print(per_subject_counts)

    incomplete_mask = (per_subject_counts == 0).any(axis=1)
    if incomplete_mask.any():
        incomplete_subjects = per_subject_counts[incomplete_mask].index.tolist()
        warnings.warn(f"Dropping subject(s) missing one condition entirely (cannot "
                       f"classify MED-vs-IMW for them): {incomplete_subjects}")
        df = df[~df["subject"].isin(incomplete_subjects)]

    print(f"[load] final: {df['subject'].nunique()} subjects, {len(df)} epochs")
    return df, feature_cols


# ============================================================================
# BLOCK 2 - INTRA-SUBJECT: STRATIFIED 10-FOLD PER SUBJECT
# ============================================================================
def _run_intra_subject_one(subject, sub_df, feature_cols, n_folds):
    """Runs in its own worker process - must be top-level (picklable), not a
    closure. Returns (rows, log_lines) instead of printing directly, since
    print() ordering across worker processes is unreliable - collect and
    print in the parent instead so output stays readable."""
    classifiers = build_classifiers()
    X = sub_df[feature_cols].values
    y = sub_df["label"].values
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    log = [f"[intra] {subject}: n_epochs={len(y)} (MED={n_pos}, IMW={n_neg})"]

    if min(n_pos, n_neg) < n_folds:
        log.append(f"    SKIPPED: smaller class has only {min(n_pos, n_neg)} epochs, "
                    f"fewer than n_folds={n_folds}")
        return [], log

    # shuffle=False: preserves original temporal order per the paper's spec.
    # Adjacent 4s epochs from a continuous recording are still likely
    # autocorrelated, so - as the paper itself notes - this should be read as
    # within-session performance, not evidence of temporal generalization.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    rows = []
    for clf_name, clf in classifiers.items():
        fold_accs = []
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            pipe = make_pipeline(clf)
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], y_pred)
            fold_accs.append(acc)
            rows.append({"subject": subject, "classifier": clf_name, "fold": fold_i,
                         "n_train": len(train_idx), "n_test": len(test_idx), "accuracy": acc})
        log.append(f"    {clf_name}: mean fold accuracy = {np.mean(fold_accs):.3f} "
                   f"(+/- {np.std(fold_accs):.3f} over {len(fold_accs)} folds)")
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
        warnings.warn("run_intra_subject produced zero rows - check the "
                       "SKIPPED messages above.")
    return fold_df


# ============================================================================
# BLOCK 3 - INTER-SUBJECT: LEAVE-ONE-SUBJECT-OUT
# ============================================================================
def _run_inter_subject_one(held_out, train_df, test_df, feature_cols):
    """Runs in its own worker process - top-level/picklable. train_df/test_df
    are passed in already split so each worker doesn't need the full df."""
    classifiers = build_classifiers()
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label"].values

    log = [f"[inter] held-out={held_out}: train n={len(y_train)} "
           f"(from {train_df['subject'].nunique()} subjects), test n={len(y_test)} "
           f"(MED={int((y_test == 1).sum())}, IMW={int((y_test == 0).sum())})"]

    # This assert is exactly the kind of check that would have caught the
    # near-100%-accuracy results seen earlier - keep it, don't remove for speed.
    assert held_out not in train_df["subject"].unique(), \
        f"LEAKAGE: {held_out} present in its own training set"

    rows = []
    for clf_name, clf in classifiers.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rows.append({"held_out_subject": held_out, "classifier": clf_name,
                     "n_train": len(y_train), "n_test": len(y_test), "accuracy": acc})
        log.append(f"    {clf_name}: accuracy = {acc:.3f}")
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
        warnings.warn("run_inter_subject produced zero rows - check subject_limit "
                       "and the loaded feature table.")
    return inter_df


# ============================================================================
# MAIN
# ============================================================================
def main():
    df, feature_cols = load_feature_table()

    if RUN_INTRA:
        print("\n" + "=" * 70)
        print("INTRA-SUBJECT (stratified 10-fold per subject, shuffle=False)")
        print("=" * 70)
        intra_fold_df = run_intra_subject(df, feature_cols, subject_limit=DEBUG_SUBJECT_LIMIT, n_jobs=N_JOBS)
        if not intra_fold_df.empty:
            intra_fold_df.to_csv(OUT_INTRA_CSV.replace(".csv", "_by_fold.csv"), index=False)
            intra_summary = (
                intra_fold_df.groupby(["subject", "classifier"])["accuracy"]
                .mean().reset_index()
            )
            intra_summary.to_csv(OUT_INTRA_CSV, index=False)
            print(f"\nSaved per-subject intra-subject accuracies to {OUT_INTRA_CSV}")
            print("\nIntra-subject accuracy, averaged across subjects, per classifier:")
            print(intra_summary.groupby("classifier")["accuracy"].agg(["mean", "std", "count"]))

    if RUN_INTER:
        print("\n" + "=" * 70)
        print("INTER-SUBJECT (leave-one-subject-out)")
        print("=" * 70)
        inter_df = run_inter_subject(df, feature_cols, subject_limit=DEBUG_SUBJECT_LIMIT, n_jobs=N_JOBS)
        if not inter_df.empty:
            inter_df.to_csv(OUT_INTER_CSV, index=False)
            print(f"\nSaved inter-subject (LOSO) accuracies to {OUT_INTER_CSV}")
            print("\nInter-subject (LOSO) accuracy across held-out subjects, per classifier:")
            print(inter_df.groupby("classifier")["accuracy"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()