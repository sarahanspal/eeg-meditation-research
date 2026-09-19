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

DEBUG CONTROLS
--------------
DEBUG_SUBJECT_LIMIT: set to e.g. 3 to smoke-test the whole pipeline on a
    handful of subjects before committing to a full run (MLP/SVM training
    across many folds/subjects is the slow part).
RUN_INTRA / RUN_INTER: toggle independently.
Every block prints shapes/checks before proceeding so a bad input surfaces
immediately rather than three stages downstream.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
    steps = [("imputer", SimpleImputer(strategy="median"))]
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

    nan_frac = df[feature_cols].isna().mean().mean()
    print(f"[load] mean fraction of NaN across remaining feature cells: {nan_frac:.4f} "
          f"(imputed per-fold downstream inside the pipeline, not here, to avoid leakage)")

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
def run_intra_subject(df, feature_cols, n_folds=N_INTRA_FOLDS, subject_limit=DEBUG_SUBJECT_LIMIT):
    classifiers = build_classifiers()
    subjects = sorted(df["subject"].unique())
    if subject_limit:
        subjects = subjects[:subject_limit]
        print(f"[intra][DEBUG] limited to {len(subjects)} subject(s): {subjects}")

    rows = []
    for si, subject in enumerate(subjects):
        sub_df = df[df["subject"] == subject].reset_index(drop=True)
        X = sub_df[feature_cols].values
        y = sub_df["label"].values
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        print(f"[intra] ({si + 1}/{len(subjects)}) {subject}: "
              f"n_epochs={len(y)} (MED={n_pos}, IMW={n_neg})")

        if min(n_pos, n_neg) < n_folds:
            warnings.warn(f"{subject}: smaller class has only {min(n_pos, n_neg)} "
                           f"epochs, fewer than n_folds={n_folds} - StratifiedKFold "
                           f"would fail or produce degenerate folds. Skipping subject.")
            continue

        # shuffle=False: preserves original temporal order per the paper's spec.
        # Adjacent 4s epochs from a continuous recording are still likely
        # autocorrelated, so - exactly as the paper itself notes - intra-subject
        # accuracy here should be read as within-session performance, not
        # evidence of temporal generalization.
        skf = StratifiedKFold(n_splits=n_folds, shuffle=False)

        for clf_name, clf in classifiers.items():
            fold_accs = []
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                pipe = make_pipeline(clf)
                pipe.fit(X[train_idx], y[train_idx])
                y_pred = pipe.predict(X[test_idx])
                acc = accuracy_score(y[test_idx], y_pred)
                fold_accs.append(acc)
                rows.append({
                    "subject": subject, "classifier": clf_name, "fold": fold_i,
                    "n_train": len(train_idx), "n_test": len(test_idx), "accuracy": acc,
                })
            print(f"    {clf_name}: mean fold accuracy = {np.mean(fold_accs):.3f} "
                  f"(+/- {np.std(fold_accs):.3f} over {len(fold_accs)} folds)")

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        warnings.warn("run_intra_subject produced zero rows - check the "
                       "min(n_pos, n_neg) < n_folds warnings above.")
    return fold_df


# ============================================================================
# BLOCK 3 - INTER-SUBJECT: LEAVE-ONE-SUBJECT-OUT
# ============================================================================
def run_inter_subject(df, feature_cols, subject_limit=DEBUG_SUBJECT_LIMIT):
    classifiers = build_classifiers()
    subjects = sorted(df["subject"].unique())
    if subject_limit:
        subjects = subjects[:subject_limit]
        print(f"[inter][DEBUG] limited to {len(subjects)} subject(s): {subjects}")

    rows = []
    for si, held_out in enumerate(subjects):
        train_df = df[df["subject"] != held_out]
        test_df = df[df["subject"] == held_out]
        X_train, y_train = train_df[feature_cols].values, train_df["label"].values
        X_test, y_test = test_df[feature_cols].values, test_df["label"].values

        print(f"[inter] ({si + 1}/{len(subjects)}) held-out={held_out}: "
              f"train n={len(y_train)} (from {train_df['subject'].nunique()} subjects), "
              f"test n={len(y_test)} (MED={int((y_test == 1).sum())}, "
              f"IMW={int((y_test == 0).sum())})")

        # This assert is exactly the kind of check that would have caught the
        # near-100%-accuracy results seen earlier in the shared Drive folder -
        # keep it, don't remove it for speed.
        assert held_out not in train_df["subject"].unique(), \
            f"LEAKAGE: {held_out} present in its own training set"

        for clf_name, clf in classifiers.items():
            pipe = make_pipeline(clf)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            rows.append({
                "held_out_subject": held_out, "classifier": clf_name,
                "n_train": len(y_train), "n_test": len(y_test), "accuracy": acc,
            })
            print(f"    {clf_name}: accuracy = {acc:.3f}")

    inter_df = pd.DataFrame(rows)
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
        intra_fold_df = run_intra_subject(df, feature_cols, subject_limit=DEBUG_SUBJECT_LIMIT)
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
        inter_df = run_inter_subject(df, feature_cols, subject_limit=DEBUG_SUBJECT_LIMIT)
        if not inter_df.empty:
            inter_df.to_csv(OUT_INTER_CSV, index=False)
            print(f"\nSaved inter-subject (LOSO) accuracies to {OUT_INTER_CSV}")
            print("\nInter-subject (LOSO) accuracy across held-out subjects, per classifier:")
            print(inter_df.groupby("classifier")["accuracy"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()