"""
diagnose_intra_subject_leakage.py
===================================
Two checks to isolate WHY intra-subject accuracy is near-ceiling for almost
every subject/classifier, before trusting or reporting that number.

CHECK 1 - univariate separability per feature
    If one (or a few) feature(s) alone near-perfectly separate MED from IMW
    (high AUC using just that single column as a score), that feature is
    very likely riding a between-recording-file artifact (impedance drift,
    differential bad-channel dropout, etc.) rather than real neural signal -
    especially if it's a feature at a channel/band with no a priori reason
    to be the strongest discriminator.

CHECK 2 - label-shuffle control (per subject, RandomForest only, for speed)
    Re-runs the exact same intra-subject 10-fold pipeline but with the
    labels randomly shuffled WITHIN that subject. If shuffled-label accuracy
    is still high (>>50%), that's a CODE BUG (e.g. a fold-splitting/ordering
    issue letting information leak regardless of the label). If shuffled
    accuracy collapses to ~50% while real-label accuracy stays near 100%,
    the code is fine - the separability is real, but likely reflects a
    genuine (if uninteresting) difference between the two recording files
    rather than a meditation-specific brain signal, and the FEATURES/
    preprocessing need attention, not the classification code.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from braboszcz_classification import (
    FEATURES_CSV, META_COLS, LABEL_MAP, load_feature_table, make_pipeline,
)

RANDOM_SEED = 42
N_SHUFFLE_SUBJECTS = 8  # how many subjects to run the (slower) shuffle control on


def check1_univariate_separability(df, feature_cols, top_n=20):
    print("=" * 70)
    print("CHECK 1: per-feature univariate AUC (MED vs IMW), pooled across all subjects")
    print("=" * 70)
    y = df["label"].values
    results = []
    for col in feature_cols:
        x = df[col].values
        mask = ~np.isnan(x)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        try:
            auc = roc_auc_score(y[mask], x[mask])
        except ValueError:
            continue
        # AUC below 0.5 just means the feature points the other way - take
        # distance from 0.5 as the "how separating is this feature" measure
        results.append((col, auc, abs(auc - 0.5)))

    res_df = pd.DataFrame(results, columns=["feature", "auc", "separation"])
    res_df = res_df.sort_values("separation", ascending=False)
    print(f"\nTop {top_n} most separating features (AUC near 0 or 1 = suspicious):")
    print(res_df.head(top_n).to_string(index=False))

    n_suspicious = (res_df["auc"].between(0.97, 1.0) | res_df["auc"].between(0.0, 0.03)).sum()
    print(f"\n{n_suspicious} feature(s) with pooled AUC >=0.97 or <=0.03 on their own - "
          f"if any of these look like an odd channel/band with no a priori reason to "
          f"be this discriminative, that's your leading suspect.")
    return res_df


def check2_label_shuffle_control(df, feature_cols, n_subjects=N_SHUFFLE_SUBJECTS, n_folds=10):
    print("\n" + "=" * 70)
    print(f"CHECK 2: label-shuffle control, RandomForest only, {n_subjects} subject(s)")
    print("=" * 70)
    rng = np.random.default_rng(RANDOM_SEED)
    subjects = sorted(df["subject"].unique())[:n_subjects]

    rows = []
    for subject in subjects:
        sub_df = df[df["subject"] == subject].reset_index(drop=True)
        X = sub_df[feature_cols].values
        y_real = sub_df["label"].values
        y_shuffled = rng.permutation(y_real)

        for label_name, y in (("real", y_real), ("shuffled", y_shuffled)):
            if min((y == 0).sum(), (y == 1).sum()) < n_folds:
                continue
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
            accs = []
            for train_idx, test_idx in skf.split(X, y):
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(random_state=RANDOM_SEED)),
                ])
                pipe.fit(X[train_idx], y[train_idx])
                accs.append(pipe.score(X[test_idx], y[test_idx]))
            mean_acc = np.mean(accs)
            rows.append({"subject": subject, "labels": label_name, "mean_accuracy": mean_acc})
            print(f"  {subject} [{label_name:>8}]: mean 10-fold accuracy = {mean_acc:.3f}")

    result_df = pd.DataFrame(rows)
    pivot = result_df.pivot(index="subject", columns="labels", values="mean_accuracy")
    print("\nSummary (real vs shuffled labels, same subjects/features/pipeline):")
    print(pivot)
    print(f"\nMean real-label accuracy:     {pivot['real'].mean():.3f}")
    print(f"Mean shuffled-label accuracy: {pivot['shuffled'].mean():.3f}")
    print(
        "\nInterpretation:\n"
        "  shuffled ~0.50  and  real ~high   -> code is fine; separability is real "
        "but likely a non-neural between-file artifact - fix the FEATURES/preprocessing.\n"
        "  shuffled ALSO high (>>0.50)       -> there is a genuine CODE BUG (e.g. an "
        "ordering/indexing leak independent of the label) - fix the PIPELINE, not the features."
    )
    return result_df


if __name__ == "__main__":
    df, feature_cols = load_feature_table(FEATURES_CSV)
    auc_df = check1_univariate_separability(df, feature_cols)
    auc_df.to_csv("diagnostic_feature_auc.csv", index=False)
    shuffle_df = check2_label_shuffle_control(df, feature_cols)
    shuffle_df.to_csv("diagnostic_label_shuffle.csv", index=False)