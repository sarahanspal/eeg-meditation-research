import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from config import PROCESSED_ROOT, FEATURES_HANDCRAFTED_ROOT

SEED = 42
FAST_CLASSIFIERS = {
    "kNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "RandomForest": RandomForestClassifier(random_state=SEED),
    "NaiveBayes": GaussianNB(),
}

def load_dataset(dataset_name, feature_root):
    feature_dir = Path(feature_root) / dataset_name
    processed_dir = Path(PROCESSED_ROOT) / dataset_name
    X_list, y_list, groups_list = [], [], []
    for feat_file in sorted(feature_dir.glob("*_features.npy")):
        subject_id = feat_file.stem.replace("_features", "")
        X = np.load(feat_file)
        y = np.load(processed_dir / f"{subject_id}_labels.npy")
        X_list.append(X); y_list.append(y)
        groups_list.append(np.full(y.shape[0], subject_id))
    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)

def run_fast_losocv(dataset_name):
    X, y, groups = load_dataset(dataset_name, FEATURES_HANDCRAFTED_ROOT)
    X_scaled = np.zeros_like(X, dtype=float)
    for subj in np.unique(groups):
        mask = groups == subj
        X_scaled[mask] = StandardScaler().fit_transform(X[mask])

    logo = LeaveOneGroupOut()
    results = {name: [] for name in FAST_CLASSIFIERS}
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        for name, clf in FAST_CLASSIFIERS.items():
            clf.fit(X_scaled[train_idx], y[train_idx])
            preds = clf.predict(X_scaled[test_idx])
            results[name].append(balanced_accuracy_score(y[test_idx], preds))
    return results

all_rows = []
for dataset in ["brandmeyer", "sky"]:
    print(f"=== {dataset} ===")
    results = run_fast_losocv(dataset)
    for clf_name, scores in results.items():
        mean_acc = np.mean(scores)
        print(f"  {clf_name}: {mean_acc:.3f}")
        all_rows.append({"dataset": dataset, "classifier": clf_name, "mean_acc": mean_acc})

df = pd.DataFrame(all_rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/losocv_fast_partial.csv", index=False)
print("\nSaved to results/losocv_fast_partial.csv")
