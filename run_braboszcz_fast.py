import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
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

X, y, groups = load_dataset("braboszcz", FEATURES_HANDCRAFTED_ROOT)
X_scaled = np.zeros_like(X, dtype=float)
for subj in np.unique(groups):
    mask = groups == subj
    X_scaled[mask] = StandardScaler().fit_transform(X[mask])

logo = LeaveOneGroupOut()
results = {name: [] for name in FAST_CLASSIFIERS}
n_folds = len(np.unique(groups))
for i, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
    print(f"fold {i+1}/{n_folds}", flush=True)
    for name, clf in FAST_CLASSIFIERS.items():
        clf.fit(X_scaled[train_idx], y[train_idx])
        preds = clf.predict(X_scaled[test_idx])
        results[name].append(balanced_accuracy_score(y[test_idx], preds))

print(f"\n{'Classifier':<15} {'Mean acc':>10}")
for name, scores in results.items():
    print(f"{name:<15} {np.mean(scores):>10.3f}")