import numpy as np
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score

from config import PROCESSED_ROOT

SEED = 42


CLASSIFIERS = {
    "NB":         GaussianNB(),
    "kNN":        KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "RF":         RandomForestClassifier(random_state=SEED),
    "SVM":        LinearSVC(C=1.0, random_state=SEED, max_iter=2000),  # swapped
    "SVM-RBF":    SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED, cache_size=1000),
    "MLP":        MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu",
                                 solver="adam", random_state=SEED, max_iter=1000),
}


def load_dataset(dataset_name, feature_root):
    feature_dir = Path(feature_root) / dataset_name
    processed_dir = Path(PROCESSED_ROOT) / dataset_name

    X_list, y_list, groups_list = [], [], []
    for feat_file in sorted(feature_dir.glob("*_features.npy")):
        subject_id = feat_file.stem.replace("_features", "")
        X = np.load(feat_file)
        y = np.load(processed_dir / f"{subject_id}_labels.npy")
        assert X.shape[0] == y.shape[0], f"{subject_id}: feature/label count mismatch"

        X_list.append(X)
        y_list.append(y)
        groups_list.append(np.full(y.shape[0], subject_id))

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)


def filter_multiclass_subjects(X, y, groups):
    """
    Removes subjects who only have one class present in their epochs.
    LOSOCV can't fairly evaluate a held-out subject with only one label --
    there's nothing to test the other class against, so balanced_accuracy_score
    becomes meaningless for that fold. Applies to any dataset, not just Brandmeyer.
    """
    valid_mask = np.ones(len(groups), dtype=bool)
    excluded = []

    for subj in np.unique(groups):
        subj_mask = groups == subj
        n_classes = len(np.unique(y[subj_mask]))
        if n_classes < 2:
            valid_mask[subj_mask] = False
            excluded.append((subj, int(subj_mask.sum()), np.unique(y[subj_mask]).tolist()))

    if excluded:
        print(f"  Excluded {len(excluded)} subject(s) with only one class present:")
        for subj, n_epochs, classes in excluded:
            print(f"    {subj}: {n_epochs} epochs, only class(es) {classes}")

    return X[valid_mask], y[valid_mask], groups[valid_mask], excluded


def standardize_per_subject(X, groups):
    X_scaled = np.zeros_like(X, dtype=float)
    for subj in np.unique(groups):
        mask = groups == subj
        X_scaled[mask] = StandardScaler().fit_transform(X[mask])
    return X_scaled


def run_losocv(dataset_name, feature_root, verbose=True):
    """
    Returns (per_subject_rows, excluded) where per_subject_rows is a list of
    dicts: {subject, classifier, accuracy} -- one row per subject per
    classifier, suitable for direct pd.DataFrame() construction and plotting.
    """
    X, y, groups = load_dataset(dataset_name, feature_root)
    X, y, groups, excluded = filter_multiclass_subjects(X, y, groups)
    X = standardize_per_subject(X, groups)

    logo = LeaveOneGroupOut()
    n_folds = len(np.unique(groups))
    rows = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_subject = groups[test_idx][0]
        if verbose:
            print(f"  [{dataset_name}] fold {fold_i+1}/{n_folds} (subject {test_subject})...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, clf in CLASSIFIERS.items():
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = balanced_accuracy_score(y_test, preds)
            rows.append({"dataset": dataset_name, "subject": test_subject,
                         "classifier": name, "accuracy": acc * 100})

    return rows, excluded


def run_cross_dataset(train_dataset, test_dataset, feature_root):
    X_train, y_train, groups_train = load_dataset(train_dataset, feature_root)
    X_test, y_test, groups_test = load_dataset(test_dataset, feature_root)

    X_train, y_train, groups_train, excluded_train = filter_multiclass_subjects(X_train, y_train, groups_train)
    X_test, y_test, groups_test, excluded_test = filter_multiclass_subjects(X_test, y_test, groups_test)

    X_train = standardize_per_subject(X_train, groups_train)
    X_test = standardize_per_subject(X_test, groups_test)

    results = {}
    for name, clf in CLASSIFIERS.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = balanced_accuracy_score(y_test, preds) * 100

    return results, excluded_train, excluded_test