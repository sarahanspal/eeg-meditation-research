import matplotlib.pyplot as plt

classifiers = ["kNN", "RandomForest", "NaiveBayes"]
accuracies = [0.520, 0.529, 0.534]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(classifiers, accuracies, color="#4C72B0")
ax.axhline(0.5, color="gray", linestyle="--", label="Chance level (50%)")
ax.set_ylabel("Balanced Accuracy")
ax.set_title("Brandmeyer: Inter-Subject (LOSOCV) Classification")
ax.set_ylim(0.4, 0.7)
ax.legend()

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f"{acc:.3f}", ha="center")

plt.tight_layout()
plt.savefig("results/brandmeyer_losocv_fast.png", dpi=150)
plt.show()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from config import PROCESSED_ROOT, FEATURES_HANDCRAFTED_ROOT

SEED = 42
CLASSIFIERS = {
    "NB":  GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "RF":  RandomForestClassifier(random_state=SEED),
    # SVM / SVM-RBF / MLP omitted for now (still running) -- add back once done
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

X, y, groups = load_dataset("brandmeyer", FEATURES_HANDCRAFTED_ROOT)
X_scaled = np.zeros_like(X, dtype=float)
for subj in np.unique(groups):
    mask = groups == subj
    X_scaled[mask] = StandardScaler().fit_transform(X[mask])

# add near the top of your LOSOCV loop, after loading groups:
valid_subjects = []
for subj in np.unique(groups):
    if len(np.unique(y[groups == subj])) >= 2:
        valid_subjects.append(subj)

mask = np.isin(groups, valid_subjects)
X_scaled, y, groups = X_scaled[mask], y[mask], groups[mask]

logo = LeaveOneGroupOut()
rows = []  # one row per (subject, classifier) -- needed for the strip plot
for train_idx, test_idx in logo.split(X_scaled, y, groups):
    test_subject = groups[test_idx][0]
    for clf_name, clf in CLASSIFIERS.items():
        clf.fit(X_scaled[train_idx], y[train_idx])
        preds = clf.predict(X_scaled[test_idx])
        acc = balanced_accuracy_score(y[test_idx], preds) * 100  # as percentage, matches their axis
        rows.append({"subject": test_subject, "classifier": clf_name, "accuracy": acc, "task": "MED/MW"})

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/brandmeyer_persubject_accuracy.csv", index=False)

# ============ Plot: Panel-A style ============
fig, ax = plt.subplots(figsize=(4, 5))

sns.stripplot(data=df, x="task", y="accuracy", hue="classifier",
              dodge=True, jitter=True, alpha=0.4, size=4, ax=ax, legend=False)

summary = df.groupby("classifier")["accuracy"].agg(["mean", "sem"]).reset_index()
n_clf = len(CLASSIFIERS)
positions = np.linspace(-0.3, 0.3, n_clf)
for pos, (_, row) in zip(positions, summary.iterrows()):
    ax.errorbar(x=pos, y=row["mean"], yerr=row["sem"] * 1.96,  # 95% CI
                fmt="o", color="black", capsize=4, markersize=6, zorder=5)
    ax.text(pos, row["mean"] + row["sem"]*1.96 + 2, row["classifier"],
            ha="center", fontsize=8)

ax.set_ylim(30, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("")
ax.set_title("Brandmeyer: MED vs. MW\n(Inter-Subject LOSOCV)")
plt.tight_layout()
plt.savefig("results/brandmeyer_panelA_style.png", dpi=150)
plt.show()