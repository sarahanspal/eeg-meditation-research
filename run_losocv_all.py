"""
Full LOSOCV, all 6 classifiers, all 3 datasets. Saves per-subject results
to CSV -- this is the single source of truth for LOSOCV numbers from here on.
No more scattered testing*.py scripts; edit harness.py for logic changes,
this script just calls it and saves output.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
from pathlib import Path
from config import FEATURES_HANDCRAFTED_ROOT
from harness import run_losocv

Path("results").mkdir(exist_ok=True)

all_rows = []
all_excluded = []

for dataset in ["brandmeyer", "sky", "braboszcz"]:
    print(f"=== {dataset} ===")
    rows, excluded = run_losocv(dataset, feature_root=FEATURES_HANDCRAFTED_ROOT)
    all_rows.extend(rows)
    for subj, n_epochs, classes in excluded:
        all_excluded.append({"dataset": dataset, "subject": subj,
                              "n_epochs": n_epochs, "classes_present": classes})

df = pd.DataFrame(all_rows)
df.to_csv("results/losocv_persubject_full.csv", index=False)

if all_excluded:
    pd.DataFrame(all_excluded).to_csv("results/losocv_excluded_subjects.csv", index=False)

print("\n=== Summary (mean balanced accuracy %) ===")
print(df.groupby(["dataset", "classifier"])["accuracy"].agg(["mean", "std", "count"]))
print(f"\nSaved: results/losocv_persubject_full.csv")
if all_excluded:
    print(f"Saved: results/losocv_excluded_subjects.csv ({len(all_excluded)} exclusions)")