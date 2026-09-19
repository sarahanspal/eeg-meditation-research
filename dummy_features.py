"""
dummy_features.py

Generates placeholder feature matrices matching B's expected output location
and shape, using REAL epoch counts/labels from Step 0 (not fake subject
structure) but RANDOM feature values. Lets the harness be built and tested
now; swap this for B's real output later without changing any harness code.
"""

import numpy as np
from pathlib import Path
from config import PROCESSED_ROOT

N_FEATURES = 70  # placeholder count; B's real set will be 29 x 14 = 406, update when ready
FEATURE_ROOT = Path("data/features/dummy")

def generate_dummy_features(dataset_name):
    dataset_dir = Path(PROCESSED_ROOT) / dataset_name
    out_dir = FEATURE_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(dataset_dir.glob("*_labels.npy"))
    for label_file in label_files:
        subject_id = label_file.stem.replace("_labels", "")
        labels = np.load(label_file)
        n_epochs = labels.shape[0]

        # random features, same epoch count as the real data -- structure is real, values are not
        features = np.random.randn(n_epochs, N_FEATURES)
        np.save(out_dir / f"{subject_id}_features.npy", features)

    print(f"{dataset_name}: generated dummy features for {len(label_files)} subjects")

if __name__ == "__main__":
    for ds in ["brandmeyer", "sky", "braboszcz"]:
        generate_dummy_features(ds)