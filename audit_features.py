"""Read-only integrity checks for the handcrafted feature store.

Run with:
    python3 audit_features.py          # fast structural audit
    python3 audit_features.py --full   # also hash every full feature file

This verifies that every ``*_features.npy`` file selected by config.py has
the expected 406 columns, is finite, is paired with the corresponding
processed labels, and is not an exact duplicate of another feature matrix.
It cannot prove scientific provenance: for that, retain the feature-extractor
code and a manifest recording its version and the raw/processed input hashes.
"""

import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from config import FEATURES_HANDCRAFTED_ROOT, PROCESSED_ROOT


DATASETS = ("brandmeyer", "sky", "braboszcz")
N_FEATURES = 406


def file_hash(path, chunk_size=1024 * 1024):
    """Return a content hash without loading the full file into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset, full=False):
    feature_dir = Path(FEATURES_HANDCRAFTED_ROOT) / dataset
    processed_dir = Path(PROCESSED_ROOT) / dataset
    feature_files = sorted(feature_dir.glob("*_features.npy"))

    if not feature_files:
        print(f"{dataset}: FAIL — no feature files in {feature_dir}")
        return False

    failures = []
    hashes = defaultdict(list)
    summaries = []

    for feature_path in feature_files:
        subject_id = feature_path.name.removesuffix("_features.npy")
        processed_label_path = processed_dir / f"{subject_id}_labels.npy"
        feature_label_path = feature_dir / f"{subject_id}_labels.npy"

        if not processed_label_path.exists():
            failures.append(f"{subject_id}: processed labels are missing")
            continue

        features = np.load(feature_path, mmap_mode="r")
        labels = np.load(processed_label_path, mmap_mode="r")
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            failures.append(f"{subject_id}: expected (epochs, {N_FEATURES}), got {features.shape}")
        if features.shape[0] != labels.shape[0]:
            failures.append(f"{subject_id}: {features.shape[0]} rows but {labels.shape[0]} labels")
        # A deterministic sparse sample makes the normal audit quick even when
        # the store is mounted from a cloud provider. ``--full`` scans every
        # value and hashes the entire file as an additional integrity check.
        row_indices = np.unique(np.linspace(0, max(features.shape[0] - 1, 0), min(8, features.shape[0]), dtype=int))
        sample = np.asarray(features[row_indices])
        if not np.isfinite(sample).all():
            failures.append(f"{subject_id}: contains NaN or infinity")

        # The feature directory retains a label copy. It must agree with the
        # canonical labels used by the training harness.
        if not feature_label_path.exists():
            failures.append(f"{subject_id}: feature-store label copy is missing")
        elif not np.array_equal(np.load(feature_label_path), labels):
            failures.append(f"{subject_id}: feature-store labels differ from processed labels")

        if full:
            if not np.isfinite(features).all():
                failures.append(f"{subject_id}: contains NaN or infinity outside the sampled rows")
            hashes[file_hash(feature_path)].append(subject_id)
        summaries.append((float(sample.mean()), float(sample.std())))

    duplicates = [ids for ids in hashes.values() if len(ids) > 1]
    if duplicates:
        failures.append(f"exact duplicate feature files: {duplicates}")

    means, stds = zip(*summaries)
    status = "PASS" if not failures else "FAIL"
    print(
        f"{dataset}: {status} — {len(feature_files)} files, "
        f"sample feature means [{min(means):.3g}, {max(means):.3g}], "
        f"sample feature SDs [{min(stds):.3g}, {max(stds):.3g}]"
    )
    for failure in failures:
        print(f"  - {failure}")
    return not failures


if __name__ == "__main__":
    full = "--full" in sys.argv
    root = Path(FEATURES_HANDCRAFTED_ROOT).resolve()
    dummy_root = (Path(__file__).parent / "data/features/dummy").resolve()
    print(f"Auditing: {root}")
    if root == dummy_root:
        print("FAIL — config.py points at dummy_features.py output, not handcrafted features.")
        raise SystemExit(1)

    ok = all(audit_dataset(dataset, full=full) for dataset in DATASETS)
    if ok:
        print("\nIntegrity checks passed. Provenance still requires the extractor and a manifest.")
    raise SystemExit(0 if ok else 1)
