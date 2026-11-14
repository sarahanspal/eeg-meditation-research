"""Inspect the SKY dataset's files and convert epoch arrays to EEG features.

Run this module directly to list files and summarize up to three MATLAB files.
Importing it has no filesystem or data-processing side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from braboszcz_feature_extraction import (
    CANONICAL_BANDS,
    COMMON_CHANNELS_14,
    band_power,
    compute_psd,
)
from config import RAW_PATHS


SKY_RENAME = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def _matlab_summary(path: Path) -> dict[str, tuple[Any, Any]]:
    """Return variable shapes and dtypes for a MATLAB file, including v7.3."""
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {
            name: (getattr(value, "shape", None), getattr(value, "dtype", None))
            for name, value in data.items()
            if not name.startswith("__")
        }
    except NotImplementedError:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is MATLAB v7.3; install h5py to inspect it"
            ) from exc
        with h5py.File(path, "r") as data:
            return {
                name: (value.shape, value.dtype)
                for name, value in data.items()
                if isinstance(value, h5py.Dataset)
            }


def inspect_sky_directory(sky_root: str | Path, max_files: int = 40,
                          max_mat_files: int = 3) -> None:
    """Print a compact inventory and MATLAB variable summaries."""
    root = Path(sky_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"SKY data directory does not exist: {root}")

    for path in sorted(root.rglob("*"))[:max_files]:
        size = path.stat().st_size if path.is_file() else ""
        print(path.relative_to(root), size)

    for path in sorted(root.rglob("*.mat"))[:max_mat_files]:
        try:
            print(path.name, _matlab_summary(path))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Could not inspect {path.name}: {exc}")


def inspect_sky_recordings_and_timings(sky_root: str | Path) -> None:
    """Print recording metadata and representative epoch timing values."""
    root = Path(sky_root).expanduser()
    edfs = sorted(root.rglob("*.edf"))
    print("EDF count:", len(edfs))
    print("First EDFs:", [p.name for p in edfs[:4]])
    print("Last EDFs:", [p.name for p in edfs[-4:]])
    print("EDF parent folders:", sorted({p.parent.name for p in edfs}))
    if edfs:
        raw = mne.io.read_raw_edf(edfs[0], preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        print("First EDF:", edfs[0])
        print("sfreq:", sfreq, "| duration min:", raw.n_times / sfreq / 60,
              "| channels:", raw.ch_names)
        print("annotations:", raw.annotations[:10] if len(raw.annotations) else "none")

    for filename in ("subjects_epochs.mat", "controls_epochs.mat"):
        path = root / filename
        if not path.exists():
            print(f"{filename}: not found")
            continue
        try:
            data = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
        except NotImplementedError:
            print(f"{filename}: MATLAB v7.3; inspect with h5py")
            continue
        if "timings" not in data:
            print(f"{filename}: no 'timings' variable; variables:",
                  [key for key in data if not key.startswith("__")])
            continue
        timings = np.asarray(data["timings"])
        print(filename, "timings shape:", timings.shape)
        if timings.ndim != 3 or not timings.size:
            print("  unexpected timing array shape")
            continue
        print("  first subject, first six:", timings[0, :, :6])
        print("  first subject, last three:", timings[0, :, -3:])
        print("  unique diffs in row:", np.unique(np.diff(timings[0], axis=1))[:8])
        if edfs:
            sfreq = raw.info["sfreq"]
            print("  row first/last seconds:", [
                (round(float(timings[0, row, 0]) / sfreq),
                 round(float(timings[0, row, -1]) / sfreq))
                for row in range(timings.shape[1])
            ])


def sky_features_from_epochs(
    subject: str,
    label: str,
    arr: np.ndarray,
    ch_names: list[str],
    sfreq: float = 256.0,
) -> pd.DataFrame:
    """Convert ``(epochs, channels, samples)`` microvolt data to band features."""
    data = np.asarray(arr)
    if data.ndim != 3:
        raise ValueError(
            "arr must have shape (n_epochs, n_channels, n_times); "
            f"got {data.shape}"
        )
    if data.shape[1] != len(ch_names):
        raise ValueError(
            f"channel count mismatch: data has {data.shape[1]}, "
            f"but {len(ch_names)} names were provided"
        )
    if not np.isfinite(data).all():
        raise ValueError("arr contains NaN or infinite values")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    names = [SKY_RENAME.get(name.upper(), name) for name in ch_names]
    if len(set(names)) != len(names):
        raise ValueError("channel names are not unique after SKY channel renaming")

    epochs = mne.EpochsArray(
        data * 1e-6,
        mne.create_info(names, sfreq, ch_types="eeg"),
        tmin=0,
        verbose=False,
        metadata=pd.DataFrame(
            {
                "subject": subject,
                "condition": label,
                "epoch_idx": np.arange(data.shape[0]),
            }
        ),
    )
    if "Cz" in epochs.ch_names:
        epochs.drop_channels(["Cz"])
    epochs.set_eeg_reference("average", projection=False, verbose=False)
    selected = [name for name in COMMON_CHANNELS_14 if name in epochs.ch_names]
    if not selected:
        raise ValueError(
            "none of the expected common EEG channels were found; "
            f"available channels: {epochs.ch_names}"
        )
    epochs.pick(selected)

    freqs, psd = compute_psd(epochs)
    features = pd.DataFrame(
        {
            f"{channel}_{band}": band_power(freqs, psd, limits)[:, channel_idx]
            for channel_idx, channel in enumerate(epochs.ch_names)
            for band, limits in CANONICAL_BANDS.items()
        }
    )
    features.insert(0, "epoch_idx", np.arange(len(epochs)))
    features.insert(0, "condition", label)
    features.insert(0, "subject", subject)
    return features


def main() -> None:
    inspect_sky_directory(RAW_PATHS["sky"])
    inspect_sky_recordings_and_timings(RAW_PATHS["sky"])


if __name__ == "__main__":
    main()
