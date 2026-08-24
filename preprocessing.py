import mne
import numpy as np
from scipy.signal import welch

TARGET_SFREQ = 256
EPOCH_LENGTH = 4
SAMPLES = TARGET_SFREQ * EPOCH_LENGTH  # 1024

COMMON_CHS = sorted(['C3','C4','F3','F4','F7','F8','O1','O2','P3','P4','P7','P8','T7','T8'])

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 30),
    'gamma': (30, 40)
}

def preprocess_raw(raw):
    raw.resample(TARGET_SFREQ, verbose=False)
    raw.filter(1, 40, verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.set_eeg_reference('average', verbose=False)

    available = [ch for ch in COMMON_CHS if ch in raw.ch_names]
    missing = [ch for ch in COMMON_CHS if ch not in raw.ch_names]
    if missing:
        print(f"  WARNING: missing channels {missing}")
    raw.pick_channels(available, verbose=False)
    return raw