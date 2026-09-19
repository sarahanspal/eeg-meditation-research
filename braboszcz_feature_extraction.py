"""
braboszcz_feature_extraction.py
================================

Feature extraction pipeline for Braboszcz, Cahn, Levy, Fernandez & Delorme (2017),
"Increased Gamma Brainwave Amplitude Compared to Control in Three Different
Meditation Traditions", PLOS ONE (OpenNeuro ds003969).

WHAT THIS SCRIPT DOES
----------------------
For a single subject, loads the already-split BIDS task runs:
    sub-XXX_task-med2_eeg.bdf     -> last 10 min of the MED block (tradition-specific
                                      practice), i.e. exactly the "MED" contrast used
                                      in the paper's Fig 3/4 analyses.
    sub-XXX_task-think2_eeg.bdf   -> last 10 min of the IMW block (instructed
                                      mind-wandering), the paper's matched control.

...and produces one row per clean epoch with:
    - relative power in 6 canonical bands (delta/theta/alpha/beta/gamma1/gamma2),
      per channel, over a common 1-70 Hz range
    - individual alpha frequency (IAF) per channel
    - homologous-pair asymmetry index (ln(right) - ln(left)) per band
    - cross-band power ratios
    - (optional) aperiodic offset/exponent per channel via FOOOF/specparam,
      fit on the condition-averaged spectrum

WHY 1-70 Hz AND NOT THE PAPER'S OWN BANDS
------------------------------------------
The paper's own gamma definition (60-110 Hz) cannot be reproduced later in the
SKY dataset (hardware bandpass 0.1-75 Hz), and each of the three source papers
uses different band edges. To keep this feature set mergeable with the other
two datasets later, everything here is computed on ONE canonical band scheme
(see CANONICAL_BANDS below) from a common 1-70 Hz spectrum, not the paper's
own idiosyncratic bands. This script is Braboszcz-only for now, but every
function is written to be reused verbatim on the other two datasets.

WHAT IS AN APPROXIMATION OF THE PAPER'S METHODS (READ BEFORE TRUSTING NUMBERS)
--------------------------------------------------------------------------------
1. Highpass filter: the paper uses a custom IIR (order 6, 0.3 Hz transition
   band) built in EEGLAB. Below we approximate this with a zero-phase
   Butterworth IIR of the same order via MNE. Edge behavior will differ
   slightly from the paper's exact filter.
2. Continuous-data artifact rejection (paper's `pop_rejcont`): re-implemented
   from the Methods description (1 s windows, 0.5 s overlap, reject runs of
   >=8 contiguous windows exceeding 17 dB in 0-10 Hz or 14 dB in 35-45 Hz).
   The paper used 35-128 Hz for the high-frequency check; that is only valid
   at the original 1024 Hz sampling, so here it is applied on the raw data
   BEFORE downsampling, then bad segments are annotated so they survive the
   later resample/epoching.
3. Bad channel detection: the paper's bad channels were identified by manual
   visual inspection (avg. 5/subject). Here this is replaced with a
   MAD-based robust-variance outlier detector, which is not equivalent -
   treat this as a placeholder for a proper manual/automated (e.g.
   `pyprep`/`autoreject`) pass before drawing scientific conclusions.
4. ICA component rejection: the paper had a human expert visually inspect
   1-5 ICA components per subject for eye/muscle topography+spectrum. Here
   `ica.find_bads_eog` / `ica.find_bads_muscle` are used as an automatic
   proxy. These are reasonable but not identical to expert visual rejection.
5. Epoch length: the paper's own spectral analysis used 1 s, non-overlapping
   epochs (matching their need for state/trait ANOVAs at fine time
   resolution). Here EPOCH_LEN_SEC defaults to 4 s, trading temporal
   granularity for a finer frequency resolution (0.25 Hz vs 1 Hz bins),
   which matters for cleanly separating your canonical band edges (e.g.
   the theta/alpha boundary at 8 Hz). Change EPOCH_LEN_SEC back to 1.0 if
   you want to match the paper's original epoching exactly.

CHANNELS
--------
COMMON_CHANNELS_14 / ASYMMETRY_PAIRS below are the confirmed 14-channel
intersection across Braboszcz / SKY / Brandmeyer (T7/T8/P7/P8 naming, not
the older T3/T4/T5/T6 - if SKY or Brandmeyer's channels.tsv uses the old
labels, that renaming needs to happen for THOSE datasets' pipelines too,
otherwise the "intersection" silently isn't one).

Braboszcz's raw .bdf channels come in natively as Biosemi cap labels
(A1..A32, B1..B32), not 10-20 names. `rename_to_standard_1020()` remaps
them positionally via MNE's built-in 'biosemi64' montage (assumes the
native A1../B32 physical order hasn't been altered upstream - it includes
a diagnostic check and will tell you at runtime whether it applied the
remap or found the channels already named).

REQUIREMENTS
------------
    pip install mne numpy scipy pandas fooof

USAGE
-----
Edit the SUBJECT_FILES dict in `main()` at the bottom, then:
    python braboszcz_feature_extraction.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz  # numpy < 2.0 fallback

try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    try:
        from specparam import SpectralModel as FOOOF  # fooof was renamed 'specparam'
        HAS_FOOOF = True
    except ImportError:
        HAS_FOOOF = False
        warnings.warn(
            "fooof/specparam not installed - aperiodic (1/f) features will be "
            "skipped. `pip install fooof` to enable them."
        )

mne.set_log_level("WARNING")

# ============================================================================
# CONFIG - dataset-specific constants. Confirm these against your actual
# channels.tsv / eeg.json before trusting the output.
# ============================================================================

# Canonical band scheme shared across all three datasets in the larger project.
# Capped at 70 Hz because SKY's hardware bandpass is 0.1-75 Hz; do not extend
# gamma2 upward if you want this feature set to stay mergeable with SKY later.
CANONICAL_BANDS = {
    "delta":  (1.0, 4.0),
    "theta":  (4.0, 8.0),
    "alpha":  (8.0, 13.0),
    "beta":   (13.0, 30.0),
    "gamma1": (30.0, 45.0),
    "gamma2": (45.0, 70.0),
}
PSD_FMIN, PSD_FMAX = 1.0, 70.0   # total-power denominator for relative band power
IAF_SEARCH_BAND = (7.0, 13.0)    # individual alpha frequency search window

# Confirmed intersection across Braboszcz / SKY / Brandmeyer (standard 10-20 names,
# post-renaming - see rename_to_standard_1020() below for how raw channels get here).
COMMON_CHANNELS_14 = [
    "C3", "C4", "F3", "F4", "F7", "F8",
    "O1", "O2", "P3", "P4", "P7", "P8", "T7", "T8",
]
# Left/right homologous pairs drawn from the list above (7 pairs -> all 14 chans).
ASYMMETRY_PAIRS = [
    ("F3", "F4"), ("F7", "F8"), ("C3", "C4"),
    ("P3", "P4"), ("P7", "P8"), ("O1", "O2"), ("T7", "T8"),
]

# Cross-band ratio definitions: name -> (numerator bands, denominator bands)
RATIO_DEFS = {
    "theta_alpha":     (["theta"], ["alpha"]),
    "alpha_beta":      (["alpha"], ["beta"]),
    "gamma_theta":     (["gamma1", "gamma2"], ["theta"]),
    "highfreq_lowfreq": (["beta", "gamma1", "gamma2"], ["delta", "theta", "alpha"]),
}

# Continuous-data artifact rejection thresholds, from the paper's Methods.
REJCONT_SEG_LEN = 1.0        # seconds
REJCONT_SEG_STEP = 0.5       # seconds (50% overlap)
REJCONT_LOW_BAND = (0.0, 10.0)
REJCONT_HIGH_BAND = (35.0, 45.0)   # capped well below the paper's 128 Hz upper
                                     # edge, which only applies at 1024 Hz raw
REJCONT_LOW_THRESH_DB = 17.0
REJCONT_HIGH_THRESH_DB = 14.0
REJCONT_MIN_CONSEC = 8

TARGET_SFREQ = 256.0
HIGHPASS_HZ = 1.0
EPOCH_LEN_SEC = 4.0   # see module docstring point 5 for rationale / how to revert
EPOCH_OVERLAP_SEC = 0.0

# CONFIRM against sub-XXX_channels.tsv: which EXG channel is the right mastoid
# the paper references to. This is a guess based on typical Biosemi wiring
# (EXG1/2 = mastoids, EXG3-6 = EOG); verify before running at scale.
RIGHT_MASTOID_CH = "EXG2"


# ============================================================================
# PREPROCESSING
# ============================================================================

def load_raw_bdf(bdf_path: str) -> mne.io.Raw:
    """Load a Biosemi .bdf file with all channels, and drop the 'Status' trigger
    channel if present (not needed for spectral analysis)."""
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    if "Status" in raw.ch_names:
        raw.drop_channels(["Status"])
    return raw


def apply_channel_types_from_bids(raw: mne.io.Raw, channels_tsv_path: str) -> mne.io.Raw:
    """Set MNE channel types (eeg/eog/misc/...) from a BIDS channels.tsv sidecar,
    instead of hardcoding channel names, since naming/ordering can vary.
    """
    ch_info = pd.read_csv(channels_tsv_path, sep="\t")
    bids_to_mne = {
        "EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg",
        "MISC": "misc", "REF": "misc", "TRIG": "stim",
    }
    type_map = {}
    for _, row in ch_info.iterrows():
        name = str(row["name"])
        bids_type = str(row.get("type", "EEG")).upper()
        if name in raw.ch_names:
            type_map[name] = bids_to_mne.get(bids_type, "misc")
    raw.set_channel_types(type_map, verbose=False)
    return raw


def rename_to_standard_1020(raw: mne.io.Raw, montage_name: str = "biosemi64") -> mne.io.Raw:
    """Rename raw's EEG channels to standard 10-20 names and attach the
    corresponding montage.

    Biosemi BDF files typically store EEG channels under the cap's native
    labels (A1..A32, B1..B32) in a fixed physical layout order that matches
    MNE's built-in 'biosemi64' standard montage 1:1 by POSITION (not by
    name). The fix is a positional remap:
        montage = mne.channels.make_standard_montage('biosemi64')
        mapping = dict(zip(raw.ch_names, montage.ch_names))
        raw.rename_channels(mapping)
        raw.set_montage(montage)

    This is correct ONLY if raw's first len(montage.ch_names) channels are
    still in that native A1..B32 order (i.e. nothing upstream already
    reordered or renamed them). If channels.tsv already lists standard 10-20
    names (some BIDS conversions do this), the positional remap would be
    WRONG (it would relabel already-correct names using cap position,
    silently corrupting the montage). So: check for overlap with standard
    10-20 names first, and only apply the positional remap if the channels
    still look like native Biosemi labels.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    standard_names = set(montage.ch_names)
    already_standard = len(set(raw.ch_names) & standard_names)

    if already_standard >= 32:
        # Channels already look like standard 10-20 names - trust them, just
        # attach the montage for positions (needed later for interpolation).
        print(f"    channel names already look standard-1020-like "
              f"({already_standard}/{len(standard_names)} overlap) - "
              f"skipping positional remap, only attaching montage.")
        raw.set_montage(montage, on_missing="warn", match_case=False, verbose=False)
        return raw

    # Otherwise assume native Biosemi A1../B32 ordering and remap positionally.
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    if len(eeg_names) != len(montage.ch_names):
        warnings.warn(
            f"EEG channel count ({len(eeg_names)}) does not match the "
            f"'{montage_name}' montage ({len(montage.ch_names)}). Positional "
            "remap may be misaligned - inspect raw.ch_names manually before "
            "trusting downstream channel labels."
        )
    mapping = dict(zip(eeg_names, montage.ch_names))
    print(f"    remapping {len(mapping)} channels positionally via "
          f"'{montage_name}' layout, e.g. {dict(list(mapping.items())[:3])} ...")
    raw.rename_channels(mapping)
    raw.set_montage(montage, on_missing="warn", match_case=False, verbose=False)
    return raw


def rereference_to_mastoid(raw: mne.io.Raw, mastoid_ch: str = RIGHT_MASTOID_CH) -> mne.io.Raw:
    """Re-reference to a single mastoid channel, matching the paper
    ('EEG data were first referenced to the right mastoid'), then drop the
    now-flat reference channel.
    """
    if mastoid_ch not in raw.ch_names:
        raise ValueError(
            f"Reference channel '{mastoid_ch}' not found in {raw.ch_names}. "
            "Check channels.tsv for the correct mastoid channel name."
        )
    raw.set_eeg_reference(ref_channels=[mastoid_ch], verbose=False)
    raw.drop_channels([mastoid_ch])
    return raw


def highpass_filter(raw: mne.io.Raw, l_freq: float = HIGHPASS_HZ) -> mne.io.Raw:
    """Zero-phase Butterworth IIR highpass, approximating the paper's 1 Hz,
    order-6, 0.3 Hz transition-band IIR filter (see module docstring point 1).
    """
    raw.filter(
        l_freq=l_freq, h_freq=None, picks="eeg",
        method="iir", iir_params=dict(order=6, ftype="butter"),
        phase="zero", verbose=False,
    )
    return raw


def detect_bad_segments(
    raw: mne.io.Raw,
    seg_len: float = REJCONT_SEG_LEN,
    seg_step: float = REJCONT_SEG_STEP,
    low_band=REJCONT_LOW_BAND,
    high_band=REJCONT_HIGH_BAND,
    low_thresh_db: float = REJCONT_LOW_THRESH_DB,
    high_thresh_db: float = REJCONT_HIGH_THRESH_DB,
    min_consec_segs: int = REJCONT_MIN_CONSEC,
) -> mne.Annotations:
    """Approximate EEGLAB's `pop_rejcont` (see module docstring point 2).

    Slides a window across the continuous EEG, computes a rough per-window
    per-channel amplitude (dB) in a low and a high frequency band, and marks
    a segment 'bad' wherever >=min_consec_segs consecutive overlapping
    windows exceed either threshold on any channel. Call this BEFORE
    downsampling, on the still-referenced-but-not-yet-filtered data, to keep
    the high-frequency check meaningful.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks="eeg") * 1e6  # V -> uV: the 17/14 dB thresholds below
                                              # are EEGLAB conventions computed on
                                              # microvolt-scale amplitude; without this
                                              # conversion the dB values never cross
                                              # threshold and nothing gets flagged.
    n_times = data.shape[1]
    seg_len_smp = int(round(seg_len * sfreq))
    step_smp = int(round(seg_step * sfreq))
    starts = np.arange(0, max(n_times - seg_len_smp, 1), step_smp)

    def band_db(psd, freqs, band):
        idx = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(idx):
            return np.full(psd.shape[0], -np.inf)
        amp = np.sqrt(np.mean(psd[:, idx], axis=-1))
        return 20 * np.log10(amp + 1e-12)

    flags = np.zeros(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        seg = data[:, s:s + seg_len_smp]
        if seg.shape[1] < seg_len_smp:
            continue
        freqs, psd = welch(seg, fs=sfreq, nperseg=seg_len_smp, axis=-1)
        low_db = band_db(psd, freqs, low_band)
        high_db = band_db(psd, freqs, high_band)
        flags[i] = bool(np.any(low_db > low_thresh_db) or np.any(high_db > high_thresh_db))

    bad_onsets, bad_durs = [], []
    run_start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and run_start is None:
            run_start = i
        elif not f and run_start is not None:
            run_len = i - run_start
            if run_len >= min_consec_segs:
                onset_sec = starts[run_start] / sfreq
                offset_sec = (starts[min(i - 1, len(starts) - 1)] + seg_len_smp) / sfreq
                bad_onsets.append(onset_sec)
                bad_durs.append(max(offset_sec - onset_sec, seg_len))
            run_start = None

    return mne.Annotations(
        onset=bad_onsets, duration=bad_durs,
        description=["bad_segment"] * len(bad_onsets),
    )


def detect_bad_channels_by_variance(raw: mne.io.Raw, z_thresh: float = 3.0) -> list:
    """Placeholder for the paper's manual bad-channel inspection (module
    docstring point 3): flags channels whose overall variance is a robust
    (MAD-based) outlier relative to the montage.
    """
    picks = mne.pick_types(raw.info, eeg=True)
    ch_names = np.array(raw.ch_names)[picks]
    data = raw.get_data(picks=picks)
    var = data.var(axis=1)
    med = np.median(var)
    mad = np.median(np.abs(var - med)) * 1.4826 + 1e-20
    z = (var - med) / mad
    return list(ch_names[np.abs(z) > z_thresh])


def run_ica_and_clean(raw: mne.io.Raw, random_state: int = 97):
    """Automatic proxy for the paper's manual ICA component rejection
    (module docstring point 4): fits Infomax ICA, flags likely eye and
    muscle components via MNE's built-in heuristics, removes them, and
    returns the cleaned raw plus which components were excluded.
    """
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    n_components = min(0.99, len(mne.pick_types(raw.info, eeg=True)) - 1)
    ica = mne.preprocessing.ICA(
        n_components=n_components, method="infomax",
        fit_params=dict(extended=True), random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw_for_ica, verbose=False)

    excluded = set()
    try:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], threshold=3.0, verbose=False)
        excluded.update(eog_inds)
    except Exception as e:
        warnings.warn(f"EOG component detection failed: {e}")
    try:
        muscle_inds, _ = ica.find_bads_muscle(raw, verbose=False)
        excluded.update(muscle_inds)
    except Exception as e:
        warnings.warn(f"Muscle component detection failed: {e}")

    ica.exclude = sorted(excluded)
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean, ica, ica.exclude


# ============================================================================
# EPOCHING
# ============================================================================

def make_fixed_epochs(raw: mne.io.Raw, duration: float = EPOCH_LEN_SEC,
                       overlap: float = EPOCH_OVERLAP_SEC) -> mne.Epochs:
    """Fixed-length epochs, automatically dropping any epoch that overlaps a
    'bad_segment' annotation (MNE's default `reject_by_annotation=True`)."""
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=duration - 1.0 / raw.info["sfreq"],
        baseline=None, preload=True, reject_by_annotation=True, verbose=False,
    )
    return epochs


# ============================================================================
# SPECTRAL FEATURES
# ============================================================================

def compute_psd(epochs: mne.Epochs, fmin: float = PSD_FMIN, fmax: float = PSD_FMAX):
    """Multitaper PSD per epoch per channel. Returns (freqs, psd) where
    psd has shape (n_epochs, n_channels, n_freqs)."""
    spectrum = epochs.compute_psd(method="multitaper", fmin=fmin, fmax=fmax, verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple,
                relative: bool = True, total_band: tuple = (PSD_FMIN, PSD_FMAX)) -> np.ndarray:
    """Integrate power within `band` (trapezoidal rule). If relative=True,
    divide by total power in `total_band` (relative power controls for
    cross-dataset/cross-amplifier gain differences - see project notes on
    why absolute uV^2 is not comparable across the three source studies).
    Returns shape (n_epochs, n_channels).
    """
    idx_band = (freqs >= band[0]) & (freqs < band[1])
    bp = np.trapezoid(psd[..., idx_band], freqs[idx_band], axis=-1)
    if relative:
        idx_total = (freqs >= total_band[0]) & (freqs < total_band[1])
        total = np.trapezoid(psd[..., idx_total], freqs[idx_total], axis=-1)
        bp = bp / (total + 1e-20)
    return bp


def individual_alpha_frequency(freqs: np.ndarray, psd: np.ndarray,
                                band: tuple = IAF_SEARCH_BAND) -> np.ndarray:
    """Per-epoch, per-channel frequency of the PSD peak within `band`.
    Returns shape (n_epochs, n_channels)."""
    idx = (freqs >= band[0]) & (freqs <= band[1])
    sub_psd = psd[..., idx]
    sub_freqs = freqs[idx]
    peak_idx = np.argmax(sub_psd, axis=-1)
    return sub_freqs[peak_idx]


def fit_aperiodic_spectrum(freqs: np.ndarray, mean_psd: np.ndarray, ch_names: list,
                            freq_range: tuple = (PSD_FMIN, PSD_FMAX)) -> pd.DataFrame:
    """Fit FOOOF/specparam to the CONDITION-AVERAGED spectrum per channel
    (fitting per single short epoch is too noisy) and return the aperiodic
    offset/exponent - a broadband '1/f slope' feature distinct from any
    single band, useful for checking whether apparent band-power effects
    are actually just a spectral tilt (see project notes).
    """
    if not HAS_FOOOF:
        return pd.DataFrame(columns=["channel", "aperiodic_offset", "aperiodic_exponent", "r_squared"])
    rows = []
    for ci, ch in enumerate(ch_names):
        fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
        try:
            fm.fit(freqs, mean_psd[ci], freq_range)
            offset, exponent = fm.aperiodic_params_
            r2 = fm.r_squared_
        except Exception:
            offset, exponent, r2 = np.nan, np.nan, np.nan
        rows.append({"channel": ch, "aperiodic_offset": offset,
                     "aperiodic_exponent": exponent, "r_squared": r2})
    return pd.DataFrame(rows)


# ============================================================================
# DERIVED FEATURES: ASYMMETRY + CROSS-BAND RATIOS
# ============================================================================

def asymmetry_index(band_df: pd.DataFrame, pairs=ASYMMETRY_PAIRS,
                     bands=CANONICAL_BANDS.keys()) -> pd.DataFrame:
    """ln(right) - ln(left) per homologous pair per band (Coan & Allen 2004 /
    same definition used in the SKY paper's Asymmetry Index)."""
    out = {}
    for left, right in pairs:
        for band in bands:
            l_col, r_col = f"{left}_{band}", f"{right}_{band}"
            if l_col in band_df.columns and r_col in band_df.columns:
                out[f"asym_{left}-{right}_{band}"] = (
                    np.log(band_df[r_col] + 1e-20) - np.log(band_df[l_col] + 1e-20)
                )
            else:
                warnings.warn(f"Skipping asymmetry pair {left}/{right}: missing column(s).")
    return pd.DataFrame(out, index=band_df.index)


def cross_band_ratios(band_df: pd.DataFrame, ch_names: list,
                       ratio_defs=RATIO_DEFS) -> pd.DataFrame:
    """Per-channel ratios between (sums of) canonical bands, e.g. theta/alpha,
    (gamma1+gamma2)/theta, high-freq/low-freq (Braboszcz's own 60-110/2-11
    ratio, recast onto the canonical scheme)."""
    out = {}
    for ch in ch_names:
        for ratio_name, (num_bands, den_bands) in ratio_defs.items():
            num_cols = [f"{ch}_{b}" for b in num_bands if f"{ch}_{b}" in band_df.columns]
            den_cols = [f"{ch}_{b}" for b in den_bands if f"{ch}_{b}" in band_df.columns]
            if not num_cols or not den_cols:
                continue
            numerator = band_df[num_cols].sum(axis=1)
            denominator = band_df[den_cols].sum(axis=1) + 1e-20
            out[f"{ch}_ratio_{ratio_name}"] = numerator / denominator
    return pd.DataFrame(out, index=band_df.index)


# ============================================================================
# FULL PIPELINE FOR ONE (SUBJECT, CONDITION) RUN
# ============================================================================

def clean_bad_channels(raw: mne.io.Raw, bad_chs: list) -> mne.io.Raw:
    """Handle bad channels via spherical-spline interpolation, with a safety
    net: interpolate_bads() can occasionally emit `divide by zero` /
    `overflow` / `invalid value` warnings from its interpolation-matrix
    computation (seen in practice - e.g. degenerate montage geometry for a
    specific channel). If that happens, the reconstructed signal for the
    affected channel(s) cannot be trusted, so drop them instead of silently
    keeping garbage.
    """
    if not bad_chs:
        return raw
    raw.info["bads"] = bad_chs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw.interpolate_bads(reset_bads=True, verbose=False)
        interp_warned = any(
            issubclass(w.category, RuntimeWarning)
            and any(k in str(w.message) for k in ("divide by zero", "overflow", "invalid value"))
            for w in caught
        )
    still_bad = [ch for ch in bad_chs if ch in raw.ch_names
                 and not np.all(np.isfinite(raw.get_data(picks=[ch])))]
    if interp_warned or still_bad:
        drop_these = still_bad if still_bad else bad_chs
        warnings.warn(
            f"interpolate_bads() looked numerically unstable "
            f"(RuntimeWarning={interp_warned}, non-finite channels={still_bad}). "
            f"Dropping {drop_these} instead of trusting interpolated data. "
            "If this keeps happening, inspect raw.get_montage().get_positions() "
            "for duplicate/zero-norm coordinates on the affected channels."
        )
        raw.drop_channels([ch for ch in drop_these if ch in raw.ch_names])
    return raw


def process_subject_condition(
    bdf_path: str,
    channels_tsv_path: str,
    subject_id: str,
    condition_label: str,
    right_mastoid_ch: str = RIGHT_MASTOID_CH,
    keep_channels: list = None,
):
    """Run the full pipeline on one BIDS EEG run and return a per-epoch
    feature DataFrame, plus (freqs, mean_psd, ch_names) for optional
    aperiodic fitting / sanity-check plotting.
    """
    print(f"[{subject_id} | {condition_label}] loading {bdf_path}")
    raw = load_raw_bdf(bdf_path)
    raw = apply_channel_types_from_bids(raw, channels_tsv_path)
    raw = rename_to_standard_1020(raw)

    # 1) artifact-segment detection BEFORE downsampling (paper ran this on
    #    the pre-downsample data so the high-frequency check stays valid)
    bad_annot = detect_bad_segments(raw)
    raw.set_annotations(bad_annot)
    print(f"    flagged {len(bad_annot)} bad segment(s), "
          f"{bad_annot.duration.sum():.1f}s total")

    # 2) re-reference, then standard preprocessing chain
    raw = rereference_to_mastoid(raw, right_mastoid_ch)
    raw.pick_types(eeg=True, eog=False, misc=False, exclude=[])
    # montage already attached by rename_to_standard_1020() above (real biosemi64
    # positions, more accurate for interpolate_bads() than a generic idealized cap)

    bad_chs = detect_bad_channels_by_variance(raw)
    if bad_chs:
        print(f"    flagged bad channel(s) by variance: {bad_chs}")
        raw = clean_bad_channels(raw, bad_chs)

    raw = highpass_filter(raw)
    raw.resample(TARGET_SFREQ, npad="auto")

    # 3) ICA cleaning
    raw_clean, ica, excluded_ics = run_ica_and_clean(raw)
    print(f"    ICA excluded components: {excluded_ics}")

    # 4) restrict to the requested channel subset (e.g. COMMON_CHANNELS_14)
    if keep_channels is not None:
        available = [ch for ch in keep_channels if ch in raw_clean.ch_names]
        missing = sorted(set(keep_channels) - set(available))
        if missing:
            warnings.warn(f"Requested channels not found and skipped: {missing}")
        raw_clean.pick_channels(available, ordered=True)

    # 5) epoch + spectral features
    epochs = make_fixed_epochs(raw_clean)
    n_dropped = len(epochs.drop_log) - len(epochs)
    print(f"    {len(epochs)} clean epochs kept, {n_dropped} dropped (artifact overlap)")

    freqs, psd = compute_psd(epochs)
    ch_names = epochs.ch_names

    band_df = pd.DataFrame(index=range(len(epochs)))
    for band_name, band_range in CANONICAL_BANDS.items():
        bp = band_power(freqs, psd, band_range)
        for ci, ch in enumerate(ch_names):
            band_df[f"{ch}_{band_name}"] = bp[:, ci]

    iaf = individual_alpha_frequency(freqs, psd)
    for ci, ch in enumerate(ch_names):
        band_df[f"{ch}_IAF"] = iaf[:, ci]

    asym_df = asymmetry_index(band_df)
    ratio_df = cross_band_ratios(band_df, ch_names)

    feat_df = pd.concat([band_df, asym_df, ratio_df], axis=1)
    feat_df.insert(0, "epoch_idx", range(len(feat_df)))
    feat_df.insert(0, "condition", condition_label)
    feat_df.insert(0, "subject", subject_id)

    mean_psd = psd.mean(axis=0)  # (n_channels, n_freqs), for aperiodic fitting
    return feat_df, freqs, mean_psd, ch_names


# ============================================================================
# MAIN - single-subject test run
# ============================================================================

def main():
    # Google Drive-synced data root (Drive desktop sync mounts as a normal
    # local path, so this works exactly like any other filesystem path -
    # just slow the first time a file is read while it downloads on demand).
    DATA_ROOT = ("/Users/sarshanspal/Library/CloudStorage/"
                 "GoogleDrive-sarahanspal@gmail.com/My Drive/meditation-eeg")
    RAW_PATHS = {
        "brandmeyer": f"{DATA_ROOT}/openneuro",   # NOTE: trailing space removed
        "sky":        f"{DATA_ROOT}/sky+control",
        "mantra":     f"{DATA_ROOT}/mantra-ds",   # not wired up yet - separate dataset
        "braboszcz":  f"{DATA_ROOT}/delorme2",
    }

    subject_id = "sub-006"
    base = Path(RAW_PATHS["braboszcz"]) / subject_id / "eeg"
    runs = {
        "MED": {
            "bdf": base / f"{subject_id}_task-med2_eeg.bdf",
            "channels_tsv": base / f"{subject_id}_task-med2_channels.tsv",
        },
        "IMW": {
            "bdf": base / f"{subject_id}_task-think2_eeg.bdf",
            "channels_tsv": base / f"{subject_id}_task-think2_channels.tsv",
        },
    }

    all_feats = []
    mean_spectra = {}  # condition -> (freqs, mean_psd, ch_names), for aperiodic fit
    for condition_label, paths in runs.items():
        feat_df, freqs, mean_psd, ch_names = process_subject_condition(
            bdf_path=str(paths["bdf"]),
            channels_tsv_path=str(paths["channels_tsv"]),
            subject_id=subject_id,
            condition_label=condition_label,
            keep_channels=COMMON_CHANNELS_14,
        )
        all_feats.append(feat_df)
        mean_spectra[condition_label] = (freqs, mean_psd, ch_names)

    feat_df = pd.concat(all_feats, ignore_index=True)
    out_path = f"{subject_id}_braboszcz_test_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(feat_df)} epoch-level feature rows to {out_path}")

    # Optional: aperiodic fit on each condition's mean spectrum
    if HAS_FOOOF:
        aperiodic_frames = []
        for condition_label, (freqs, mean_psd, ch_names) in mean_spectra.items():
            ap_df = fit_aperiodic_spectrum(freqs, mean_psd, ch_names)
            ap_df.insert(0, "condition", condition_label)
            ap_df.insert(0, "subject", subject_id)
            aperiodic_frames.append(ap_df)
        aperiodic_df = pd.concat(aperiodic_frames, ignore_index=True)
        ap_out_path = f"{subject_id}_braboszcz_aperiodic_features.csv"
        aperiodic_df.to_csv(ap_out_path, index=False)
        print(f"Saved aperiodic features to {ap_out_path}")

    # Quick sanity summary: per-condition mean relative alpha/gamma power,
    # averaged across epochs and the two occipital channels, as a crude
    # replication check against the paper's headline effects.
    for band in ("alpha", "gamma1", "gamma2"):
        cols = [c for c in feat_df.columns if c.startswith(("O1_", "O2_")) and c.endswith(f"_{band}")]
        if cols:
            summary = feat_df.groupby("condition")[cols].mean().mean(axis=1)
            print(f"Mean occipital relative {band} power by condition:\n{summary}\n")


if __name__ == "__main__":
    main()
