"""
Brandmeyer MED vs MW EEG analysis
=================================

Dataset:
    OpenNeuro ds001787

Brandmeyer & Delorme:
    Reduced mind wandering in experienced meditators and associated EEG correlates

This script performs two analyses:

METHOD 1
--------
Per-subject condition means.

For each subject × electrode × band:

    MED_mean = mean(log10 absolute band power across MED trials)
    MW_mean  = mean(log10 absolute band power across MW trials)

    difference = MED_mean - MW_mean

These subject-level differences are the quantities that should later
be entered into a paired subject-level statistical analysis across
subjects.

METHOD 2
--------
Trial-level linear mixed-effects model:

    logpower ~ condition * group + (1 + condition | subject)

This is only fit when multiple subjects are available.

For a single subject, the script does NOT pretend that a mixed-effects
model is valid. It simply saves the trial-level data for later
multi-subject analysis.

IMPORTANT
---------
This script is a harmonized band-power implementation.

The original Brandmeyer paper used Morlet wavelet time-frequency
analysis. Here we use multitaper PSD on the exact Brandmeyer 10-s
epochs so that the resulting features can be harmonized with the
other datasets in the project.

The actual Brandmeyer condition definition is retained:

    Q1 > Q2  -> MED
    Q2 > Q1  -> MW
    Q1 == Q2 -> ignored
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import mne

from scipy import stats

# statsmodels is only required for Method 2
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = Path(
    "/Users/sarshanspal/Library/CloudStorage/"
    "GoogleDrive-sarahanspal@gmail.com/My Drive/meditation-eeg"
)

# IMPORTANT:
# The local OpenNeuro directory currently has a trailing space in its name.
# Keep this value in sync with the directory on disk.
RAW_PATH = DATA_ROOT / "openneuro "


# -------------------------------------------------------------------------
# Subject to run
# -------------------------------------------------------------------------

SUBJECT_ID = "sub-002"


# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------

OUTPUT_DIR = Path.cwd() / "brandmeyer_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# EEG channels used for harmonization with your other datasets
# -------------------------------------------------------------------------

COMMON_CHANNELS_14 = [
    "C3", "C4",
    "F3", "F4",
    "F7", "F8",
    "O1", "O2",
    "P3", "P4",
    "P7", "P8",
    "T7", "T8",
]


# -------------------------------------------------------------------------
# Canonical frequency bands
# -------------------------------------------------------------------------

CANONICAL_BANDS = {
    "delta":  (1.0, 4.0),
    "theta":  (4.0, 8.0),
    "alpha":  (8.0, 13.0),
    "beta":   (13.0, 30.0),
    "gamma1": (30.0, 45.0),
    "gamma2": (45.0, 70.0),
}


# -------------------------------------------------------------------------
# PSD configuration
# -------------------------------------------------------------------------

PSD_FMIN = 1.0
PSD_FMAX = 70.0


# -------------------------------------------------------------------------
# Brandmeyer epoch
# -------------------------------------------------------------------------

EPOCH_TMIN = -10.05
EPOCH_TMAX = -0.05


# -------------------------------------------------------------------------
# Small positive floor before log10
#
# PSD values should be positive. This prevents log10(0) in pathological
# numerical cases.
# -------------------------------------------------------------------------

LOG_FLOOR = 1e-20


# =============================================================================
# DATASET EVENT CODING
# =============================================================================

PROBE_ONSET_VALUE = 128

# BIDS event specification for this dataset:
#
# 2  -> response 1
# 4  -> response 2
# 8  -> response 3
# 16 -> involuntary response
#
# For the actual Q1/Q2/Q3 responses, we interpret the first three
# response events following a 128 probe onset as Q1, Q2, Q3.
#
# The keypad rating itself is therefore:
#
# 2  -> 0
# 4  -> 1
# 8  -> 2
#
# and 16 is not treated as a normal rating.
#
# The dataset's event documentation distinguishes response event
# numbers from question order; the question order is determined by
# their position within each probe.
RESPONSE_VALUES = {
    2: 0,
    4: 1,
    8: 2,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def find_brandmeyer_files(subject_id):
    """
    Robustly find the BDF and events TSV for a Brandmeyer subject.

    Handles both:

        sub-XXX/eeg/
        sub-XXX/ses-01/eeg/

    because the dataset contains both directory layouts.
    """

    subject_dir = RAW_PATH / subject_id

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Subject directory does not exist:\n{subject_dir}"
        )

    bdfs = sorted(subject_dir.rglob(
        f"{subject_id}*task-meditation*_eeg.bdf"
    ))

    if not bdfs:
        bdfs = sorted(subject_dir.rglob("*.bdf"))

    if not bdfs:
        raise FileNotFoundError(
            f"No BDF file found under:\n{subject_dir}"
        )

    if len(bdfs) > 1:
        print("WARNING: Multiple BDF files found:")
        for p in bdfs:
            print("   ", p)
        print("Using:", bdfs[0])

    bdf_path = bdfs[0]

    # Prefer the exact BIDS-matching events filename
    exact_events = bdf_path.with_name(
        bdf_path.name.replace(
            "_eeg.bdf",
            "_events.tsv"
        )
    )

    if exact_events.exists():
        events_path = exact_events
    else:
        events_candidates = sorted(
            bdf_path.parent.glob("*events.tsv")
        )

        if not events_candidates:
            raise FileNotFoundError(
                f"No events.tsv found next to:\n{bdf_path}"
            )

        events_path = events_candidates[0]

    # Channels TSV
    exact_channels = bdf_path.with_name(
        bdf_path.name.replace(
            "_eeg.bdf",
            "_channels.tsv"
        )
    )

    if exact_channels.exists():
        channels_path = exact_channels
    else:
        channel_candidates = sorted(
            bdf_path.parent.glob("*channels.tsv")
        )

        channels_path = (
            channel_candidates[0]
            if channel_candidates
            else None
        )

    return bdf_path, events_path, channels_path


# =============================================================================
# CHANNEL HANDLING
# =============================================================================

def remap_biosemi64_channel_names(raw):
    """Replace BioSemi acquisition labels (A1--B32) with 10-20 names.

    The Brandmeyer BDF files store the 64 scalp electrodes as ``A1`` through
    ``B32``.  Those labels are the acquisition order of MNE's ``biosemi64``
    montage, not anatomical channel names.  Without this conversion none of
    ``COMMON_CHANNELS_14`` can be selected later in the pipeline.
    """

    biosemi_labels = [
        *(f"A{i}" for i in range(1, 33)),
        *(f"B{i}" for i in range(1, 33)),
    ]

    present_labels = [
        name for name in biosemi_labels
        if name in raw.ch_names
    ]

    if not present_labels:
        return raw

    if len(present_labels) != len(biosemi_labels):
        warnings.warn(
            "Found only some BioSemi A1--B32 channels; leaving their names "
            "unchanged rather than applying a potentially incorrect map."
        )
        return raw

    montage = mne.channels.make_standard_montage("biosemi64")
    rename_map = dict(zip(biosemi_labels, montage.ch_names))
    raw.rename_channels(rename_map)

    return raw

def mark_auxiliary_channels(raw):
    """
    Mark BioSemi auxiliary channels as non-EEG where possible.

    This prevents MNE from trying to assign EEG montage positions
    to channels such as EXG1, GSR1, Resp, Temp, etc.
    """

    aux_names = [
        "EXG1", "EXG2", "EXG3", "EXG4",
        "EXG5", "EXG6", "EXG7", "EXG8",
        "GSR1", "GSR2",
        "Erg1", "Erg2",
        "Resp", "Plet", "Temp",
    ]

    type_map = {}

    for ch in aux_names:
        if ch in raw.ch_names:
            # MNE accepts several non-EEG channel types.
            # Misc is appropriate for channels that are not analyzed.
            type_map[ch] = "misc"

    if type_map:
        raw.set_channel_types(
            type_map,
            on_unit_change="ignore",
            verbose=False,
        )

    return raw


def standardize_channel_names(raw):
    """
    Standardize obvious capitalization variants.

    Does not perform positional remapping because Brandmeyer BDF files
    already contain standard EEG channel names.
    """

    rename_map = {}

    for ch in raw.ch_names:

        clean = ch.strip()

        # Common capitalization corrections
        replacements = {
            "c3": "C3",
            "c4": "C4",
            "f3": "F3",
            "f4": "F4",
            "f7": "F7",
            "f8": "F8",
            "o1": "O1",
            "o2": "O2",
            "p3": "P3",
            "p4": "P4",
            "p7": "P7",
            "p8": "P8",
            "t7": "T7",
            "t8": "T8",
        }

        if clean in replacements:
            rename_map[ch] = replacements[clean]

    if rename_map:
        raw.rename_channels(rename_map)

    return raw


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_raw(raw):
    """
    Brandmeyer-compatible preprocessing.

    Paper:
        - average reference
        - 256 Hz sampling
        - 2 Hz high-pass
        - average reference again

    The supplied dataset is already at 256 Hz.
    """

    print("Original sampling rate:", raw.info["sfreq"])
    print("Original channels:", len(raw.ch_names))

    # Brandmeyer BDFs use BioSemi acquisition labels (A1--B32).  Map those
    # labels before looking for the anatomical channels used in this analysis.
    raw = remap_biosemi64_channel_names(raw)

    # Standardize names
    raw = standardize_channel_names(raw)

    # Mark auxiliary channels before montage
    raw = mark_auxiliary_channels(raw)

    # ---------------------------------------------------------------------
    # Montage
    # ---------------------------------------------------------------------

    try:
        montage = mne.channels.make_standard_montage("biosemi64")

        raw.set_montage(
            montage,
            match_case=False,
            on_missing="ignore",
            verbose=False,
        )

    except Exception as exc:
        print("Montage warning:", exc)

    # ---------------------------------------------------------------------
    # Average reference
    # ---------------------------------------------------------------------

    raw.set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )

    # ---------------------------------------------------------------------
    # High-pass at 2 Hz
    # ---------------------------------------------------------------------

    raw.filter(
        l_freq=2.0,
        h_freq=None,
        method="fir",
        phase="zero",
        verbose=False,
    )

    # ---------------------------------------------------------------------
    # Ensure 256 Hz
    # ---------------------------------------------------------------------

    if not np.isclose(raw.info["sfreq"], 256.0):

        raw.resample(
            256,
            npad="auto",
            verbose=False,
        )

    # ---------------------------------------------------------------------
    # Average reference again, as in paper
    # ---------------------------------------------------------------------

    raw.set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )

    print("After preprocessing:", raw.info["sfreq"], "Hz")

    return raw


# =============================================================================
# EVENTS
# =============================================================================

def load_events(events_path):
    """
    Load BIDS events.tsv and validate its structure.
    """

    events = pd.read_csv(
        events_path,
        sep="\t",
    )

    required = {
        "onset",
        "trial_type",
        "value",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            f"events.tsv is missing columns: {sorted(missing)}"
        )

    events["onset"] = pd.to_numeric(
        events["onset"],
        errors="coerce",
    )

    events["value_num"] = pd.to_numeric(
        events["value"],
        errors="coerce",
    )

    events["trial_type_clean"] = (
        events["trial_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    events = events.dropna(
        subset=["onset", "value_num"]
    ).reset_index(drop=True)

    return events


def find_q1_q2_events(events):
    """
    Parse Brandmeyer experience-sampling probes.

    Every probe starts with:

        value = 128

    The subsequent response events belong to:

        response 1 = Q1 meditation depth
        response 2 = Q2 mind wandering
        response 3 = Q3 drowsiness

    Classification:

        Q1 > Q2 -> MED
        Q2 > Q1 -> MW
        Q1 = Q2 -> ignored
    """

    probe_indices = np.flatnonzero(
        events["value_num"].to_numpy() == PROBE_ONSET_VALUE
    )

    rows = []

    for probe_number, start_idx in enumerate(probe_indices):

        if probe_number + 1 < len(probe_indices):
            end_idx = probe_indices[probe_number + 1]
        else:
            end_idx = len(events)

        block = events.iloc[
            start_idx + 1:end_idx
        ].copy()

        # Normal response events only
        response_mask = (
            block["trial_type_clean"].eq("response")
            &
            block["value_num"].isin(RESPONSE_VALUES.keys())
        )

        responses = block.loc[
            response_mask
        ].copy()

        if len(responses) < 2:
            continue

        # First response = Q1
        q1_trigger = int(
            responses.iloc[0]["value_num"]
        )

        # Second response = Q2
        q2_trigger = int(
            responses.iloc[1]["value_num"]
        )

        q1 = RESPONSE_VALUES[q1_trigger]
        q2 = RESPONSE_VALUES[q2_trigger]

        # Third response = Q3 if present
        q3 = np.nan

        if len(responses) >= 3:
            q3_trigger = int(
                responses.iloc[2]["value_num"]
            )

            if q3_trigger in RESPONSE_VALUES:
                q3 = RESPONSE_VALUES[q3_trigger]

        # -------------------------------------------------------------
        # Brandmeyer classification
        # -------------------------------------------------------------

        if q1 > q2:
            condition = "MED"

        elif q2 > q1:
            condition = "MW"

        else:
            condition = "EQUAL_IGNORED"

        rows.append({
            "probe_idx": probe_number,
            "q1_onset": float(
                events.iloc[start_idx]["onset"]
            ),
            "q1_rating": q1,
            "q2_rating": q2,
            "q3_rating": q3,
            "condition": condition,
        })

    probes = pd.DataFrame(rows)

    if probes.empty:
        raise ValueError(
            "No valid Q1/Q2 probes were found."
        )

    return probes


# =============================================================================
# EPOCHING
# =============================================================================

def make_brandmeyer_epochs(raw, probes):
    """
    Extract Brandmeyer 10-second epochs.

    Each epoch runs:

        -10.05 s -> -0.05 s

    relative to Q1 onset.
    """

    # Keep only common channels
    available = [
        ch for ch in COMMON_CHANNELS_14
        if ch in raw.ch_names
    ]

    missing = [
        ch for ch in COMMON_CHANNELS_14
        if ch not in raw.ch_names
    ]

    print("Available common channels:", available)

    if missing:
        print("WARNING - missing common channels:", missing)

    if len(available) == 0:
        raise ValueError(
            "None of the requested common EEG channels are present."
        )

    raw_eeg = raw.copy().pick(
        available
    )

    sfreq = raw_eeg.info["sfreq"]

    data_rows = []
    metadata_rows = []

    for _, probe in probes.iterrows():

        condition = probe["condition"]

        # Brandmeyer discards equal ratings
        if condition == "EQUAL_IGNORED":
            continue

        q1_onset = float(
            probe["q1_onset"]
        )

        # Q1 onset is in seconds from recording start
        q1_sample = int(
            round(q1_onset * sfreq)
        )

        start_sample = int(
            round(
                q1_sample +
                EPOCH_TMIN * sfreq
            )
        )

        stop_sample = int(
            round(
                q1_sample +
                EPOCH_TMAX * sfreq
            )
        )

        if start_sample < 0:
            continue

        if stop_sample >= raw_eeg.n_times:
            continue

        epoch_data = raw_eeg.get_data(
            start=start_sample,
            stop=stop_sample,
        )

        # Expected ~10 seconds
        if epoch_data.shape[1] < int(9.9 * sfreq):
            continue

        data_rows.append(epoch_data)

        metadata_rows.append({
            "subject": SUBJECT_ID,
            "probe_idx": int(probe["probe_idx"]),
            "condition": condition,
            "q1_rating": float(probe["q1_rating"]),
            "q2_rating": float(probe["q2_rating"]),
            "q3_rating": probe["q3_rating"],
            "q1_onset": q1_onset,
        })

    if not data_rows:
        raise ValueError(
            "No valid 10-second Brandmeyer epochs were created."
        )

    data = np.stack(
        data_rows,
        axis=0,
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    # All epochs should have identical length
    n_times = data.shape[-1]

    events = np.column_stack([
        np.arange(len(data)),
        np.zeros(len(data), dtype=int),
        np.ones(len(data), dtype=int),
    ])

    epochs = mne.EpochsArray(
        data,
        raw_eeg.info.copy(),
        events=events,
        event_id={"brandmeyer": 1},
        tmin=EPOCH_TMIN,
        metadata=metadata,
        verbose=False,
    )

    return epochs


# =============================================================================
# SPECTRAL FEATURES
# =============================================================================

def extract_band_power(epochs):
    """
    Extract absolute multitaper band power.

    Output is log10(power).

    Dimensions:

        trial × channel × band
    """

    print()
    print("Extracting spectral features...")

    # MNE multitaper PSD
    spectrum = epochs.compute_psd(
        method="multitaper",
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        bandwidth=None,
        adaptive=False,
        low_bias=True,
        normalization="full",
        verbose=False,
    )

    psd = spectrum.get_data()

    freqs = spectrum.freqs

    ch_names = epochs.ch_names

    # -------------------------------------------------------------
    # Prevent zero / negative numerical values
    # -------------------------------------------------------------

    psd = np.maximum(
        psd,
        LOG_FLOOR,
    )

    feature_rows = []

    for trial_idx in range(
        psd.shape[0]
    ):

        row = {
            "subject": SUBJECT_ID,
            "trial_idx": trial_idx,
        }

        metadata = epochs.metadata.iloc[
            trial_idx
        ]

        row["probe_idx"] = int(
            metadata["probe_idx"]
        )

        row["condition"] = metadata[
            "condition"
        ]

        row["q1_rating"] = metadata[
            "q1_rating"
        ]

        row["q2_rating"] = metadata[
            "q2_rating"
        ]

        row["q3_rating"] = metadata[
            "q3_rating"
        ]

        row["q1_onset"] = metadata[
            "q1_onset"
        ]

        # ---------------------------------------------------------
        # Each electrode × band
        # ---------------------------------------------------------

        for band_name, (
            f_low,
            f_high,
        ) in CANONICAL_BANDS.items():

            freq_mask = (
                (freqs >= f_low)
                &
                (freqs < f_high)
            )

            if not np.any(freq_mask):
                continue

            # Mean PSD within frequency band
            band_power = psd[
                trial_idx,
                :,
                freq_mask,
            ].mean(axis=-1)

            log_band_power = np.log10(
                np.maximum(
                    band_power,
                    LOG_FLOOR,
                )
            )

            for ch_idx, ch_name in enumerate(
                ch_names
            ):

                row[
                    f"{ch_name}_{band_name}_logpower"
                ] = log_band_power[ch_idx]

        feature_rows.append(row)

    feature_df = pd.DataFrame(
        feature_rows
    )

    return feature_df


# =============================================================================
# METHOD 1
# =============================================================================

def method_1_subject_means(
    trial_features,
    subject_id,
):
    """
    Method 1:

    Average trial-level log-power separately for MED and MW.

    This produces one row per:

        subject × electrode × band

    with:

        MED_mean_logpower
        MW_mean_logpower
        MED_minus_MW
        percent_change
    """

    print_header(
        "METHOD 1: PER-SUBJECT CONDITION MEANS"
    )

    feature_columns = [
        c
        for c in trial_features.columns
        if c.endswith("_logpower")
    ]

    rows = []

    for feature in feature_columns:

        parts = feature.split("_")

        # Expected:
        # C3_theta_logpower
        channel = parts[0]
        band = parts[1]

        med_values = trial_features.loc[
            trial_features["condition"] == "MED",
            feature,
        ].dropna()

        mw_values = trial_features.loc[
            trial_features["condition"] == "MW",
            feature,
        ].dropna()

        if len(med_values) == 0:
            continue

        if len(mw_values) == 0:
            continue

        med_mean = med_values.mean()
        mw_mean = mw_values.mean()

        difference = (
            med_mean - mw_mean
        )

        # Percentage change in the log-power scale
        percent_change = np.nan

        if mw_mean != 0:
            percent_change = (
                difference / abs(mw_mean)
            ) * 100.0

        rows.append({
            "subject": subject_id,
            "channel": channel,
            "band": band,

            "n_MED": len(med_values),
            "n_MW": len(mw_values),

            "MED_mean_logpower": med_mean,
            "MW_mean_logpower": mw_mean,

            "MED_minus_MW": difference,

            "percent_change_logpower": percent_change,
        })

    result = pd.DataFrame(rows)

    return result


# =============================================================================
# METHOD 1 — OPTIONAL WITHIN-SUBJECT PERMUTATION
# =============================================================================

def within_subject_permutation_test(
    trial_features,
    feature,
    n_permutations=10000,
    random_state=42,
):
    """
    One-subject permutation sanity check.

    This is NOT a substitute for the group-level analysis.

    Null hypothesis:
        MED and MW labels are exchangeable.

    Statistic:
        mean(MED) - mean(MW)
    """

    rng = np.random.default_rng(
        random_state
    )

    valid = trial_features[
        ["condition", feature]
    ].dropna()

    values = valid[feature].to_numpy()

    labels = (
        valid["condition"]
        .to_numpy()
    )

    observed = (
        values[labels == "MED"].mean()
        -
        values[labels == "MW"].mean()
    )

    permuted = np.empty(
        n_permutations
    )

    n_med = np.sum(
        labels == "MED"
    )

    for i in range(
        n_permutations
    ):

        shuffled = rng.permutation(
            values
        )

        permuted[i] = (
            shuffled[:n_med].mean()
            -
            shuffled[n_med:].mean()
        )

    p_value = (
        np.sum(
            np.abs(permuted)
            >= abs(observed)
        ) + 1
    ) / (
        n_permutations + 1
    )

    return observed, p_value


# =============================================================================
# METHOD 2
# =============================================================================

def method_2_mixed_model(
    trial_features,
    output_prefix,
):
    """
    Trial-level mixed-effects analysis.

    Intended model:

        logpower ~ condition * group
                   + (1 + condition | subject)

    statsmodels syntax:

        logpower ~ condition * group

        groups = subject
        re_formula = ~condition

    With one subject, this model is not identifiable as a
    multi-subject mixed-effects model, so the function exits
    gracefully after saving the trial-level data.
    """

    print_header(
        "METHOD 2: TRIAL-LEVEL MIXED-EFFECTS MODEL"
    )

    unique_subjects = (
        trial_features["subject"]
        .nunique()
    )

    print(
        f"Subjects available: {unique_subjects}"
    )

    if unique_subjects < 2:

        print()
        print(
            "Only one subject is present."
        )
        print(
            "A subject-level random-effects model "
            "cannot be validly estimated."
        )
        print(
            "Trial-level data have been saved for "
            "the future multi-subject analysis."
        )

        return None

    if not HAS_STATSMODELS:

        print()
        print(
            "statsmodels is not installed."
        )
        print(
            "Install it with:"
        )
        print(
            "    pip install statsmodels"
        )

        return None

    # -------------------------------------------------------------
    # Group column
    # -------------------------------------------------------------
    #
    # We deliberately do not invent group membership here.
    # For the full dataset, provide a group column in the combined
    # trial-level CSV or merge the participant metadata before fitting.
    #

    if "group" not in trial_features.columns:

        print()
        print(
            "No 'group' column is available."
        )
        print(
            "The full model requires expert/non-expert "
            "group membership."
        )
        print(
            "Trial-level data have been saved."
        )

        return None

    # -------------------------------------------------------------
    # Condition coding
    # -------------------------------------------------------------

    df = trial_features.copy()

    df = df[
        df["condition"].isin(
            ["MW", "MED"]
        )
    ].copy()

    df["condition"] = pd.Categorical(
        df["condition"],
        categories=["MW", "MED"],
    )

    df["group"] = pd.Categorical(
        df["group"]
    )

    # -------------------------------------------------------------
    # Fit one model per electrode × band
    # -------------------------------------------------------------

    feature_columns = [
        c
        for c in df.columns
        if c.endswith("_logpower")
    ]

    model_rows = []

    for feature in feature_columns:

        parts = feature.split("_")

        channel = parts[0]
        band = parts[1]

        model_df = df[
            [
                "subject",
                "condition",
                "group",
                feature,
            ]
        ].dropna()

        if model_df["subject"].nunique() < 2:
            continue

        model_df = model_df.rename(
            columns={
                feature: "logpower"
            }
        )

        try:

            model = smf.mixedlm(
                "logpower ~ condition * group",
                data=model_df,
                groups=model_df["subject"],
                re_formula="~condition",
            )

            fit = model.fit(
                reml=False,
                method="lbfgs",
                maxiter=1000,
                disp=False,
            )

            for term in fit.params.index:

                model_rows.append({
                    "channel": channel,
                    "band": band,
                    "term": term,
                    "coef": fit.params[
                        term
                    ],
                    "se": fit.bse[
                        term
                    ],
                    "p_value": fit.pvalues[
                        term
                    ],
                })

        except Exception as exc:

            warnings.warn(
                f"Mixed model failed for "
                f"{channel} {band}: {exc}"
            )

    results = pd.DataFrame(
        model_rows
    )

    output_path = (
        OUTPUT_DIR
        / f"{output_prefix}_mixed_models.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved mixed-model results to:\n"
        f"{output_path}"
    )

    return results


# =============================================================================
# SUBJECT SUMMARY
# =============================================================================

def print_subject_summary(
    probes,
    epochs,
    trial_features,
):
    """
    Print a concise sanity check.
    """

    print_header(
        "BRANDMEYER SUBJECT SUMMARY"
    )

    print(
        f"Total Q1/Q2 probes: "
        f"{len(probes)}"
    )

    print(
        f"MED probes: "
        f"{np.sum(probes['condition'] == 'MED')}"
    )

    print(
        f"MW probes: "
        f"{np.sum(probes['condition'] == 'MW')}"
    )

    print(
        f"Equal/ignored probes: "
        f"{np.sum(probes['condition'] == 'EQUAL_IGNORED')}"
    )

    print()

    print(
        f"Valid EEG epochs: "
        f"{len(epochs)}"
    )

    print(
        "Trial features:",
        len(trial_features),
        "rows"
    )

    print()

    print(
        trial_features[
            "condition"
        ].value_counts()
    )


# =============================================================================
# SANITY CHECK
# =============================================================================

def print_theta_alpha_sanity_check(
    method1_results
):
    """
    Print theta/alpha results for the 14 channels.

    This is particularly useful because Brandmeyer reported
    theta and alpha condition differences.
    """

    print_header(
        "THETA / ALPHA SANITY CHECK"
    )

    subset = method1_results[
        method1_results["band"].isin(
            ["theta", "alpha"]
        )
    ].copy()

    if subset.empty:
        print(
            "No theta/alpha results available."
        )
        return

    display_columns = [
        "channel",
        "band",
        "n_MED",
        "n_MW",
        "MED_mean_logpower",
        "MW_mean_logpower",
        "MED_minus_MW",
    ]

    print(
        subset[
            display_columns
        ].to_string(index=False)
    )


# =============================================================================
# SAVE PROBE INFORMATION
# =============================================================================

def save_probe_table(
    probes,
    subject_id,
):
    """
    Save the behavioral probe classification.

    This makes it possible to audit exactly which trials were
    classified MED, MW, or ignored.
    """

    output_path = (
        OUTPUT_DIR
        / f"{subject_id}_brandmeyer_probe_classification.csv"
    )

    probes.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved probe classification to:\n"
        f"{output_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "BRANDMEYER MED vs MW ANALYSIS"
    )

    # =====================================================================
    # FIND FILES
    # =====================================================================

    bdf_path, events_path, channels_path = (
        find_brandmeyer_files(
            SUBJECT_ID
        )
    )

    print()
    print("BDF:", bdf_path)
    print("Events:", events_path)

    if channels_path is not None:
        print(
            "Channels:",
            channels_path
        )
    else:
        print(
            "Channels: not found "
            "(not required for this analysis)"
        )

    # =====================================================================
    # LOAD RAW EEG
    # =====================================================================

    raw = mne.io.read_raw_bdf(
        str(bdf_path),
        preload=True,
        verbose=False,
    )

    # =====================================================================
    # PREPROCESS
    # =====================================================================

    raw = preprocess_raw(
        raw
    )

    # =====================================================================
    # LOAD EVENTS
    # =====================================================================

    events_df = load_events(
        events_path
    )

    # =====================================================================
    # CLASSIFY PROBES
    # =====================================================================

    probes = find_q1_q2_events(
        events_df
    )

    print()
    print(
        f"Found {len(probes)} Q1/Q2 probe pairs."
    )

    # =====================================================================
    # SAVE PROBE TABLE
    # =====================================================================

    save_probe_table(
        probes,
        SUBJECT_ID
    )

    # =====================================================================
    # EPOCH
    # =====================================================================

    epochs = make_brandmeyer_epochs(
        raw,
        probes,
    )

    # =====================================================================
    # FEATURE EXTRACTION
    # =====================================================================

    trial_features = extract_band_power(
        epochs
    )

    # =====================================================================
    # SAVE TRIAL-LEVEL FEATURES
    # =====================================================================

    trial_output = (
        OUTPUT_DIR
        / f"{SUBJECT_ID}_brandmeyer_trial_level_logpower.csv"
    )

    trial_features.to_csv(
        trial_output,
        index=False,
    )

    print()
    print(
        f"Saved trial-level features to:\n"
        f"{trial_output}"
    )

    # =====================================================================
    # SUBJECT SUMMARY
    # =====================================================================

    print_subject_summary(
        probes,
        epochs,
        trial_features,
    )

    # =====================================================================
    # METHOD 1
    # =====================================================================

    subject_means = method_1_subject_means(
        trial_features,
        SUBJECT_ID,
    )

    method1_output = (
        OUTPUT_DIR
        / f"{SUBJECT_ID}_brandmeyer_MED_vs_MW_subject_means.csv"
    )

    subject_means.to_csv(
        method1_output,
        index=False,
    )

    print(
        f"\nSaved Method 1 results to:\n"
        f"{method1_output}"
    )

    # =====================================================================
    # THETA / ALPHA CHECK
    # =====================================================================

    print_theta_alpha_sanity_check(
        subject_means
    )

    # =====================================================================
    # ONE-SUBJECT PERMUTATION SANITY CHECK
    # =====================================================================

    print_header(
        "ONE-SUBJECT PERMUTATION SANITY CHECK"
    )

    # Only a sanity check — NOT group inference.
    for channel in ["F3", "F4", "C3", "C4", "P3", "P4"]:

        for band in ["theta", "alpha"]:

            feature = (
                f"{channel}_{band}_logpower"
            )

            if feature not in trial_features.columns:
                continue

            try:

                observed, p = (
                    within_subject_permutation_test(
                        trial_features,
                        feature,
                        n_permutations=10000,
                    )
                )

                print(
                    f"{feature:30s} "
                    f"MED-MW = {observed: .5f}   "
                    f"permutation p = {p:.4f}"
                )

            except Exception as exc:

                print(
                    f"{feature}: permutation failed: "
                    f"{exc}"
                )

    # =====================================================================
    # METHOD 2
    # =====================================================================

    mixed_results = method_2_mixed_model(
        trial_features,
        SUBJECT_ID,
    )

    # =====================================================================
    # FINAL MESSAGE
    # =====================================================================

    print_header(
        "ANALYSIS COMPLETE"
    )

    print(
        "Outputs:"
    )

    print(
        f"  Probe classifications:\n"
        f"    {OUTPUT_DIR / f'{SUBJECT_ID}_brandmeyer_probe_classification.csv'}"
    )

    print(
        f"  Trial-level log-power:\n"
        f"    {trial_output}"
    )

    print(
        f"  Method 1 subject means:\n"
        f"    {method1_output}"
    )

    print()

    print(
        "For this one-subject run, Method 1 is descriptive and "
        "the permutation test is only a within-subject sanity check."
    )

    print(
        "The mixed-effects model is intentionally not fitted with "
        "one subject. Run it after combining the trial-level files "
        "from all Brandmeyer subjects."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
