"""
describe_features.py
=====================
Prints the full feature schema that braboszcz_feature_extraction.py produces:
every column name, which group it belongs to, what it measures, and a total
dimension count - without needing to load any EEG data.

USAGE
-----
    python describe_features.py                                 # just print the schema
    python describe_features.py sub-006_braboszcz_test_features.csv   # also cross-check
                                                                  # an actual output file
                                                                  # against the schema
                                                                  # (catches things like a
                                                                  # missing asymmetry pair
                                                                  # if a channel got dropped)

Requires braboszcz_feature_extraction.py to be importable (same folder or on PYTHONPATH) -
only its config constants are used, not its MNE-dependent processing functions.
"""

import sys
import pandas as pd

from braboszcz_feature_extraction import (
    CANONICAL_BANDS, COMMON_CHANNELS_14, ASYMMETRY_PAIRS, RATIO_DEFS, IAF_SEARCH_BAND,
)


def describe_feature_schema(
    channels=COMMON_CHANNELS_14,
    bands=CANONICAL_BANDS,
    pairs=ASYMMETRY_PAIRS,
    ratios=RATIO_DEFS,
) -> pd.DataFrame:
    """Build a DataFrame with one row per feature column: name, group, meaning."""
    rows = []

    # --- per-channel relative band power ---
    for ch in channels:
        for band, (lo, hi) in bands.items():
            rows.append({
                "feature": f"{ch}_{band}",
                "group": "band_power",
                "meaning": (
                    f"Relative power in the {band} band ({lo}-{hi} Hz) at channel {ch}: "
                    f"integrated PSD in that range divided by total 1-70 Hz power in the "
                    f"same epoch. Unitless (0-1ish). Relative, not absolute uV^2, so it "
                    f"isn't confounded by amplifier gain differences across datasets."
                ),
            })

    # --- per-channel individual alpha frequency ---
    for ch in channels:
        rows.append({
            "feature": f"{ch}_IAF",
            "group": "individual_alpha_frequency",
            "meaning": (
                f"Frequency (Hz, within {IAF_SEARCH_BAND[0]}-{IAF_SEARCH_BAND[1]} Hz) of "
                f"the PSD peak at channel {ch} in this epoch - the dominant alpha rhythm's "
                f"location, independent of its amplitude."
            ),
        })

    # --- homologous-pair asymmetry, per band ---
    for left, right in pairs:
        for band in bands:
            rows.append({
                "feature": f"asym_{left}-{right}_{band}",
                "group": "asymmetry",
                "meaning": (
                    f"ln({right}_{band}) - ln({left}_{band}). Positive = right-hemisphere "
                    f"dominance in that band at this pair; 0 = symmetric; negative = "
                    f"left-hemisphere dominance."
                ),
            })

    # --- per-channel cross-band ratios ---
    for ch in channels:
        for ratio_name, (num_bands, den_bands) in ratios.items():
            rows.append({
                "feature": f"{ch}_ratio_{ratio_name}",
                "group": "cross_band_ratio",
                "meaning": (
                    f"[{' + '.join(num_bands)}] power / [{' + '.join(den_bands)}] power "
                    f"at channel {ch}, same-epoch relative-power inputs."
                ),
            })

    return pd.DataFrame(rows)


def main():
    schema_df = describe_feature_schema()
    n_by_group = schema_df.groupby("group").size().sort_values(ascending=False)

    print(f"Channels ({len(COMMON_CHANNELS_14)}): {COMMON_CHANNELS_14}")
    print(f"Bands ({len(CANONICAL_BANDS)}): {CANONICAL_BANDS}")
    print(f"Asymmetry pairs ({len(ASYMMETRY_PAIRS)}): {ASYMMETRY_PAIRS}")
    print(f"Ratio definitions ({len(RATIO_DEFS)}): {list(RATIO_DEFS.keys())}")
    print(f"\nTOTAL FEATURE DIMENSIONS: {len(schema_df)}")
    print("\nBreakdown by group:")
    print(n_by_group.to_string())

    print("\nFull feature list:")
    for _, row in schema_df.sort_values(["group", "feature"]).iterrows():
        print(f"  [{row['group']:>26}] {row['feature']:<32} {row['meaning']}")

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        actual_df = pd.read_csv(csv_path)
        meta_cols = {"subject", "condition", "epoch_idx"}
        actual_features = set(actual_df.columns) - meta_cols
        expected_features = set(schema_df["feature"])

        missing = sorted(expected_features - actual_features)
        extra = sorted(actual_features - expected_features)

        print(f"\n--- Cross-check against {csv_path} ---")
        print(f"  {len(actual_features)} feature columns in file "
              f"vs {len(expected_features)} expected from current config.")
        if missing:
            print(f"  MISSING from file (expected but absent - often means a channel "
                  f"was dropped as bad and never made it back into keep_channels): "
                  f"{missing}")
        if extra:
            print(f"  EXTRA in file (not in current schema - config may have changed "
                  f"since this file was generated): {extra}")
        if not missing and not extra:
            print("  Schema matches exactly.")


if __name__ == "__main__":
    main()