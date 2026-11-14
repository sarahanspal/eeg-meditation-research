"""Run the think1-vs-think2 file-control and calibration-weight check.

Examples:
    python run_calibration_controls.py --control
    python run_calibration_controls.py --weights
    python run_calibration_controls.py --control --weights

This script performs extraction/model fitting only when invoked directly.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import run_batch_extraction_braboszcz as rb
from calibration import CFG, load_features, run_calibration, summarize


FEATURES_CSV = "braboszcz_all_subjects_features.csv"
INTRA_CSV = "braboszcz_intra_subject_accuracies.csv"


def eligible_subjects() -> tuple[list[str], pd.DataFrame]:
    source = pd.read_csv(FEATURES_CSV)
    classified = set(pd.read_csv(INTRA_CSV).subject.unique())
    subjects = sorted(classified & set(source.subject.unique()))
    if not subjects:
        raise ValueError("No subjects are shared by the feature table and intra-subject results.")
    return subjects, source


def run_think_file_control() -> pd.DataFrame:
    subjects, source = eligible_subjects()
    style_by_subject = source.groupby("subject")["style"].first()

    # batch_extract reads RUNS when called; its module has no import-time run.
    old_runs = rb.RUNS
    try:
        rb.RUNS = {"IMW1": "task-think1"}
        think1, failures = rb.batch_extract(subject_ids=subjects, n_jobs=rb.N_JOBS)
    finally:
        rb.RUNS = old_runs

    if failures:
        pd.DataFrame(failures, columns=["subject", "condition", "reason"]).to_csv(
            "braboszcz_control_think1_failures.csv", index=False
        )
    think1.insert(1, "style", think1.subject.map(style_by_subject))

    imw2 = source[source.condition == "IMW"].copy().assign(condition="IMW2")
    control = pd.concat([think1, imw2], ignore_index=True, sort=False)
    control_path = "braboszcz_control_think1_vs_think2.csv"
    control.to_csv(control_path, index=False)

    cfg = dict(CFG)
    cfg.update(
        FEATURES_CSV=control_path,
        OUT_CSV="calibration_control_think.csv",
        POS_LABEL="IMW2",
        NEG_LABEL="IMW1",
        CLASSIFIERS=["SVM-linear"],
        FRACTIONS=[0.0, 0.05, 0.20],
        N_REPEATS=3,
    )
    df, features = load_features(cfg)
    df = df[df.subject.isin(subjects)].reset_index(drop=True)
    result = run_calibration(df, features, cfg)
    summary, _ = summarize(result)
    summary.to_csv("calibration_control_think_summary.csv", index=False)
    return summary


def run_weight_sensitivity(n_subjects: int = 8) -> pd.DataFrame:
    subjects, _ = eligible_subjects()
    base_cfg = dict(CFG)
    df, features = load_features(base_cfg)
    available = sorted(set(subjects) & set(df.subject.unique()))
    if not available:
        raise ValueError("No evaluated subjects remain after the calibration minimum-unit filter.")
    rng = np.random.default_rng(CFG["SEED"])
    chosen = sorted(rng.choice(available, size=min(n_subjects, len(available)), replace=False).tolist())
    df = df[df.subject.isin(chosen)].reset_index(drop=True)

    results = []
    for weight in (20, 100, 500):
        cfg = dict(CFG)
        cfg.update(
            OUT_CSV=f"calibration_weight_{weight}.csv",
            FRACTIONS=[0.05],
            CAL_WEIGHT=weight,
        )
        result = run_calibration(df, features, cfg, subjects=chosen)
        result["cal_weight"] = weight
        results.append(result)

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv("calibration_weight_sensitivity.csv", index=False)
    # summarize each weight separately so the weight remains part of the grouping.
    summary = pd.concat(
        [summarize(group)[0].assign(cal_weight=weight)
         for weight, group in combined.groupby("cal_weight")],
        ignore_index=True,
    )
    summary.to_csv("calibration_weight_sensitivity_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true", help="extract and compare think1 vs think2")
    parser.add_argument("--weights", action="store_true", help="compare calibration weights on 8 subjects")
    args = parser.parse_args()
    if not args.control and not args.weights:
        parser.error("choose --control, --weights, or both")
    if args.control:
        run_think_file_control()
    if args.weights:
        run_weight_sensitivity()


if __name__ == "__main__":
    main()
