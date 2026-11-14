# EEG Meditation Research

Decoding meditation vs. mind-wandering from EEG across three open datasets: **Braboszcz** (Himalayan Yoga, Vipassana, Isha Shoonya meditators and controls), **Brandmeyer** (MED vs. MW probes) and **SKY** (pre/post rest). All three use a shared 14-channel preprocessing and feature pipeline.

The final analysis is the **harmonized 14-channel pipeline**. Its full write-up (methods, cohorts, numbers and limitations) is in [`results/harmonized/README.md`](results/harmonized/README.md).

## Repository layout

```
.
├── README.md                  ← this file
├── requirements.txt           ← pinned Python dependencies
├── config.example.py          ← copy to config.py and set DATA_ROOT
├── *.py                       ← analysis scripts (grouped below; run from repo root)
├── *.csv / *.png (root)       ← early-phase (pre-harmonization) results
├── brandmeyer_outputs/        ← early Brandmeyer example outputs
└── results/
    ├── harmonized/            ← FINAL results: tables, logs, manifest.json, README
    ├── figures/               ← paper figures (fig1–fig3, PDF + PNG)
    ├── paper_numbers.csv      ← every number cited in the paper, with its source file
    └── methods_facts.md       ← methods facts pulled from code + tables
```

Scripts stay flat in the repo root because they import each other (`loaders`, `preprocessing`, `harness`, …) and use paths relative to the root.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.py config.py   # then edit DATA_ROOT to point at your raw data
```

Raw EEG is **not** in this repo. Download the source datasets from OpenNeuro and point `DATA_ROOT` in `config.py` at them.

## Scripts by stage

### 1. Final harmonized pipeline (14 channels)

| Stage | Script |
|---|---|
| Re-extraction (Braboszcz, Brandmeyer, SKY) | `reextract_harmonized.py`, `extract_braboszcz_ctr.py` |
| Cohorts | `cohort_reconciliation.py`, `braboszcz_classifier_cohort.py`, `cohort_by_analysis.py`, `braboszcz_run_order.py` |
| Main classification | `combined_analysis.py`, `combined_supplementary.py` |
| Meditation-style models | `braboszcz_style_classifiers.py`, `braboszcz_style_statistics.py` |
| Calibration | `calibration_decomposition.py`, `calibration_curve.py` |
| Baselines | `csp_harmonized_epochs.py`, `csp_harmonized_loso.py`, `run_csp_after_extraction.sh`, `riemann_loso.py` |
| Robustness | `feature_specification_curve.py`, `preproc_variants.py`, `preproc_sensitivity.py`, `aperiodic_theta.py`, `brandmeyer_sanity.py` |
| Trait / experience | `trait_experience_analysis.py`, `trait_ctr_vs_experienced.py`, `dose_response_years.py` |
| Numerical warnings check | `numerical_warning_diagnosis_probe.py`, `numerical_warning_blas_comparison.py` |
| Paper outputs | `paper_numbers.py`, `methods_facts.py`, `make_figures.py`, `plot_research_summary.py`, `write_harmonized_manifest.py` |

### 2. Shared infrastructure

`loaders.py`, `preprocessing.py`, `experience_mapping.py`, `harness.py`, `csp_harness.py`, `permutations.py`, `run_step0.py`

### 3. Early-phase exploration (pre-harmonization, superseded)

These are kept for the record. Their outputs are the CSV/PNG files in the repo root.

- **Braboszcz:** `braboszcz_feature_extraction.py`, `run_batch_extraction_braboszcz*.py`, `braboszcz_classification*.py`, `run_3c_style_pairwise.py`, `diagnose_intra_subject_leakage.py`, `run_braboszcz_fast.py`
- **Brandmeyer / SKY:** `brandmeyer-med-mw.py`, `bradmeyer-features.py`, `decode_brandmeyer_probes.py`, `sky-inspect.py`, `run_sky+b.py`
- **Cross-dataset / misc:** `run_losocv_all.py`, `run_losocv_by_tradition.py`, `calibration.py`, `run_calibration_controls.py`, `explainability.py`, `per_feature.py`, `audit_features.py`, `describe_features.py`, `dummy_features.py`
- **Plots:** `plot.py`, `plot_panelA.py`, `plot_tsne.py`

## Not tracked (regenerable or too large)

Raw EEG, epoch caches (`*.npz`), preprocessing-variant caches, the full per-epoch feature tables (`*_all_subjects_features.csv`, 10–85 MB each) and virtual environments. See `.gitignore`. Rerun the extraction scripts to regenerate them.
