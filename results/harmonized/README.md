# Harmonized extraction and analysis

The outputs in this directory are generated separately from the original feature CSVs.

## Extraction

Run `.venv/bin/python reextract_harmonized.py` from the repository root. By default it uses the same Braboszcz and Brandmeyer subject cohorts as the existing feature tables, plus the SKY cohort. `--all-available` expands the Braboszcz and Brandmeyer cohorts; `--limit N` runs a small subject subset.

The common pipeline uses the 14 shared EEG channels, average reference over those channels, 50 and 60 Hz notches, a 1 Hz sixth-order zero-phase Butterworth high-pass, 256 Hz sampling, shared subject-level extended Infomax ICA, non-overlapping four-second epochs, and the same canonical feature definitions. Braboszcz task think1 is retained as the file identity control.

SKY timing values in `subjects_epochs.mat` are seconds and are converted to samples. The Braboszcz bad-segment detector's fixed dB threshold flags at least 95% of SKY's EDF recordings; SKY epoch annotation rejection is therefore skipped and documented in the output metadata. This means SKY shares the spectral preprocessing and feature schema, while its continuous bad-segment rejection remains a limitation.

## Analysis scripts

- `combined_analysis.py`: within-dataset LOSO, pooled LOSO, cross-dataset transfer, dataset identity probe, and buffered Braboszcz controls.
- `combined_supplementary.py`: subject bootstrap intervals, sign-flip tests, feature sign consistency, and learning curves.
- `calibration_decomposition.py`: population baseline, threshold-only update, and full model update, with think1/think2 control.
- `trait_experience_analysis.py`: subject-level experience contrasts stratified by dataset and state, with age and years-of-practice analyses where metadata exist.
- `feature_specification_curve.py`: feature-family and logistic regularization sensitivity on fixed subject-held-out splits.
- `write_harmonized_manifest.py`: package versions, parameters, script hashes, and output hashes.
- `plot_research_summary.py`: summary figure.

The Braboszcz MED/IMW and Brandmeyer MED/MW analyses share the meditating-versus-mind-wandering label construct. SKY pre/post rest is kept as a separate task. Trait analyses are stratified by dataset and state because these source datasets do not share one identical state definition.

## Completed run (2026-11-11)

The harmonized outputs contain 24,036 Braboszcz rows from 56 subjects (MED, IMW2, and the think1/IMW1 file control), 618 Brandmeyer rows from 15 subjects (MED/MW), and 2,930 SKY rows from 37 subjects (pre/post). All three tables have 196 common feature columns and no non-finite feature values. Braboszcz and Brandmeyer extraction had zero failed subjects. SKY sky-12 initially contained duplicate epoch starts; duplicate starts are now deduplicated and sky-12 was re-extracted successfully. The final SKY failure table is empty.

Before applying the style-matched classifier rule below, LOSO scores used all 56 Braboszcz subjects. Those preliminary classifier summaries are superseded by the matched-cohort figures below. The harmonized feature tables themselves retain all 56 meditation-style subjects for descriptive and trait analyses.

On the matched classifier cohort, Braboszcz LOSO scores are 0.570 within, 0.592 pooled, and 0.546 cross-dataset. Brandmeyer is 0.526 within, 0.541 pooled, and 0.544 cross-dataset. Dataset identity is 0.638. Blocked controls remain elevated: 0.671 for think2 versus think1 and 0.607 for the first versus second half of think2, compared with 0.878 for MED versus IMW2. Treat state-decoding scores as confound-sensitive.

The corrected matched-cohort Braboszcz learning curve uses same-dataset, style-balanced training sets: balanced accuracy was 0.505 with three training subjects, peaked at 0.567 with 24, and was 0.561 with 36; the intervals overlap broadly. Brandmeyer has only 14 classifier subjects, so its curve is estimated at 3, 6, and 12 training subjects and remains near chance. In matched-cohort calibration (54 Braboszcz subjects), mean MED/IMW2 accuracy was 0.572 before calibration, 0.582 after a threshold-only update, and 0.634 after full model updating. The think1/think2 control also gained 0.036 with full updating; the mean null-adjusted gain was 0.027 (median 0.036; 38/54 subjects positive), so this is modest and exploratory. In the matched feature specification curve, mean Braboszcz accuracy averaged over feature families and regularization settings was 0.590 with all features, 0.556 without gamma, and 0.547 without gamma and ratios. Brandmeyer was near chance across these specifications (0.545 with all features; 0.505 without gamma and ratios). Intervals overlap broadly. This curve varies feature families and logistic regularization; it does not vary epoch length, notch/reference choice, or ICA inclusion. The Riemannian and EEGNet baselines were not run because their dependencies are unavailable in the current environment.

The trait cohort is unbalanced in Braboszcz (55 experienced and one novice at the 3-year threshold), so do not interpret its experience contrast as a reliable group comparison. During classifier fitting, scikit-learn emitted repeated matrix-multiplication overflow/invalid warnings. This was investigated (see Numerical warnings below): it is a numpy 2.0.2 + Apple Accelerate artifact, not a data problem, and results are unaffected. MNE also emitted ICA/interpolation matrix warnings for some recordings. The SKY bad-segment detector exception is described above and in `manifest.json`.

## Braboszcz matched-style classifier cohort

For classifier analyses, the eligible harmonized Braboszcz cohort has 18 HYT, 18 VIP, and 20 ISY subjects. The classifier cohort uses 18 per style: all HYT and VIP subjects and a reproducible seeded subset of 18 ISY subjects (`seed=20261111`). CTR and TM are excluded. The style selection and reasons are recorded in `braboszcz_classifier_cohort.csv`; the full 56-subject meditation-only table remains available for trait/descriptive work.

On the matched cohort, combined Braboszcz LOSO balanced accuracy is 0.570 within-dataset, 0.592 pooled, and 0.546 cross-dataset; dataset identity probe is 0.638. By style, held-subject MED/IMW2 accuracy is 0.603 for HYT, 0.565 for VIP, and 0.483 for ISY. The MED-only pairwise style macro balanced accuracy is 0.556 (HYT/VIP), 0.648 (HYT/ISY), and 0.669 (VIP/ISY); three-class macro balanced accuracy is 0.461 (chance 0.333). These subject-held-out models classify meditation style from MED epochs and should be treated as exploratory. Within-subject blocked MED/IMW2 remains high across styles (0.866 HYT, 0.870 ISY, 0.899 VIP), alongside elevated temporal/file controls.

The earlier `csp_matched_loso_subject.csv` used the legacy processed arrays and is superseded by the harmonized CSP run described below; do not quote it. PyRiemann and PyTorch/EEGNet are not installed. The feature specification curve covers feature families and regularization only; epoch-length, notch, reference, and ICA ablations still require separate full re-extractions and have not been run.

The compact output summary is `research_summary.png`; data hashes, code hashes, package versions, and preprocessing details are recorded in `manifest.json`.


## CSP on harmonized epochs (supersedes `csp_matched_loso_subject.csv`)

`csp_harmonized_epochs.py` re-creates the harmonized MED and IMW2 epochs (same preprocessing and epoching as the feature tables; epoch counts verified equal to the feature table) and `csp_harmonized_loso.py` runs CSP (6 components, Ledoit-Wolf, log-variance) fit inside each training fold, with LDA and logistic regression, leave-one-subject-out over the matched 54-subject cohort. Output: `csp_harmonized_loso_subject.csv`. Balanced accuracy per subject, then averaged: **0.550** (N=54, bootstrap 95% CI 0.516-0.584, sign-flip p=0.006) for both LDA and logistic regression. Per-subject range 0.19-0.88.

Why the old run showed 5-26% subjects (sub-018, sub-010, sub-071, sub-076, sub-022): the held-out test sets were balanced (e.g. sub-018 303/312) and the metric was already balanced accuracy, and labels were consistent (task-name based), so it was not a metric or polarity bug. The legacy arrays used a different contrast (med1breath+med2 vs think1+think2 files) and preprocessing. On the harmonized epochs those five subjects score 0.50, 0.48, 0.70, 0.50 and 0.50. The legacy `results/csp_losocv_persubject.csv` was never overwritten (git-unmodified).

Caveat: in 26 of 54 folds the classifier predicted one class for at least 95% of the held-out epochs. Unnormalized CSP log-variance carries a per-subject amplitude offset, so the pooled threshold often misplaces a held-out subject. The 0.550 should be read as a lower bound for CSP; a per-subject normalized variant has not been run.

## Cohort Ns by analysis

`cohort_by_analysis.csv` (and `cohort_by_analysis_exclusions.csv`) list N, subject-list hash and exclusion reasons per analysis, derived from the inclusion rules and asserted equal to the subjects in each output file. All differences are rule-based, not bugs: classifier LOSO is 54 Braboszcz + 14 Brandmeyer (sub-017 has MED epochs only); feature sign consistency is 54 + 12 = 66 (needs >=5 epochs per class; three Brandmeyer subjects fall short); blocked controls and the paired control are 53 for MED vs IMW2 (sub-097 has 7 MED epochs; blocked CV needs >=20 per class), while think2 vs think1 is 54. Note sub-097 remains in LOSO and sign consistency with only 7 MED epochs.

## Braboszcz recording order

`braboszcz_run_order.csv`, from BDF header acquisition timestamps (no scans.tsv exists). In all 54 matched subjects think1 and think2 are consecutive (about 32 s between end of think1 and start of think2; think1 always first) and the two meditation runs are consecutive. Order is counterbalanced: 27 subjects think1>think2>med1breath>med2, 27 med1breath>med2>think1>think2. med2 and think2 start about 87 min apart (range 69-110 min), med2 first in half the subjects. This contradicts the dataset README's description of thinking blocks interspaced between meditation blocks. Consequence: the think1 vs think2 control (adjacent runs) is not conservative for MED vs IMW2 (runs about 87 min apart); it is a lower bound on drift/file effects. The within-think2 first-vs-second-half control is the cleaner drift floor. The counterbalanced order allows a direct test of a time-order effect by splitting scores by order group; this has not been run.

## Numerical warnings

The sklearn matmul overflow/invalid-value warnings reproduce on a single fold, but features are finite and bounded after the log transform and after scaling (range -8.2 to 13; no |x|>1e3; no constant columns; `numerical_warning_feature_profile.csv`). The flagged matmul results are finite and match einsum to 2.7e-15. Under numpy 1.26.4 (OpenBLAS) the same 9 fits give 0 warnings and identical balanced accuracy to 6 decimals (`numerical_warning_blas_comparison.log`). Cause: spurious floating-point flags from numpy 2.0.2 with Apple Accelerate on macOS 15. No feature values changed, so no combined LOSO or control reruns were needed. The environment is unchanged; pinning `numpy<2` would remove the warnings.

## Additions (Parts B and C): confound and control analyses

All numbers below are in `results/paper_numbers.csv` (claim ids, N, CI, p, FDR p, source file, script); caveats are in `results/paper_numbers_notes.csv`. Methods facts are in `results/methods_facts.md`. Figures are in `results/figures/` (PDF and PNG).

**Order and independent check (A4 follow-up).** The header-derived run order matches `participants.tsv` `first_session` in 92 of 92 subjects that have it (53 of 54 in the matched cohort).

**Pipeline inconsistency found.** In `combined_analysis.main()` (and `calibration_decomposition.main()`), the think1/think2 and think2-halves controls are fit on the raw feature table, whereas MED vs IMW2 uses log-transformed features. With the log transform on both sides (`preproc_sensitivity.py`, main row) the file control is 0.675 instead of 0.671; the real-minus-control gap is 0.202. The existing control numbers elsewhere in this README remain as originally computed.

**B1 trait comparison (`trait_ctr_vs_experienced.csv`).** Extracted 31 CTR subjects with the same harmonized pipeline (`braboszcz_ctr_features.csv`, no failures; their "MED" block is breath counting). Prespecified outcome: mean log relative theta over F3/F4/F7/F8. Experienced (matched 54) vs CTR (31): theta during IMW2 d = 0.29 (95% CI -0.12 to 0.73; Mann-Whitney p = 0.25); during MED d = 0.07 (p = 0.38); the within-subject MED-minus-IMW2 effect was larger in CTR (0.61) than in experienced (0.34), d = -0.41 (CI -0.87 to 0.04; Mann-Whitney p = 0.25; permutation p = 0.07; age-adjusted p = 0.066; mean age 46.8 vs 39.8). No contrast is significant after FDR. A secondary all-14-channel version gave d = -0.54 (permutation p = 0.018, Mann-Whitney p = 0.066), not significant after FDR. So the theta increase is not specific to experienced meditators. Brandmeyer: 10 experts and 5 novices among the 15 subjects; only the MED-state comparison could be run (9 vs 5; expert theta lower, d = -0.98, p = 0.15); the MW state has only 4 novices, so it was not run.

**B2 dose-response (`dose_response_years.csv`, N = 53; one subject lacks years of practice).** Frontal theta effect vs years of practice: Spearman rho = 0.06, p = 0.65 (partial given age rho = -0.02). LOSO within-dataset accuracy vs years: rho = 0.32, p = 0.019 (partial given age 0.33); pooled 0.26, p = 0.061. Four tests, no multiplicity correction; exploratory.

**B3 Riemannian baseline (`riemann_loso_subject.csv`).** Tangent space + logistic regression on 14-channel Ledoit-Wolf covariances of the harmonized epochs, same 54 subjects and LOSO folds. MED vs IMW2: 0.511 (CI 0.490-0.535) vs the feature model 0.570; paired difference -0.058 (CI -0.092 to -0.023, p = 0.002). Think2 vs think1 under LOSO is at chance for both models (feature 0.512, Riemannian 0.507), so the subject-held-out feature-model score exceeds its file-control by 0.057 (CI 0.016-0.099, p = 0.011). EEGNet was not run.

**B4 preprocessing sensitivity (`preproc_sensitivity.csv`, Braboszcz matched cohort).** Real-minus-control blocked gap: main 0.202, 2 s epochs 0.187, 8 s 0.216, ICA off 0.177, 64-channel average reference 0.273; every interval excludes zero. LOSO: 0.570, 0.569, 0.566, 0.510 (ICA off: chance), 0.613 (64-channel reference). Frontal theta effect positive in 76%, 74%, 79%, 65%, 76% of subjects respectively. The 64-channel variant flagged and interpolated more channels (8 in the smoke subject) and raised blocked accuracy to 0.951, so decoding scores are reference-sensitive. The main-pipeline reproduction by `preproc_variants.py` matched the cached epochs exactly for all 108 subject-condition pairs.

**B5 aperiodic control (`aperiodic_sign_consistency.csv`, 61,496 fits, mean R2 0.943).** Frontal aperiodic exponent and offset do not differ between MED and IMW2 (57% of subjects positive each, FDR p = 0.41). Periodic theta above the aperiodic fit is positive in 80% of subjects (FDR p = 0.0004); theta peak height (NaN when no peak, present in about 34-39% of epochs) 67% (FDR p = 0.033); log relative theta after removing the epoch-level exponent 72% (FDR p = 0.005). The theta increase is therefore not a 1/f slope shift. fooof 1.1.1 was used (specparam is only available as a release candidate for Python 3.9).

**B6 Brandmeyer.** All 326 probes (292 with two epochs, 34 with one) stay within one fold in every CV used (both CVs group by subject). Per-subject mean ratings (Med, MW, Sleep): experts 1.80, 1.29, 0.85; novices 1.34, 1.38, 0.51 (Table 1 values recorded in the repo: 1.85, 1.14, 0.69 and 1.39, 1.59, 0.73; not independently verified). LOSO is 0.526-0.544 (N = 14; p = 0.13-0.22; FDR 0.20-0.22): a null result, reported without tuning.

**Methods-sheet findings.** Bad channels and ICA are shared across a subject's runs, so they cannot differ between conditions; epochs retained after bad-segment rejection do not differ for MED vs IMW2 (0.902 vs 0.894, p = 0.34, N = 87) and show a small unadjusted difference for think1 vs think2 (0.922 vs 0.894, p = 0.07). SKY subjects had a median of 10 ICs removed (range 6-12) out of at most about 13 components.

**Calibration curve (`calibration_curve_summary.csv`, Figure 3).** Gain in balanced accuracy (full update minus population model) at about 23, 58, 114, 228 and 342 s of calibration data, 54 subjects, both tasks log-transformed. MED vs IMW2: 0.030, 0.063, 0.099, 0.145, 0.174. Think2 vs think1 control: 0.022, 0.039, 0.060, 0.083, 0.092. The control gains too, so most of the calibration gain is not specific to meditation. The null-adjusted gain is 0.008 (CI -0.006 to 0.021, p = 0.24) at 23 s, 0.024 at 58 s (CI 0.004-0.044, p = 0.017; the original single-amount run gave 0.027), and rises to 0.082 at 342 s (CI 0.038-0.126, p < 0.001).
