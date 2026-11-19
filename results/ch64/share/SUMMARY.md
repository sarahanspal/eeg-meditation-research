# 64-channel re-analysis: summary

## What was run
64-ch features were extracted for both datasets and run through four scripts.
- `population_64`: LOSO MED vs MW, label control, learning curve, theta vs accuracy.
- `classifier_comparison_64`: 8 models plus a Friedman test.
- `theta_shift_64`: sign consistency, CTR vs meditators, recording order.
- `style_classification_64`: style decoding, with the dummy reseeded per subject.

`paper_numbers_ch64.csv` puts 14-ch and 64-ch side by side. Figures are the existing PNGs resampled to 150 dpi.

## N per analysis
| Analysis | Braboszcz | Brandmeyer |
|---|---|---|
| LOSO / classifiers / learning curve | 54 | 13 (14-ch: 14) |
| Theta sign consistency | 53 | 11 |
| CTR vs meditators (theta) | 53 vs 29 | – |
| Style pairwise / 3-class | 36 / 54 | 68 incl. Brandmeyer |
| Meditators vs CTR (style) | 54 vs 30 | – |

## What changed from 14-ch
- **LOSO MED vs MW drops.** Braboszcz pooled 0.592 → 0.543, within 0.570 → 0.521. Nothing survives FDR. The label control is unchanged (0.512 vs 0.511).
- **The learning curve is flat** (~0.51–0.52). At 14-ch it rose to 0.567.
- **Theta sign consistency holds.** F3, F4, P4, T8 and F7 are 70–78% positive, FDR < 0.02.
- **Gaussian NB is best** for MED vs MW (0.566, FDR 0.036).
- **Style logistic is lower:** HYT vs ISY 0.648 → 0.579.

## Failed or missing
- **Aperiodic:** `No space left on device` at sub-008. Two subjects are cached and there are no summaries.
- **Calibration:** no main-run output; the run was likely killed. Its figure exists only in `smoke/` and is not included.
- **Extraction failures:** Braboszcz sub-027, 056, 058 and 059; Brandmeyer sub-003.

## Looks wrong
- **Brandmeyer N is inconsistent:** 13 / 11 / 14 across scripts. Style uses 14 even though sub-003 failed extraction.
- **Braboszcz theta N is 53, not 54.** Recording order covers only 52.
- **Brandmeyer within-LOSO is 0.461**, below chance.
- **The style dummy is below chance:** ~0.486, vote 0.31.
- **The trait sign may have flipped** (14-ch d −0.41 vs 64-ch difference +0.21), but the groups and metric differ.
- **The dataset-identity probe** (0.638 vs 0.853) uses different epochs.
- **The 64-ch FDR family is larger**, so p_fdr is not directly comparable.
