# 64-channel final results

Rule: a subject is evaluated only with >= 5 epochs in every condition the analysis uses (excluded subjects stayed in training, nothing refit). Dummy = expected accuracy, p = 1.

## Final N
| Analysis | Braboszcz | Brandmeyer |
|---|---|---|
| MED vs MW (classifiers, population) | 53 (was 54) | 11 (was 13) |
| IMW2 vs IMW1 control | 52 (was 54) | - |
| Theta sign consistency | 53 | 11 |
| Style pairwise / three-class | 36 / 54 | 68 incl. 14 Brandmeyer |
| Meditators vs CTR, MED | 54 vs 30 | - |
| Meditators vs CTR, IMW2 | 53 vs 29 | - |
| Calibration | 52 | 7 |
| Aperiodic | 52-53 | - |

## Calibration exclusion rule
At least 6 units per class and k + 2 units per class; amounts under 2 units skipped (Brandmeyer 8 s). Null-adjusted gain: subjects in both runs (n = 44-51). Full update: +0.028 (8 s) to +0.096 (128 s) per class, FDR-significant at all amounts. Threshold only: not significant.

## Changes above 0.01
- Brandmeyer MED vs MW within logistic 0.461 to 0.485 (kNN 0.498 to 0.538, MLP 0.466 to 0.499); learning curve +0.02 to +0.03.
- IMW2 vs IMW1 logistic 0.511 to 0.521.
- Style dummy 0.486 to 0.500; IMW2 MLP and SVM-RBF +0.013 to +0.014.
- Braboszcz MED vs MW accuracies: all within 0.01. Real minus control: no model significant after FDR.

Aperiodic was re-run after the Drive failure and is complete.
