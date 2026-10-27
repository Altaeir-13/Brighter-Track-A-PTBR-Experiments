# BRIGHTER PT-BR — T2/T3 ensemble feasibility

## 1. Motivation

Test whether the observed T2/T3 error complementarity can be converted into cross-fitted performance using only existing probabilities.

## 2. Leakage-safe protocol

For each frozen outer fold, T2 and T3 inner-OOF probabilities were joined strictly by ID. Every probability came from an inner model that excluded that example from training. EN1 thresholds and EN2 alpha plus thresholds were selected only on outer-train inner-OOF predictions. Outer-validation labels were never used for selection. No model was trained or loaded; the test remained **SEALED — NOT LOADED OR EVALUATED**.

EN2 searched alpha 0.00–1.00 in steps of 0.05. Thresholds were optimized per label on 0.01–0.99. Alpha ties prefer the value closest to 0.50 and then the larger T2 weight; threshold ties prefer the largest threshold.

## 3. Systems

T2 is BERTweetBR + ASL with nested thresholds. T3 is BERTimbau Large + BCE with nested thresholds. EN1 fixes equal probability weights; EN2 tunes one global T2 weight per outer fold.

## 4. Equal-weight ensemble

EN1 reaches Macro-F1 0.5548, a delta of -0.0037 versus T2. Equal averaging alone does not improve the primary system.

## 5. Tuned-weight ensemble

EN2 reaches Macro-F1 0.5691, delta +0.0106. The mean T2 weight is 0.870 ± 0.076 (range 0.75–0.95), leaving mean T3 weight 0.130.

## 6. Main results

| System | Macro-F1 | Micro-F1 | Macro AP | Neutral FP | Cardinality |
|---|---:|---:|---:|---:|---:|
| T2 | 0.5585 | 0.6750 | 0.6018 | 0.4425 | 1.1014 |
| T3 | 0.5023 | 0.6248 | 0.5360 | 0.4292 | 1.0746 |
| EN1 | 0.5548 | 0.6657 | 0.6003 | 0.4971 | 1.2131 |
| EN2 | 0.5691 | 0.6846 | 0.6002 | 0.4159 | 1.0639 |

## 7. Minority labels

| System | disgust F1/AP | fear F1/AP | surprise F1/AP |
|---|---:|---:|---:|
| T2 | 0.2714/0.2341 | 0.3536/0.3656 | 0.4919/0.5524 |
| T3 | 0.2146/0.1550 | 0.2864/0.3102 | 0.4289/0.4528 |
| EN1 | 0.2683/0.1956 | 0.3708/0.3969 | 0.4918/0.5446 |
| EN2 | 0.2780/0.2118 | 0.3846/0.3578 | 0.4990/0.5490 |

EN2 improves minority decision F1 versus T2 for disgust, fear, and surprise, but AP decreases for all three. Thus the gain comes from the jointly selected decision surface and modest probability fusion, not improved minority ranking.

## 8. Neutral and cardinality

EN2 improves Neutral FP 0.4425 → 0.4159. Cardinality moves 1.1014 → 1.0639, closer to gold 0.8821.

## 9. Ensemble weights

| Fold | T2 alpha | T3 weight | Inner Macro-F1 |
|---:|---:|---:|---:|
| 0 | 0.90 | 0.10 | 0.5672 |
| 1 | 0.90 | 0.10 | 0.5622 |
| 2 | 0.75 | 0.25 | 0.5595 |
| 3 | 0.85 | 0.15 | 0.5475 |
| 4 | 0.95 | 0.05 | 0.5588 |

Alpha consistently favors T2 (0.75–0.95). T3 contributes a small but nonzero 5–25%; the selection does not collapse to alpha=1.

## 10. Thresholds

EN2 threshold stability:

| Label | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| anger | 0.692 | 0.045 | 0.62 | 0.74 |
| disgust | 0.454 | 0.043 | 0.39 | 0.49 |
| fear | 0.464 | 0.039 | 0.41 | 0.51 |
| joy | 0.694 | 0.034 | 0.64 | 0.73 |
| sadness | 0.600 | 0.041 | 0.54 | 0.65 |
| surprise | 0.472 | 0.030 | 0.43 | 0.51 |

## 11. Error correction

EN2 corrects 98 exact-labelset errors made by T2 and destroys 60 exact-labelset successes. Net subset-accuracy corrections: +38.

Per-label transition counts are available in `ensemble_error_analysis.csv`. For fear the ensemble recovers more false negatives than true positives it loses; for disgust and surprise, the small F1 gains arise mainly from removing false positives rather than increasing recall.

## 12. Bootstrap

| Comparison | Delta Macro-F1 | 95% percentile CI |
|---|---:|---:|
| EN1_vs_T2 | -0.0037 | [-0.0165, +0.0096] |
| EN2_vs_T2 | +0.0106 | [+0.0011, +0.0202] |
| EN2_vs_T3 | +0.0668 | [+0.0495, +0.0844] |

Intervals use 10,000 paired instance bootstrap samples, seed 42, and are reported descriptively.

## 13. Computational trade-off

The ensemble requires two separate networks: 134,907,654 T2 parameters plus 334,402,566 T3 parameters (469,310,220 across both). Historical sequential OOF inference is approximately 48.3s versus 4.6s for T2 alone (10.5×). Approximate deployable checkpoint storage is 1.75 GiB versus 0.50 GiB for T2.

## 14. Final selection

**Decision A — ensemble approved.** EN2 gains +0.0106, exceeds the predeclared +0.005 practical threshold, improves Neutral FP and cardinality, and uses a reasonably stable nonzero T3 contribution. Freeze EN2 as the primary candidate and retain T2 as the final standalone baseline.

## 15. Next step

Freeze the selected configuration and its five cross-fitted alpha/threshold sets. Do not open or evaluate the test in this task. Any final-test protocol requires separate explicit authorization and a predeclared method for deriving one deployment alpha/threshold vector without test labels.

## Validation

Source OOF hashes were unchanged; both ensemble OOFs have 2,426 unique IDs and identical outer folds. Dataset train/dev hashes match frozen raw/interim/curated manifests. No preprocessing, label changes, sampling, augmentation, translation, synthetic/external data, or Track B was used. Test: **SEALED — NOT LOADED OR EVALUATED**.
