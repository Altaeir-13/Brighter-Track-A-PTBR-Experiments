# BRIGHTER PT-BR — leakage-safe threshold calibration

## 1. Motivation

Evaluate whether per-emotion decision thresholds improve the three selected systems without changing encoders, losses, data, probabilities, or ranking.

## 2. Leakage-safe protocol

For each frozen outer fold, thresholds were learned exclusively from three-fold iterative-stratified inner-OOF predictions over outer-train. Outer-validation labels never entered model training, grid search, or tie-breaking. Existing outer OOF probabilities were reused unchanged. The test remained **SEALED — NOT LOADED OR EVALUATED**.

## 3. Nested calibration design

The grid was 0.01–0.99 in steps of 0.01. Each label threshold independently maximized its F1; since Macro-F1 is the unweighted mean of six label-wise F1 terms and each term depends only on its label decisions, this separately maximizes every term on calibration data. Numerical ties selected the largest threshold, predeclared to favor fewer false positives. Seed 42 and exact source-system hyperparameters were retained.

## 4. Threshold distributions

| System | Label | Mean | SD | Min | Max |
|---|---|---:|---:|---:|---:|
| T1 | anger | 0.422 | 0.008 | 0.41 | 0.43 |
| T1 | disgust | 0.262 | 0.015 | 0.24 | 0.28 |
| T1 | fear | 0.288 | 0.020 | 0.27 | 0.31 |
| T1 | joy | 0.428 | 0.008 | 0.42 | 0.44 |
| T1 | sadness | 0.368 | 0.008 | 0.36 | 0.38 |
| T1 | surprise | 0.318 | 0.011 | 0.31 | 0.33 |
| T2 | anger | 0.734 | 0.026 | 0.70 | 0.76 |
| T2 | disgust | 0.500 | 0.010 | 0.49 | 0.51 |
| T2 | fear | 0.512 | 0.015 | 0.49 | 0.53 |
| T2 | joy | 0.744 | 0.018 | 0.72 | 0.76 |
| T2 | sadness | 0.614 | 0.017 | 0.60 | 0.64 |
| T2 | surprise | 0.522 | 0.008 | 0.51 | 0.53 |
| T3 | anger | 0.318 | 0.099 | 0.21 | 0.47 |
| T3 | disgust | 0.152 | 0.037 | 0.10 | 0.20 |
| T3 | fear | 0.106 | 0.011 | 0.09 | 0.12 |
| T3 | joy | 0.442 | 0.047 | 0.39 | 0.50 |
| T3 | sadness | 0.312 | 0.049 | 0.27 | 0.39 |
| T3 | surprise | 0.132 | 0.013 | 0.12 | 0.15 |

## 5. Main results

| System | Fixed Macro-F1 | Thresholded Macro-F1 | Delta | 95% bootstrap CI | Micro-F1 after | Neutral FP after | Cardinality after | disgust F1 | fear F1 | surprise F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.3974 | 0.5152 | +0.1179 | [+0.0891, +0.1464] | 0.6197 | 0.6180 | 1.2370 | 0.2722 | 0.3557 | 0.4528 |
| T2 | 0.5258 | 0.5585 | +0.0327 | [+0.0240, +0.0417] | 0.6750 | 0.4425 | 1.1014 | 0.2714 | 0.3536 | 0.4919 |
| T3 | 0.3976 | 0.5023 | +0.1046 | [+0.0813, +0.1269] | 0.6248 | 0.4292 | 1.0746 | 0.2146 | 0.2864 | 0.4289 |

AP was exactly invariant for every system because probabilities were copied unchanged; only binary decisions varied. The earlier published B2 value 0.3958 is the mean of the five outer-fold Macro-F1 values; 0.3976 is the Macro-F1 recomputed once on the concatenated 2,426-row OOF and is therefore the correct paired reference for this cross-fitted comparison.

## 6. Minority labels

| System | Label | AP | F1 fixed → thresholded | Precision after | Recall after | Predicted prevalence after |
|---|---|---:|---:|---:|---:|---:|
| T1 | disgust | 0.2466 | 0.1474 → 0.2722 | 0.1822 | 0.5375 | 0.0973 |
| T1 | fear | 0.3571 | 0.1527 → 0.3557 | 0.2699 | 0.5214 | 0.0932 |
| T1 | surprise | 0.4767 | 0.2512 → 0.4528 | 0.3840 | 0.5517 | 0.1031 |
| T2 | disgust | 0.2341 | 0.2632 → 0.2714 | 0.1900 | 0.4750 | 0.0824 |
| T2 | fear | 0.3656 | 0.3438 → 0.3536 | 0.2675 | 0.5214 | 0.0940 |
| T2 | surprise | 0.5524 | 0.4706 → 0.4919 | 0.3805 | 0.6954 | 0.1311 |
| T3 | disgust | 0.1550 | 0.0000 → 0.2146 | 0.1760 | 0.2750 | 0.0515 |
| T3 | fear | 0.3102 | 0.0650 → 0.2864 | 0.2014 | 0.4957 | 0.1187 |
| T3 | surprise | 0.4528 | 0.2642 → 0.4289 | 0.3401 | 0.5805 | 0.1224 |

T3 recovers **22 true-positive disgust examples** (recall 0.2750 of support 80), versus zero at threshold 0.50. Its mean disgust threshold is 0.152 ± 0.037 (range 0.10–0.20); predicted prevalence becomes 0.0515 against gold 0.0330. This confirms usable ranking signal, but F1 0.2146 remains below T1/T2 and the between-fold range indicates moderate instability.

Fear improves 0.0650 → 0.2864 at threshold 0.106 ± 0.011; surprise improves 0.2642 → 0.4289 at 0.132 ± 0.013.

## 7. ASL behavior

T2 remains the best overall system. Calibration raises majority-label thresholds while keeping minority thresholds near 0.50, reducing ASL overprediction and increasing Macro-F1.

## 8. Weighted Focal behavior

T1's low, highly stable thresholds convert strong minority ranking into recall and F1, but overshoot gold cardinality and sharply increase Neutral FP.

## 9. BERTimbau Large behavior

Thresholding alone raises T3 Macro-F1 by +0.1046, close to T1 but below T2. It substantially recovers every minority label, proving that fixed 0.50 hid ranking signal. Nevertheless, specialized losses remain useful: T1/T2 retain higher minority AP/F1, and T2 retains higher overall Macro-F1.

## 10. Neutral examples

| System | Neutral FP fixed → thresholded | Exact neutral accuracy fixed → thresholded |
|---|---:|---:|
| T1 | 0.2124 → 0.6180 | 0.7876 → 0.3820 |
| T2 | 0.7257 → 0.4425 | 0.2743 → 0.5575 |
| T3 | 0.2935 → 0.4292 | 0.7065 → 0.5708 |

## 11. Label cardinality

| System | Fixed | Thresholded | Gold | Absolute error before → after |
|---|---:|---:|---:|---:|
| T1 | 0.5470 | 1.2370 | 0.8821 | 0.3351 → 0.3549 |
| T2 | 1.4477 | 1.1014 | 0.8821 | 0.5655 → 0.2193 |
| T3 | 0.6925 | 1.0746 | 0.8821 | 0.1896 → 0.1925 |

## 12. Multilabel-only evaluation

| System | Macro-F1 fixed → after | Micro-F1 fixed → after | Jaccard fixed → after | Subset accuracy fixed → after |
|---|---:|---:|---:|---:|
| T1 | 0.3926 → 0.6454 | 0.5091 → 0.6982 | 0.3489 → 0.5619 | 0.0754 → 0.2263 |
| T2 | 0.6870 → 0.6616 | 0.7492 → 0.7175 | 0.6366 → 0.5913 | 0.2849 → 0.2626 |
| T3 | 0.3950 → 0.5842 | 0.5552 → 0.6419 | 0.4043 → 0.4951 | 0.0726 → 0.1480 |

## 13. Bootstrap comparison

Paired instance bootstrap used 10,000 resamples, seed 42. Intervals are percentile intervals and are reported descriptively, without automatic significance language.

| System | Delta Macro-F1 | 95% CI |
|---|---:|---:|
| T1 | +0.1179 | [+0.0891, +0.1464] |
| T2 | +0.0327 | [+0.0240, +0.0417] |
| T3 | +0.1046 | [+0.0813, +0.1269] |

## 14. Threshold stability

T1 is the most stable. T2 is stable for the minority thresholds and moderately variable for anger/joy. T3 has stable fear/surprise thresholds but moderate disgust variability (0.10–0.20) and high anger variability (0.21–0.47), a validity caveat.

## 15. Error complementarity

| Pair | Example disagreement | Label disagreement | A correct/B wrong | B correct/A wrong | Probability corr. | Error corr. | Multilabel disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1/T2 | 0.4287 | 0.0903 | 203 | 398 | 0.7798 | 0.5803 | 0.5279 |
| T1/T3 | 0.5367 | 0.1292 | 322 | 442 | 0.7185 | 0.4235 | 0.7039 |
| T2/T3 | 0.4658 | 0.1103 | 359 | 284 | 0.7919 | 0.4594 | 0.6788 |

Exclusive exact-correct/error counts:

| System | Exact correct | Exclusively correct | Exclusive error | Multilabel exclusively correct |
|---|---:|---:|---:|---:|
| T1 | 1074 | 121 | 240 | 20 |
| T2 | 1269 | 158 | 82 | 25 |
| T3 | 1194 | 202 | 201 | 15 |

Minority disagreement rates range from 0.0585 to 0.1129; the nonzero exclusive-correct counts justify studying an ensemble later, but do not by themselves authorize building one.

## 16. Candidate selection

Select **T2 (BERTweetBR + ASL + nested thresholds)** as the primary system and **T3 (BERTimbau Large + BCE + nested thresholds)** as the complementary secondary candidate. T2 has the best Macro-F1 and best balance of neutral/cardinality behavior. T1 exceeds T3 in Macro-F1 and minority AP/F1 at much lower calibration cost, but its Neutral FP (0.6180) and cardinality (1.2370) are materially worse. T3 is more complementary to T2: it is exclusively exact-correct on 202 examples overall (versus 121 for T1) and wins 284 paired exact decisions where T2 fails (versus 203 for T1), while retaining better neutral/cardinality behavior. T1 remains an important calibrated ablation, but not one of the two selected systems.

## 17. Next experiment

Recommendation only — **do not execute automatically**: freeze T2 as the primary system and perform a narrowly scoped, leakage-safe ensemble feasibility study using T2/T3 predictions. T3's additional exclusive correct decisions and lower error correlation with T2 justify this analysis despite its higher computational cost. Do not train BERTimbau Large + ASL/Weighted Focal unless a later, separately authorized study specifically targets encoder–loss interaction.

## Validation

All thresholded OOF files contain 2,426 unique ordered IDs with identical outer folds and bitwise-identical source probabilities. Train/dev raw, interim, and curated hashes match frozen records. No preprocessing, label modification, sampling, augmentation, translation, synthetic/external data, or Track B was used. Test policy: **SEALED — NOT LOADED OR EVALUATED**.
