# BRIGHTER PT-BR — post-hoc generalization gap analysis

## 1. Objective

Investigate the frozen development-to-test gap without changing the predictive experiment. This analysis uses only preserved OOF/final probability, prediction, metric, calibration, and ensemble artifacts. It performs no training, model loading, official-test loading, inference, threshold search, or parameter modification.

## 2. Frozen Experiment Context

Primary EN2 remains frozen at nested development Macro-F1 0.5691 and final test Macro-F1 0.2852 (difference -0.2839). Secondary T2 remains 0.5585 and 0.1569 (difference -0.4016). EN2 probabilities are fixed as 0.70×T2 + 0.30×T3. For probability-scale transfer diagnostics only, the same frozen fusion is applied to preserved development OOF probabilities; the unbiased performance estimate remains the nested cross-fitted result.

## 3. Label Distribution

| Label | Gold dev | Gold test | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| anger | 0.3199 | 0.3140 | -0.0059 | -1.83% |
| disgust | 0.0330 | 0.0319 | -0.0011 | -3.28% |
| fear | 0.0482 | 0.0624 | +0.0142 | +29.48% |
| joy | 0.2650 | 0.2493 | -0.0157 | -5.93% |
| sadness | 0.1443 | 0.1541 | +0.0098 | +6.81% |
| surprise | 0.0717 | 0.0705 | -0.0012 | -1.66% |

The largest absolute per-label prevalence change was 0.0157. These changes are too small, descriptively, to match the magnitude of the Macro-F1 gaps; H1 is therefore not supported by label prevalence alone.

## 4. Cardinality Shift

Gold cardinality was 0.882110 in development and 0.882300 in test (delta +0.000190).

| System | Predicted dev | Predicted test | Delta | Test/dev |
|---|---:|---:|---:|---:|
| EN2 | 1.0639 | 0.3459 | -0.7180 | 0.325 |
| T2 | 1.1014 | 0.3616 | -0.7398 | 0.328 |

EN2 emitted only 32.5% of its nested development label cardinality; T2 emitted 32.8%. This decision-rate collapse occurred while gold cardinality was almost unchanged.

## 5. Probability Distribution Shift

Across labels, mean probability shifts averaged T2=-0.0379, T3=-0.0070, and EN2=-0.0286. The largest negative mean shifts were T2/sadness=-0.2565 and T3/joy=-0.0206.

Full quantiles (min, p01, p05, p10, p25, median, p75, p90, p95, p99, max), Wasserstein distances, and fixed-histogram Jensen–Shannon divergences are in the machine-readable probability tables.

## 6. Threshold Crossing Analysis

| System | Label | Threshold | Dev crossing | Test crossing | Delta |
|---|---|---:|---:|---:|---:|
| T2 | anger | 0.66 | 0.3669 | 0.1518 | -0.2150 |
| T2 | disgust | 0.53 | 0.0408 | 0.0000 | -0.0408 |
| T2 | fear | 0.52 | 0.0845 | 0.0013 | -0.0832 |
| T2 | joy | 0.75 | 0.2613 | 0.0741 | -0.1872 |
| T2 | sadness | 0.65 | 0.1785 | 0.0018 | -0.1767 |
| T2 | surprise | 0.57 | 0.0882 | 0.1325 | +0.0443 |
| EN2 | anger | 0.67 | 0.3042 | 0.1024 | -0.2018 |
| EN2 | disgust | 0.36 | 0.1105 | 0.0013 | -0.1091 |
| EN2 | fear | 0.43 | 0.0515 | 0.0081 | -0.0434 |
| EN2 | joy | 0.65 | 0.2601 | 0.1608 | -0.0993 |
| EN2 | sadness | 0.58 | 0.1575 | 0.0036 | -0.1539 |
| EN2 | surprise | 0.49 | 0.0581 | 0.0696 | +0.0115 |

Frozen-threshold crossing rates fell for most labels, directly explaining the lower predicted cardinality. This is a transfer diagnostic only: no alternative threshold or oracle test F1 was computed.

## 7. Per-label Generalization

| Label | P dev | P test | R dev | R test | F1 dev | F1 test |
|---|---:|---:|---:|---:|---:|---:|
| anger | 0.7827 | 0.7895 | 0.7938 | 0.2575 | 0.7882 | 0.3883 |
| disgust | 0.2011 | 0.3333 | 0.4500 | 0.0141 | 0.2780 | 0.0270 |
| fear | 0.2941 | 0.6111 | 0.5556 | 0.0791 | 0.3846 | 0.1401 |
| joy | 0.8025 | 0.8659 | 0.8025 | 0.5586 | 0.8025 | 0.6791 |
| sadness | 0.5923 | 0.7500 | 0.7514 | 0.0175 | 0.6625 | 0.0342 |
| surprise | 0.3909 | 0.4452 | 0.6897 | 0.4395 | 0.4990 | 0.4423 |

Final EN2 macro precision remained 0.6325, whereas macro recall fell to 0.2277; the dominant decision-level pattern is conservatism rather than indiscriminate false positives.

## 8. Minority Classes

Disgust recall changed 0.4500→0.0141, fear 0.5556→0.0791, and surprise 0.6897→0.4395. Sadness also collapsed (0.7514→0.0175), so the gap is not confined to the three pre-designated minority labels.

## 9. T2 Probability Behavior

T2 predicted cardinality changed 1.1014→0.3616. Its most negative mean score shift was sadness (-0.2565). Surprise behaved differently at the decision level: final recall remained 0.4713, while sadness recall was 0.0087. Their distinct score locations relative to frozen thresholds, not a changed decision rule, account descriptively for that contrast.

## 10. T3 Component Probability Behavior

T3 mean scores also shifted (-0.0070 averaged across labels), with the largest negative label shift in joy (-0.0206). This argues against a purely T2-specific probability-scale failure. No T3 standalone final Macro-F1 was calculated or reported.

## 11. EN2 Ensemble Behavior

EN2 final Macro-F1 (0.2852) exceeded frozen T2 (0.1569), and its neutral FP was lower (0.0942 vs 0.1867). Thus T3 mitigated some T2 failure, but the fixed fusion could not prevent the shared downward score/crossing shift.

## 12. Threshold-Free Ranking Analysis

**POST-HOC DIAGNOSTIC METRIC — NOT USED FOR MODEL SELECTION.**

| System | Macro AP dev | Macro AP test | Delta |
|---|---:|---:|---:|
| T2 | 0.6018 | 0.2591 | -0.3427 |
| EN2 | 0.6002 | 0.5042 | -0.0961 |

AP changed as well as thresholded F1, indicating that threshold transfer is not the sole issue. The relative size varies by label; the calibration CSV contains every per-label value.

## 13. Calibration Diagnostics

**POST-HOC DIAGNOSTIC METRICS — NOT USED FOR MODEL SELECTION.**

| System | Brier dev | Brier test | ECE-10 dev | ECE-10 test |
|---|---:|---:|---:|---:|
| T2 | 0.1488 | 0.1845 | 0.2852 | 0.2468 |
| EN2 | 0.1275 | 0.1215 | 0.2496 | 0.1965 |

ECE uses ten fixed equal-width bins. Brier has priority; ECE is included only as a simple descriptive supplement. Score-by-gold-class tables separate positive and negative score movement.

## 14. Potential Explanations

The evidence is compatible with multiple non-exclusive mechanisms: modest label-prevalence differences, downward probability-scale transfer, ranking degradation, final-model retraining variance, rare-label instability, and covariate/domain shift. The present artifacts do not identify a unique cause.

## 15. Evidence For/Against Each Hypothesis

| Hypothesis | Evidence supporting | Evidence against | Status |
|---|---|---|---|
| H1 label prevalence shift | Largest absolute per-label prevalence change was 0.0157. | Gold cardinality changed only +0.000190; prevalence changes are modest relative to the performance gap. | **NOT SUPPORTED** |
| H2 probability calibration shift | Mean score shifts averaged -0.0286 for EN2 and predicted cardinality fell -0.7180; crossing rates collapsed for most labels. | EN2 macro Brier did not deteriorate (0.1275→0.1215), and score movement conflates retraining with test-distribution effects. | **PARTIALLY SUPPORTED** |
| H3 ranking degradation | EN2 macro AP changed 0.6002→0.5042, so threshold-free ranking also degraded. | Some labels, especially surprise and joy, retained materially better ranking/recall than the rare conservative labels. | **PARTIALLY SUPPORTED** |
| H4 final-model retraining shift | Frozen final models moved probability scales relative to fold OOF: mean shifts T2=-0.0379, T3=-0.0070. | No matched inference of fold and final models on the same held-out texts exists, so retraining cannot be isolated from test shift. | **PARTIALLY SUPPORTED** |
| H5 T2-specific failure | T2 had the largest standalone test gap and mean score shift -0.0379, versus only -0.0070 for T3. | EN2 also remained conservative and some T2 labels shifted upward, so the failure was heterogeneous rather than uniformly T2-wide. | **PARTIALLY SUPPORTED** |
| H6 ensemble failure | The frozen fusion inherited low component scores and remained far below nested development. | EN2 test Macro-F1 0.2852 exceeded T2 0.1569; T3 therefore mitigated rather than solely caused T2 conservatism. | **NOT SUPPORTED** |
| H7 textual/domain shift | Development-to-test score and ranking changes are compatible with covariate/domain differences. | No permitted frozen text-level representation supports direct covariate analysis in this task. | **INCONCLUSIVE** |

## 16. Implications for the Article

The article should report the robust nested estimate (0.5691) and the frozen final result (0.2852) without hiding the gap. The post-hoc evidence makes threshold transfer between cross-validation model distributions and single full-development retrained models a central methodological risk in small imbalanced multilabel tasks.

Historical context must remain separate: an earlier experiment used a previously balanced/augmented dataset, BERTimbau Large, Focal Loss, demojize, thresholds, and a Safety Net, with leaderboard Macro-F1 0.55746. The current experiment used original Track A, no augmentation/preprocessing/Safety Net, nested selection, ASL/BERTweetBR and a frozen ensemble. The values 0.55746 and 0.2852 are not directly comparable because protocols, data, and possibly evaluation endpoints differ.

## 17. Limitations

- The dataset and rare-label supports are small; original training had only 75 disgust positives.
- Final models used one frozen seed.
- OOF distributions aggregate fold-specific models, while deployment uses one full-development model per component.
- Test text was not reopened, so H7 textual/domain shift could not be directly measured.
- Probability distances are descriptive, not causal tests.
- ECE depends on fixed binning and is secondary to Brier.
- One train–test overlap known pre-test remains in the official protocol.

## 18. Conclusions

The clearest evidence is a heterogeneous probability-scale and frozen-threshold transfer failure after final retraining/application, accompanied by a predicted-cardinality collapse despite nearly unchanged gold cardinality. Threshold-free AP also declined, so the gap cannot be reduced to thresholds alone. T2 shifted much more than T3 on average, making a T2-dominant component effect partially supported; T3 was comparatively stable and its contribution mitigated, but did not eliminate, the final conservatism. Retraining-induced calibration shift cannot be isolated from test/domain shift with the frozen artifacts, so causal attribution remains appropriately limited.
