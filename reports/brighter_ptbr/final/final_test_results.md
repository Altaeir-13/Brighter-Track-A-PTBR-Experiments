# BRIGHTER PT-BR — final test results

## 1. Frozen protocol

BRIGHTER Track A PT-BR, revision `419566ac8f46f951b51584be36c13e5363008144`, 2,226 official test examples. Primary EN2 fused frozen T2/T3 probabilities with alpha 0.70/0.30 and thresholds `anger=.67, disgust=.36, fear=.43, joy=.65, sadness=.58, surprise=.49`. Secondary T2 used thresholds `anger=.66, disgust=.53, fear=.52, joy=.75, sadness=.65, surprise=.57`. No preprocessing, data modification, recalibration, or post-test parameter change occurred.

## 2. Execution integrity

Final inference was executed exactly once. T2/T3 probabilities, EN2 fusion, frozen decisions, submissions, and metrics were persisted before a reporting-only Matplotlib cache permission failure. The pipeline stopped as pre-registered. Recovery consumed only frozen result/output artifacts: no model was loaded, no test dataset was reopened, no inference was repeated, and no hyperparameter changed.

## 3. Development estimates

Leakage-safe nested Macro-F1: T2 = **0.5585**; EN2 = **0.5691**.

## 4. Deployment calibration

T2 = 0.5787; EN2 = 0.5897. **DEPLOYMENT CALIBRATION ONLY — NOT AN UNBIASED PERFORMANCE ESTIMATE.**

## 5. Final test

| System | Macro-F1 | Micro-F1 | Macro Precision | Macro Recall | Hamming | Jaccard | Subset Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN2 (primary) | 0.2852 | 0.4221 | 0.6325 | 0.2277 | 0.1183 | 0.2152 | 0.4353 |
| T2 standalone (secondary) | 0.1569 | 0.2405 | 0.4034 | 0.1486 | 0.1575 | 0.1059 | 0.3014 |

## 6. Per-label results

| System | Label | Precision | Recall | F1 | AP | Support | Predicted prevalence |
|---|---|---:|---:|---:|---:|---:|---:|
| EN2 | anger | 0.7895 | 0.2575 | 0.3883 | 0.6987 | 699 | 0.1024 |
| EN2 | disgust | 0.3333 | 0.0141 | 0.0270 | 0.1288 | 71 | 0.0013 |
| EN2 | fear | 0.6111 | 0.0791 | 0.1401 | 0.2884 | 139 | 0.0081 |
| EN2 | joy | 0.8659 | 0.5586 | 0.6791 | 0.7909 | 555 | 0.1608 |
| EN2 | sadness | 0.7500 | 0.0175 | 0.0342 | 0.6695 | 343 | 0.0036 |
| EN2 | surprise | 0.4452 | 0.4395 | 0.4423 | 0.4489 | 157 | 0.0696 |
| T2 | anger | 0.4379 | 0.2117 | 0.2854 | 0.4087 | 699 | 0.1518 |
| T2 | disgust | 0.0000 | 0.0000 | 0.0000 | 0.0571 | 71 | 0.0000 |
| T2 | fear | 0.3333 | 0.0072 | 0.0141 | 0.0798 | 139 | 0.0013 |
| T2 | joy | 0.6485 | 0.1928 | 0.2972 | 0.4949 | 555 | 0.0741 |
| T2 | sadness | 0.7500 | 0.0087 | 0.0173 | 0.2286 | 343 | 0.0018 |
| T2 | surprise | 0.2508 | 0.4713 | 0.3274 | 0.2854 | 157 | 0.1325 |

## 7. Minority labels

EN2 retained the strongest minority result for surprise (F1 0.4423) but recovered very few disgust (recall 0.0141) and fear (recall 0.0791) positives. T2 recovered no disgust positives and only one fear true positive. These are frozen descriptive findings, not grounds for post-test adjustment.

## 8. Neutral behavior

| System | Exact Neutral Accuracy | Neutral FP |
|---|---:|---:|
| EN2 | 0.9058 | 0.0942 |
| T2 | 0.8133 | 0.1867 |

## 9. Cardinality

| System | Gold | Predicted | Delta |
|---|---:|---:|---:|
| EN2 | 0.8823 | 0.3459 | -0.5364 |
| T2 | 0.8823 | 0.3616 | -0.5207 |

Development EN2 predicted cardinality was approximately 1.0639; final test cardinality was 0.3459 against gold 0.8823. The final system emitted substantially fewer positive labels, consistent with very low recalls for disgust (0.0141), fear (0.0791), and sadness (0.0175). No recalibration was performed.

## 10. Development-to-test gap

EN2 changed from 0.5691 nested development Macro-F1 to 0.2852 on final test, a difference of -0.2839. T2 changed from 0.5585 to 0.1569, a difference of -0.4016.

## 11. Interpretation

The gaps are compatible with several non-exclusive hypotheses: train/dev-to-test distribution shift; probability calibration shift after final retraining; threshold-transfer failure; seed/model variance; rare-label uncertainty; and differences between cross-fitted OOF models and a single final model trained on all 2,426 development examples. This recovery does not test or privilege any causal explanation. The result is valid under the frozen protocol and is not treated as erroneous merely because it is lower than development.

## 12. Threats to validity

- The development dataset is small for neural multilabel estimation.
- Label imbalance is severe; the original training split contains only 75 disgust positives.
- One train–test text overlap was known before final evaluation and retained under the official split protocol.
- Deployment alpha 0.70 differed from the nested alpha range/mean, increasing transfer uncertainty.
- Development and test scores/distributions differ markedly.
- Final models used a single frozen seed.
- Per-label thresholds may be sensitive to prevalence and score-distribution changes.
- OOF estimates aggregate multiple fold-specific models, whereas deployment used one model per component trained on the full development pool.

The search space remains closed. EN2 remains the primary pre-registered system and T2 the secondary baseline regardless of relative test performance.
