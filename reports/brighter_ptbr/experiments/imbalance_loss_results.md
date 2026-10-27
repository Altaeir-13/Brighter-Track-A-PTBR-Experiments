# BRIGHTER PT-BR — imbalance loss ablation

## A. Audit

The existing reports, registry, fold metrics, confusion matrices, persisted CV assignments, manifests, baseline code, B1/B2 model provenance, and all three baseline OOF files were inspected. The development pool is the deliberate union of official train+dev (2,426 unique examples). All OOF files use the same ordered IDs and the same five persisted folds (seed 42). Train/dev SHA-256 hashes match the frozen manifest.

The official test remains **SEALED — NOT LOADED OR EVALUATED**. A direct `--final-test` smoke test failed before configuration/data access, as required. No preprocessing, sampling, balancing, augmentation, synthetic data, external data, or label modification was used. This directory is not a Git repository, so `git_commit` is recorded as `NOT_A_GIT_REPOSITORY` rather than fabricated.

## B. Implementations

All variants use BERTimbau Base, max length 128, AdamW, LR 2e-5, four epochs, weight decay 0.01, 10% linear warmup, FP16, physical batch 8 × accumulation 2 = effective batch 16, seed 42, identical folds, and threshold 0.5.

- **L1 Weighted BCE:** for class c, `w_c = n_c^- / n_c^+`, and `L_c = -[w_c y log σ(z) + (1-y) log(1-σ(z))]`. Counts use only the outer-fold training partition.
- **L2 Focal:** `L_c = (1-p_t)^2 BCE(z,y)`, with `p_t = exp(-BCE)` and alpha=1. Gamma 0 exactly recovers BCE under the same mean reduction.
- **L3 Weighted Focal:** `alpha_c = N/(n_c^+ k)` (k=6), then `L_c = alpha_c (1-p_t)^2 BCE(z,y)`. The cited paper is ambiguous about adapting its class-level weight to both binary signs; the mathematically literal implementation multiplies the complete per-class focal term, not only positives.
- **L4 ASL:** positive `L+ = -(1-p)^gamma+ log p`; negative `p_m=max(p-0.05,0)`, `L- = -p_m^gamma- log(1-p_m)`, with `gamma+=0`, `gamma-=4`. Computation is float32 over logits under autocast.

Formulations were checked against [Hanif et al. (2025)](https://aclanthology.org/2025.semeval-1.279/) and the [official Ridnik et al. ASL implementation](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py).

## C. Results

Primary selection uses OOF Macro-F1; fold mean ± sample SD is included for stability.

| Model/Loss | OOF Macro-F1 | Fold Macro-F1 mean ± SD | Micro-F1 | Hamming | Jaccard | AP disgust | AP fear | AP surprise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 Weighted BCE | 0.4470 | 0.4482 ± 0.0065 | 0.5432 | 0.1906 | 0.4022 | 0.1440 | 0.2427 | 0.3872 |
| L2 Focal | 0.3789 | 0.3786 ± 0.0081 | 0.6344 | 0.0960 | 0.4210 | 0.1353 | 0.1994 | 0.3869 |
| L3 Weighted Focal | 0.4024 | 0.4004 ± 0.0238 | 0.5745 | 0.1057 | 0.3546 | 0.1987 | 0.2302 | 0.3948 |
| L4 ASL | 0.4617 | 0.4624 ± 0.0200 | 0.5873 | 0.1646 | 0.4422 | 0.1297 | 0.2029 | 0.3751 |

### OOF metrics by class

| Loss | Emotion | Precision | Recall | F1 | AP | Gold + | Predicted + | Gold prev. | Pred. prev. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | anger | 0.5653 | 0.7758 | 0.6540 | 0.7189 | 776 | 1065 | 0.3199 | 0.4390 |
| L1 | disgust | 0.1114 | 0.5500 | 0.1853 | 0.1440 | 80 | 395 | 0.0330 | 0.1628 |
| L1 | fear | 0.1435 | 0.5812 | 0.2301 | 0.2427 | 117 | 474 | 0.0482 | 0.1954 |
| L1 | joy | 0.6282 | 0.8647 | 0.7277 | 0.8197 | 643 | 885 | 0.2650 | 0.3648 |
| L1 | sadness | 0.3899 | 0.7743 | 0.5187 | 0.5906 | 350 | 695 | 0.1443 | 0.2865 |
| L1 | surprise | 0.2589 | 0.6264 | 0.3664 | 0.3872 | 174 | 421 | 0.0717 | 0.1735 |
| L2 | anger | 0.7127 | 0.6714 | 0.6914 | 0.7655 | 776 | 731 | 0.3199 | 0.3013 |
| L2 | disgust | 0.0000 | 0.0000 | 0.0000 | 0.1353 | 80 | 1 | 0.0330 | 0.0004 |
| L2 | fear | 1.0000 | 0.0085 | 0.0169 | 0.1994 | 117 | 1 | 0.0482 | 0.0004 |
| L2 | joy | 0.7769 | 0.7418 | 0.7589 | 0.8292 | 643 | 614 | 0.2650 | 0.2531 |
| L2 | sadness | 0.6312 | 0.5429 | 0.5837 | 0.6322 | 350 | 301 | 0.1443 | 0.1241 |
| L2 | surprise | 0.6970 | 0.1322 | 0.2222 | 0.3869 | 174 | 33 | 0.0717 | 0.0136 |
| L3 | anger | 0.6713 | 0.5026 | 0.5748 | 0.6842 | 776 | 581 | 0.3199 | 0.2395 |
| L3 | disgust | 0.4762 | 0.1250 | 0.1980 | 0.1987 | 80 | 21 | 0.0330 | 0.0087 |
| L3 | fear | 0.4615 | 0.0513 | 0.0923 | 0.2302 | 117 | 13 | 0.0482 | 0.0054 |
| L3 | joy | 0.7726 | 0.6921 | 0.7301 | 0.8072 | 643 | 576 | 0.2650 | 0.2374 |
| L3 | sadness | 0.6555 | 0.4457 | 0.5306 | 0.5697 | 350 | 238 | 0.1443 | 0.0981 |
| L3 | surprise | 0.6667 | 0.1839 | 0.2883 | 0.3948 | 174 | 48 | 0.0717 | 0.0198 |
| L4 | anger | 0.5711 | 0.8750 | 0.6911 | 0.7777 | 776 | 1189 | 0.3199 | 0.4901 |
| L4 | disgust | 0.1588 | 0.3375 | 0.2160 | 0.1297 | 80 | 170 | 0.0330 | 0.0701 |
| L4 | fear | 0.2204 | 0.3504 | 0.2706 | 0.2029 | 117 | 186 | 0.0482 | 0.0767 |
| L4 | joy | 0.6079 | 0.8802 | 0.7192 | 0.8344 | 643 | 931 | 0.2650 | 0.3838 |
| L4 | sadness | 0.3712 | 0.8029 | 0.5077 | 0.6341 | 350 | 757 | 0.1443 | 0.3120 |
| L4 | surprise | 0.2564 | 0.6379 | 0.3657 | 0.3751 | 174 | 433 | 0.0717 | 0.1785 |

## D. Comparison with B1 and B2

| Loss | Δ OOF Macro-F1 vs B1 | Δ vs B2 | Δ Micro-F1 vs B1 | Δ Neutral FP vs B1 |
|---|---:|---:|---:|---:|
| L1 Weighted BCE | +0.1226 | +0.0494 | -0.0690 | +0.4838 |
| L2 Focal | +0.0545 | -0.0188 | +0.0222 | +0.0251 |
| L3 Weighted Focal | +0.0779 | +0.0047 | -0.0377 | -0.0295 |
| L4 ASL | +0.1373 | +0.0641 | -0.0249 | +0.4882 |

## E. Disgust

| Loss | Any positive? | Predicted + | Precision | Recall | F1 | AP | Gold prev. | Pred. prev. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | yes | 395 | 0.1114 | 0.5500 | 0.1853 | 0.1440 | 0.0330 | 0.1628 |
| L2 | yes | 1 | 0.0000 | 0.0000 | 0.0000 | 0.1353 | 0.0330 | 0.0004 |
| L3 | yes | 21 | 0.4762 | 0.1250 | 0.1980 | 0.1987 | 0.0330 | 0.0087 |
| L4 | yes | 170 | 0.1588 | 0.3375 | 0.2160 | 0.1297 | 0.0330 | 0.0701 |

All weighted/asymmetric variants make positive disgust predictions and lift F1 above zero. L3 gives the best disgust AP and precision with prevalence closest to gold; L1 gives the highest recall but overpredicts substantially. L4 improves F1 but its precision and AP are weaker than L3.

## F. Threshold diagnostics

**DIAGNOSTIC ONLY — THRESHOLD SELECTED AND EVALUATED ON SAME OOF.**

B1/B2 were swept from 0.01 to 0.90 by 0.01. See `threshold_diagnostics.md` and its full CSV. Key minority points:

| Model | Emotion | Best diagnostic threshold | Maximum diagnostic F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| B1 | disgust | 0.09 | 0.1838 | 0.1108 | 0.5375 |
| B1 | fear | 0.07 | 0.1494 | 0.0875 | 0.5128 |
| B1 | surprise | 0.08 | 0.1678 | 0.0972 | 0.6149 |
| B2 | disgust | 0.11 | 0.2207 | 0.1524 | 0.4000 |
| B2 | fear | 0.18 | 0.3604 | 0.3810 | 0.3419 |
| B2 | surprise | 0.17 | 0.4764 | 0.4375 | 0.5230 |

These values show ranking signal, especially for B2 fear/surprise, but cannot be reported as unbiased generalization performance.

## G. Cost

| Loss | Parameters | Training total (s) | Inference (s) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|
| L1 Weighted BCE | 108,927,750 | 448.3 | 3.89 | 1.92 |
| L2 Focal | 108,927,750 | 450.7 | 3.94 | 1.92 |
| L3 Weighted Focal | 108,927,750 | 467.0 | 4.08 | 1.92 |
| L4 ASL | 108,927,750 | 457.6 | 3.89 | 1.92 |

## H. Decision

Advance **L4 ASL** and **L3 Weighted Focal** (maximum two). L4 has the best OOF/fold Macro-F1 and strong minority recall, but its neutral false-positive rate is high and must remain a selection caveat. L3 is the complementary conservative candidate: best disgust AP/precision, substantially lower neutral FP, and no severe majority-label collapse, although its fold variance is higher. L1 is stable and high-Macro-F1 but is not advanced because its neutral FP and overprediction are nearly as severe as L4 while its ranking evidence is less compelling than L3. L2 improves on B1 but does not recover disgust at 0.5.

## I. Next phase

Run the frozen BERTweetBR/Albertina encoder × BCE/Weighted-Focal/ASL matrix without threshold tuning. Retain BCE as the encoder-effect control. Leakage-safe nested per-label threshold calibration remains a later phase. **Track B is OUT OF SCOPE BY EXPERIMENTAL DESIGN**; the active project uses BRIGHTER Track A only.

## Validation

All 40 unit/integration tests pass. Each new OOF contains exactly 2,426 unique IDs with persisted folds and finite probabilities. Train/dev hashes remain unchanged. Fold checkpoints, config hashes, package versions, timings, VRAM, timestamps, OOF paths/hashes, and train-only class statistics are persisted. The test remains sealed.
