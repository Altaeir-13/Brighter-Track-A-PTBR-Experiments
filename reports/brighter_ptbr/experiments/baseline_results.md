# BRIGHTER PT-BR — First baseline battery

## 1. Experimental protocol

Track A at frozen revision `419566ac8f46f951b51584be36c13e5363008144`. The development pool contains official train+dev (2426 examples). The official test set is **SEALED — NOT LOADED OR EVALUATED**.

Raw `text_raw` was used without manual lowercasing, cleaning, normalization, balancing, sampling, augmentation, class weights, or threshold optimization.

## 2. Cross-validation strategy

Five persisted iterative multilabel-stratified folds use seed 42. Every development ID has exactly one OOF prediction from each model; train/validation IDs and exact texts are disjoint in every fold. See `cv_distribution.md`.

## 3. Models

- B0: word (1–2) + character `char_wb` (3–5) TF-IDF, One-vs-Rest Logistic Regression.
- B1: `neuralmind/bert-base-portuguese-cased`, six independent logits, unweighted BCE.
- B2: `neuralmind/bert-large-portuguese-cased`, same objective and folds.

## 4. Training configuration

Transformers: max length 128, dynamic padding, AdamW, learning rate 2e-5, four epochs, weight decay 0.01, 10% linear warmup, fp16, effective batch size 16. Base uses physical batch 8 × accumulation 2; Large uses physical batch 1 × accumulation 16.

## 5. Overall results

| Model | Macro-F1 mean ± SD | Micro-F1 mean | Macro precision | Macro recall | Hamming loss | Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.1694 ± 0.0120 | 0.3385 | 0.4198 | 0.1144 | 0.1234 | 0.1677 |
| B1 | 0.3238 ± 0.0204 | 0.6119 | 0.3545 | 0.3036 | 0.0988 | 0.4055 |
| B2 | 0.3958 ± 0.0212 | 0.6459 | 0.5506 | 0.3611 | 0.0929 | 0.4315 |

All values use the fixed threshold 0.5. Mean and sample SD are calculated over the five validation folds.

## 6. Per-label results

| Model | Emotion | Precision | Recall | F1 | Support | AP |
|---|---|---:|---:|---:|---:|---:|
| B0 | anger | 0.7404 | 0.2977 | 0.4246 | 776 | 0.6174 |
| B0 | disgust | 0.0000 | 0.0000 | 0.0000 | 80 | 0.1006 |
| B0 | fear | 0.0000 | 0.0000 | 0.0000 | 117 | 0.1517 |
| B0 | joy | 0.8644 | 0.3173 | 0.4642 | 643 | 0.7189 |
| B0 | sadness | 0.8929 | 0.0714 | 0.1323 | 350 | 0.4903 |
| B0 | surprise | 0.0000 | 0.0000 | 0.0000 | 174 | 0.2670 |
| B1 | anger | 0.7137 | 0.6521 | 0.6815 | 776 | 0.7538 |
| B1 | disgust | 0.0000 | 0.0000 | 0.0000 | 80 | 0.0937 |
| B1 | fear | 0.0000 | 0.0000 | 0.0000 | 117 | 0.0873 |
| B1 | joy | 0.7651 | 0.7496 | 0.7573 | 643 | 0.8199 |
| B1 | sadness | 0.6419 | 0.4200 | 0.5078 | 350 | 0.5684 |
| B1 | surprise | 0.0000 | 0.0000 | 0.0000 | 174 | 0.1547 |
| B2 | anger | 0.7363 | 0.6946 | 0.7149 | 776 | 0.7969 |
| B2 | disgust | 0.0000 | 0.0000 | 0.0000 | 80 | 0.1550 |
| B2 | fear | 0.6667 | 0.0342 | 0.0650 | 117 | 0.3102 |
| B2 | joy | 0.7848 | 0.7372 | 0.7602 | 643 | 0.8359 |
| B2 | sadness | 0.6300 | 0.5400 | 0.5815 | 350 | 0.6649 |
| B2 | surprise | 0.7368 | 0.1609 | 0.2642 | 174 | 0.4528 |

## 7. Zero-label behavior

| Model | Zero support | Exact Neutral Accuracy | Neutral False Positive Rate |
|---|---:|---:|---:|
| B0 | 678 | 0.9115 | 0.0885 |
| B1 | 678 | 0.7153 | 0.2847 |
| B2 | 678 | 0.7065 | 0.2935 |

## 8. Multi-label behavior

| Model | Group | N | Subset accuracy | Jaccard | Hamming loss |
|---|---|---:|---:|---:|---:|
| B0 | single_label | 1390 | 0.2568 | 0.2576 | 0.1296 |
| B0 | multi_label | 358 | 0.0000 | 0.1364 | 0.3049 |
| B1 | single_label | 1390 | 0.6079 | 0.6180 | 0.0860 |
| B1 | multi_label | 358 | 0.0391 | 0.3487 | 0.2453 |
| B2 | single_label | 1390 | 0.6309 | 0.6489 | 0.0785 |
| B2 | multi_label | 358 | 0.0726 | 0.4043 | 0.2291 |

## 9. Precision–Recall analysis

OOF precision–recall curves and per-label Average Precision were generated for Base and Large. They are diagnostic only; no thresholds were selected.

## 10. Computational cost

| Model | Parameters | Training (s) | Inference (s) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|
| B0 | n/a | 2.5 | 0.27 | n/a |
| B1 | 108,927,750 | 506.9 | 4.29 | 1.92 |
| B2 | 334,402,566 | 6803.4 | 43.73 | 5.58 |

GPU timings are hardware-specific. Seeds and deterministic settings were fixed, but PyTorch reported a non-bitwise-deterministic memory-efficient attention backward kernel; exact reproduction can vary across GPU/driver/library stacks.

## 11. Observations

- The highest five-fold Macro-F1 in this battery is B2 at 0.3958 ± 0.0212.
- Minority-label OOF F1 above zero was observed for B2:fear=0.065, B2:surprise=0.264; remaining zeros must be interpreted jointly with AP/PR curves, not as absence of ranking signal.
- Differences between thresholded F1 and Average Precision indicate that ranking quality and the fixed operating point must be kept conceptually separate.
- Neutral and genuinely multilabel cases exhibit different failure profiles; both should remain explicit in later ablations.

## 12. Next experimental questions

The current evidence motivates controlled train/dev-only ablations of loss weighting, alternative losses, threshold calibration, maximum length, and OOF-based calibration. It does not establish that any particular method will improve performance. The official test remains sealed until the final protocol is frozen.

### Data modifications

- preprocessing: NONE
- balancing: NONE
- sampling: NONE
- augmentation: NONE
- synthetic data: NONE
