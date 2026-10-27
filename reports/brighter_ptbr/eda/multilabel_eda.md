# BRIGHTER PT-BR Track A — Multilabel EDA

## 1. Dataset Version

Dataset: `brighter-dataset/BRIGHTER-emotion-categories`; configuration: `ptbr`; Track A; revision `419566ac8f46f951b51584be36c13e5363008144`.

## 2. Experimental Protocol

`train + dev` may support methodological development. **TEST SET — DESCRIPTIVE ONLY.** Test statistics below document the corpus and must not guide model, loss, threshold, preprocessing, augmentation, sampling, or hyperparameter choices.

No dataset content was modified by this analysis.

## 3. Label Distribution

| Emotion | Train % | Dev % | Test % (descriptive only) |
|---|---|---|---|
| anger | 32.26 | 29.00 | 31.40 |
| disgust | 3.37 | 2.50 | 3.19 |
| fear | 4.90 | 4.00 | 6.24 |
| joy | 26.10 | 31.00 | 24.93 |
| sadness | 14.47 | 14.00 | 15.41 |
| surprise | 6.87 | 10.50 | 7.05 |

### train

| Emotion | Positive | Negative | Prevalence (%) | Pos/Neg ratio |
|---|---|---|---|---|
| anger | 718 | 1508 | 32.26 | 0.4761 |
| disgust | 75 | 2151 | 3.37 | 0.0349 |
| fear | 109 | 2117 | 4.90 | 0.0515 |
| joy | 581 | 1645 | 26.10 | 0.3532 |
| sadness | 322 | 1904 | 14.47 | 0.1691 |
| surprise | 153 | 2073 | 6.87 | 0.0738 |

### dev

| Emotion | Positive | Negative | Prevalence (%) | Pos/Neg ratio |
|---|---|---|---|---|
| anger | 58 | 142 | 29.00 | 0.4085 |
| disgust | 5 | 195 | 2.50 | 0.0256 |
| fear | 8 | 192 | 4.00 | 0.0417 |
| joy | 62 | 138 | 31.00 | 0.4493 |
| sadness | 28 | 172 | 14.00 | 0.1628 |
| surprise | 21 | 179 | 10.50 | 0.1173 |

### test — TEST SET — DESCRIPTIVE ONLY

| Emotion | Positive | Negative | Prevalence (%) | Pos/Neg ratio |
|---|---|---|---|---|
| anger | 699 | 1527 | 31.40 | 0.4578 |
| disgust | 71 | 2155 | 3.19 | 0.0329 |
| fear | 139 | 2087 | 6.24 | 0.0666 |
| joy | 555 | 1671 | 24.93 | 0.3321 |
| sadness | 343 | 1883 | 15.41 | 0.1822 |
| surprise | 157 | 2069 | 7.05 | 0.0759 |

## 4. Zero-, Single- and Multi-Label Instances

| Split | Zero | Single | Multi (≥2) |
|---|---|---|---|
| train | 632 | 1261 | 333 |
| dev | 46 | 129 | 25 |
| test | 616 | 1280 | 330 |

The six all-zero outputs are treated as `none`; no seventh `neutral` class was created.

## 5. Label Cardinality and Density

| Split | Cardinality | Density | Emotional cardinality | Emotional density |
|---|---|---|---|---|
| train | 0.8796 | 0.1466 | 1.2284 | 0.2047 |
| dev | 0.9100 | 0.1517 | 1.1818 | 0.1970 |
| test | 0.8823 | 0.1471 | 1.2199 | 0.2033 |

## 6. Labelset Distribution

Training observes **32 / 64** possible labelsets; 8 occur once and 12 occur fewer than five times. The most common labelset is `none` (632 examples). See `labelsets.csv` and the training UpSet plot.

## 7. Label Co-occurrence

Co-occurrence matrices were computed for train and dev only. Diagonals equal label-positive counts. No test co-occurrence analysis is used for methodological interpretation.

## 8. Conditional Relationships

The ordered training matrix reports `P(column label | row label)`, so opposing directions are intentionally distinct. Training Jaccard and lift matrices provide symmetric overlap and independence-relative views.

## 9. Label Imbalance

Training MeanIR is **4.2198**. Labels above MeanIR are: **disgust, fear, surprise**. SCUMBLE is **0.037882**, computed on 1594 emotional examples; 632 zero-label examples are excluded because the original active-label formula is undefined for an empty labelset.

SCUMBLE uses `1 − geometric_mean(IRLbl_active) / arithmetic_mean(IRLbl_active)` per non-empty instance, then averages instance scores. SCUMBLELbl averages those scores over instances containing each label. This follows Charte et al., [original HAIS formulation](https://doi.org/10.1007/978-3-319-07617-1_10) and the [extended analysis](https://doi.org/10.1016/j.neucom.2016.08.158).

## 10. Minority/Majority Concurrence

| Minority label | Positive | Alone | With majority | Most frequent co-label | P(co-label | minority) |
|---|---|---|---|---|---|
| disgust | 75 | 8 | 65 | anger | 0.8000 |
| fear | 109 | 52 | 48 | anger | 0.2569 |
| surprise | 153 | 61 | 83 | anger | 0.3072 |

## 11. Text Characteristics

Training text lengths are summarized without changing text. Emoji occur in 0 texts (0.00%), with 0 emoji occurrences. Social-media indicators include URLs in 0 texts, mentions in 0, hashtags in 0, and non-ASCII characters in 1719.

## 12. Token Length Analysis

Tokenizer: `neuralmind/bert-base-portuguese-cased` at revision `94d69c95f98f7d5b2a8700c420230ae10def0baa`, with special tokens and no truncation. Training token p95=76.0, p99=144.8, max=585. >64: 191 (8.58%); >128: 28 (1.26%); >256: 6 (0.27%); >512: 2 (0.09%).

## 13. Train–Dev Distribution Comparison

Largest absolute label-prevalence difference: **joy** at **4.90 percentage points**. Rare dev positive counts requiring caution: disgust=5, fear=8.

## 14. Data Quality Notes

The known exact train–test overlap remains untouched; see `../cross_split_leakage.md`. It was not used for any methodological selection. Curated-input SHA-256 hashes were verified before analysis.

## 15. Implications for Training

- Imbalance is relevant: `disgust, fear, surprise` exceed MeanIR; the maximum IRLbl is 9.57.
- The observed SCUMBLE and per-minority concurrence counts indicate that frequency and concurrence should both be monitored; simple oversampling can replicate majority co-labels and is therefore not automatically safe.
- Weighted BCE merits controlled train/dev evaluation because positive frequencies differ substantially. Focal Loss and Asymmetric Loss also merit comparison, but EDA alone does not establish a winner.
- Augmentation is not yet justified by descriptive statistics alone. If tested later, it should be an ablation based only on train/dev.
- Aggressive text preprocessing is not justified by this EDA: textual emoticons, repeated punctuation, numbers, HTML-like entities, and non-ASCII characters are present and may carry emotion information; emoji, URLs, mentions, hashtags, and uppercase signals were not observed in train.
- Token-length evidence supports evaluating `MAX_LENGTH` candidates [128, 256, 512]; the final value must be selected on train/dev efficiency and validation behavior, never test.
- The official dev set has only 200 examples and very small counts for some labels (disgust=5, fear=8). Per-label threshold calibration may be unstable; out-of-fold train predictions should be considered alongside the official dev protocol.

### Dataset modifications

- text preprocessing: NONE
- label modification: NONE
- balancing: NONE
- sampling: NONE
- augmentation: NONE
- synthetic data: NONE
