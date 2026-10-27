# BRIGHTER PT-BR — Data inventory

## Dataset

- Dataset: `brighter-dataset/BRIGHTER-emotion-categories`
- Configuration: `ptbr`
- Revision SHA: `419566ac8f46f951b51584be36c13e5363008144`
- Download timestamp (UTC): `2026-08-26T02:44:11+00:00`
- Processing: `none`

## Splits

| Split | Number of instances |
|---|---:|
| train | 2226 |
| dev | 400 |
| test | 4452 |

## Columns

| Split | Column | Type | Missing values |
|---|---|---|---:|
| train | `id` | `str` | 0 |
| train | `text` | `str` | 0 |
| train | `text_raw` | `str` | 0 |
| train | `anger` | `int64` | 0 |
| train | `disgust` | `int64` | 0 |
| train | `fear` | `int64` | 0 |
| train | `joy` | `int64` | 0 |
| train | `sadness` | `int64` | 0 |
| train | `surprise` | `int64` | 0 |
| train | `emotions` | `object` | 0 |
| dev | `id` | `str` | 0 |
| dev | `text` | `str` | 0 |
| dev | `text_raw` | `str` | 0 |
| dev | `anger` | `int64` | 0 |
| dev | `disgust` | `int64` | 0 |
| dev | `fear` | `int64` | 0 |
| dev | `joy` | `int64` | 0 |
| dev | `sadness` | `int64` | 0 |
| dev | `surprise` | `int64` | 0 |
| dev | `emotions` | `object` | 0 |
| test | `id` | `str` | 0 |
| test | `text` | `str` | 0 |
| test | `text_raw` | `str` | 0 |
| test | `anger` | `int64` | 0 |
| test | `disgust` | `int64` | 0 |
| test | `fear` | `int64` | 0 |
| test | `joy` | `int64` | 0 |
| test | `sadness` | `int64` | 0 |
| test | `surprise` | `int64` | 0 |
| test | `emotions` | `object` | 0 |

## Basic label distribution

### train

| Emotion | Positive | Negative | Positive % |
|---|---:|---:|---:|
| anger | 718 | 1508 | 32.26% |
| disgust | 75 | 2151 | 3.37% |
| fear | 109 | 2117 | 4.90% |
| joy | 581 | 1645 | 26.10% |
| sadness | 322 | 1904 | 14.47% |
| surprise | 153 | 2073 | 6.87% |

### dev

| Emotion | Positive | Negative | Positive % |
|---|---:|---:|---:|
| anger | 116 | 284 | 29.00% |
| disgust | 10 | 390 | 2.50% |
| fear | 16 | 384 | 4.00% |
| joy | 124 | 276 | 31.00% |
| sadness | 56 | 344 | 14.00% |
| surprise | 42 | 358 | 10.50% |

### test

| Emotion | Positive | Negative | Positive % |
|---|---:|---:|---:|
| anger | 1398 | 3054 | 31.40% |
| disgust | 142 | 4310 | 3.19% |
| fear | 278 | 4174 | 6.24% |
| joy | 1110 | 3342 | 24.93% |
| sadness | 686 | 3766 | 15.41% |
| surprise | 314 | 4138 | 7.05% |

## Number of emotions per text

| Num. labels | train | dev | test |
|---:|---:|---:|---:|
| 0 | 632 | 92 | 1232 |
| 1 | 1261 | 258 | 2560 |
| 2 | 303 | 44 | 612 |
| 3 | 29 | 6 | 48 |
| 4 | 1 | 0 | 0 |
| 5 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 |

### Label cardinality

| Split | Cardinality |
|---|---:|
| train | 0.879605 |
| dev | 0.910000 |
| test | 0.882300 |

## Text length distributions

| Split | Unit | Min | Mean | Median | P95 | Max |
|---|---|---:|---:|---:|---:|---:|
| train | characters | 11 | 113.58 | 76.00 | 280.00 | 2665 |
| train | words | 3 | 20.74 | 14.00 | 53.00 | 463 |
| dev | characters | 15 | 101.17 | 68.50 | 267.10 | 889 |
| dev | words | 3 | 18.61 | 13.00 | 48.00 | 158 |
| test | characters | 10 | 110.99 | 76.00 | 280.00 | 1219 |
| test | words | 3 | 20.22 | 14.00 | 51.00 | 221 |

## Integrity

### train

- Missing IDs: 0
- Duplicated ID rows beyond the first occurrence: 0
- Exact duplicated text rows beyond the first occurrence: 0
- Null texts: 0
- Empty texts: 0
- Whitespace-only texts: 0
- Invalid or null label values: 0

### dev

- Missing IDs: 0
- Duplicated ID rows beyond the first occurrence: 0
- Exact duplicated text rows beyond the first occurrence: 200
- Null texts: 0
- Empty texts: 0
- Whitespace-only texts: 0
- Invalid or null label values: 0

### test

- Missing IDs: 0
- Duplicated ID rows beyond the first occurrence: 0
- Exact duplicated text rows beyond the first occurrence: 2226
- Null texts: 0
- Empty texts: 0
- Whitespace-only texts: 0
- Invalid or null label values: 0

### Cross-split overlap

| Pair | Shared IDs | Shared exact texts |
|---|---:|---:|
| train × dev | 0 | 0 |
| train × test | 0 | 1 |
| dev × test | 0 | 0 |

### Warnings and errors

- dev: 200 exact duplicated text rows beyond the first occurrence
- test: 2226 exact duplicated text rows beyond the first occurrence
- train__test: shared IDs=0, shared exact texts=1

> Factual inventory only. No preprocessing, balancing, augmentation, removal, or automatic correction was applied.
