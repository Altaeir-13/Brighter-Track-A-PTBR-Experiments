# BRIGHTER PT-BR development-fold distribution

Frozen revision: `419566ac8f46f951b51584be36c13e5363008144`. Development pool: official train + dev (2426 examples).

The official test set is **SEALED — NOT LOADED OR EVALUATED**. The documented train–test overlap is not used here.

Method: 5-fold iterative multilabel stratification; seed 42. Exact ID and text disjointness was verified for every train/validation partition.

| Fold | N | Zero-label | Multilabel | anger + | disgust + | fear + | joy + | sadness + | surprise + |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 485 | 133 | 67 | 155 | 16 | 23 | 128 | 70 | 35 |
| 1 | 485 | 138 | 77 | 155 | 16 | 24 | 129 | 70 | 34 |
| 2 | 486 | 138 | 72 | 155 | 16 | 23 | 129 | 70 | 35 |
| 3 | 485 | 129 | 67 | 155 | 16 | 23 | 128 | 70 | 35 |
| 4 | 485 | 140 | 75 | 156 | 16 | 24 | 129 | 70 | 35 |

## Prevalence (%)

| Fold | anger | disgust | fear | joy | sadness | surprise |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31.96 | 3.30 | 4.74 | 26.39 | 14.43 | 7.22 |
| 1 | 31.96 | 3.30 | 4.95 | 26.60 | 14.43 | 7.01 |
| 2 | 31.89 | 3.29 | 4.73 | 26.54 | 14.40 | 7.20 |
| 3 | 31.96 | 3.30 | 4.74 | 26.39 | 14.43 | 7.22 |
| 4 | 32.16 | 3.30 | 4.95 | 26.60 | 14.43 | 7.22 |
