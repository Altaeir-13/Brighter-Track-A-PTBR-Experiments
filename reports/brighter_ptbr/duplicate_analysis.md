# BRIGHTER PT-BR — Duplicate and Track A × Track C analysis

- Revision: `419566ac8f46f951b51584be36c13e5363008144`
- Grouping key: exact `text_raw` value; no normalization was applied.
- Complete per-group audit: `metadata/brighter_ptbr/duplicate_groups.jsonl`.

## DEV

| Measure | Count |
|---|---:|
| Total rows | 400 |
| Unique texts | 200 |
| Duplicate groups | 200 |
| Groups size = 2 | 200 |
| Groups size > 2 | 0 |
| Same text + same six labels | 200 |
| Same text + conflicting labels | 0 |
| Exact full-row duplicates | 0 |
| Identical `emotions` | 200 |
| Track A / Track C pairs | 200 |
| Groups differing only in `id` | 200 |

### Representative examples

| Text SHA-256 (prefix) | IDs | Classification | Differing columns |
|---|---|---|---|
| `03fede6b857032f7` | `ptbr_dev_track_a_00001`, `ptbr_dev_track_c_00112` | same_text_same_labels | `id` |
| `282db96d0e352049` | `ptbr_dev_track_a_00002`, `ptbr_dev_track_c_00181` | same_text_same_labels | `id` |
| `62d79e574c5868c6` | `ptbr_dev_track_a_00003`, `ptbr_dev_track_c_00081` | same_text_same_labels | `id` |

## TEST

| Measure | Count |
|---|---:|
| Total rows | 4452 |
| Unique texts | 2226 |
| Duplicate groups | 2226 |
| Groups size = 2 | 2226 |
| Groups size > 2 | 0 |
| Same text + same six labels | 2226 |
| Same text + conflicting labels | 0 |
| Exact full-row duplicates | 0 |
| Identical `emotions` | 2226 |
| Track A / Track C pairs | 2226 |
| Groups differing only in `id` | 2226 |

### Representative examples

| Text SHA-256 (prefix) | IDs | Classification | Differing columns |
|---|---|---|---|
| `ebd9ab1b0beeaf1f` | `ptbr_test_track_a_00001`, `ptbr_test_track_c_01421` | same_text_same_labels | `id` |
| `4c869507b535c3d6` | `ptbr_test_track_a_00002`, `ptbr_test_track_c_02079` | same_text_same_labels | `id` |
| `fc6fc32d87ded27c` | `ptbr_test_track_a_00003`, `ptbr_test_track_c_00697` | same_text_same_labels | `id` |

## Empirical conclusion

Every duplicated text group in both splits has exactly two rows: one explicit Track A ID and one explicit Track C ID. In every group, `text`, `text_raw`, the six binary labels, and `emotions` are identical; only `id` differs. Track C is therefore a complete ID-renamed reuse of the Track A instances in these PT-BR splits at this frozen revision.

## Curated Track A basic statistics

| Split | Rows | Unique IDs | Unique texts | Duplicate texts | Anger + | Disgust + | Fear + | Joy + | Sadness + | Surprise + | Zero-label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2226 | 2226 | 2226 | 0 | 718 | 75 | 109 | 581 | 322 | 153 | 632 |
| dev | 200 | 200 | 200 | 0 | 58 | 5 | 8 | 62 | 28 | 21 | 46 |
| test | 2226 | 2226 | 2226 | 0 | 699 | 71 | 139 | 555 | 343 | 157 | 616 |

> These are structural counts only. No IRLbl, MeanIR, co-occurrence, concurrence, correlation, preprocessing, or modeling was performed.
