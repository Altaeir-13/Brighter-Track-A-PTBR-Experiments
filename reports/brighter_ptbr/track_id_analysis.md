# BRIGHTER PT-BR — Track ID analysis

- Revision: `419566ac8f46f951b51584be36c13e5363008144`
- Analysis uses original IDs without normalization.

## train

- Rows: 2226
- First IDs: `ptbr_train_track_a_00001, ptbr_train_track_a_00002, ptbr_train_track_a_00003, ptbr_train_track_a_00004, ptbr_train_track_a_00005, ptbr_train_track_a_00006, ptbr_train_track_a_00007, ptbr_train_track_a_00008, ptbr_train_track_a_00009, ptbr_train_track_a_00010`
- Last IDs: `ptbr_train_track_a_02217, ptbr_train_track_a_02218, ptbr_train_track_a_02219, ptbr_train_track_a_02220, ptbr_train_track_a_02221, ptbr_train_track_a_02222, ptbr_train_track_a_02223, ptbr_train_track_a_02224, ptbr_train_track_a_02225, ptbr_train_track_a_02226`
- Unparsed IDs: 0
- Split mismatches: 0
- Configuration mismatches: 0

| Prefix | Track | Count | First ID | Last ID | Suffix range | Width | Contiguous from 1 |
|---|---|---:|---|---|---|---|---|
| `ptbr_train_track_a` | A | 2226 | `ptbr_train_track_a_00001` | `ptbr_train_track_a_02226` | 1–2226 | [5] | True |

## dev

- Rows: 400
- First IDs: `ptbr_dev_track_a_00001, ptbr_dev_track_a_00002, ptbr_dev_track_a_00003, ptbr_dev_track_a_00004, ptbr_dev_track_a_00005, ptbr_dev_track_a_00006, ptbr_dev_track_a_00007, ptbr_dev_track_a_00008, ptbr_dev_track_a_00009, ptbr_dev_track_a_00010`
- Last IDs: `ptbr_dev_track_c_00191, ptbr_dev_track_c_00192, ptbr_dev_track_c_00193, ptbr_dev_track_c_00194, ptbr_dev_track_c_00195, ptbr_dev_track_c_00196, ptbr_dev_track_c_00197, ptbr_dev_track_c_00198, ptbr_dev_track_c_00199, ptbr_dev_track_c_00200`
- Unparsed IDs: 0
- Split mismatches: 0
- Configuration mismatches: 0

| Prefix | Track | Count | First ID | Last ID | Suffix range | Width | Contiguous from 1 |
|---|---|---:|---|---|---|---|---|
| `ptbr_dev_track_a` | A | 200 | `ptbr_dev_track_a_00001` | `ptbr_dev_track_a_00200` | 1–200 | [5] | True |
| `ptbr_dev_track_c` | C | 200 | `ptbr_dev_track_c_00001` | `ptbr_dev_track_c_00200` | 1–200 | [5] | True |

## test

- Rows: 4452
- First IDs: `ptbr_test_track_a_00001, ptbr_test_track_a_00002, ptbr_test_track_a_00003, ptbr_test_track_a_00004, ptbr_test_track_a_00005, ptbr_test_track_a_00006, ptbr_test_track_a_00007, ptbr_test_track_a_00008, ptbr_test_track_a_00009, ptbr_test_track_a_00010`
- Last IDs: `ptbr_test_track_c_02217, ptbr_test_track_c_02218, ptbr_test_track_c_02219, ptbr_test_track_c_02220, ptbr_test_track_c_02221, ptbr_test_track_c_02222, ptbr_test_track_c_02223, ptbr_test_track_c_02224, ptbr_test_track_c_02225, ptbr_test_track_c_02226`
- Unparsed IDs: 0
- Split mismatches: 0
- Configuration mismatches: 0

| Prefix | Track | Count | First ID | Last ID | Suffix range | Width | Contiguous from 1 |
|---|---|---:|---|---|---|---|---|
| `ptbr_test_track_a` | A | 2226 | `ptbr_test_track_a_00001` | `ptbr_test_track_a_02226` | 1–2226 | [5] | True |
| `ptbr_test_track_c` | C | 2226 | `ptbr_test_track_c_00001` | `ptbr_test_track_c_02226` | 1–2226 | [5] | True |

## Conclusion

Track A is directly and reproducibly identifiable from the original `id` field. Every ID matches `ptbr_<split>_track_<track>_<numeric suffix>`; no fallback based on text deduplication is needed. `train` contains Track A only, while `dev` and `test` contain complete Track A and Track C partitions.
