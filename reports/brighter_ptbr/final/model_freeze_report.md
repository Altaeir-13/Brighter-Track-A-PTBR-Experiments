# BRIGHTER PT-BR — model freeze report

State: **FROZEN_PRE_TEST**. Test: **SEALED — NOT LOADED OR EVALUATED**.

## 1. Frozen dataset

BRIGHTER Track A PT-BR revision `419566ac8f46f951b51584be36c13e5363008144`; official train+dev only, 2,426 examples. Development IDs SHA-256: `d1e2b0b539c5085e7e67eda3708ff05d301a387bf956f5e2a412c82718b7916b`.

Data policy: preprocessing, label changes, sampling, augmentation, translation, synthetic/external data = NONE; Track B = OUT OF SCOPE.

## 2. Final systems and training

T2: `melll-uff/bertweetbr`, ASL gamma_pos=0, gamma_neg=4, clip=0.05; 4 epochs, LR 2e-5, AdamW, weight decay 0.01, warmup 0.1, effective batch 16, max length 128, seed 42. Trained on 2426 examples in 138.0s; peak VRAM 2.30 GiB; 134,907,654 parameters.

T3: `neuralmind/bert-large-portuguese-cased`, BCEWithLogitsLoss; same frozen optimizer/schedule/epochs/max length/effective batch/seed. Trained on 2426 examples in 2173.5s; peak VRAM 5.58 GiB; 334,402,566 parameters.

No validation split, early stopping, tuning, additional seed, or test observation was used.

## 3. Frozen deployment decision

Alpha T2 = **0.70**; T3 weight = **0.30**.

| Label | Final threshold | Nested mean ± SD | Nested range | Consistent |
|---|---:|---:|---:|---:|
| anger | 0.67 | 0.692 ± 0.045 | 0.62–0.74 | yes |
| disgust | 0.36 | 0.454 ± 0.043 | 0.39–0.49 | no |
| fear | 0.43 | 0.464 ± 0.039 | 0.41–0.51 | yes |
| joy | 0.65 | 0.694 ± 0.034 | 0.64–0.73 | yes |
| sadness | 0.58 | 0.600 ± 0.041 | 0.54–0.65 | yes |
| surprise | 0.49 | 0.472 ± 0.030 | 0.43–0.51 | yes |

Alpha nested range was 0.75–0.95; global deployment alpha 0.70 lies outside it. This value was not manually changed. The likely reason is the less regularized full-OOF optimization, which uses all 2,426 OOF labels for deployment selection rather than five separate outer-train calibration sets. This is recorded as a deployment stability warning.

## 4. Scientific estimate versus deployment calibration

The frozen unbiased development estimate remains nested EN2 Macro-F1 = **0.5691**. Global deployment calibration Macro-F1 = 0.5897 is labeled: **DEPLOYMENT CALIBRATION ONLY — HYPERPARAMETERS SELECTED ON THE SAME OOF SET. NOT AN UNBIASED PERFORMANCE ESTIMATE.** It must never replace the nested estimate.

## 5. Test protocol

The pre-registered protocol is in `final_test_protocol.md`. EN2 must be reported as primary and T2 standalone as secondary. Alpha, thresholds, models, label order, and metrics cannot change after test access. T3 is a component, not a third candidate.

## 6. Hashes and environment

T2 model manifest SHA-256: `ca9a9d55c72e224e9c893cf200f111373ff2dfba52903da4e5afede3932be01b`.
T3 model manifest SHA-256: `9002c34f5e31282ad4599d87fe3cb86797859de1fc3e87f2aebd46fe13ec5549`.
Final EN2 config SHA-256: `71f1cb1b486380bd7a07537dc7eafb6bcce144a84325a45e2d52114a80c5f8ef`.
Final T2 config SHA-256: `3a8152b86c7a34b404faf495a49f615f5a6270ab27193ffeac48c7b9679d85be`.

Hardware: NVIDIA GeForce RTX 3060 Ti; declared 8 GB VRAM and 32 GB RAM. Software versions are persisted in the final manifest.

## 7. Immutability statement

The search space is closed. No decision may change after the test is opened. This task did not authorize or perform test access.
