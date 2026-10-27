# BRIGHTER PT-BR — encoder × loss experiment

## 1. Experimental question

We tested whether social-media PT-BR pretraining (BERTweetBR) or a DeBERTa-family PT-BR encoder (Albertina) improves Track A multilabel emotion recognition, and whether Weighted Focal/ASL transfer across encoders.

## 2. Protocol

Only encoder and loss varied. The immutable 2,426-example development pool, exact persisted five folds (seed 42), AdamW/LR 2e-5, four epochs, weight decay 0.01, linear warmup, max length 128, effective batch 16, checkpoint rule and fixed threshold 0.50 were retained. No preprocessing, sampling, augmentation, translation, external/synthetic data, label changes, Track B, or test access occurred. Weighted Focal weights use only each fold's training subset; ASL is gamma_pos=0, gamma_neg=4, clip=0.05.

Albertina was run in FP32 because its DeBERTa attention mask overflows under both FP16 and BF16 autocast in the installed PyTorch/Transformers combination. This technical exception was established by real-batch forward/backward smoke tests before E4; all other protocol choices stayed fixed.

## 3. Models

BERTimbau Base/Large are Portuguese BERT controls. [BERTweetBR](https://huggingface.co/melll-uff/bertweetbr) is RoBERTa-like and pretrained on Brazilian tweets; its frozen tokenizer had optional normalization disabled, so raw text was preserved. [Albertina 100M PT-BR](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder) is a PT-BR DeBERTa-family encoder. All models emit six independent logits; sigmoid is used once only for probabilities.

## 4. Main results

| Encoder | Loss | Macro-F1 | Micro-F1 | AP disgust | F1 disgust | AP fear | F1 fear | AP surprise | F1 surprise | Neutral FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERTimbau Base (B1) | BCE | 0.3244 | 0.6122 | 0.0937 | 0.0000 | 0.0873 | 0.0000 | 0.1547 | 0.0000 | 0.2847 |
| BERTimbau Base (L3) | Weighted Focal | 0.4024 | 0.5745 | 0.1987 | 0.1980 | 0.2302 | 0.0923 | 0.3948 | 0.2883 | 0.2552 |
| BERTimbau Base (L4) | ASL | 0.4617 | 0.5873 | 0.1297 | 0.2160 | 0.2029 | 0.2706 | 0.3751 | 0.3657 | 0.7729 |
| BERTimbau Large (B2) | BCE | 0.3976 | 0.6461 | 0.1550 | 0.0000 | 0.3102 | 0.0650 | 0.4528 | 0.2642 | 0.2935 |
| BERTweetBR (E1) | BCE | 0.3263 | 0.6344 | 0.1675 | 0.0000 | 0.1273 | 0.0000 | 0.1480 | 0.0000 | 0.2670 |
| BERTweetBR (E2) | Weighted Focal | 0.3974 | 0.5901 | 0.2466 | 0.1474 | 0.3571 | 0.1527 | 0.4767 | 0.2512 | 0.2124 |
| BERTweetBR (E3) | ASL | 0.5258 | 0.6408 | 0.2341 | 0.2632 | 0.3656 | 0.3438 | 0.5524 | 0.4706 | 0.7257 |
| Albertina 100M PT-BR (E4) | BCE | 0.2804 | 0.5326 | 0.0980 | 0.0000 | 0.1121 | 0.0000 | 0.2110 | 0.0000 | 0.2965 |
| Albertina 100M PT-BR (E5) | Weighted Focal | 0.2664 | 0.4403 | 0.0945 | 0.0220 | 0.1200 | 0.0462 | 0.2287 | 0.1852 | 0.2050 |
| Albertina 100M PT-BR (E6) | ASL | 0.3953 | 0.5322 | 0.1142 | 0.1608 | 0.1191 | 0.1768 | 0.2393 | 0.2933 | 0.8230 |

## 5. Minority labels

E2 improved ranking over L3 for disgust (0.2466 vs 0.1987), fear (0.3571 vs 0.2302), and surprise (0.4767 vs 0.3948). Its disgust F1 fell to 0.1474: precision stayed 0.4667, but recall was only 0.0875. E5 did not transfer this benefit reliably and was fold-unstable.

E3 obtained disgust F1 0.2632, fear F1 0.3438, and surprise F1 0.4706; its strong Macro-F1 is therefore not just a majority-label effect.

## 6. Encoder effect

Under BCE, E1-B1 and E4-B1 isolate the approximate encoder changes. BERTweetBR was essentially neutral; Albertina was negative under the frozen optimization protocol.

## 7. Loss effect

Within BERTweetBR, Weighted Focal improves over BCE but ASL is much stronger. Within Albertina, Weighted Focal is negative and unstable, while ASL substantially rescues performance. Therefore loss efficacy is not encoder-invariant.

## 8. Interaction

| Contrast | Δ Macro-F1 | Δ Micro-F1 | Δ AP disgust | Δ AP fear | Δ AP surprise | Δ Neutral FP |
|---|---:|---:|---:|---:|---:|---:|
| E1 - B1 | +0.0019 | +0.0222 | +0.0738 | +0.0400 | -0.0067 | -0.0177 |
| E2 - E1 | +0.0711 | -0.0442 | +0.0791 | +0.2298 | +0.3286 | -0.0546 |
| E3 - E1 | +0.1995 | +0.0065 | +0.0666 | +0.2383 | +0.4043 | +0.4587 |
| E4 - B1 | -0.0441 | -0.0796 | +0.0043 | +0.0248 | +0.0562 | +0.0118 |
| E5 - E4 | -0.0140 | -0.0923 | -0.0035 | +0.0079 | +0.0178 | -0.0914 |
| E6 - E4 | +0.1149 | -0.0004 | +0.0162 | +0.0070 | +0.0283 | +0.5265 |
| E2 - L3 | -0.0050 | +0.0156 | +0.0479 | +0.1268 | +0.0819 | -0.0428 |
| E3 - L4 | +0.0641 | +0.0535 | +0.1044 | +0.1627 | +0.1773 | -0.0472 |
| E5 - L3 | -0.1360 | -0.1342 | -0.1042 | -0.1102 | -0.1660 | -0.0501 |
| E6 - L4 | -0.0664 | -0.0551 | -0.0155 | -0.0838 | -0.1358 | +0.0501 |

The divergent E2/E5 and E3/E6 contrasts demonstrate a material encoder × loss interaction rather than an additive encoder-only or loss-only effect.

## 9. Calibration behavior

This section describes threshold-decision prevalence, not formal probability calibration.

| Model | Gold cardinality | Predicted cardinality | Delta |
|---|---:|---:|---:|
| B1 | 0.8821 | 0.6463 | -0.2358 |
| L3 | 0.8821 | 0.6088 | -0.2733 |
| L4 | 0.8821 | 1.5111 | +0.6290 |
| B2 | 0.8821 | 0.6925 | -0.1896 |
| E1 | 0.8821 | 0.6072 | -0.2749 |
| E2 | 0.8821 | 0.5470 | -0.3351 |
| E3 | 0.8821 | 1.4477 | +0.5655 |
| E4 | 0.8821 | 0.5837 | -0.2984 |
| E5 | 0.8821 | 0.4509 | -0.4312 |
| E6 | 0.8821 | 1.5066 | +0.6245 |

**E3 ASL vs L4:** Neutral FP is 0.7257 vs 0.7729; predicted cardinality is 1.4477 vs gold 0.8821. Largest prevalence deltas: sadness +0.1406, anger +0.1257, joy +0.1129. The recall gain must therefore be read together with precision and neutral behavior.

**E6 ASL vs L4:** Neutral FP is 0.8230 vs 0.7729; predicted cardinality is 1.5066 vs gold 0.8821. Largest prevalence deltas: anger +0.1822, sadness +0.1764, joy +0.1521. The recall gain must therefore be read together with precision and neutral behavior.

**ASL conclusion.** E3 still superpredicts, but less than L4 (both cardinality delta and Neutral FP improve). Its recall increase is accompanied by higher minority F1/AP and higher Micro-F1, so the precision loss is compensated in aggregate, though decision behavior remains aggressive. E6 superpredicts slightly more than L4, ranks every minority label worse, and loses Macro/Micro-F1; its recall does not compensate for the precision and neutral-behavior degradation.

**Weighted Focal conclusion.** E2 raises AP for all three minorities and keeps Neutral FP below L3. Disgust precision remains high, but recall drops enough that disgust F1 falls; fear F1 improves, while surprise F1 falls despite improved AP. Thus BERTweetBR improves minority ranking more consistently than fixed-threshold recovery. E5 reduces minority AP/F1 and is not a successful transfer of L3.

## 10. Computational cost

| Model | Params | Training min | Mean/fold min | OOF inference s | Peak VRAM GiB | Physical × accum = effective | Examples/s |
|---|---:|---:|---:|---:|---:|---|---:|
| E1 | 134,907,654 | 8.46 | 1.69 | 4.30 | 2.30 | 8 × 2 = 16 | 76.44 |
| E2 | 134,907,654 | 8.84 | 1.77 | 4.99 | 2.30 | 8 × 2 = 16 | 73.16 |
| E3 | 134,907,654 | 8.50 | 1.70 | 4.62 | 2.30 | 8 × 2 = 16 | 76.12 |
| E4 | 139,196,934 | 16.24 | 3.25 | 10.47 | 2.32 | 8 × 2 = 16 | 39.84 |
| E5 | 139,196,934 | 17.42 | 3.48 | 10.95 | 2.32 | 8 × 2 = 16 | 37.13 |
| E6 | 139,196,934 | 16.22 | 3.24 | 10.45 | 2.32 | 8 × 2 = 16 | 39.89 |

## 11. Selection

Advance at most three systems:

1. **E3 BERTweetBR + ASL** — best Macro-F1 and strong fold stability/minority F1; retain the explicit overprediction caveat.
2. **E2 BERTweetBR + Weighted Focal** — lower Macro-F1 but best minority ranking and controlled Neutral FP, complementary to E3.
3. **B2 BERTimbau Large + BCE** — retained as a stronger-encoder, calibrated-behavior comparator and likely error-complementary control.

E4–E6 are not selected: Albertina does not offset its slower FP32 cost or lower performance. E5 is additionally unstable across folds.

## 12. Next experiment

Recommended but **not executed**: BERTimbau Large + Weighted Focal, followed by leakage-safe per-label threshold calibration for the selected systems. No ensemble, augmentation, translation, Track B, or final-test evaluation was run.

## Validation

All six new OOF artifacts contain exactly 2,426 unique examples with identical ordered IDs/folds, finite [0,1] probabilities, matching saved SHA-256 hashes, and unchanged train/dev hashes. The official test remains **SEALED — NOT LOADED OR EVALUATED**.
