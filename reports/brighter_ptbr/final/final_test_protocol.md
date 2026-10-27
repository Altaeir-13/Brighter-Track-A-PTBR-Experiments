# BRIGHTER PT-BR — pre-registered final-test protocol

Status: **PRE-REGISTERED; TEST SEALED AND NOT ACCESSED**.

This document was written before final full-development training and before any test access.

## 1. Declared systems

- PRIMARY: EN2 final — T2 BERTweetBR+ASL and T3 BERTimbau Large+BCE.
- SECONDARY: T2 standalone final.
- No T3 standalone, EN1, T1, Albertina, or other candidate may be evaluated as a competing final system.

## 2. Frozen decision rule

`p = 0.70 * p_T2 + 0.30 * p_T3`.

Threshold order: `anger, disgust, fear, joy, sadness, surprise`.

EN2 thresholds: `anger=0.67, disgust=0.36, fear=0.43, joy=0.65, sadness=0.58, surprise=0.49`.

T2 standalone thresholds: `anger=0.66, disgust=0.53, fear=0.52, joy=0.75, sadness=0.65, surprise=0.57`.

## 3. Metrics

Primary: Macro-F1. Secondary: Micro-F1, per-label Precision/Recall/F1, Hamming Loss, sample Jaccard, Subset Accuracy, Exact Neutral Accuracy, Neutral False Positive Rate, and label cardinality.

## 4. Immutable evaluation rules

1. No alpha change after test access.
2. No threshold calibration or selection on test.
3. No model/loss/seed change.
4. No choice between T2 and EN2 based on test; EN2 is primary and T2 is always reported secondarily.
5. Exactly one separately authorized final evaluation is permitted.
6. No preprocessing, label modification, sampling, augmentation, translation, synthetic/external data, or Track B.

## 5. Pre-test provenance

Development IDs: 2426; SHA-256 `d1e2b0b539c5085e7e67eda3708ff05d301a387bf956f5e2a412c82718b7916b`. OOF audit: 2426 unique IDs with matched labels/folds.

The unbiased pre-test development estimate remains nested EN2 Macro-F1 = 0.5691. Any full-OOF deployment calibration score is explicitly not an unbiased performance estimate.
