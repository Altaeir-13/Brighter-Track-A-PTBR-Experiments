# B1/B2 OOF threshold diagnostics

**DIAGNOSTIC ONLY — THRESHOLD SELECTED AND EVALUATED ON SAME OOF**

The sweep spans 0.01–0.90 in increments of 0.01. These values describe the existing OOF predictions; they are not leakage-safe estimates of a calibrated model.

| Model | Emotion | AP | Best diagnostic threshold | Maximum diagnostic F1 | Precision | Recall | Predicted + | Gold + | Max score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | anger | 0.7538 | 0.35 | 0.7007 | 0.6498 | 0.7603 | 908 | 776 | 0.9285 |
| B1 | disgust | 0.0937 | 0.09 | 0.1838 | 0.1108 | 0.5375 | 388 | 80 | 0.2604 |
| B1 | fear | 0.0873 | 0.07 | 0.1494 | 0.0875 | 0.5128 | 686 | 117 | 0.2404 |
| B1 | joy | 0.8199 | 0.46 | 0.7615 | 0.7609 | 0.7621 | 644 | 643 | 0.9601 |
| B1 | sadness | 0.5684 | 0.26 | 0.5732 | 0.5066 | 0.6600 | 456 | 350 | 0.8675 |
| B1 | surprise | 0.1547 | 0.08 | 0.1678 | 0.0972 | 0.6149 | 1101 | 174 | 0.2460 |
| B2 | anger | 0.7969 | 0.36 | 0.7239 | 0.7040 | 0.7448 | 821 | 776 | 0.9868 |
| B2 | disgust | 0.1550 | 0.11 | 0.2207 | 0.1524 | 0.4000 | 210 | 80 | 0.3967 |
| B2 | fear | 0.3102 | 0.18 | 0.3604 | 0.3810 | 0.3419 | 105 | 117 | 0.6594 |
| B2 | joy | 0.8359 | 0.30 | 0.7707 | 0.7348 | 0.8103 | 709 | 643 | 0.9869 |
| B2 | sadness | 0.6649 | 0.40 | 0.6059 | 0.5950 | 0.6171 | 363 | 350 | 0.9767 |
| B2 | surprise | 0.4528 | 0.17 | 0.4764 | 0.4375 | 0.5230 | 208 | 174 | 0.8856 |

Thresholds selected here must not be carried into outer-fold evaluation. Confirmatory calibration requires an inner calibration split/OOF process inside each outer training fold.
