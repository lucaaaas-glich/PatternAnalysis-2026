# Frozen experiment results

Primary policy: YOLOv8n, confidence0.05, selected on validation and confirmed before test. Comparator: YOLOv8s at the same threshold. Both30epochs/640/batch16/seed3710/FP32; each original COCO pretrained checkpoint. Test job661498 COMPLETED/0:0 in58seconds; downloaded summary.json completed. These are image-level results with known/unknown lesion leakage, NOT patient/lesion-independent clinical evidence.

## Held-out standard detection AP

| Model | mAP50 | mAP50:95 | Benign AP50 | Malignant AP50 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.7042 | 0.5095 | 0.9099 | 0.4985 |
| YOLOv8s | 0.6945 | 0.4968 | 0.8951 | 0.4938 |

Library score-ranking AP uses standard rectangular validation. It is not accuracy or mean IoU; library automatic P/R are not the fixed0.05 policy. Small differences in one run do not establish statistical significance.

## Fixed0.05 policy (square predict, matching IoU>=0.5)

| Model/class | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| n/Benign | 291 | 115 | 20 | 0.7167 | 0.9357 |
| n/Malignant | 38 | 58 | 39 | 0.3958 | 0.4935 |
| s/Benign | 293 | 132 | 18 | 0.6894 | 0.9421 |
| s/Malignant | 38 | 54 | 39 | 0.4130 | 0.4935 |

FP includes unmatched/duplicate/wrong-class detections; it does not mean distinct patients or unnecessary biopsies.

Image malignant alerts (ignore localisation): n alerts41/77malignant and36/311benign, misses36malignant; sensitivity0.5325 and benign false-alert rate0.1158. s alerts38/77malignant and37/311benign, misses39malignant; sensitivity0.4935 and benign false-alert rate0.1190. This is an experimental referral proxy, not biopsy burden measured in a clinic. No automatic safety-clear or treatment recommendation is supported.

## IoU denominators

| Model/class | Same-class best IoU, all GT including missing=0 | Class-agnostic best IoU, all GT | Mean IoU of matched TP only |
| --- | ---: | ---: | ---: |
| n/Benign | 0.8198 | 0.8551 | 0.8656 |
| n/Malignant | 0.4416 | 0.8574 | 0.8384 |
| s/Benign | 0.8204 | 0.8547 | 0.8617 |
| s/Malignant | 0.4092 | 0.8472 | 0.8233 |

Best-box IoU is an oracle diagnostic (selects highest overlap), ignores extra boxes and must be paired with FP/Recall. TP-only IoU excludes misses. The high class-agnostic/TP-only malignant values must not be used to claim overall class-correct IoU>=0.8 success. Classification is a major weakness; causal visual triggers remain to be examined. Indeterminate has no held-out truth and cannot be evaluated.

## Speed scope

| Model | Parameters before fusion | Mean latency (ms) | Median latency (ms) | FPS |
| --- | ---: | ---: | ---: | ---: |
| n | 3,011,433 | 12.77 | 10.24 | 78.33 |
| s | 11,136,761 | 12.91 | 10.37 | 77.49 |

A100 FP32 batch1,640square;10warmup calls then50predecoded images; CUDA synchronisation. Includes Python predict API/preprocessing/forward/NMS, excludes disk decoding/rendering/network/JSON serialisation. Not mobile/Mac or pure-forward throughput. Single short measurement and system effects limit speed conclusions; no meaningful speed superiority established from this small difference.

## Five failure cases reviewed

Deterministically selected primary-model errors, not a representative random sample:

- ISIC_0014897: malignant wrong-class candidate.
- ISIC_0009875: benign false-alert candidate.
- ISIC_0015071: malignant alert with failed localisation match.
- ISIC_0012178: duplicate same-class candidate.
- ISIC_0012758: malignant miss selected by small GT pixel-box area (not physical size).

Five downloaded GT/n/s figures and cached predictions were reviewed on 2026-10-15. [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md) records measured errors, visible features, hypotheses and demo explanations. The fifth case is a malignant-class miss despite good localisation (n IoU0.833, s0.867), not an absent small-lesion box. The third shows that an image alert can still fail localisation matching; s has a valid malignant match plus a conflicting benign FP. One-to-one matching penalises duplicate boxes.

Only existing predictions were inspected; no threshold/model tuning after test. Hair/ink/contrast causes are not established. Original records and figures remain in ignored outputs. Student understanding and final README/demo packaging are still pending.
