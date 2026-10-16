# COMP3710 Project 3 restart: YOLOv8 lesion detection

This is an experimental implementation in preparation, not a completed assignment report. The real ISIC cohort has not been retrained or scored with this code. The code integrates Ultralytics 8.3.221; the YOLO network, loss and training engine are third-party implementations, not my own network design. Acceptance of this packaging under the course's `modules.py` requirement is still awaiting clarification.

## Data and evaluation boundary

The candidate study uses all 1,098 Task 1 images with recorded lesion IDs: 824 benign and 274 malignant. The other 1,496 images are excluded from every phase. The single Indeterminate image is outside this cohort; it is not relabelled benign. The new head has exactly two classes, Benign=0 and Malignant=1.

| Split | Benign | Malignant | Total |
| --- | ---: | ---: | ---: |
| Training | 576 | 192 | 768 |
| Validation | 124 | 41 | 165 |
| Supplementary reassessment | 124 | 41 | 165 |

`plan_group_splits.py` forms connected components from all recorded lesion and patient links before selecting the cohort. It then assigns whole groups, using seed 3710 and approximate 70/15/15 class and size balance. No missing medical IDs are imputed. There are no overlapping recorded lesion/patient IDs between these sets, but 708 selected images still lack patient IDs. Full patient independence is not established.

The original image-level pilot and its test results were already inspected. Repartitioning or deleting old local artifacts does not undo that exposure. The new evaluation is supplementary reassessment, not a fresh blind test. Both models must start from the original COCO weights; old fitted checkpoints cannot be transferred into this comparison. The teacher has not yet confirmed whether the restricted cohort is acceptable.

## Working code

| File | Responsibility |
| --- | --- |
| `modules.py` | Constructs the actual two-class PyTorch DetectionModel using the pinned upstream implementation; checks loaded class mapping. |
| `dataset.py` | Checks manifest hashes, labels and known groups; verifies image decoding; stages image links and label copies; supplies the loader actually called by the trainer. |
| `train.py` | Produces a plan, trains and saves models, validates, and runs locked supplementary reassessment. |
| `predict.py` | Loads the selected saved checkpoint, predicts one image and writes an overlay and JSON. |
| `evaluation.py` | Class-correct detection matching, separate image alerts, explicit IoU denominators, validation threshold selection, and consistent NMS policy. |
| `experiment.json` | Shared n/s experiment draft, data hashes, original-weight hashes and operating-point rules. |
| `smoke_test.py` | Small synthetic end-to-end software check for both n and s. It produces no ISIC evidence. |
| `test_*.py` | Grouping, label, metric and experiment-boundary regression tests. |

The model builder and loader calls are recorded in each run's `project_model_interface.json` and `project_loader_calls.jsonl`. These show actual integration; matching required filenames alone does not establish course compliance. Model internals and upstream source locations are explained in the project-level `MODEL_COMPONENTS_zh.txt`.

## Experiment draft

YOLOv8n is the predeclared primary model and YOLOv8s the comparator. Both use their own hash-verified original COCO initialization. They share 640-pixel inputs, batch 16, FP32, seed 3710, 60 epochs, AdamW (lr0=0.001, beta1=0.9, weight decay=0.0005), a linear learning-rate schedule ending at 1% of its initial value, and 3 warm-up epochs. Early stopping is disabled. Nominal batch size is also 16, giving one optimizer update per batch: 48 batches/epoch and 2,880 planned updates per model. Actual update counts are saved. This budget is not a claim of convergence.

The initial augmentation policy is horizontal/vertical flips with probability 0.5 each. Mosaic, MixUp, CutMix, oversampling, color changes and affine distortion are off. Optional Albumentations is removed from the runtime transform list so installing it cannot silently add blur or CLAHE. Some disabled transform objects remain present in the library's composition with zero probability/gain. This conservative policy preserves color and whole-lesion context; no performance benefit has yet been demonstrated.

Both training validation and prediction use square, aspect-preserving preprocessing and class-aware NMS at 0.7, maximum 300 detections, and a score floor of 0.001. Unlike the library's default multi-label validation, our validator retains only the highest-scoring class per candidate, matching ordinary prediction. This is an explicit new protocol choice, so the resulting AP is not directly interchangeable with old default-validation results.

`best.pt` is selected by maximum validation macro mAP50:95 over the fixed budget; the pinned trainer retains the later epoch on an exact tie. That rule is the same for n and s. The final model is not selected after comparing supplementary reassessment results.

The operating threshold is selected only on validation using the fixed grid in `experiment.json`. The provisional classroom budget allows malignant alerts on at most 10% of benign validation images (at most 12 of 124). It has no established clinical justification. Within that budget, select maximum malignant **detection** recall, then precision, fewer benign alerts, and finally the higher threshold. If no grid point is feasible, report that fact and do not issue a selection lock. Do not expand the grid after looking at reassessment results.

All score filtering uses `confidence > threshold`, matching the upstream NMS boundary. A detection TP additionally needs the correct class and IoU ≥ 0.5. Each GT can match at most one prediction. Wrong-class or badly located predictions can produce both an FP and an FN. Image-alert sensitivity does not require IoU and does not count actual biopsies.

AP uses the low-score sweep, not the chosen operating threshold. At the operating point, per-class IoU is reported separately for all GT with the best class-correct box (missing prediction=0), all GT ignoring class, and matched TPs only. The teacher's intended denominator for the IoU ≥ 0.8 course target remains unresolved. Confidence is not a calibrated probability of cancer.

## Running from the restart project root

The examples below assume `recognition/YOLO_s4905914/`, `data/known_lesion_v1/` and the source snapshots have been transferred together. Data and weights are ignored by Git; reproduce or separately supply the audited manifests and source snapshots. Do not commit them. The local macOS environment snapshot is recorded in `tested-environment-macos-py313.txt`. Linux/CUDA compatibility, GPU memory and timing still need a school-host preflight.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r recognition/YOLO_s4905914/requirements.txt
python recognition/YOLO_s4905914/fetch_weights.py --output weights/original_coco
python -m unittest discover -s recognition/YOLO_s4905914 -p 'test_*.py' -v
python recognition/YOLO_s4905914/train.py plan \
  --config recognition/YOLO_s4905914/experiment.json --prepared data/known_lesion_v1
```

Small CPU integration check (synthetic images only; choose an unused output path):

```bash
python recognition/YOLO_s4905914/smoke_test.py \
  --base-config recognition/YOLO_s4905914/experiment.json \
  --weights-dir weights/original_coco --output outputs/synthetic_smoke
```

Before real training, resolve cohort acceptance and the course questions, review the draft policy, and save `status="locked_before_training"` and `ready_for_full_training=true` in the experiment configuration. This is an explicit gate, not a claim that confirmation has already occurred. A different approved cohort requires new manifests, hashes, counts and a new configuration version; do not bypass the checks.

On the school host, stage the declared images and verify all files decode. The default uses the paths in the manifests. `--image-root /actual/image/directory` is available when the same source files are mounted elsewhere. Images are symlinked, never edited; labels/cache go in the new staging directory.

```bash
python recognition/YOLO_s4905914/dataset.py \
  --prepared data/known_lesion_v1 --config recognition/YOLO_s4905914/experiment.json \
  --output data/runtime_v1

python recognition/YOLO_s4905914/train.py fit \
  --config recognition/YOLO_s4905914/experiment.json --prepared data/known_lesion_v1 \
  --stage data/runtime_v1 --weights-dir weights/original_coco --model n \
  --output outputs/n_v1 --device 0
```

Run s using the same command with `--model s --output outputs/s_v1`. Use the school's approved GPU allocation procedure; these commands do not request a scheduler allocation. They refuse to overwrite an existing run directory. Training saves `best.pt`, `last.pt`, `args.yaml`, `results.csv`, loss/metric plots, the environment, source/config/data hashes and runtime measurements. CUDA peak allocation is not total device memory. Library timing is not complete mobile application latency.

Validate and select the operating point:

```bash
python recognition/YOLO_s4905914/train.py evaluate \
  --config recognition/YOLO_s4905914/experiment.json --prepared data/known_lesion_v1 \
  --stage data/runtime_v1 --checkpoint outputs/n_v1/fit/weights/best.pt \
  --split val --output outputs/n_v1_validation --device 0
```

Review validation, confirm the declared comparison is complete, and only then run supplementary reassessment with the generated lock. This command cannot sweep thresholds and checks the checkpoint, configuration, membership and recorded training provenance:

```bash
python recognition/YOLO_s4905914/train.py evaluate \
  --config recognition/YOLO_s4905914/experiment.json --prepared data/known_lesion_v1 \
  --stage data/runtime_v1 --checkpoint outputs/n_v1/fit/weights/best.pt \
  --split reassessment --lock outputs/n_v1_validation/selection_lock.json \
  --output outputs/n_v1_reassessment --device 0

python recognition/YOLO_s4905914/predict.py \
  --config recognition/YOLO_s4905914/experiment.json \
  --checkpoint outputs/n_v1/fit/weights/best.pt \
  --lock outputs/n_v1_validation/selection_lock.json \
  --image /path/to/example.jpg --output outputs/example_prediction --device 0
```

## Checks completed locally

28 unit tests passed. Both n and s completed a one-epoch CPU run on 8 synthetic training images, with 4 synthetic validation images and 4 synthetic reassessment images. Each run performed 4 optimizer steps; saved/reloaded two-class checkpoints; produced a finite (1, 6, 8400) decoded output for a 640×640 input; and completed validation selection, locked reassessment and prediction overlays. These synthetic metrics are not ISIC results. `IMPLEMENTATION_VERIFICATION.json` at the restart project root records the evidence and outstanding checks. Eight installed upstream source files matched the downloaded v8.3.221 tag exactly.

## Work still required for submission

The new cohort's GPU preflight, real training, validation comparison, locked reassessment, failure-case analysis and final report are not complete. A local software smoke test does not establish the feasibility check-off hurdle. Check-off status, the current deadline, component packaging, cohort acceptance and IoU aggregation still need confirmation. A GitHub PR, final README/PDF and the later review response are also outstanding.

This repository records progressive local implementation work. It has not been pushed or submitted. The final report must describe actual results and the history of exposed data; this preparation README is not a substitute.

## Sources and attribution

- [Pinned Ultralytics source, v8.3.221](https://github.com/ultralytics/ultralytics/tree/v8.3.221). The network, losses, datasets, training engine and AP implementation remain upstream AGPL-3.0 components.
- [Pinned detection trainer](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/models/yolo/detect/train.py), [validator](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/models/yolo/detect/val.py), and [metrics](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/utils/metrics.py).
- [Official original-weight assets](https://github.com/ultralytics/assets/releases/tag/v8.3.0). `weights_registry.json` records the downloaded SHA-256 values; these are locally recorded checksums, not publisher signatures.
- Course assessment draft supplied by the student, pages 7 and 12–13. Local filename says v2.3; cover says Draft/Version 2.2. Latest teaching announcements take precedence.
