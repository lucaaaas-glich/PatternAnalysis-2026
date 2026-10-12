# Skin Lesion Detection with YOLO

COMP3710 Assignment 3 — Project 3 (Normal Difficulty)

Student ID: s4905914

## Objective

Detect skin lesions in ISIC 2017/2018 dermoscopic images using YOLOv7 or newer. Segmentation masks will be converted to YOLO-format bounding boxes before training.

## Current status

Latest verified milestone (2026-10-13): CPU label-preparation job 655099 completed in 11m07s with exit code 0:0. The downloaded summary reports 2,594/2,594 valid pairs, 2,594 exported labels and zero issues. Local checks found 2,594 unique CSV IDs and zero label-format/normalised-coordinate/CSV-consistency errors. AI-assisted inspection of all six previews found boxes consistent with the displayed masks, including the smallest box (0.39096% image area) and a full-image box (100%). This is sample annotation QA, not exhaustive manual review or model evaluation. Patient metadata, diagnosis mapping and independent splits remain unresolved. Historical milestone notes below describe their status at the time, not the latest status.

- Repository and `topic-recognition` branch initialised.
- Windows CUDA environment verified with an NVIDIA GeForce RTX 4070 SUPER.
- ISIC2018 official validation audit: 100 images paired with 100 nonempty binary masks; dimensions match.
- Converted all 100 masks into single-lesion YOLO boxes, and visually inspected six seeded previews.
- Five boundary/pairing/export tests pass on Python 3.12.7, NumPy 2.3.3, Pillow 11.3.0.
- Rangpur A100 environment verified on 2026-10-12: job 652217, a100-3, COMPLETED in 45 seconds, exit code 0:0.
- Remote stack: Python 3.11.16, PyTorch 2.14.0+cu130, CUDA 13.0; NVIDIA A100-PCIE-40GB, CUDA and GPU matrix multiplication passed.
- Formal training is planned on Rangpur A100 via Slurm; Windows handles development and initial annotation checks. See [A100_GUIDE.md](A100_GUIDE.md) and `check_a100.sh` for reproducible environment verification.
- Patient metadata, diagnosis labels, final splits, baseline training, evaluation, threshold analysis, and failure cases are pending.
- School training audit job 653933 executed on a100-8 and ended after 3m51s with FAILED/1:0 because all 2,594 uniquely paired, readable, size-matched masks failed the current strict 0/1/255 pixel-value check. This is not evidence that lesions are absent or all data are corrupt. Actual mask modes/pixel ranges and preprocessing provenance must be inspected before changing label handling. No candidate CSV was found beneath the audited ISIC2018 root; metadata may exist elsewhere.
- Added `audit_data.py` for input integrity checks and CSV-header inventory without exporting labels or creating splits. Ten combined tests pass on bundled Python 3.12.14. The tool passed an executed audit of the existing 100 local validation pairs; the school training audit executed but failed the strict mask-value rule, as detailed above.
- The student executed a three-mask diagnostic: ISIC_0000000/0000001/0000003 are L-mode uint8, range 0–255, with 230/215/245 unique intensities. Added `diagnose_masks.py` to compare candidate thresholds and create previews without altering originals or exporting training labels. Four new tests passed at that milestone (14 combined); local binary validation data were used only to verify figure layout/output. School previews have since been executed and reviewed as detailed below; threshold approval and school-preprocessing provenance remain unverified.
- School previews were generated and downloaded; AI-assisted visual review found the main lesion regions retained at candidate threshold 128, with 0.55–1.01% intermediate pixels in these three samples. Official reference masks were downloaded from ISIC (26MB archive), verified as 0/255 for these IDs, and resized with NEAREST to the school image dimensions. Candidate-versus-reference box IoUs are 0.997653, 0.973333, and 0.990000; these measure annotation consistency, NOT model performance. `check_mask_reference.py` makes the check reproducible; 16 combined tests pass. Full-dataset original-mask preparation, patient metadata, diagnosis scope, and splits remain pending. No threshold was approved automatically.
- Official-mask full-preparation tools are ready: `prepare_official_labels.py`, its CPU Slurm worker, and `submit_official_labels.sh`. Eight new tests pass (24 combined), both shell scripts pass Bash syntax checks, and the real cached archive checksum and its 2,594 mask filenames were checked. On 2026-10-13 the student's terminal confirmed the reference checksum and submission of job 655099; completion and exported label counts have NOT yet been checked. Labels export only after every pair passes; sources remain unchanged, and class 0 remains localisation-only lesion.

## Data inspection and demo

See [PROJECT_MAP.md](PROJECT_MAP.md) for the living project mind map, verified lecture connections, current progress, and code-to-concept mapping. Update it at each milestone or when a new problem/operation changes the workflow.

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for the image/mask/box concepts, a worked coordinate example, and a demonstration explanation in Chinese.

Download the official validation image archive and matching Task 1 ground truth from:

- https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip
- https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip

Extract them under `data/raw/` at repository root. Install the data-only dependencies with `python -m pip install -r recognition/YOLO_s4905914/requirements-data.txt`.
Use the verified Python 3.12 interpreter, then run from repository root:

```text
python recognition/YOLO_s4905914/dataset.py --images data/raw/ISIC2018_Task1-2_Validation_Input --masks data/raw/ISIC2018_Task1_Validation_GroundTruth --output data/inspection/isic2018_val
python -m unittest discover -s recognition/YOLO_s4905914 -p test_dataset.py -v
```

Outputs are `summary.json`, `annotations.csv`, 100 label TXT files, and six preview PNGs inside the ignored output directory.
Each preview shows input, human mask, and the derived ground-truth box. It is not a prediction or an evaluation score.
The observed box area covers 2.15% to 99.58% of the image: lesion scale varies substantially.

The current class is `0 = lesion`, for annotation-pipeline verification only. The assignment also calls for classification performance; diagnosis metadata and category mapping must be established before the final model scope is fixed.
Official validation membership does not establish patient independence. No patient-separated train/validation/test split has yet been verified.
The inspection set must not be repurposed as the held-out final test set after tuning.

## Assessment requirements verified

The supplied draft specification permits pretrained YOLOv7-or-newer models, sets a test localization IoU target of at least 0.8, and requires a baseline comparison, diagnosis/classification consideration, patient-independent evaluation, and threshold/failure analysis.
It also requires a one-page feasibility review and compulsory tutor check-off, plus a PDF export of the final README for Turnitin.
Its midpoint consultation explicitly does not require a live demonstration, slides, or oral examination. Any separately announced demo requirements still need to be followed.

## Artificial Intelligence Usage Disclosure

ChatGPT/Codex assisted with repository setup, mask-to-box conversion, annotation checks, tests, explanatory documentation, and a living learning map checked against relevant lecture-transcript excerpts.
Verification so far consists of executed boundary/pairing/export tests and AI-assisted visual inspection of six real ISIC previews.
Student review and understanding of these results remain to be confirmed; no student audit, final training, or final metric is claimed here.
Dataset sources: [ISIC official data](https://challenge.isic-archive.com/data/) and Codella et al., [ISIC 2018 challenge](https://arxiv.org/abs/1902.03368).
Label format: [Ultralytics detection dataset documentation](https://docs.ultralytics.com/datasets/detect/).

## Planned evaluation

- mAP@0.50
- mAP@0.50:0.95
- Precision and recall
- Bounding-box IoU
- Inference speed / FPS
- Confidence-threshold sensitivity versus false-positive trade-off
- Three to five representative failure cases

Datasets, model weights, and generated experiment outputs are intentionally excluded from Git.
