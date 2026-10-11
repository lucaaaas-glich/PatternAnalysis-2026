# Skin Lesion Detection with YOLO

COMP3710 Assignment 3 — Project 3 (Normal Difficulty)

Student ID: s4905914

## Objective

Detect skin lesions in ISIC 2017/2018 dermoscopic images using YOLOv7 or newer. Segmentation masks will be converted to YOLO-format bounding boxes before training.

## Current status

- Repository and `topic-recognition` branch initialised.
- Windows CUDA environment verified with an NVIDIA GeForce RTX 4070 SUPER.
- ISIC2018 official validation audit: 100 images paired with 100 nonempty binary masks; dimensions match.
- Converted all 100 masks into single-lesion YOLO boxes, and visually inspected six seeded previews.
- Five boundary/pairing/export tests pass on Python 3.12.7, NumPy 2.3.3, Pillow 11.3.0.
- Formal training is planned on Rangpur A100 via Slurm; Windows handles development and initial annotation checks.
- Patient metadata, diagnosis labels, final splits, baseline training, evaluation, threshold analysis, and failure cases are pending.

## Data inspection and demo

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

ChatGPT/Codex assisted with repository setup, mask-to-box conversion, annotation checks, tests, and explanatory documentation.
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
