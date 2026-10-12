#!/bin/bash
# Run in the school's SSH terminal from repository root, not Windows PowerShell.
# The small verified download runs here; full image decoding is scheduled by Slurm.
set -euo pipefail
test -f recognition/YOLO_s4905914/prepare_official_labels.py
source "$HOME/miniconda3/bin/activate"
conda activate torch
python recognition/YOLO_s4905914/prepare_official_labels.py --download-reference-only
sbatch recognition/YOLO_s4905914/prepare_official_labels.sh
