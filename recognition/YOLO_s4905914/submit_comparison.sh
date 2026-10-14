#!/bin/bash
# Download/verify weights on login node; train only in Slurm GPU allocation.
set -euo pipefail
test -x .venv-yolo/bin/python
test -f outputs/yolo_runtime/image_level_3710_v1/data.yaml
.venv-yolo/bin/python recognition/YOLO_s4905914/download_comparison_weights.py
sbatch recognition/YOLO_s4905914/comparison_a100.sh
