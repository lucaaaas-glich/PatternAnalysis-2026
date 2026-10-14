#!/bin/bash
#SBATCH --job-name=lesion-threshold-val
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=yolo_threshold_%j.out
#SBATCH --error=yolo_threshold_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
export YOLO_OFFLINE=true
.venv-yolo/bin/python recognition/YOLO_s4905914/scan_thresholds.py \
  --data outputs/yolo_runtime/image_level_3710_v1/data.yaml \
  --baseline outputs/baseline/yolov8n_661165/train/weights/best.pt \
  --comparison outputs/comparison/yolov8s_661355/train/weights/best.pt \
  --output "outputs/thresholds/validation_${SLURM_JOB_ID}"
