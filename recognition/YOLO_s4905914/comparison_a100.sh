#!/bin/bash
#SBATCH --job-name=lesion-yolov8s-comparison
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=yolo_comparison_%j.out
#SBATCH --error=yolo_comparison_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
export YOLO_OFFLINE=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-yolo/bin/python recognition/YOLO_s4905914/train.py \
  --profile comparison \
  --data outputs/yolo_runtime/image_level_3710_v1/data.yaml \
  --weights weights/yolov8s.pt \
  --output "outputs/comparison/yolov8s_${SLURM_JOB_ID}"
