#!/bin/bash
#SBATCH --job-name=lesion-yolo-smoke
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=yolo_smoke_%j.out
#SBATCH --error=yolo_smoke_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
export YOLO_OFFLINE=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-yolo/bin/python recognition/YOLO_s4905914/train.py \
  --data outputs/yolo_runtime/image_level_3710_v1/data.yaml \
  --weights weights/yolov8n.pt \
  --output "outputs/smoke/yolov8n_${SLURM_JOB_ID}"
