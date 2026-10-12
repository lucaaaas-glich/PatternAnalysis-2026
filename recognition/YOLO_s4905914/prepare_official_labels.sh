#!/bin/bash
# CPU-only full preparation. Submit from repository root after reference download.
#SBATCH --job-name=lesion-label-prep
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=official_label_prep_%j.out
#SBATCH --error=official_label_prep_%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
printf 'Official label preparation job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
python recognition/YOLO_s4905914/prepare_official_labels.py \
  --output "outputs/official_labels/school_nn_${SLURM_JOB_ID}"
