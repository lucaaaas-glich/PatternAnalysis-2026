#!/bin/bash
# Submit from repository root. CPU-only input audit, not YOLO training.
#SBATCH --job-name=lesion-data-audit
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=school_data_audit_%j.out
#SBATCH --error=school_data_audit_%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"
test -f recognition/YOLO_s4905914/audit_data.py
# Reuse the previously verified course environment. No GPU is requested.
# Omit an explicit memory request, matching the working course setup.
source "$HOME/miniconda3/bin/activate"
conda activate torch
printf 'Input audit job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
python recognition/YOLO_s4905914/audit_data.py \
  --images /home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2 \
  --masks /home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2 \
  --metadata-root /home/groups/comp3710/ISIC2018 \
  --report "outputs/data_audit/school_training_${SLURM_JOB_ID}.json"
