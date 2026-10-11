#!/bin/bash
# Adapted from the student's Demo2 train_oasis_vae_a100.sh:
# https://github.com/lucaaaas-glich/comp3710/blob/main/lab2-pattern-recognition/train_oasis_vae_a100.sh
# Submit from repository root: sbatch recognition/YOLO_s4905914/check_a100.sh
# Short environment check only; this does not train or evaluate a detector.
#SBATCH --job-name=lesion-gpu-check
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=lesion_gpu_check_%j.out
#SBATCH --error=lesion_gpu_check_%j.err

set -euo pipefail
# Rangpur's course nodes advertise RealMemory=1 to Slurm; use the partition
# default, as in the working Demo2 scripts, instead of requesting --mem=16G.
source "$HOME/miniconda3/bin/activate"
conda activate torch
printf 'Job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import json
import sys
from pathlib import Path
import torch

assert torch.cuda.is_available(), 'CUDA unavailable inside GPU allocation'
x = torch.ones((128, 128), device='cuda')
result = x @ x
torch.cuda.synchronize()
assert torch.all(result == 128).item(), 'GPU arithmetic check failed'
print(json.dumps({
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'cuda_available': True,
    'gpu_matmul_passed': True,
    'course_dataset_exists': Path('/home/groups/comp3710/ISIC2018').is_dir(),
}, indent=2))
PY
