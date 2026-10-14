#!/bin/bash
# Run on login node from repository root. Dependency setup only, no training.
set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
test -f recognition/YOLO_s4905914/assemble_yolo.py
if test -e .venv-yolo; then
  printf 'Existing .venv-yolo preserved. Stop and inspect it; do not overwrite.\n'
  exit 1
fi
# Fail before creating an overlay if the existing torch/torchvision stack is broken.
python - <<'PY'
import torch
import torchvision
print('Base PyTorch:', torch.__version__)
print('Base torchvision:', torchvision.__version__)
PY
python -m venv --system-site-packages .venv-yolo
.venv-yolo/bin/python - <<'PY'
from pathlib import Path
from importlib.metadata import version
pins = '\n'.join(f'{name}=={version(name)}' for name in ('torch', 'torchvision', 'numpy')) + '\n'
Path('.venv-yolo/base-constraints.txt').write_text(pins, encoding='utf-8')
print('Preserving base versions:\n' + pins)
PY
# Install only into the overlay, with base GPU versions constrained.
# Missing dependencies may be installed here, never into the conda torch base.
.venv-yolo/bin/python -m pip install --constraint .venv-yolo/base-constraints.txt 'ultralytics==8.3.221'
.venv-yolo/bin/python - <<'PY'
import torch
import torchvision
import ultralytics
from pathlib import Path
from importlib.metadata import version
actual = '\n'.join(f'{name}=={version(name)}' for name in ('torch', 'torchvision', 'numpy')) + '\n'
assert actual == Path('.venv-yolo/base-constraints.txt').read_text(encoding='utf-8'), 'Base version changed'
print('Ultralytics:', ultralytics.__version__)
print('PyTorch:', torch.__version__, 'compiled CUDA:', torch.version.cuda)
print('torchvision:', torchvision.__version__)
print('Dependencies imported. GPU execution must be checked in a Slurm allocation.')
print('No model downloaded or trained.')
PY
