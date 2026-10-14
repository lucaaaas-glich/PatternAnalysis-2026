#!/bin/bash
# Run from repository root on login node; download only before submitting GPU job.
set -euo pipefail
test -x .venv-yolo/bin/python
test -f outputs/yolo_runtime/image_level_3710_v1/data.yaml
export MPLBACKEND=Agg
.venv-yolo/bin/python - <<'PY'
from pathlib import Path
from ultralytics import YOLO
weights = Path('weights/yolov8n.pt')
weights.parent.mkdir(exist_ok=True)
if not weights.exists():
    print('Obtaining official COCO-pretrained YOLOv8n checkpoint; no training on login node.')
    YOLO(str(weights), task='detect')
if not weights.is_file():
    raise RuntimeError('Weights download did not produce expected file')
print('Weights available:', weights)
PY
sbatch recognition/YOLO_s4905914/smoke_a100.sh
