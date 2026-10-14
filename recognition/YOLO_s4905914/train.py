"""Two-epoch GPU smoke check, NOT a completed baseline or test evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from modules import load_detector


def smoke_config(data, output):
    return dict(data=str(Path(data).resolve()), project=str(Path(output).resolve()),
                name='train', exist_ok=False, epochs=2, imgsz=320, batch=8,
                device=0, workers=2, seed=3710, deterministic=True,
                amp=False, cache=False, val=True, plots=True, save=True,
                optimizer='auto', close_mosaic=0)


def run_smoke(data, weights, output):
    import torch
    import ultralytics
    import yaml
    data, weights, output = map(Path, (data, weights, output))
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside a Slurm GPU allocation, not the login node')
    if ultralytics.__version__ != '8.3.221':
        raise RuntimeError('Use the verified .venv-yolo environment')
    if not data.is_file() or not weights.is_file() or output.exists():
        raise ValueError('Inputs must exist; output must be new')
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    if config.get('names') != ['Benign', 'Malignant', 'Indeterminate']:
        raise ValueError('Unexpected diagnosis classes')
    if config.get('train') != 'images/train' or config.get('val') != 'images/val':
        raise ValueError('Smoke check must use train/val only')
    for name in ('train', 'val'):
        if not (Path(config['path']) / config[name]).is_dir():
            raise ValueError('Missing runtime train/val directory')
    options = smoke_config(data, output)
    provenance = dict(purpose='Pipeline smoke check, not final evaluation',
                      status='started', job_id=os.environ['SLURM_JOB_ID'],
                      torch=torch.__version__, ultralytics=ultralytics.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                      config=options, test_split_used=False,
                      split_unit='image', patient_independent=False, lesion_independent=False)
    with weights.open('rb') as handle:
        provenance['weights_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    with data.open('rb') as handle:
        provenance['data_yaml_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    model = load_detector(weights)
    output.mkdir(parents=True)
    record = output / 'smoke.json'
    record.write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    try:
        model.train(**options)
        checkpoint = output / 'train' / 'weights' / 'last.pt'
        if not checkpoint.is_file():
            raise RuntimeError('Training returned without last.pt')
        provenance.update(status='completed', last_checkpoint=str(checkpoint))
    except Exception as error:
        provenance.update(status='failed', error=str(error))
        raise
    finally:
        record.write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    print('SMOKE CHECK COMPLETED. No held-out test evaluation performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_smoke(args.data, args.weights, args.output)

