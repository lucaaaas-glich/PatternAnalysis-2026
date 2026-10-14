"""GPU smoke/baseline training with validation only, never held-out test use."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from modules import load_detector

PRETRAINED_SHA256 = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
COMPARISON_SHA256 = '1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a'
EXPERIMENT_DATA_SHA256 = '78a243b872f21040acf5bb4abfa0b7b9a003766eca739178e3e160e2c959ff41'


def smoke_config(data, output):
    return dict(data=str(Path(data).resolve()), project=str(Path(output).resolve()),
                name='train', exist_ok=False, epochs=2, imgsz=320, batch=8,
                device=0, workers=2, seed=3710, deterministic=True,
                amp=False, cache=False, val=True, plots=True, save=True,
                optimizer='auto', close_mosaic=0)


def training_config(data, output, profile='smoke'):
    options = smoke_config(data, output)
    if profile in ('baseline', 'comparison'):
        options.update(epochs=30, imgsz=640, batch=16, workers=1,
                       close_mosaic=10, patience=100, resume=False, fraction=1.0)
    elif profile != 'smoke':
        raise ValueError('Unknown training profile')
    options['split'] = 'val'
    return options


def verify_initial_weights(weights, profile):
    with Path(weights).open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if profile == 'baseline' and digest != PRETRAINED_SHA256:
        raise ValueError('Baseline requires original verified pretrained weights, not smoke best.pt')
    if profile == 'comparison' and digest != COMPARISON_SHA256:
        raise ValueError('Comparison requires original verified YOLOv8s checkpoint')
    return digest


def run_training(data, weights, output, profile='smoke'):
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
        raise ValueError('Training must use train/val only')
    for name in ('train', 'val'):
        if not (Path(config['path']) / config[name]).is_dir():
            raise ValueError('Missing runtime train/val directory')
    options = training_config(data, output, profile)
    purposes = {'baseline': 'Baseline training with validation; no final test evaluation',
                'comparison': 'YOLOv8s matched-protocol comparison; validation only',
                'smoke': 'Pipeline smoke check, not final evaluation'}
    purpose = purposes[profile]
    provenance = dict(purpose=purpose, profile=profile,
                      status='started', job_id=os.environ['SLURM_JOB_ID'],
                      torch=torch.__version__, ultralytics=ultralytics.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                      config=options, test_split_used=False,
                      split_unit='image', patient_independent=False, lesion_independent=False)
    provenance['weights_sha256'] = verify_initial_weights(weights, profile)
    with data.open('rb') as handle:
        provenance['data_yaml_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    if profile == 'comparison' and provenance['data_yaml_sha256'] != EXPERIMENT_DATA_SHA256:
        raise ValueError('Comparison data config differs from executed baseline')
    model = load_detector(weights)
    output.mkdir(parents=True)
    record = output / f'{profile}.json'
    record.write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    try:
        model.train(**options)
        checkpoint = output / 'train' / 'weights' / 'last.pt'
        if not checkpoint.is_file():
            raise RuntimeError('Training returned without last.pt')
        best = output / 'train' / 'weights' / 'best.pt'
        if not best.is_file():
            raise RuntimeError('Training returned without best.pt')
        provenance.update(status='completed', last_checkpoint=str(checkpoint), best_checkpoint=str(best),
                          checkpoint_selection='Ultralytics validation fitness, not test data')
    except Exception as error:
        provenance.update(status='failed', error=str(error))
        raise
    finally:
        record.write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    print(f'{profile.upper()} COMPLETED. No held-out test evaluation performed.')


def run_smoke(data, weights, output):
    return run_training(data, weights, output, profile='smoke')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=['smoke', 'baseline', 'comparison'], default='smoke')
    args = parser.parse_args()
    run_training(args.data, args.weights, args.output, args.profile)

