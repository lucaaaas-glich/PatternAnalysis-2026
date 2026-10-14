"""Validation-only threshold sweep for both trained models, never reads test images."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from modules import load_detector
from threshold_metrics import THRESHOLDS, evaluate

EXPECTED_CHECKPOINT_HASHES = ['2d58977b18c3386152e7579119d458233241507a8810935ccfd67ffaabf00a49',
                              '0abef51d79d61344d813efa5547994816eec3bc1f22bbc07b1676cbf75ee3878']


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def scan(data, baseline, comparison, output):
    import torch
    import ultralytics
    import yaml
    from PIL import Image
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Submit as a Slurm GPU job')
    if ultralytics.__version__ != '8.3.221':
        raise ValueError('Use the verified overlay environment')
    data, baseline, comparison, output = map(Path, (data, baseline, comparison, output))
    if output.exists():
        raise ValueError('Preserve existing evidence; output must be new')
    if not baseline.is_file() or not comparison.is_file():
        raise ValueError('Both trained checkpoints must exist')
    if [sha256(baseline), sha256(comparison)] != EXPECTED_CHECKPOINT_HASHES:
        raise ValueError('Checkpoint hashes differ from downloaded best.pt evidence')
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    if config.get('val') != 'images/val' or config.get('names') != ['Benign', 'Malignant', 'Indeterminate']:
        raise ValueError('Unexpected validation config')
    runtime = Path(config['path'])
    if output.resolve() == runtime.resolve() or runtime.resolve() in output.resolve().parents:
        raise ValueError('Output must be outside runtime dataset')
    images = sorted(p for p in (runtime/'images'/'val').iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'})
    if len(images) != 388 or len({p.stem for p in images}) != 388:
        raise ValueError('Expected fixed 388-image validation set')
    ground_truth = {}
    for image in images:
        with Image.open(image) as photo:
            width, height = photo.size
        tokens = (runtime/'labels'/'val'/f'{image.stem}.txt').read_text(encoding='utf-8').strip().split()
        if len(tokens) != 5:
            raise ValueError('Expected exactly one annotated box')
        cls = int(tokens[0])
        x, y, w, h = map(float, tokens[1:])
        if not all(0 <= v <= 1 for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError('Invalid ground truth coordinates')
        if min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7:
            raise ValueError('Ground truth box outside image')
        ground_truth[image.stem] = {'image_id': image.stem, 'gt_class': cls,
            'gt_box': [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]}
    if Counter(r['gt_class'] for r in ground_truth.values()) != {0: 311, 1: 77}:
        raise ValueError('Validation truth differs from fixed split')
    output.mkdir(parents=True)
    protocol = {'purpose': 'Validation operating-point analysis, NOT test or clinical deployment',
                'job_id': os.environ['SLURM_JOB_ID'], 'test_split_used': False,
                'image_count': 388, 'benign_images': 311, 'malignant_images': 77,
                'imgsz': 640, 'rect': False, 'prediction_conf_floor': 0.001,
                'nms_iou': 0.7, 'max_det': 300, 'agnostic_nms': False, 'half': False,
                'matching_rule': 'Confidence-ordered greedy, correct class, IoU>=0.5, one truth per image',
                'data_yaml_sha256': sha256(data), 'ultralytics': ultralytics.__version__,
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                'models': {}, 'status': 'started',
                'limitations': ['Image split has known/unknown lesion leakage.',
                    'Cached predictions contain post-NMS selected classes, not all class probabilities.',
                    'Alert rates ignore localisation; not interchangeable with detection recall.',
                    'False alerts are a referral proxy, NOT actual biopsy counts.',
                    'Fixed square predict preprocessing differs from trainer rectangular validation.',
                    'No mAP, calibrated cancer probability or formal FPS measured here.',
                    'Mean matched IoU excludes missed lesions and is not an overall IoU claim.']}
    record_file = output/'summary.json'
    record_file.write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    try:
        for name, checkpoint in [('yolov8n', baseline), ('yolov8s', comparison)]:
            model = load_detector(checkpoint)
            if [model.names[i] for i in range(3)] != config['names']:
                raise ValueError('Checkpoint class mapping mismatch')
            records, seen = [], set()
            results = model.predict(source=str(runtime/'images'/'val'), stream=True,
                imgsz=640, batch=16, rect=False, device=0, conf=0.001, iou=0.7,
                max_det=300, agnostic_nms=False, half=False, augment=False, save=False, verbose=False)
            for result in results:
                identifier = Path(result.path).stem
                if identifier not in ground_truth or identifier in seen:
                    raise ValueError('Unexpected or duplicate prediction image')
                seen.add(identifier)
                predictions = [{'box': box, 'confidence': score, 'class_id': int(cls)}
                    for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
                records.append(dict(ground_truth[identifier], predictions=predictions))
            if seen != set(ground_truth):
                raise ValueError('Incomplete predictions')
            (output/f'{name}_predictions.json').write_text(json.dumps(records)+'\n', encoding='utf-8')
            rows = [evaluate(records, threshold) for threshold in THRESHOLDS]
            protocol['models'][name] = {'checkpoint_sha256': sha256(checkpoint), 'sweep': rows,
                'images_at_max_det_cap': sum(len(r['predictions']) >= 300 for r in records)}
            print(f'{name}: {len(records)} validation images, {len(rows)} thresholds')
        plot_sweep(protocol, output/'threshold_tradeoff.png')
        protocol['status'] = 'completed'
    except Exception as error:
        protocol.update(status='failed', error=str(error))
        raise
    finally:
        record_file.write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    print('VALIDATION THRESHOLD SCAN COMPLETED. Test set not used.')


def plot_sweep(protocol, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, result in protocol['models'].items():
        rows = result['sweep']
        axes[0].plot([r['detection_by_class']['1']['recall'] for r in rows],
                     [r['detection_by_class']['1']['precision'] for r in rows], 'o-', label=name)
        axes[1].plot([r['image_level_malignant_alert']['benign_false_alert_rate'] for r in rows],
                     [r['image_level_malignant_alert']['sensitivity'] for r in rows], 'o-', label=name)
    axes[0].set(xlabel='Malignant detection recall (IoU >= 0.5)', ylabel='Malignant detection precision')
    axes[1].set(xlabel='Benign image false-alert rate', ylabel='Malignant image alert sensitivity')
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle('Validation only: fixed post-NMS predictions; not clinical evidence')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data', 'baseline', 'comparison', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    scan(args.data, args.baseline, args.comparison, args.output)
