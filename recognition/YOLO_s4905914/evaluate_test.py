"""Final held-out test report for a frozen validation-selected policy.

No training, threshold sweep, checkpoint selection or hyperparameter tuning.
Standard AP and fixed-threshold metrics have distinct preprocessing/definitions.
"""

import argparse
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from modules import load_detector
from report_metrics import localisation_report
from scan_thresholds import EXPECTED_CHECKPOINT_HASHES, sha256
from threshold_metrics import evaluate

POLICY = {'primary_model': 'yolov8n', 'comparison_model': 'yolov8s',
          'confidence': 0.05, 'matching_iou': 0.5, 'nms_iou': 0.7,
          'imgsz': 640, 'max_det': 300, 'half': False,
          'selection_data': 'validation_661469', 'selection_confirmed_by_user': True,
          'validation_summary_sha256': 'cc7308f53ddd2e8275183338ff605d44e44671f9a408c6fb78a5ce6cfbb96af4',
          'selection_reason': 'Validation n at .05 retained same 48 malignant alerts as .01 with 30 rather than 43 benign alerts'}


def load_test_truth(runtime):
    from PIL import Image
    images = sorted(p for p in (runtime/'images'/'test').iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'})
    if len(images) != 388 or len({p.stem for p in images}) != 388:
        raise ValueError('Expected the fixed 388-image held-out test set')
    truth = {}
    for image in images:
        with Image.open(image) as photo:
            width, height = photo.size
        tokens = (runtime/'labels'/'test'/f'{image.stem}.txt').read_text(encoding='utf-8').split()
        if len(tokens) != 5:
            raise ValueError('One truth box per test image required')
        cls = int(tokens[0])
        x, y, w, h = map(float, tokens[1:])
        if not all(0 <= v <= 1 for v in (x,y,w,h)) or min(w,h) <= 0:
            raise ValueError('Invalid test annotation')
        if min(x-w/2,y-h/2) < -1e-7 or max(x+w/2,y+h/2) > 1+1e-7:
            raise ValueError('Test box outside image')
        truth[image.stem] = {'image_id': image.stem, 'gt_class': cls,
            'gt_box': [(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]}
    if Counter(row['gt_class'] for row in truth.values()) != {0:311,1:77}:
        raise ValueError('Test class counts differ from frozen split')
    return images, truth


def predict_test(model, runtime, truth):
    records, seen = [], set()
    for result in model.predict(source=str(runtime/'images'/'test'), stream=True,
            imgsz=640, rect=False, batch=16, device=0, conf=.001, iou=.7,
            max_det=300, half=False, agnostic_nms=False, augment=False, save=False, verbose=False):
        identifier = Path(result.path).stem
        if identifier not in truth or identifier in seen:
            raise ValueError('Unexpected test inference ID')
        seen.add(identifier)
        predictions = [{'box': box, 'confidence': score, 'class_id': int(cls)}
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
        records.append(dict(truth[identifier], predictions=predictions))
    if seen != set(truth):
        raise ValueError('Incomplete held-out predictions')
    return records


def measure_fps(model, images):
    import torch
    from PIL import Image
    # Images are decoded before timing; measure batch-1 API+preprocess+forward+NMS.
    decoded = []
    for path in images[:50]:
        with Image.open(path) as image:
            decoded.append(image.convert('RGB'))
    options = dict(imgsz=640, rect=False, device=0, conf=.05, iou=.7, max_det=300,
                   half=False, agnostic_nms=False, augment=False, save=False, verbose=False, batch=1)
    for i in range(10):
        model.predict(source=decoded[i % len(decoded)], **options)
    torch.cuda.synchronize()
    durations = []
    for image in decoded:
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(source=image, **options)
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-start)
    return {'batch':1, 'warmup_calls':10, 'timed_images':len(durations),
            'fps':len(durations)/sum(durations),
            'mean_latency_ms':1000*statistics.mean(durations),
            'median_latency_ms':1000*statistics.median(durations),
            'scope':'FP32 batch1 Python predict API including preprocess/forward/NMS; excludes disk decode, render, network and result serialisation',
            'hardware':'A100; not mobile/Mac deployment throughput'}


def run_test(data, baseline, comparison, output):
    import torch
    import ultralytics
    import yaml
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run in Slurm GPU allocation')
    if ultralytics.__version__ != '8.3.221':
        raise ValueError('Unexpected Ultralytics version')
    data, baseline, comparison, output = map(Path, (data,baseline,comparison,output))
    if output.exists():
        raise ValueError('Preserve previous test evidence; output must be new')
    if [sha256(baseline),sha256(comparison)] != EXPECTED_CHECKPOINT_HASHES:
        raise ValueError('Frozen trained checkpoint hash mismatch')
    if sha256(data) != '78a243b872f21040acf5bb4abfa0b7b9a003766eca739178e3e160e2c959ff41':
        raise ValueError('Dataset config changed since training')
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    if config.get('test') != 'images/test' or config.get('names') != ['Benign','Malignant','Indeterminate']:
        raise ValueError('Unexpected test configuration')
    runtime = Path(config['path'])
    if runtime.resolve() in output.resolve().parents or runtime.resolve() == output.resolve():
        raise ValueError('Output must be separate from runtime dataset')
    images, truth = load_test_truth(runtime)
    report = {'status':'started', 'policy':POLICY, 'test_split_used':True,
              'job_id':os.environ['SLURM_JOB_ID'], 'test_images':388,
              'torch':torch.__version__, 'ultralytics':ultralytics.__version__,
              'gpu':torch.cuda.get_device_name(0), 'models':{},
              'limitations':['Image-level split has known and unknown lesion leakage; not patient/lesion independent.',
                'Indeterminate has no test truth; per-class performance unavailable.',
                'AP summarises score rankings; library P/R are automatic diagnostic operating points, not frozen .05 policy.',
                'AP uses standard rectangular validation; frozen metrics use square predict matching validation threshold scan.',
                'Best-box mean IoU is an oracle localisation diagnostic, not TP-only IoU or AP.',
                'Missing same-class boxes contribute zero to all-truth IoU.',
                'Image false alerts are a referral proxy, not actual biopsy counts.',
                'Do not change threshold/model/hyperparameters based on this test report.']}
    output.mkdir(parents=True)
    summary = output/'summary.json'
    summary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    try:
        for name, checkpoint in [('yolov8n',baseline),('yolov8s',comparison)]:
            model = load_detector(checkpoint)
            if [model.names[i] for i in range(3)] != config['names']:
                raise ValueError('Model class mapping mismatch')
            parameter_count = sum(p.numel() for p in model.model.parameters())
            metrics = model.val(data=str(data.resolve()), split='test', imgsz=640,
                batch=16, workers=1, device=0, half=False, conf=.001, iou=.7,
                max_det=300, augment=False, plots=True, project=str(output.resolve()),
                name=name+'_ap', exist_ok=False)
            per_class = {}
            for index, cls in enumerate(metrics.box.ap_class_index):
                p,r,ap50,ap = map(float, metrics.box.class_result(index))
                per_class[str(int(cls))] = {'library_auto_precision':p,'library_auto_recall':r,
                                           'AP50':ap50,'AP50_95':ap}
            records = predict_test(model,runtime,truth)
            (output/f'{name}_predictions.json').write_text(json.dumps(records)+'\n',encoding='utf-8')
            fixed = evaluate(records, POLICY['confidence'], POLICY['matching_iou'])
            report['models'][name] = {'checkpoint_sha256':sha256(checkpoint),
                'parameters_before_val_fusion':parameter_count,
                'standard_ap':{'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),'by_class':per_class},
                'fixed_threshold_metrics':fixed,
                'localisation_iou':localisation_report(records,POLICY['confidence']),
                'images_at_max_det_cap':sum(len(r['predictions'])>=300 for r in records),
                'speed':measure_fps(model,images)}
            summary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(name, 'test AP50=',metrics.box.map50,'AP50:95=',metrics.box.map,
                  'frozen malignant recall=',fixed['detection_by_class']['1']['recall'])
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        summary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('FROZEN TEST EVALUATION COMPLETED. No model/threshold tuning performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data','baseline','comparison','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args()
    run_test(args.data,args.baseline,args.comparison,args.output)
