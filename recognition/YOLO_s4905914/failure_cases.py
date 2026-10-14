"""Deterministic post-test error examples; no training, inference or tuning.

Select measurable failure types, then render existing cached predictions.
Visual causes (hair/contrast/ink) require manual review, not automatic attribution.
"""

import argparse
import json
from pathlib import Path

from threshold_metrics import box_iou


def select_cases(records, threshold=.05):
    pools = {name: [] for name in ('malignant_wrong_class', 'benign_false_alert',
                                  'malignant_alert_bad_localisation', 'duplicate_same_class',
                                  'malignant_miss_small_pixel_box')}
    for row in records:
        kept = [p for p in row['predictions'] if p['confidence'] >= threshold]
        correct = [p for p in kept if p['class_id'] == row['gt_class'] and box_iou(row['gt_box'], p['box']) >= .5]
        malignant = [p for p in kept if p['class_id'] == 1]
        wrong_benign = [p for p in kept if p['class_id'] == 0 and box_iou(row['gt_box'], p['box']) >= .5]
        if row['gt_class'] == 1 and not correct:
            if wrong_benign:
                pools['malignant_wrong_class'].append((max(p['confidence'] for p in wrong_benign), row))
            if malignant:
                pools['malignant_alert_bad_localisation'].append((max(p['confidence'] for p in malignant), row))
            l,t,r,b = row['gt_box']
            pools['malignant_miss_small_pixel_box'].append((-(r-l)*(b-t), row))
        if row['gt_class'] == 0 and malignant:
            pools['benign_false_alert'].append((max(p['confidence'] for p in malignant), row))
        if len(correct) > 1:
            pools['duplicate_same_class'].append((len(correct), row))
    selected, seen = [], set()
    for category, candidates in pools.items():
        for score, row in sorted(candidates, key=lambda pair: (-pair[0], pair[1]['image_id'])):
            if row['image_id'] not in seen:
                selected.append({'category':category, 'image_id':row['image_id'],
                                 'selection_score':score, 'visual_cause':'pending manual review'})
                seen.add(row['image_id'])
                break
    return selected


def render_case(case, truth, comparison, image_path, destination, threshold):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    with Image.open(image_path) as source:
        original = source.convert('RGB')
    names = ['Benign','Malignant','Indeterminate']
    font = ImageFont.load_default(size=18)
    frame = Image.new('RGB',(1800,600),'#f6f7fa')
    draw = ImageDraw.Draw(frame)
    draw.text((12,12),f"{case['image_id']} | {case['category']} | frozen confidence {threshold}",font=font,fill='#182438')
    draw.text((12,565),'TEST error analysis only. Visual causes require review; no threshold/model tuning.',font=font,fill='#182438')
    groups = [('Ground truth (derived from human mask)', [{'box':truth['gt_box'], 'class_id':truth['gt_class']}]),
              ('YOLOv8n frozen predictions', sorted([p for p in truth['predictions'] if p['confidence']>=threshold],key=lambda p:-p['confidence'])),
              ('YOLOv8s frozen predictions', sorted([p for p in comparison['predictions'] if p['confidence']>=threshold],key=lambda p:-p['confidence']))]
    for index,(title,boxes) in enumerate(groups):
        panel = original.copy()
        painter = ImageDraw.Draw(panel)
        box_font = ImageFont.load_default(size=max(13,original.width//40))
        for prediction in boxes[:6]:
            cls = prediction['class_id']
            colour = ['#2563eb','#06b6d4','#a855f7'][cls]
            l,t,r,b = prediction['box']
            painter.rectangle((l,t,r,b),outline=colour,width=max(2,original.width//180))
            text = names[cls]
            if 'confidence' in prediction:
                text += f" {prediction['confidence']:.3f} IoU={box_iou(truth['gt_box'],prediction['box']):.2f}"
            painter.text((max(0,l),max(0,t-box_font.size-4)),text,font=box_font,fill=colour,stroke_width=1,stroke_fill='white')
        fitted = ImageOps.contain(panel,(580,455),Image.Resampling.LANCZOS)
        frame.paste(fitted,(index*600+(600-fitted.width)//2,90+(455-fitted.height)//2))
        draw.text((index*600+12,55),title,font=font,fill='#182438')
        if index:
            draw.text((index*600+12,540),f'Shown {min(6,len(boxes))}/{len(boxes)} boxes; descending confidence',font=font,fill='#182438')
    frame.save(destination)


def create(test_root, runtime, output, select_only=False):
    test_root, output = Path(test_root), Path(output)
    if output.exists():
        raise ValueError('Output exists; preserve earlier evidence')
    if output.resolve()==test_root.resolve() or test_root.resolve() in output.resolve().parents:
        raise ValueError('Output must be outside original test results')
    summary = json.loads((test_root/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed' or summary['policy']['confidence'] != .05:
        raise ValueError('Require completed frozen .05 test report')
    records = json.loads((test_root/'yolov8n_predictions.json').read_text(encoding='utf-8'))
    comparison = json.loads((test_root/'yolov8s_predictions.json').read_text(encoding='utf-8'))
    record_map = {r['image_id']:r for r in records}
    comparison_map = {r['image_id']:r for r in comparison}
    if len(records)!=388 or len(record_map)!=388 or record_map.keys()!=comparison_map.keys():
        raise ValueError('Cached test IDs mismatch')
    if any(record_map[i]['gt_class']!=comparison_map[i]['gt_class'] or
           record_map[i]['gt_box']!=comparison_map[i]['gt_box'] for i in record_map):
        raise ValueError('Ground truth differs between cached models')
    cases = select_cases(records)
    if len(cases) < 3:
        raise ValueError('Fewer than three distinct measurable error examples; review selection')
    image_paths = {}
    if not select_only:
        if runtime is None:
            raise ValueError('Runtime dataset required to render images')
        runtime = Path(runtime).resolve()
        if runtime == output.resolve() or runtime in output.resolve().parents:
            raise ValueError('Output must be outside source runtime')
        for case in cases:
            matches = [p for p in (runtime/'images'/'test').glob(case['image_id']+'.*')
                       if p.suffix.lower() in {'.jpg','.jpeg'} and p.is_file()]
            if len(matches)!=1:
                raise ValueError('Cannot uniquely locate selected test image')
            image_paths[case['image_id']] = matches[0]
    output.mkdir(parents=True)
    for case in cases:
        print(case['category'],case['image_id'])
        if not select_only:
            identifier=case['image_id']
            render_case(case,record_map[identifier],comparison_map[identifier],image_paths[identifier],
                        output/(identifier+'.png'),.05)
    report = {'purpose':'Post-test qualitative error review; no model/threshold changes',
              'source_test_job':summary['job_id'], 'primary_model':'yolov8n',
              'threshold':.05, 'cases':cases, 'images_rendered':not select_only,
              'limitations':['Selection is deterministic and deliberately error-focused, not a representative random sample.',
                  'Pixel-box size is not physical lesion size or normalised image area.',
                  'Visual artefacts cannot establish causal explanations without controlled tests.',
                  'Panels show at most six highest-score boxes, and disclose the displayed/total count.']}
    (output/'cases.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('CASE SELECTION COMPLETE. Visual review still required.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root',type=Path,required=True)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--select-only',action='store_true')
    args=parser.parse_args()
    create(args.test_root,args.runtime,args.output,args.select_only)
