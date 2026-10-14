"""Pure metrics for one annotated lesion per image; fixed cached post-NMS boxes.

Detection requires correct class AND IoU>=0.5. Image triage only asks whether
any malignant prediction triggers an alert; it does NOT prove localisation.
"""

THRESHOLDS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(records, threshold, match_iou=0.5):
    if not 0 <= threshold <= 1 or not 0 < match_iou <= 1 or not records:
        raise ValueError('Valid threshold, matching IoU and nonempty records required')
    if len({r['image_id'] for r in records}) != len(records):
        raise ValueError('Duplicate image ID')
    stats = {str(c): {'tp': 0, 'fp': 0, 'fn': 0, 'matched_ious': []} for c in range(3)}
    triage = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for record in records:
        gt_class = record['gt_class']
        if gt_class not in (0, 1):
            raise ValueError('Validation triage protocol requires benign/malignant truth only')
        kept = sorted((p for p in record['predictions'] if p['confidence'] >= threshold),
                      key=lambda p: p['confidence'], reverse=True)
        matched = False
        for prediction in kept:
            if prediction['class_id'] not in range(3):
                raise ValueError('Unexpected predicted class')
            cls = str(prediction['class_id'])
            overlap = box_iou(record['gt_box'], prediction['box'])
            if prediction['class_id'] == gt_class and overlap >= match_iou and not matched:
                stats[cls]['tp'] += 1
                stats[cls]['matched_ious'].append(overlap)
                matched = True
            else:
                stats[cls]['fp'] += 1
        if not matched:
            stats[str(gt_class)]['fn'] += 1
        alert = any(p['class_id'] == 1 for p in kept)
        triage['tp' if alert else 'fn'] += int(gt_class == 1)
        triage['fp' if alert else 'tn'] += int(gt_class == 0)
    for values in stats.values():
        values['precision'] = ratio(values['tp'], values['tp']+values['fp'])
        values['recall'] = ratio(values['tp'], values['tp']+values['fn'])
        values['f1'] = ratio(2*values['tp'], 2*values['tp']+values['fp']+values['fn'])
        overlaps = values.pop('matched_ious')
        values['mean_iou_of_true_positives_only'] = ratio(sum(overlaps), len(overlaps))
    triage.update(sensitivity=ratio(triage['tp'], triage['tp']+triage['fn']),
                  benign_false_alert_rate=ratio(triage['fp'], triage['fp']+triage['tn']),
                  alert_precision=ratio(triage['tp'], triage['tp']+triage['fp']),
                  alert_count=triage['tp']+triage['fp'])
    return {'threshold': threshold, 'match_iou': match_iou,
            'detection_by_class': stats, 'image_level_malignant_alert': triage}
