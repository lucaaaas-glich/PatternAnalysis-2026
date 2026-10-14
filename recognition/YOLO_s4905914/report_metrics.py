"""IoU diagnostics with explicit denominators; no missed truth is dropped."""
from threshold_metrics import box_iou, ratio


def localisation_report(records, threshold):
    report = {}
    for cls in range(3):
        same, agnostic = [], []
        for row in records:
            if row['gt_class'] != cls:
                continue
            predictions = [p for p in row['predictions'] if p['confidence'] >= threshold]
            agnostic.append(max((box_iou(row['gt_box'], p['box']) for p in predictions), default=0.0))
            same.append(max((box_iou(row['gt_box'], p['box']) for p in predictions
                             if p['class_id'] == cls), default=0.0))
        report[str(cls)] = {'truth_count': len(same),
            'mean_same_class_best_iou_all_truths': ratio(sum(same), len(same)),
            'mean_class_agnostic_best_iou_all_truths': ratio(sum(agnostic), len(agnostic)),
            'fraction_same_class_iou_ge_0_8': ratio(sum(v >= .8 for v in same), len(same))}
    return report
