"""Report the completed exploratory runs without changing selection locks."""
import hashlib
import json


def add_expanded_section(code, add, page, table, pct):
    provenance = json.loads((code / 'expanded_evidence_provenance.json').read_text())
    for name, record in provenance.items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == record['sha256'], name
    comparison = json.loads((code / 'expanded_ns_results_670000.json').read_text())
    expansion = json.loads((code / 'expanded_n_results_669929.json').read_text())
    plan = json.loads((code / 'expanded_ns_comparison_plan.json').read_text())
    decision = json.loads((code / 'coursework_model_decision.json').read_text())
    assert comparison['status'] == expansion['status'] == 'passed'
    assert comparison['plan_sha256'] == hashlib.sha256((code / 'expanded_ns_comparison_plan.json').read_bytes()).hexdigest()
    assert plan['primary_fixed_thresholds'] == [0.05, 0.5]
    assert comparison['primary_points_with_strict_joint_improvement'] == []
    assert comparison['all_grid_points_with_strict_joint_improvement'] == [0.7]
    assert not comparison['reassessment_inference'] and not comparison['original_study_replaced']
    assert [(d['role'], d['model'], d['confidence']) for d in decision['choices']] == [
        ('coursework_primary', 'YOLOv8n', 0.5), ('expanded_development_candidate', 'YOLOv8s', 0.5)]
    models = comparison['models']
    rows = {r['threshold']: r for r in comparison['same_threshold_comparisons']}
    assert models['n']['checkpoint_sha256'] != models['s']['checkpoint_sha256']

    page()
    add('''## Exploratory expansion: design and results

After the main study was completed, the training set was expanded to investigate the effect of using more of the available data. The original 768 training images were retained and all 1,495 eligible binary-labelled images with missing lesion IDs were added. The sole Indeterminate image remained excluded. Validation and reassessment membership and labels were unchanged; only validation was scored in the new runs. These results do not replace the main study.

The added images also lack patient IDs. Recorded links are still checked, and no exact decoded-image duplicates cross the splits. Unrecorded lesion or patient links may nevertheless connect training and evaluation images. Missing IDs were not fabricated, and deduplication is not proof of independence.''')
    table(['Design feature', 'Main study', 'Expanded exploration'], [
        ['Training images (Benign / Malignant)', '768 (576 / 192)', '2,263 (1,826 / 437)'],
        ['Validation images (Benign / Malignant)', '165 (124 / 41)', 'Same images and labels'],
        ['Epochs / optimizer calls, each model', '60 / 2,880', '60 / 8,520'],
        ['Missing-lesion training images', '0', '1,495'],
    ])
    add('''Expanded n and s both started from their respective original COCO weights. They used the same 60 epochs, seed 3710, FP32, 640-pixel input, batch and nominal batch 16, AdamW, learning-rate schedule and flips-only augmentation. Each completed 8,520 optimizer calls; all original image file hashes and per-call learning rates match between these two runs. Best EMA selection still maximizes validation macro mAP50:95.

Relative to restricted n, expanded n raises malignant AP50 from 0.3591 to 0.5311 on the same validation images. At confidence 0.05 its malignant TP increases from 25 to 33, but benign false-alert images also increase from 38 to 55. Training-set size, class balance and optimizer-call budget changed together, so this is not an isolated causal effect of sample count.''')
    table(['Expanded-model validation AP', 'YOLOv8n', 'YOLOv8s'], [
        ['Macro mAP50', *[f'{models[m]["mAP50"]:.4f}' for m in ('n', 's')]],
        ['Macro mAP50:95', *[f'{models[m]["mAP50_95"]:.4f}' for m in ('n', 's')]],
        *[[f'{label} {metric.replace("_", ":")}', *[f'{models[m]["ap"][cls][metric]:.4f}' for m in ('n', 's')]]
          for cls, label in [('0', 'Benign'), ('1', 'Malignant')]
          for metric in ('AP50', 'AP50_95')],
    ])
    add('s improves the macro average while its malignant AP50 and AP50:95 are lower. Macro improvement therefore does not establish improvement in the malignant class.')
    add('Training-call time was %.2f / %.2f minutes for n/s, with peak allocated CUDA memory %.2f / %.2f GiB and selected EMA epochs %d / %d. The models have 3,011,238 / 11,136,374 parameters. Jobs 669929 and 670000 ran on separate A100 nodes; these times are descriptive, not a controlled speed comparison.' % (
        models['n']['training_seconds'] / 60, models['s']['training_seconds'] / 60,
        models['n']['peak_cuda_allocated_bytes'] / 2**30, models['s']['peak_cuda_allocated_bytes'] / 2**30,
        models['n']['best_epoch'], models['s']['best_epoch']))

    page()
    add('''## Expanded n/s: common operating thresholds

Before training s, the comparison specified common confidence thresholds 0.05 and 0.50 as the primary points, plus 0.70 as an additional reference because expanded n had already selected it. Strict joint improvement required more malignant TP and fewer benign-image false alerts. The 14-point grid was retained without refinement. Both models use the same class-aware NMS at 0.7 and correct-class IoU >= 0.5 matching. Confidence is not a calibrated cancer probability.''')
    common_rows = []
    for t in [*plan['primary_fixed_thresholds'], plan['additional_fixed_reference_threshold']]:
        for title, render in [
            ('Malignant TP / FP / FN', lambda p: f'{p["malignant_tp"]} / {p["malignant_fp"]} / {p["malignant_fn"]}'),
            ('Malignant recall / precision', lambda p: f'{pct(p["malignant_recall"])} / {pct(p["malignant_precision"])}'),
            ('Benign images with malignant alert', lambda p: f'{p["benign_false_alert_images"]}/124 ({pct(p["benign_false_alert_rate"])})'),
        ]:
            common_rows.append([f'{t:.2f}: {title}', *[render(rows[t][m]) for m in ('n', 's')]])
    table(['Confidence and measure', 'YOLOv8n', 'YOLOv8s'], common_rows)
    add('''At both primary thresholds, s reduces benign false alerts but also reduces malignant recall. At 0.70 it detects one more malignant image and falsely alerts on two fewer benign images. This is the only grid point meeting the strict joint-improvement rule. Pairing the images shows 11 malignant detections in common, one detected only by n, two only by s, and 27 missed by both. The five n false alerts and three s false alerts occur on different benign images. This small net difference is not uniform improvement across cases or evidence of stable superiority.

Malignant FP counts are prediction-box counts, whereas benign false alerts count images. They have different denominators. For example, n has 85 malignant FP boxes at 0.05 but 55 benign images with malignant alerts. An alert itself does not require adequate localization.''')
    add('### Same false-alert budget, separately selected thresholds')
    table(['Measure', 'Expanded n', 'Expanded s'], [
        ['Existing validation-selected confidence', *[f'{models[m]["selected_threshold"]:.2f}' for m in ('n', 's')]],
        ['Malignant detections / recall', *[
            f'{models[m]["selected_point"]["classes"]["Malignant"]["tp"]}/41 ({pct(models[m]["selected_point"]["classes"]["Malignant"]["recall"])})'
            for m in ('n', 's')]],
        ['Benign images with malignant alert', *[
            f'{models[m]["selected_point"]["image_alerts"]["benign_fp"]}/124 ({pct(models[m]["selected_point"]["image_alerts"]["benign_false_alert_rate"])})'
            for m in ('n', 's')]],
    ])
    add('''Both selected points satisfy the original validation budget of at most 12 benign false-alert images. s has greater recall within this constraint, but also more false alerts. This supports s at 0.50 as an exploratory development candidate; it is a different-threshold comparison, not simultaneous improvement at a common threshold. The report keeps the original known-lesion n at 0.50 as its primary experiment.''')

    page()
    add('''## Expanded n/s: complete threshold grid

All 14 declared points are shown, including those unfavorable to s. Malignant TP has denominator 41; benign false-alert images have denominator 124. These are validation results, not new test results.''')
    table(['Confidence', 'n M TP', 's M TP', 'n B alerts', 's B alerts'], [
        [f'{t:.2f}', rows[t]['n']['malignant_tp'], rows[t]['s']['malignant_tp'],
         rows[t]['n']['benign_false_alert_images'], rows[t]['s']['benign_false_alert_images']]
        for t in plan['full_threshold_grid']])
    add('''![Expanded n/s validation trade-off](assets/expanded_threshold_tradeoff.png)

Markers are evaluated points; connecting lines do not validate intermediate thresholds. All 28 operating points were independently replayed from saved predictions. AP uses the pinned Ultralytics implementation and was not independently recomputed.

Only one seed and 41 malignant validation images were used. Validation has been reused during development and missing identity links may inflate results. The architecture and initialization-transfer differences remain. No expanded model was evaluated on supplementary reassessment, and no original checkpoint or threshold lock was changed.''')
