"""Assemble the report README from audited experimental records and explanations."""
from pathlib import Path
import json
from report_expanded import add_expanded_section
from report_known_optimization import add_known_optimization
from report_course_closeout import add_course_closeout


def build():
    code=Path(__file__).resolve().parent
    val=json.loads((code/'validation_results_669143.json').read_text())
    final=json.loads((code/'reassessment_results_669815.json').read_text())
    assert val['status']==final['status']=='passed'
    v=val['models'];r=final['models'];parts=[]
    def add(t):parts.append(t.strip()+'\n')
    def page():add('<!-- pagebreak -->')
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows))
    def pct(x):return f'{100*x:.2f}%'
    def point(d):return d['operating_point']
    def mc(d):return point(d)['classes']['Malignant']
    def f4(x):return f'{x:.4f}'
    add(r'''# Lesion detection and classification with YOLOv8

COMP3710 Assignment 3, Project 3 | s4905914 | Report updated 22 October 2026

This project compares YOLOv8n and YOLOv8s for finding skin lesions and classifying them as benign or malignant. The main question is whether either model can detect more malignant lesions while keeping false alerts on benign images low. YOLOv8n was chosen as the main baseline before training. The main experiment uses images with known lesion IDs; a later experiment adds more training images and is reported separately.

Both models completed 60 training epochs, but malignant detection remained weak. In the supplementary evaluation, n detected 10/41 malignant lesions at confidence 0.50, while s detected 6/41 at 0.80. Their mAP50:95 values were 0.3619 and 0.3527. n also raised malignant alerts on 16/124 benign images, above the 10% limit used to select its threshold on validation. The models and thresholds were kept unchanged after these results.

The current development choice is YOLOv8n with light geometry at confidence 0.40, selected using only the known-lesion training and validation sets. It detects 18/41 malignant validation lesions with 12/124 benign alerts. The later bounded follow-up retained this model. The opening n/s reassessment figures refer to the original baseline experiment; the geometry candidate has not yet completed reassessment. The expanded-data experiment is retained as historical exploration and is excluded from the current choice.

Zhao's reply supplied on 22 October requires overall and per-class IoU, Accuracy, and additional classification metrics for imbalance. The 0.8 IoU requirement includes an evidence-supported best-effort provision. The final sections add these metrics and the follow-up analysis. Imported Ultralytics components and personal tutor check-off still need confirmation. These experiments do not establish clinical suitability.''')
    optimization=json.loads((code/'known_optimization_results_20261021.json').read_text())
    decision=optimization['final_decision']
    if decision['status']=='completed' and 'confirmation' in optimization:
        confirmation=optimization['confirmation']
        base=confirmation['reports']['n_baseline_seed3711']['selected']
        candidate=confirmation['reports'][confirmation['candidate']+'_seed3711']['selected']
        outcome='met' if decision['confirmation_passed'] else 'did not meet'
        chosen='n with light geometry' if decision['candidate']=='n_geometry' else 'original n'
        add(f"In the second-seed validation check, the geometry candidate detected {candidate['classes']['Malignant']['tp']}/41 malignant lesions versus {base['classes']['Malignant']['tp']}/41 for the repeated baseline, with {candidate['image_alerts']['benign_fp']} versus {base['image_alerts']['benign_fp']} benign alerts. It {outcome} the prespecified confirmation rule. The current choice is {chosen} at confidence {decision['confidence']:.2f}, using its seed-3710 checkpoint. This comparison reuses validation images and is not new test evidence.")
    add(r'''## Data and scope

The dataset contains 2,594 school-provided dermoscopic photographs. Each bounding box is the smallest rectangle aligned with the image axes that encloses the lesion in its official binary mask. The masks were resized to the photo dimensions using nearest-neighbour interpolation before creating the boxes. Class labels came from official metadata matched by image ID. The checked boxes were reused for this restart; the original school photos and masks were unchanged [1,2].

The metadata lists 2,074 benign, 519 malignant and one indeterminate image. The main experiment uses the 1,098 images with a recorded lesion_id and excludes the other 1,496. The later experiment adds 1,495 of those images to training because they have benign or malignant labels. The indeterminate image is excluded rather than assigned a diagnosis. The model classes are Benign=0 and Malignant=1. Every image has one annotated lesion, so this study does not test scenes with multiple lesions.''')
    table(['Split','Benign / Malignant','Images'],[['Train','576 / 192',768],['Validation','124 / 41',165],['Supplementary reassessment','124 / 41',165]])
    add(r'''

Before selecting the subset, images linked by any recorded lesion or patient ID were grouped together, including indirect links through other images. The groups were then split using seed 3710. No recorded lesion or patient ID appears in more than one set. The subset contains 1,083 recorded lesions and 1,043 groups for splitting; a group is not a new patient ID. There are still 708 images without patient_id, so complete patient independence cannot be confirmed.''')
    page()
    add(r'''## One-page feasibility review

This section restates the plan written on 17 October, with simpler wording added on 21 October. It describes what was planned before training; the measured results appear later. The original plan is kept in `FEASIBILITY_REVIEW.md`.''')
    # Keep report generation portable when only the coursework module is copied.
    review=(code/'FEASIBILITY_REVIEW_PLAIN.txt').read_text().splitlines()
    headings={'Purpose and scope','Data and model choice','Five testable acceptance criteria','Evidence, risks and next decision','Course and authorship notes'}
    body=[]
    for line in review[3:]:
        body.append(('### '+line) if line in headings else line)
    add('\n'.join(body))
    page()
    add(r'''## Model and implementation

YOLOv8 predicts lesion boxes and classes in one network [3]. Its backbone uses convolutions to learn image features. The neck combines features at different scales, and the detection head predicts boxes and class scores at three resolutions. Conv blocks combine convolution, normalization and activation; C2f combines feature branches; SPPF pools features over different areas. Box and class predictions use separate branches. These components come from Ultralytics. The YOLOv7 paper provides background on speed and accuracy in object detection [4], while the actual models here are YOLOv8n and YOLOv8s.

Training uses CIoU for box regression, distribution focal loss (DFL) for box distances, and binary cross-entropy for classification. DFL measures part of the localization error, so a falling dfl_loss does not by itself show better classification. Prediction confidence is not a calibrated probability of cancer. Non-maximum suppression (NMS) removes some overlapping boxes of the same predicted class. It can still leave more than one box in an image.

`modules.py` builds the two-class model and loads saved models. `dataset.py` provides the data loader used by `train.py`, which trains and evaluates the model, saves checkpoints and records losses and metrics. `predict.py` loads a checkpoint and its saved validation threshold, then writes predictions and an overlay image. Helper scripts handle grouping, checking and reporting. The network layers come from Ultralytics rather than student-written implementations; whether this meets the course component requirement still needs confirmation.

### Fixed training protocol''')
    table(['Setting','YOLOv8n','YOLOv8s'],[
        ['Initial weights','Original COCO n','Original COCO s'],['Library','Ultralytics 8.3.221','Same'],
        ['Epochs / completed optimizer calls','60 / 2,880','60 / 2,880'],['Image size / batch / nominal batch','640 / 16 / 16','Same'],
        ['Seed / arithmetic','3710 / FP32','Same'],['Optimizer','AdamW','Same'],
        ['Initial LR / weight decay','0.001 / 0.0005','Same'],['Warm-up / LR schedule','3 epochs / linear','Same'],
        ['Augmentation','Horizontal/vertical flips, p=0.5 each','Same'],
        ['Mosaic / MixUp / oversampling','Off / off / none','Same'],['Early stopping / resumed training','Disabled / no','Same'],
        ['Selected epoch by macro mAP50:95','32','42']])
    add(r'''The backbone, neck and prediction branches were trained; the fixed DFL integration coordinates were not. Under the library learning-rate schedule, the first optimizer call uses zero learning rate and the final epoch uses 0.0000265. Recorded calls were checked against this schedule. Setting both batch size and nominal batch size to 16 avoids a difference in gradient accumulation between the models.

The two sizes do not load exactly the same proportion of their pretrained state: n loads 319/355 entries and s loads 349/355, including buffers. Changing to two classes reduces n's classification hidden width from 80 to 64; s keeps width 128 and replaces the final outputs. Both load the backbone, neck and box-regression state. Their comparison therefore includes initialization differences as well as model size.''')
    page()
    add(r'''## Evaluation and selection rules

Images are resized to fit a 640-pixel square while keeping their aspect ratio, then padded (letterboxing). Synthetic images were used to check that the validation loader and predictor produce identical input pixels. Evaluation keeps scores above 0.001, uses one predicted class per box and class-aware NMS at IoU 0.7, with at most 300 boxes. This differs from default multi-label validation, so AP values from the two policies should not be mixed.

The checkpoint is selected by the highest validation macro mAP50:95 over 60 epochs. Scores are rounded to five decimal places, with a later epoch chosen when they tie. Validation uses EMA weights, saved in FP16 and reloaded in FP32 by Ultralytics. Training validation uses batch 32; the later evaluation requests batch 16. These execution details are recorded for reproducibility, but their effects on score differences were not isolated.

A TP requires the correct class, confidence strictly above the selected threshold and IoU >= 0.5. Each GT matches the highest-confidence eligible prediction once. Unmatched predictions count as FP for their predicted class; unmatched GTs count as FN. A wrong-class or poorly localized box can therefore contribute both FP and FN. An image-level malignant alert requires any retained malignant prediction, regardless of IoU. Alert counts and detection recall are reported separately.

AP50 and AP50:95 are calculated from the low-score sweep; macro mAP weights both classes equally. The operating threshold maximises malignant detection recall while allowing alerts on at most 12 of 124 benign validation images (10%). Ties favour higher precision, fewer benign alerts, then a higher threshold. The grid was fixed before training: 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 and 0.95. This alert limit is an experimental comparison rule, not a course passing threshold or a clinical standard.

![Validation threshold trade-off](assets/threshold_tradeoff.png)

Each marker is one of the 14 tested thresholds; the joining lines do not represent extra tests. The rule selected 0.50 for n and 0.80 for s. At 0.50, s detects 22 malignant validation cases but alerts on 28 benign cases. At 0.70, it has 14 malignant TP and 14 benign alerts. Both exceed the limit of 12. No extra thresholds were added after viewing the results.

![Training history](assets/training_curves.png)

Validation classification loss is plotted on a logarithmic scale so both early spikes and later changes are visible. All recorded values are finite. The cause of the spikes was not tested. Completing 60 epochs does not establish convergence, and an earlier best checkpoint alone does not establish overfitting.''')
    page()
    add(r'''## Main experiment: results

Validation and reassessment each contain 124 benign and 41 malignant images, with different membership. Each model was evaluated once on reassessment using its fixed checkpoint and threshold. Earlier exposure to some images means this is not a fresh blind test.''')
    table(['AP measure','n validation / reassessment','s validation / reassessment'],[
        ['Macro mAP50',f'{v["n"]["mAP50"]:.4f} / {r["n"]["mAP50"]:.4f}',f'{v["s"]["mAP50"]:.4f} / {r["s"]["mAP50"]:.4f}'],
        ['Macro mAP50:95',f'{v["n"]["mAP50_95"]:.4f} / {r["n"]["mAP50_95"]:.4f}',f'{v["s"]["mAP50_95"]:.4f} / {r["s"]["mAP50_95"]:.4f}'],
        *[[f'{name} {metric.replace("_", ":")}',*[f'{v[m]["ap"][cls][metric]:.4f} / {r[m]["ap"][cls][metric]:.4f}' for m in ('n','s')]] for cls,name in [('0','Benign'),('1','Malignant')] for metric in ('AP50','AP50_95')]])
    add('### Locked operating points on supplementary reassessment')
    table(['Measure','YOLOv8n','YOLOv8s'],[
        ['Confidence',.50,.80],
        ['Malignant TP / FP / FN',*[f'{mc(r[m])["tp"]} / {mc(r[m])["fp"]} / {mc(r[m])["fn"]}' for m in ('n','s')]],
        ['Malignant precision',*[pct(mc(r[m])['precision']) for m in ('n','s')]],
        ['Malignant detection recall',*[f'{mc(r[m])["tp"]}/41 ({pct(mc(r[m])["recall"])})' for m in ('n','s')]],
        ['Benign TP / FP / FN',*[f'{point(r[m])["classes"]["Benign"]["tp"]} / {point(r[m])["classes"]["Benign"]["fp"]} / {point(r[m])["classes"]["Benign"]["fn"]}' for m in ('n','s')]],
        ['Benign precision / recall',*[f'{pct(point(r[m])["classes"]["Benign"]["precision"])} / {pct(point(r[m])["classes"]["Benign"]["recall"])}' for m in ('n','s')]],
        ['Malignant image alerts',*[f'{point(r[m])["image_alerts"]["malignant_tp"]}/41 ({pct(point(r[m])["image_alerts"]["sensitivity"])})' for m in ('n','s')]],
        ['Benign image false alerts',*[f'{point(r[m])["image_alerts"]["benign_fp"]}/124 ({pct(point(r[m])["image_alerts"]["benign_false_alert_rate"])})' for m in ('n','s')]]])
    add(r'''On validation, n detects 13/41 malignant lesions (31.71%) with 11/124 benign alerts; s detects 11/41 (26.83%) with 7/124 alerts. Reassessment detection counts fall to 10 and 6. n's 16 benign alerts exceed the validation limit, while s produces three. The thresholds were kept fixed.

Both models detect five malignant reassessment cases, only n detects five, only s detects one, and both miss 30. n has higher AP and recall; s has higher precision and fewer alerts. One seed and small samples do not establish a reliable winner. Scores from the old image-level experiment are not directly comparable because its splits and evaluation settings differed.''')
    add('### Localization on supplementary reassessment')
    table(['Mean IoU at locked operating points','YOLOv8n','YOLOv8s'],[
        [f'{name}: {label}',*[f'{point(r[m])["classes"][name]["iou"][key]["mean"]:.4f} (N={point(r[m])["classes"][name]["iou"][key]["denominator"]})' for m in ('n','s')]]
        for name in ('Benign','Malignant') for label,key in [('all GT, correct class','all_gt_class_correct'),('all GT, ignore class','all_gt_class_agnostic'),('matched TP only','tp_only')]])
    add("All-GT means use the best retained box under the stated class rule, with missing predictions scored zero. TP-only means exclude misses. Both malignant TP-only means are below 0.8, and the all-GT means are lower. The teacher's later reply requires overall and per-class reporting without specifying the treatment of missed or wrong-class predictions. The final sections provide pooled overall IoU alongside these definitions and discuss the best-effort provision.")
    add_expanded_section(code,add,page,table,pct)
    page()
    add(r'''## Main experiment: runtime and memory

Training ran on an A100 in job 669143. Reassessment and speed measurement ran on an A100-PCIE-40GB in job 669815. Speed was measured using validation images, FP32, 640-pixel input and one image at a time. Labels were not used to select a model in this test. Each model had 10 warm-up calls, followed by three passes through 165 images: 495 timed calls. GPU synchronization before and after each call ensured that timing included completed GPU work.''')
    table(['Measurement','YOLOv8n','YOLOv8s'],[
        ['Training parameters',*[f'{v[m]["parameters"]:,}' for m in ('n','s')]],
        ['Training-call elapsed time',*[f'{v[m]["training_seconds"]/60:.2f} min' for m in ('n','s')]],
        ['Peak training PyTorch CUDA allocation',*[f'{v[m]["peak_cuda_allocated_GiB"]:.2f} GiB' for m in ('n','s')]],
        ['Library inference-stage mean',*[f'{r[m]["benchmark"]["library_inference"]["mean_ms"]:.2f} ms' for m in ('n','s')]],
        ['File-to-prediction wall mean',*[f'{r[m]["benchmark"]["wall"]["mean_ms"]:.2f} ms' for m in ('n','s')]],
        ['File-to-prediction median / p95',*[f'{r[m]["benchmark"]["wall"]["median_ms"]:.2f} / {r[m]["benchmark"]["wall"]["p95_ms"]:.2f} ms' for m in ('n','s')]],
        ['Serial file-to-prediction throughput',*[f'{r[m]["benchmark"]["serial_images_per_second"]:.2f} images/s' for m in ('n','s')]],
        ['Peak inference PyTorch CUDA allocation',*[f'{r[m]["benchmark"]["peak_cuda_allocated_bytes_after_warmup"]/2**20:.2f} MiB' for m in ('n','s')]]])
    add(r'''File-to-prediction time includes reading and decoding the image, preprocessing, device transfer, model execution, NMS, threshold filtering and copying predictions back to the CPU. It excludes startup, warm-up, drawing, saving files and the user interface. The library inference time covers only the model stage. Throughput is 1,000 divided by mean elapsed milliseconds for these single-image calls.

n was measured before s, and repeated image reads may have benefited from caching. GPU clocks and execution order were not controlled separately. Although s was faster in this run, this is not enough to conclude that it is generally faster. These A100 measurements do not represent phone performance or a complete clinical application. Allocated tensor memory also differs from total GPU memory usage.

### Verification

The preparation checks included 40 unit tests, complete n/s runs on synthetic data, pretrained-weight checks and real-image checks. All 1,098 photos were decoded, compared with school-side file hashes, and checked for dimensions, orientation and box coordinates. No exact decoded-image duplicates crossed the splits. This cannot rule out shared patients where IDs are missing.

A separate NumPy matcher reproduced all 28 main-study validation threshold results and both fixed reassessment results from saved predictions. It checked TP/FP/FN, image alerts and IoU means, along with image membership, labels, configuration and file hashes. AP values still come from Ultralytics; AP was not independently reimplemented or recalculated.

The training and evaluation source files matched the recorded runs, and the checkpoints and threshold records were unchanged. Changes to the transfer packager were separate from model execution. The resource check also recalculated means and throughput from all 495 timings per model. These checks support the consistency of the code and records; they do not verify every diagnosis or make the study a blind clinical evaluation.''')
    page()
    add(r'''## Five failure examples from reassessment

For each failure category, the image with the lowest ID was selected, without using an image twice. These examples illustrate errors rather than forming a random sample. Dashed boxes show GT; solid boxes show all predictions above the model's fixed threshold. B and M mean benign and malignant. Image proportions are preserved. Hair, rulers or other visible features are not assumed to cause an error without a controlled test.''')
    explanations=[
        'The lesion is malignant, but n predicts benign with confidence 0.737 and IoU 0.676. The box overlaps the lesion well enough for the matching rule, but its class is wrong: a benign FP and a malignant FN. s returns no box above 0.80. Its saved lower-score predictions contain a malignant box at 0.427, which the fixed threshold removes. This identifies the output errors, but does not explain why the models gave different scores.',
        'The lesion is benign, but n predicts malignant with confidence 0.758 and IoU 0.857. Good overlap does not fix the wrong class: this is a malignant FP and a benign FN. s returns no box above 0.80. It avoids a malignant alert, but also fails to detect the benign lesion, so it should not be described as a correct benign prediction.',
        'Both models predict malignant, but their boxes are inaccurate. n has confidence 0.610 and IoU 0.436; s has confidence 0.876 and IoU 0.220. Both raise an image alert, but neither reaches the IoU 0.5 detection requirement. Each adds a malignant FP and FN. The higher confidence of s does not mean its box is more accurate.',
        'n predicts two benign boxes for one benign lesion. Their confidences are 0.635/0.546 and GT IoUs are 0.623/0.627. The higher-confidence eligible box matches the GT and counts as TP; the other counts as FP. The boxes overlap each other by IoU 0.663, below the NMS threshold of 0.7, so both remain. Lowering NMS might remove this duplicate but could change other results. It was not adjusted after seeing this case. s returns no box above 0.80.',
        'Neither model returns a box above its selected threshold for this malignant lesion. The saved predictions after NMS contain a malignant box overlapping the lesion at score 0.00531 for n and 0.01192 for s. Both fall below 0.50/0.80. The boxes were removed by score filtering, so this case should not be described as having no localized candidate at all. The thresholds were not lowered to rescue this already inspected case.'
    ]
    assert [c['image_id'] for c in final['cases']]==['ISIC_0012989','ISIC_0013114','ISIC_0014072','ISIC_0013000','ISIC_0013198']
    for i,(case,explanation) in enumerate(zip(final['cases'],explanations),1):
        if i in (3,5):page()
        add(f'### Case {i}: {case["category"]}\n\n{case["image_id"]}; GT: {"Malignant" if case["class_id"] else "Benign"}.')
        add(f'![Case {i}: {case["image_id"]}]({case["figure"]})')
        add(explanation)
    add(r'''## Model choice and next steps

The original experiment retains the main-study n model at confidence 0.50 and its s comparison at 0.80. n was the baseline selected before the experiment, and both models have completed evaluation with fixed thresholds. This choice does not claim that n has the highest score among every later run. The main split separates known lesion and patient links, but some patient links are still unknown. This paragraph records the original experiment; the current development decision follows the known-lesion-only comparison at the end of the report.

Before the latest teacher advice, expanded s at 0.50 was considered a development candidate: it detects 16/41 malignant validation lesions with 10/124 benign alerts. That recommendation is now historical because the current scope excludes unknown-lesion-ID training images. The old decision remains in `coursework_model_decision.json`; `known_optimization_decision_20261021.json` records the current decision using only the known-lesion cohort.

The main weakness is malignant detection when false alerts are limited. The final section compares a fixed batch of known-lesion-only ablations, with a prespecified stopping and seed-confirmation rule. No further open-ended search is planned on this validation set. Course questions, personal check-off and submission preparation remain. A benign prediction or no alert cannot rule out malignancy.

The longer training-budget comparison is now reported in the final section. Further work could examine class-balanced sampling with actual sampling rates measured, or a YOLO detector followed by a separate classifier. These ideas remain untested in this study and would need a new evaluation plan; the two-stage approach would also need to meet the course requirements. Better identity and diagnosis records would strengthen that work. Hair removal, cropping and lesion-size effects have not been isolated by these experiments.''')
    page()
    add(r'''## Reproducibility and use

Saved main training and validation results are in `outputs/provisional_20261017_669143`; reassessment and timing results are in `outputs/reassessment_20261017_669815`. Expanded n/s results are in `outputs/expanded_training_20261017_669929` and `outputs/expanded_s_20261018_670000`. Raw outputs and weights are excluded from Git, while summary JSON files and report figures are included. `expanded_evidence_provenance.json` identifies the copied exploration records. The PDF is generated from this README. Earlier commands are kept in `RUNBOOK.md`.

### Restoring the recorded study layout

In the course checkout, experimental sources are stored under this module's `research_sources/` directory. Restore them into a new working directory before using commands that refer to top-level `studies/`. The helper verifies source hashes and copies this module plus the recorded study sources; it does not copy photos, metadata, labels, prediction caches or weights. Run subsequent commands from the restored workspace.

```bash
python recognition/YOLO_s4905914/restore_study_sources.py \
  --destination ../comp3710-reproduction
cd ../comp3710-reproduction
```

### Environment and inputs

The school runs used Python 3.13.16, PyTorch 2.8.0, torchvision 0.23.0 and Ultralytics 8.3.221. `requirements.txt` fixes the ML package versions, and `requirements-linux-a100.lock.txt` records the tested Linux environment. PDF generation uses the separate `requirements-report.txt`. Use `experiment_provisional.json` to reproduce the original main experiment; `experiment.json` is the original disabled draft.

To reproduce the main experiment, restore the image lists and labels in `data/known_lesion_v1`, the recorded metadata snapshots and the unchanged school images. These inputs and trained weights are not included in Git. The local transfer utility packages image lists, labels and source records, but not photos, masks or weights; it checks file integrity. To recreate labels in a new directory, use `prepare_dataset.py` with `sources.json` and `PROTOCOL.json`. It checks source hashes and refuses to overwrite existing output.

In a separate Python 3.13 environment, install the recorded packages, download the original weights and run the checks below. Real training requires an allocated GPU and should not run on a login node:

```bash
python -m pip install -r recognition/YOLO_s4905914/requirements.txt \
  -c recognition/YOLO_s4905914/requirements-linux-a100.lock.txt
python recognition/YOLO_s4905914/fetch_weights.py --output weights/original_coco
python -m unittest discover -s recognition/YOLO_s4905914 -p 'test_*.py' -v
python recognition/YOLO_s4905914/train.py plan \
  --config recognition/YOLO_s4905914/experiment_provisional.json \
  --prepared data/known_lesion_v1
```

`dataset.py` creates links to the original photos and copies the labels, leaving source photos unchanged. Training and evaluation commands are in `RUNBOOK.md`; select `experiment_provisional.json` for this experiment. The current candidate uses `studies/known_lesion_optimization_20261021/n_geometry.json` and the checkpoint/lock recorded in `known_optimization_decision_20261021.json`. Do not pair the original baseline config with the geometry checkpoint. The original school launchers depend on historical paths. On a new machine, check the inputs and GPU first, then use the direct commands in a new GPU allocation.

### Original baseline: saved-model example

After restoring the checkpoint and its validation threshold record, this command saves prediction JSON and an overlay image. Replace the example image path with a dermoscopic photograph. The output confidence is not a cancer probability.

```bash
python recognition/YOLO_s4905914/predict.py \
  --config recognition/YOLO_s4905914/experiment_provisional.json \
  --checkpoint outputs/provisional_20261017_669143/train_n/fit/weights/best.pt \
  --lock outputs/provisional_20261017_669143/val_n/selection_lock.json \
  --image /path/to/image.jpg --output outputs/new_prediction --device 0
```

To check saved main-study results without running inference again, use `compare_validation.py` and `analyse_reassessment.py`. Expanded-study code, configurations and commands are in `studies/expanded_training_20261017/` at the project root; `compare_ns.py` recalculates all 28 saved threshold results. `build_final_report.py` and `report_expanded.py` build this README, and `export_readme_pdf.py` creates the PDF and its file-hash record. A fixed seed does not guarantee identical numerical results on every platform.''')
    page()
    add(r'''## Limitations and course status

Missing identity information is the main data limitation. The official CSV and all 26 API pages matched all 2,594 images, but 1,496 still lacked lesion IDs and 2,204 lacked patient IDs. Task 3 grouping IDs did not match this Task 1 set, and the checked 2019/2020 metadata added no missing lesion links. This does not rule out other sources. The main split keeps known links together, but full patient independence is unproven and using a subset may change the mix of cases. Adding unknown-ID images in the later experiment also means full lesion independence cannot be established. Exact-image deduplication cannot resolve missing identities.

The earlier image-level experiment and its test failures had already been viewed. The current validation set contains 120 images from old training, 22 from old validation and 23 from old test. Reassessment contains 122 from old training, 22 from old validation and 21 from old test. The new models started from original COCO weights, but that does not erase prior knowledge of the images. Reassessment is therefore a retrospective check, not a new blind test. It must not guide further model or threshold choices while being described as independent evaluation.

Each evaluation set has only 41 malignant images. The original and expanded experiments each used one training seed; any seed-confirmation runs for the later optimization are listed separately. Diagnosis labels came from metadata and were not independently checked by clinicians. The comparisons do not establish a single cause of the classification errors. Class imbalance, label definitions and visible image features remain possible factors requiring separate evidence. There was no mobile-device test, probability calibration, measurement of real referrals or biopsies, or clinical safety study. The original reassessment results do not demonstrate the 0.8 target. The supplied teacher reply now requires Accuracy and additional imbalance-sensitive metrics, plus overall and per-class IoU. These are added later with explicit denominators. The treatment of wrong-class or missed predictions was not specified in that reply.

YOLOv8 is allowed by the listed version requirement and falls in the Normal difficulty band. Use of the known-lesion subset follows the teacher advice reported on 21 October; imported model components still need confirmation. The teacher's metric clarification is recorded separately and does not establish full course acceptance. Tutor check-off and the exact deadline also need checking. This code and report have not yet been submitted through a new course PR or Turnitin. The final material needs student review and may need changes after tutor feedback.

## Artificial Intelligence Usage Disclosure

OpenAI ChatGPT/Codex helped with planning, metadata and grouping checks, mask-to-box checks, training and evaluation code, diagnostic scripts, tests, resource measurements, figures, report writing and explanations. The YOLO network, layers, losses, trainer and AP calculation come from Ultralytics 8.3.221. These were not written from scratch for this assignment. This disclosure covers the assistance recorded in this project.

Checks included inspecting the pinned library code, comparing file hashes and preprocessed tensors, checking pretrained-weight loading, running the original 40 unit tests, seven additional metric tests and synthetic n/s pipelines, checking real photos, independently matching saved detections and reviewing the rendered PDF. AP itself was not independently recalculated. These checks cannot prove that every diagnosis or conclusion is correct. The student is responsible for reviewing, correcting and explaining the final work, including the parts prepared with AI assistance.

## References

1. ISIC. [Official challenge datasets and ground-truth downloads](https://challenge.isic-archive.com/data/). Task 1 photographs/masks are the source of the lesion images and box geometry used here.
2. ISIC Archive. [Collection 63](https://api.isic-archive.com/collections/63/). Official image metadata and locally preserved CSV/API responses supply diagnosis and recorded identity fields. The audit and its coverage limits are documented in `ID_RECHECK_zh.txt`.
3. Ultralytics. [Version 8.3.221 source](https://github.com/ultralytics/ultralytics/tree/v8.3.221), particularly nn/tasks.py, nn/modules, models/yolo/detect, utils/loss.py and utils/metrics.py. Upstream code is attributed under its AGPL-3.0 terms. `MODEL_COMPONENTS_zh.txt` at the project root records the component and initialization audit.
4. Wang, C.-Y., Bochkovskiy, A., and Liao, H.-Y. M. (2022). [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696). Background reference for real-time object detection; the present runs use YOLOv8.
5. COMP3710/COMP7710 assessment specification supplied by the student, filename `COMP3710_Report_2026_draft_v2.3.pdf`, cover marked DRAFT / Version 2.2. Project 3 and submission requirements are on pages 6-7 and 11-16. Previously checked Ed guidance is recorded in `COURSE_REQUIREMENTS_AUDIT_zh.txt`; later official course clarifications take precedence.
''')
    add_known_optimization(code,add,page,table,pct)
    add_course_closeout(code,add,page,table,pct)
    (code/'README.md').write_text('\n'.join(parts))
    print(code/'README.md')


if __name__=='__main__':build()
