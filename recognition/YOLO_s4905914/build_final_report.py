"""Assemble the report README from audited experimental records and explanations."""
from pathlib import Path
import json
from report_expanded import add_expanded_section


def build():
    code=Path(__file__).resolve().parent;root=code.parents[1]
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
    add('''# Lesion detection and classification with YOLOv8

COMP3710 Assignment 3, Project 3 | s4905914 | Updated 18 October 2026

The main experiment compares YOLOv8n, the predeclared primary, with YOLOv8s on an ISIC2018 Task 1 cohort with known lesion IDs. It includes fresh 60-epoch training, validation selection, locked supplementary reassessment, resource measurements and five failure examples. A later exploration adds training images with missing IDs and compares n/s on the same validation images. Its results are reported separately.

On supplementary reassessment, n has mAP50:95 0.3619 and detects 10/41 malignant lesions at confidence 0.50; s has mAP50:95 0.3527 and detects 6/41 at 0.80. n generates malignant alerts on 16/124 benign images, exceeding the 10% budget that was satisfied on validation. Neither model demonstrates adequate malignant screening sensitivity. n remains the primary by the original protocol, and neither its threshold nor the model policy was changed after reassessment.

Course acceptance of the restricted cohort and third-party component packaging remains unconfirmed. Personal practical check-off has not been verified. These are experimental results, not a claim that all assessment requirements have been accepted or that the prototype is clinically safe.

## Data and scope

The source consists of 2,594 school-provided dermoscopic photographs. Bounding boxes are the smallest axis-aligned rectangles around the corresponding official binary masks, resized to photo dimensions with nearest-neighbour interpolation. The verified coordinates are reused in this restart. Class labels come from official metadata matched by image ID, not from the masks. The school source photos and masks were not altered. The official dataset and metadata sources are listed in References [1,2].

The full metadata contains 2,074 Benign, 519 Malignant and one Indeterminate image. Only 1,098 have a recorded lesion_id; these form the main study, excluding the other 1,496 from all its phases. The later exploration adds 1,495 binary-labelled missing-ID images to training only. Indeterminate is excluded, not relabelled. The outputs are Benign=0 and Malignant=1. Each image has one annotated lesion; multi-lesion performance is not established.''')
    table(['Split','Benign / Malignant','Images'],[['Train','576 / 192',768],['Validation','124 / 41',165],['Supplementary reassessment','124 / 41',165]])
    add('''![Data and experiment flow](assets/pipeline.png)

Groups are connected components of all recorded lesion/patient links before cohort filtering. With seed 3710, recorded lesion and patient IDs do not cross splits. There are 1,083 distinct recorded lesions and 1,043 assignment groups; groups are not fabricated patient IDs. The cohort still has 708 images without patient_id. Full patient independence is therefore unproven.''')
    page()
    add('''## One-page feasibility review

The following is the pre-training written plan prepared on 17 October. It is retained as a dated planning record. Its statements about unmeasured runtime refer to that stage; actual results appear later. A written review does not establish that tutor check-off was completed.''')
    review=(root/'FEASIBILITY_REVIEW.txt').read_text().splitlines()
    headings={'Purpose and scope','Data and model choice','Five testable acceptance criteria','Evidence, risks and next decision','Course and authorship notes'}
    body=[]
    for line in review[3:]:
        body.append(('### '+line) if line in headings else line)
    add('\n'.join(body))
    page()
    add('''## Model and implementation

YOLOv8 is a one-stage detector. A convolutional backbone extracts spatial features; the neck combines information across scales; the detection head predicts box distances and class scores at three resolutions. In the pinned implementation, Conv blocks combine convolution, normalization and activation, C2f blocks combine feature branches, and SPPF pools features at multiple effective scales. Box and class prediction branches are separate. These are upstream PyTorch implementations, not original components written for this assignment [3]. YOLOv7 provides background on real-time detection and the accuracy-speed trade-off [4]; the models trained here are YOLOv8n and YOLOv8s.

The loss combines CIoU box regression, distribution focal loss (DFL) for box distances, and binary-cross-entropy class loss. DFL is a localization component, so a lower dfl_loss alone is not evidence of better malignant classification. The score is not a calibrated malignancy probability. After inference, single-label class-aware NMS removes some overlapping same-class boxes; it does not enforce one prediction per image.

`modules.py` constructs the actual two-class DetectionModel and loads trained models. `dataset.py` supplies the loader used by `train.py`. `train.py` performs fitting and evaluation, saves provenance and checkpoints, and records losses and metrics. `predict.py` loads a saved checkpoint with its validation selection lock and saves JSON plus a prediction overlay. Helpers implement grouping, audits, preprocessing, scoring and reporting. The course's acceptance of integrating Conv/C2f/SPPF/Detect from Ultralytics rather than including local implementations remains unresolved.

### Fixed training protocol''')
    table(['Setting','YOLOv8n','YOLOv8s'],[
        ['Initial weights','Original COCO n','Original COCO s'],['Library','Ultralytics 8.3.221','Same'],
        ['Epochs / completed optimizer calls','60 / 2,880','60 / 2,880'],['Image size / batch / nominal batch','640 / 16 / 16','Same'],
        ['Seed / arithmetic','3710 / FP32','Same'],['Optimizer','AdamW','Same'],
        ['Initial LR / weight decay','0.001 / 0.0005','Same'],['Warm-up / LR schedule','3 epochs / linear','Same'],
        ['Augmentation','Horizontal/vertical flips, p=0.5 each','Same'],
        ['Mosaic / MixUp / oversampling','Off / off / none','Same'],['Early stopping / resumed training','Disabled / no','Same'],
        ['Selected epoch by macro mAP50:95','32','42']])
    add('''The whole backbone, neck and prediction branches remain trainable; the fixed DFL integral coordinates do not. The first optimizer call has zero LR and the final epoch uses LR 0.0000265 under the pinned upstream schedule. Each recorded optimizer call is checked against that schedule. Equal batch and nominal batch avoid differing gradient-accumulation budgets.

Initialization is not identical across sizes: n transfers 319/355 state entries and s 349/355. Counts include buffers. n's classification hidden width changes from 80 to 64, whereas s retains width 128 and replaces the final outputs. Both transfer backbone, neck and box-regression state. This difference belongs in the comparison; the experiment cannot attribute outcomes solely to parameter count.''')
    page()
    add('''## Evaluation and selection rules

Evaluation uses one aspect-preserving letterbox resize to 640, followed by padding; it does not stretch images. The custom validation loader and ordinary predictor were checked for identical input pixels on synthetic shapes. Scores use a 0.001 floor, single-label class-aware NMS at IoU 0.7 and at most 300 boxes. This differs from default multi-label validation, so AP should not be mixed with results from a different validation policy.

The best checkpoint maximizes validation macro mAP50:95 over 60 epochs. Upstream training scores are rounded to five decimal places; the later epoch wins exact ties. Validation uses EMA weights. Upstream saves EMA in FP16 and reloads to FP32. The selected training-time score and the subsequently reloaded-checkpoint score need not be numerically identical. Internal training validation uses batch 32; standalone evaluation requests batch 16. Those are execution differences, not a controlled explanation for score discrepancies.

A TP needs the correct class, confidence strictly above the operating threshold, and IoU >= 0.5. Each GT matches at most one prediction, choosing the highest-confidence eligible candidate. Every remaining prediction is FP for its predicted class. An unmatched GT is FN. A wrong-class or badly located box can therefore contribute both FP and FN. An image-level malignant alert requires any retained malignant box, without an IoU condition. These alerts are not actual biopsy counts.

AP50 and AP50:95 use the low-score sweep. Macro mAP weights the two classes equally. The operating threshold instead maximizes malignant detection recall subject to malignant alerts on at most 10% of benign validation images: at most 12 of 124. Ties use precision, fewer benign alerts, then higher threshold. The grid, fixed before training, is 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 and 0.95. This is a classroom budget, not a clinically established standard.

![Validation threshold trade-off](assets/threshold_tradeoff.png)

Markers are the 14 evaluated thresholds; connecting lines do not imply that intermediate thresholds were selected. n selected 0.50 and s selected 0.80. At the shared threshold 0.50, s detects 22 malignant validation cases but alerts on 28 benign cases, exceeding the budget. At 0.70 it has 14 malignant TP and 14 benign alerts, still exceeding the budget. The grid was not refined after seeing these results.

![Training history](assets/training_curves.png)

Classification loss on validation uses a logarithmic axis to show early spikes and later detail. All logged scalar values are finite; the cause of early spikes was not isolated. Completing 60 epochs does not prove convergence, and selecting an earlier best checkpoint does not by itself prove overfitting.''')
    page()
    add('''## Main-study quantitative results

Validation and supplementary reassessment each contain 124 benign and 41 malignant images, but they are different image sets. Reassessment used the already selected best checkpoints and thresholds once per model. No reassessment threshold sweep, checkpoint search or model replacement was performed.''')
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
    add('''n's validation malignant detection recall was 13/41 (31.71%) with 11/124 benign alerts; s's was 11/41 (26.83%) with 7/124 benign alerts. On reassessment these detection counts fall to 10 and 6. n's 16 benign false alerts exceed the validation budget, while s's three remain below it. The budget was a selection constraint on validation, not a guarantee for another sample. The locks were retained despite this result.

Of the 41 malignant reassessment images, both models detect five, n alone detects five, s alone detects one, and both miss 30. n has higher AP and recall here; s has higher operating-point precision and fewer false alerts. One seed and small class counts do not establish a general or statistically significant winner. Comparing these values with the old image-level pilot would confound changes in data, grouping and evaluation protocol.''')
    add('### Localization on supplementary reassessment')
    table(['Mean IoU at locked operating points','YOLOv8n','YOLOv8s'],[
        [f'{name}: {label}',*[f'{point(r[m])["classes"][name]["iou"][key]["mean"]:.4f} (N={point(r[m])["classes"][name]["iou"][key]["denominator"]})' for m in ('n','s')]]
        for name in ('Benign','Malignant') for label,key in [('all GT, correct class','all_gt_class_correct'),('all GT, ignore class','all_gt_class_agnostic'),('matched TP only','tp_only')]])
    add('''All-GT means use the best retained box under the stated class condition, with missing predictions scored zero. TP-only excludes misses and has a different denominator. Even malignant TP-only means are below 0.8 for both models here; no reported malignant mean establishes the course localization target. The teaching team's intended aggregation rule remains unconfirmed.''')
    add_expanded_section(code,add,page,table,pct)
    page()
    add('''## Main-study resource measurements and scope

Training ran on an A100 host in job 669143. Locked reassessment and a separately declared resource test ran on NVIDIA A100-PCIE-40GB in job 669815. The latter used validation images only, without labels for selection, with FP32, 640-pixel input and serial batch-one calls. Ten warm-up calls preceded three passes over all 165 validation paths, giving 495 timed calls per model. CUDA synchronization bracketed each wall-time measurement.''')
    table(['Measurement','YOLOv8n','YOLOv8s'],[
        ['Training parameters',*[f'{v[m]["parameters"]:,}' for m in ('n','s')]],
        ['Training-call elapsed time',*[f'{v[m]["training_seconds"]/60:.2f} min' for m in ('n','s')]],
        ['Peak training PyTorch CUDA allocation',*[f'{v[m]["peak_cuda_allocated_GiB"]:.2f} GiB' for m in ('n','s')]],
        ['Library inference-stage mean',*[f'{r[m]["benchmark"]["library_inference"]["mean_ms"]:.2f} ms' for m in ('n','s')]],
        ['File-to-prediction wall mean',*[f'{r[m]["benchmark"]["wall"]["mean_ms"]:.2f} ms' for m in ('n','s')]],
        ['File-to-prediction median / p95',*[f'{r[m]["benchmark"]["wall"]["median_ms"]:.2f} / {r[m]["benchmark"]["wall"]["p95_ms"]:.2f} ms' for m in ('n','s')]],
        ['Serial file-to-prediction throughput',*[f'{r[m]["benchmark"]["serial_images_per_second"]:.2f} images/s' for m in ('n','s')]],
        ['Peak inference PyTorch CUDA allocation',*[f'{r[m]["benchmark"]["peak_cuda_allocated_bytes_after_warmup"]/2**20:.2f} MiB' for m in ('n','s')]]])
    add('''The wall measurement includes file reading and decoding, preprocessing, device transfer, model execution, NMS, filtering at the locked threshold and copying retained predictions to CPU. It excludes model startup, warm-up, drawing overlays, saving output and application UI. The library inference stage excludes the surrounding pipeline. Throughput is 1,000 divided by mean wall milliseconds for this serial workload, not reciprocal median latency or batched peak FPS.

Models were timed sequentially in n-then-s order, and repeated paths may benefit from operating-system or network file caches. GPU clocks, kernel execution and order effects were not independently controlled. s happened to have lower measured latency here; that is an observation of this run, not proof that a larger model is generally faster. Neither the library-stage timing nor the full measured call represents mobile-device or complete clinic-application speed. Peak allocated tensor memory is not total GPU memory usage.

### Verification

The initial preparation passed 40 unit tests, complete synthetic n/s pipelines, an exact initialization-transfer audit and a real-input audit. All 1,098 photos were decoded and matched to school-side SHA-256 values; dimensions, orientation and label coordinates were checked. No exact decoded-image duplicates crossed splits. That check does not prove missing patient identities are independent.

The completed validation audit independently reproduced all 28 declared threshold points with a NumPy matcher. The supplementary audit reproduced both locked operating points, including TP/FP/FN, image alerts and IoU means, from saved predictions. It checked image membership and GT coordinates, configuration, manifests, checkpoint and selection-lock hashes. AP remains the fixed Ultralytics implementation's output; it was not independently recomputed.

All training/evaluation source files needed by reassessment matched the completed training records. The local transfer packager was revised separately, outside that import path. Checkpoint and threshold files were not changed. The resource audit checked all 495 wall observations per model and recomputed the reported means and serial throughput. These checks support software and record consistency; they do not validate every diagnosis label or turn development evidence into a blinded clinical study.''')
    page()
    add('''## Five main-study supplementary failure cases

Cases are selected by the lowest image ID satisfying each stated failure category, without reusing an image. They illustrate mechanisms in the saved output and are not a random sample. Dashed outlines show GT; solid outlines show every prediction retained at the model's locked threshold. B and M mean predicted Benign and Malignant. The photos retain their original aspect ratios. Visible artefacts are not treated as causal explanations without a controlled experiment.''')
    explanations=[
        'The GT is malignant. n retains a benign prediction with confidence 0.737 and IoU 0.676, so localization alone does not produce a malignant TP. It contributes a benign FP and a malignant FN. s retains no prediction at 0.80; its low-score cache contains a localized malignant candidate at 0.427. That candidate is excluded by the already selected threshold. This does not establish why the models assigned different scores.',
        'The GT is benign. n predicts Malignant with confidence 0.758 and IoU 0.857. It is a malignant FP and a benign FN despite good spatial overlap. s retains no box at 0.80, avoiding the malignant alert but still leaving the benign GT undetected. It is therefore inaccurate to describe s as a correct benign detection on this image.',
        'The GT is malignant. n retains a malignant box with confidence 0.610 and IoU 0.436; s retains one at confidence 0.876 and IoU 0.220. Both trigger image alerts, but both boxes fail the IoU 0.5 matching rule and each produces a malignant FP and FN. The larger confidence of the s box does not imply better localization.',
        'The GT is benign. n retains two benign boxes at confidence 0.635/0.546 and GT IoU 0.623/0.627. The first eligible, higher-confidence box is the TP; the other is FP because one GT can match only once. Their pairwise IoU is 0.663, below the NMS threshold 0.7, so their survival is consistent with that NMS setting. Lowering NMS might remove this duplicate, but could affect other cases; no such change was selected from reassessment. s retains no box at 0.80.',
        'The GT is malignant and neither model retains a prediction at its operating point. The post-NMS cache contains a localized malignant candidate at score 0.00531 for n and 0.01192 for s. These outputs are below 0.50/0.80. This supports a score-filtering description of the miss, not a claim that the image was never localized internally. Lowering thresholds for this particular case after seeing the supplementary result would violate the fixed evaluation protocol.'
    ]
    assert [c['image_id'] for c in final['cases']]==['ISIC_0012989','ISIC_0013114','ISIC_0014072','ISIC_0013000','ISIC_0013198']
    for i,(case,explanation) in enumerate(zip(final['cases'],explanations),1):
        if i in (3,5):page()
        add(f'### Case {i}: {case["category"]}\n\n{case["image_id"]}; GT: {"Malignant" if case["class_id"] else "Benign"}.')
        add(f'![Case {i}: {case["image_id"]}]({case["figure"]})')
        add(explanation)
    add('''## Engineering recommendation

Retain the known-lesion main-study n checkpoint at confidence 0.50 for the coursework prototype, with its original s comparator at 0.80. This preserves the declared experiment and its completed reassessment; it is not a claim that n has the best score across all later runs. Both recorded lesion and patient links are isolated in the main cohort, although complete patient independence and course acceptance remain unconfirmed.

Within the expanded-data exploration, s at 0.50 is the development candidate under the same 10% validation false-alert budget: it detects 16/41 malignant images with 10/124 benign alerts, compared with expanded n at 0.70 detecting 12/41 with 5/124 alerts. This recommendation follows inspection of the exploratory results; it was not the original primary-model declaration. Unknown identity links and repeated validation use prevent promoting it as an independently validated replacement. The exact checkpoints, configurations and existing locks are recorded in `coursework_model_decision.json`.

The main limitation remains malignant detection under the false-alert constraint. Expanded s reduces false alerts but loses recall at the primary shared thresholds; its small advantage at 0.70 is insufficient to establish stable superiority. Stop additional performance-driven tuning for this revision and finish the report review, course clarifications, check-off and submission preparation. Do not retune on reassessment cases. A benign output or absent alert cannot rule out malignancy.

For a future, separately declared study, prioritize trustworthy grouping/diagnostic provenance and a new evaluation plan. On development data, compare one factor at a time: a fixed augmentation schedule with a longer update budget; then class-balanced sampling with measured actual sampling frequencies. A localization-plus-classification design could be investigated because well-located wrong-class boxes exist, but its value is unproven and it must retain a qualifying YOLO detector or obtain course approval. A controlled artefact-removal or crop experiment is needed before attributing errors to hair, rulers or lesion size. None of these future experiments is claimed as completed here.''')
    page()
    add(r'''## Reproducibility and use

The main training/validation run is `outputs/provisional_20261017_669143`; the locked supplementary/resource run is `outputs/reassessment_20261017_669815`. Expanded n/s runs are `outputs/expanded_training_20261017_669929` and `outputs/expanded_s_20261018_670000`. Raw outputs and weights are excluded from Git. Aggregate JSON files and figures are included; `expanded_evidence_provenance.json` records the copied exploration evidence. This README is the prose source for the PDF prepared for submission; `RUNBOOK.md` preserves historical commands.

### Environment and inputs

The school run used Python 3.13.16, PyTorch 2.8.0, torchvision 0.23.0 and Ultralytics 8.3.221. `requirements.txt` pins the ML packages and `requirements-linux-a100.lock.txt` records the tested Linux environment. The report exporter uses a separate environment defined by `requirements-report.txt`. The precise configuration is `experiment_provisional.json`; `experiment.json` remains the original disabled draft.

Restore the audited manifests and labels under `data/known_lesion_v1`, the recorded source snapshots and unchanged school images. A plain Git checkout intentionally omits data and trained weights. The local preparation transfer utility includes manifests, labels and provenance but not photographs, masks or weights. It validates package integrity, not scientific correctness. Use `prepare_dataset.py` with `sources.json` and `PROTOCOL.json` if rebuilding labels into a new directory; it verifies input hashes and refuses to overwrite existing outputs.

In a separate Python 3.13 environment, install the fixed packages, fetch the recorded original weights, and run checks. Use an allocated GPU for real training, not a login node:

```bash
python -m pip install -r recognition/YOLO_s4905914/requirements.txt \
  -c recognition/YOLO_s4905914/requirements-linux-a100.lock.txt
python recognition/YOLO_s4905914/fetch_weights.py --output weights/original_coco
python -m unittest discover -s recognition/YOLO_s4905914 -p 'test_*.py' -v
python recognition/YOLO_s4905914/train.py plan \
  --config recognition/YOLO_s4905914/experiment_provisional.json \
  --prepared data/known_lesion_v1
```

`dataset.py` stages symlinks to the declared original photographs and copied labels without modifying source images. The direct fit/evaluate commands are documented in `RUNBOOK.md`; use `experiment_provisional.json` for this study. Original school launchers record the historical jobs and expect that project's existing evidence paths. For a fresh machine, perform input and GPU checks and use the documented direct commands within a new GPU allocation rather than assuming those historical paths exist.

### Saved-model example

With the completed checkpoint and validation lock restored, the following creates JSON and an image overlay. The input path is a user-supplied dermoscopic photograph; output scores are not cancer probabilities.

```bash
python recognition/YOLO_s4905914/predict.py \
  --config recognition/YOLO_s4905914/experiment_provisional.json \
  --checkpoint outputs/provisional_20261017_669143/train_n/fit/weights/best.pt \
  --lock outputs/provisional_20261017_669143/val_n/selection_lock.json \
  --image /path/to/image.jpg --output outputs/new_prediction --device 0
```

To audit archived main-study results without inference, use `compare_validation.py` and `analyse_reassessment.py`. Expanded-study sources, fixed configurations and commands are under `studies/expanded_training_20261017/` at the project root; `compare_ns.py` replays all 28 saved operating points. `build_final_report.py` and `report_expanded.py` assemble this README from the copied evidence; `export_readme_pdf.py` exports its prose, tables and figures with source hashes. Platform execution can affect numerical reproducibility despite the fixed seed.''')
    page()
    add('''## Limitations and course status

The most important limitation is identity information. Rechecking the official CSV and all 26 API pages found 2,594 matched images but still 1,496 missing lesion IDs and 2,204 missing patient IDs. Task 3 grouping IDs do not intersect this Task 1 set; checked 2019/2020 metadata filled no missing lesion links. Other sources may exist. The main cohort isolates recorded lesion/patient links but cannot prove full patient independence. Restriction may change the case mix. Adding unknown-ID images in the exploration also removes any complete lesion-independence assurance; exact-image deduplication cannot restore it.

The prior image-level pilot and its test failures were inspected. Of the new validation images, 120 were in the old training set, 22 in old validation and 23 in old test. Of the new reassessment images, 122 were in old training, 22 in old validation and 21 in old test. Both new models started from original COCO weights rather than old fitted checkpoints, but this cannot remove the researcher's historical exposure. This is retrospective supplementary evaluation, not a restored blind test. It is now fully exposed and must not be reused to choose further models or thresholds while retaining an independent-evaluation claim.

Only one training seed and 41 malignant images per evaluation split were used. Diagnostic classes were taken from metadata and were not independently clinically adjudicated. There is no evidence here isolating imbalance, insufficient training, augmentation or visible artefacts as the cause of errors. There is no mobile-device test, calibrated malignancy probability, real referral/biopsy count or clinical safety demonstration. The course's intended IoU aggregation and classification performance criterion remain unresolved; the results do not demonstrate the stated malignant localization target under the reported means.

YOLOv8 meets the listed version option and Normal difficulty band. The report discloses rather than resolves the pending acceptance of the known-lesion cohort and imported model components. Personal feasibility check-off and the exact submission deadline must be confirmed in the course system. This local report and code have not been submitted through a new course PR or Turnitin. The student must review the final report before submission. Subsequent teacher feedback may require changes; no retrospective approval is assumed.

## Artificial Intelligence Usage Disclosure

OpenAI ChatGPT/Codex assisted with planning, metadata and grouping checks, mask-box provenance checks, training/evaluation integration, diagnostic scripts, tests, resource measurement, figures, report drafting and concept explanations. The YOLO architecture, layers, losses, trainer and AP implementation are supplied by Ultralytics 8.3.221; they are not student-authored network components. No other AI tool usage is inferred.

Verification included fixed-source inspection, hash checks, actual preprocessing tensor comparisons, original-weight transfer checks, 40 unit tests, synthetic end-to-end n/s runs, real-photo integrity audits, independent operating-point matching and inspection of the rendered report. AP itself was not independently implemented or recomputed. These checks do not prove every clinical label or all scientific claims correct. The student remains responsible for reading, correcting and explaining all submitted code and results; this assistance must not be represented as entirely unaided work.

## References

1. ISIC. [Official challenge datasets and ground-truth downloads](https://challenge.isic-archive.com/data/). Task 1 photographs/masks are the source of the lesion images and box geometry used here.
2. ISIC Archive. [Collection 63](https://api.isic-archive.com/collections/63/). Official image metadata and locally preserved CSV/API responses supply diagnosis and recorded identity fields. The audit and its coverage limits are documented in `ID_RECHECK_zh.txt`.
3. Ultralytics. [Version 8.3.221 source](https://github.com/ultralytics/ultralytics/tree/v8.3.221), particularly nn/tasks.py, nn/modules, models/yolo/detect, utils/loss.py and utils/metrics.py. Upstream code is attributed under its AGPL-3.0 terms. `MODEL_COMPONENTS_zh.txt` at the project root records the component and initialization audit.
4. Wang, C.-Y., Bochkovskiy, A., and Liao, H.-Y. M. (2022). [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696). Background reference for real-time object detection; the present runs use YOLOv8.
5. COMP3710/COMP7710 assessment specification supplied by the student, filename `COMP3710_Report_2026_draft_v2.3.pdf`, cover marked DRAFT / Version 2.2. Project 3 and submission requirements are on pages 6-7 and 11-16. Previously checked Ed guidance is recorded in `COURSE_REQUIREMENTS_AUDIT_zh.txt`; later official course clarifications take precedence.
''')
    (code/'README.md').write_text('\n'.join(parts))
    print(code/'README.md')


if __name__=='__main__':build()
