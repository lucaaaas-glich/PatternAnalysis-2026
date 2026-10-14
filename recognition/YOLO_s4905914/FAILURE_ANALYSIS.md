# Five frozen-test failure cases

Reviewed on 2026-10-15 using the downloaded GT/n/s PNGs and cached predictions from test job 661498. Primary model: YOLOv8n; comparator: YOLOv8s; fixed confidence: 0.05. No new inference, retraining, or test-based threshold selection was performed.

The left panel uses a bounding box derived from the human mask. Its diagnosis class comes from matched ISIC metadata, not the mask or our interpretation of the photograph. Prediction IoU measures spatial overlap regardless of class; a detection TP requires the correct class AND IoU >= 0.5, with one prediction matched to the one annotated lesion.

## Observed errors

| Image ID / ground truth | YOLOv8n evidence | YOLOv8s evidence | Interpretation |
| --- | --- | --- | --- |
| ISIC_0014897 / Malignant | Benign 0.889; IoU 0.919 | Benign 0.842; IoU 0.882 | Both locate well but misclassify: each gives a benign FP and malignant FN, with no malignant alert. |
| ISIC_0009875 / Benign | Malignant 0.733; IoU 0.792 | Malignant 0.655; IoU 0.840 | Both falsely alert on a real lesion, not a demonstrated background-object hallucination: malignant FP and benign FN. |
| ISIC_0015071 / Malignant | Malignant 0.518; IoU 0.425 | Benign 0.257 / IoU 0.598; Malignant 0.179 / IoU 0.689 | n alerts but fails detection matching: malignant FP and FN. s has a malignant TP plus an unmatched benign FP; its conflicting classes are not two GT lesions. |
| ISIC_0012178 / Benign | Three benign boxes above 0.05 | Five benign boxes above 0.05 | One GT allows only one TP: n has one benign TP plus two FP; s has one benign TP plus four FP. Neither creates a malignant image alert. |
| ISIC_0012758 / Malignant | Benign 0.737; IoU 0.833 | Benign 0.748; IoU 0.867 | Both locate well but miss the malignant class. This is not an absent-box localisation failure: benign FP and malignant FN, with no malignant alert. |

Values are rounded from downloaded cached test records. Some duplicate-case labels overlap in the PNG; use the records rather than guesses from obscured text. n's scores are 0.308, 0.190 and 0.121. Their pairwise prediction-to-prediction IoUs are 0.659, 0.628 and 0.414, all below NMS 0.7: NMS need not suppress them despite their overlap with the same GT. NMS compares predictions with each other; detection matching compares a prediction with GT.

The last image was selected under `malignant_miss_small_pixel_box`: it has the smallest GT pixel-box area among remaining unmatched malignant candidates after earlier IDs were selected. Its box area is 9,116 pixels, not physical size or a normalised comparison. The name does not prove that size caused failure. Preserve the original category and `cases.json`, but describe a malignant-class miss despite successful spatial localisation.

## Visible features versus unproven causes

| Image | Visible features | Hypothesis only |
| --- | --- | --- |
| ISIC_0014897 | Non-uniform pigmentation, darker patch inside a broader brown region | Appearance may be confused with benign examples. High confidence does not establish correctness or calibration. |
| ISIC_0009875 | Multi-lobed outline and non-uniform brown colouring | Appearance overlap between diagnosis classes may contribute. These features are not themselves a diagnosis. |
| ISIC_0015071 | Irregular dark pattern, bright/bubble-like areas and corner text | Context or boundary ambiguity may influence the oversized n box; no controlled test isolates their effects. |
| ISIC_0012178 | Diffuse pink/orange region, low-contrast boundary, ruler near the bottom | Boundary/scale ambiguity may contribute to overlapping boxes. Ruler presence does not prove the model used it. |
| ISIC_0012758 | Compact brown lesion with a darker centre, surrounded by skin | Scale or class-feature ambiguity is a future hypothesis, not an established cause. Good overlap rules out absent localisation here. |

Do not claim hair/ink caused an error without controlled evidence. These deliberately selected failures are not a representative random sample; their counts do not estimate failure-type prevalence. AI-assisted visual review is complete; student understanding has not been assessed.

## Engineering implications

- Evaluate localisation and classification separately: high class-agnostic IoU cannot compensate for malignant-class misses.
- Lower confidence trades sensitivity against extra boxes/alerts but does not relabel a confident benign prediction as malignant. Future settings must be selected on validation within a new documented experiment, not these exposed test cases.
- More capacity alone did not eliminate these errors. s improves one localisation case but retains class errors and duplicates; five examples cannot establish a model-wide advantage.
- Class imbalance, richer diagnosis supervision, boundary ambiguity and validation-only NMS selection are future investigations, not implemented fixes or proven causes. Fresh independent evaluation would be needed for improvement claims.
- Image-level splits have known/unknown lesion leakage, so these are not lesion/patient-independent results. Alerts are referral proxies, not measured biopsy decisions; no automatic safety clearance is supported.

## Demo: explain in your own words

1. **高 IoU 仍然出错：**“第1张答案是恶性，框的位置很好，但模型说良性。检测正确需要位置和类别同时正确，所以仍然漏掉一个恶性目标。0.889是模型分数，不是正确率或患病概率。”
2. **误警报：**“第2张确实有病灶，但答案是良性，模型判成恶性。这是类别误报，不是证明它把毛发识别成病灶。”
3. **警报不等于合格检测：**“第3张n给恶性警报，但框太大，IoU约0.43，低于匹配标准0.5。s的恶性框合格，却还多给了一个良性框。”
4. **重复框为什么算FP：**“第4张只有一个答案目标，最多匹配一个TP。n剩下两个框算FP。NMS比较预测框之间的重叠，评估比较预测框与答案的重叠，不能混淆。”
5. **不能只读自动类别名：**“第5张被选为小像素框恶性漏检，但两模型都定位到了，真正错的是类别。像素面积小不证明现实中病灶小，也不证明尺寸导致错误。”

Overall: “I inspected five errors from frozen test predictions, separated classification, localisation and duplicate-detection failures, and compared both models on the same images. Visible artefacts suggest hypotheses, not causal explanations. I did not tune the model or threshold on these test examples.”

## Evidence files, excluded from Git

- Windows frozen records: `outputs/frozen_661498/summary.json` and `{yolov8n,yolov8s}_predictions.json`; school originals: `outputs/final_test/frozen_661498/`.
- Windows reviewed figures: `outputs/test_661498/{image_id}.png` and `cases.json`; school originals: `outputs/failure_cases/test_661498/`.
- Implementation: `failure_cases.py` selects/renders; `threshold_metrics.py` counts matches.

This review does not overwrite experiment records or photographs. Figure reuse permissions and final README packaging remain to be checked; original images are not added to the course repository by this change.
