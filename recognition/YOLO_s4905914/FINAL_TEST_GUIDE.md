# 冻结方案后评估test：不再按test调参

用户已确认主要实验方案YOLOv8n、置信度0.05，YOLOv8s同阈值对比。阈值来源为val job661469：n在.05警报48/77恶性、30/311良性误警报；.01同48恶性却43良性误警报。检测TP47/77与图片警报48/77不同。此选择是作业实验操作点，不是安全临床阈值或自动放行规则。原始val摘要SHA256写入POLICY。没有按test选择阈值或checkpoint。

学校终端：

```bash
cd ~/comp3710-a3
git pull --ff-only
sbatch recognition/YOLO_s4905914/test_a100.sh
```

不要重复提交；脚本任务编号对应新输出outputs/final_test/frozen_JOBID。日志yolo_test_JOBID.out/.err。输入为固定data.yaml和已下载检查的两模型best.pt；校验SHA、类别映射，检查test为388张/311良性/77恶性。不训练、不扫描test阈值、不重选模型。执行前写started记录，失败保留已取得结果，成功写completed。当前学校执行待确认。

## 指标口径

- 标准mAP：Ultralytics val split=test，640、rect验证、conf底限.001、NMS IoU.7；mAP50与mAP50:95含跨置信度排序，按官方API输出，不能称为固定.05 Precision。库自动P/R单独标注仅作诊断，不采用其test最佳操作点。
- 冻结方案：同val阈值扫描的固定640方形predict，底限.001缓存，筛至.05，类别正确且IoU>=.5按置信度一对一匹配，输出TP/FP/FN和P/R。图片警报另统计，忽略定位，不等于检测Recall。
- IoU：各真实类别的所有GT均进分母。每个GT取阈值以上同类别框的最大IoU；无同类框记0，报告均值及IoU>=.8覆盖比例。这是best-box oracle诊断，会忽略额外框，因此结合FP及Recall看，不能替代检测指标。另报忽略类别的best-box均值、只匹配TP平均IoU。定义不能混用，也不保证符合老师对IoU目标的唯一解释。
- FPS：A100、FP32、batch1、固定640方形、.05阈值、10次预热后50张按ID排序测试图片，GPU同步计时，包含Python predict接口/预处理/forward/NMS，不含磁盘解码、画图、网络及JSON序列化。保存平均/中位耗时与FPS；不是手机、Mac或纯forward速度。

输出summary.json：冻结策略、标准AP、固定阈值指标、各类IoU、速度、参数及哈希；两个_predictions.json保存测试GT与预测，用于事后失败分析，不用于再调参。各模型_ap目录保存标准评估曲线/混淆矩阵。

继续明确披露图片级划分存在已知/未知病灶交叉，不能称为病灶/患者独立。第三类只1训练样本、无held-out真值，AP/P/R/IoU缺失为不可评估，不伪造第三类满分。假阳性警报是转诊代理，不能当作真实活检次数。测完若结果不理想如实报告，不重新按test调阈值。故障诊断重跑应保留原证据与说明，不伪装独立test。
