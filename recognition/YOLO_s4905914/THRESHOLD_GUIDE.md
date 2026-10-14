# 验证集置信度阈值：检出与误报的代价

当前证据：YOLOv8s job661355 COMPLETED/0:0，29分10秒；comparison.json、args.yaml、权重及图已下载复核。YOLOv8n与s的训练配置除输出路径相同，data.yaml哈希相同，两者未使用test。s验证mAP50 0.715/mAP50:95 0.516、恶性P0.486/R0.545；n为0.774/0.554、恶性P0.640/R0.532。一次同预算实验中s并未整体胜出，不是“更大永远更差”的证明。不能将小幅Recall变化当作显著或临床优势。

学校终端执行：

```bash
cd ~/comp3710-a3
git pull --ff-only
sbatch recognition/YOLO_s4905914/threshold_a100.sh
```

GPU任务仅加载两个已训练best.pt，各预测388张验证图一次，再用缓存扫描14个阈值：0.01、0.05、0.1、0.15、0.2、0.25、0.3、0.4、0.5、0.6、0.7、0.8、0.9、0.95。不训练、不读取test图片/标签，不下载新模型。检查模型哈希、类别、val图片数量与311良性/77恶性分布。输出新目录outputs/thresholds/validation_JOBID，保留旧文件；日志yolo_threshold_JOBID.out/.err。

## 必须分开的两种统计

1. 检测：预测类别正确且框IoU>=0.5，按置信度降序一对一匹配。一个人工框最多算一个TP，重复预测算FP，未匹配人工框算FN。输出每类TP/FP/FN、Precision/Recall/F1。分母为0时记null，不伪造0或1。仅匹配TP的平均IoU只是条件统计，不能掩盖漏检或当作整体IoU目标已达标。
2. 图片恶性警报：一张图至少有一个恶性预测达到阈值就触发警报。分母为77张恶性图的敏感度、311张良性图的误警报率，另报总警报数和警报Precision。这里不要求位置匹配，所以它不等于检测Recall。误警报仅是作业里的转诊负担代理，不是真实活检次数。

置信度是模型分数，不是经过校准的患癌概率。降低门槛只会保留更多已有预测，不能凭空创造不存在的恶性类别预测；错分需单独分析。NMS重叠抑制阈值0.7与检测匹配IoU0.5不是同一个用途，也不是mask二值化阈值。

## 结果与限制

summary.json：协议、checkpoint哈希、各阈值统计；两个模型_predictions.json：每张图的GT与post-NMS框/类别/分数；threshold_tradeoff.png：检测PR与图片警报代价图。不自动选择临床阈值。待结果后再在val上约定选择规则，之后冻结方案再运行test。

统一predict采用640固定方形padding、FP32、无TTA、NMS IoU0.7/类别相关、conf floor0.001/max_det300。两模型使用同协议；与trainer的rect验证预处理和自动F1操作点不同，因此扫描P/R不要求等于训练日志。缓存只保留NMS后的selected类别，不能恢复全部类别概率。报告达到max_det上限的图片数量，底限以下预测不可恢复。不是mAP计算或正式FPS测量。已知/未知病灶跨集合的限制继续披露，test独立性不宣称。当前工具及统计测试完成，学校扫描尚未执行。
