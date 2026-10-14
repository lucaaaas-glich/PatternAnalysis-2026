# YOLOv8s对比：同预算，不预设胜负

baseline job661165的训练记录已下载：baseline.json=completed，30轮，原YOLOv8n预训练重新开始，640/batch16/workers1/seed3710/FP32/optimizer auto/close_mosaic10。测试集未用。最佳权重验证结果：全体P0.726/R0.714/mAP50 0.774/mAP50:95 0.554；良性P0.811/R0.896/mAP50 0.906/mAP50:95 0.675；恶性P0.640/R0.532/mAP50 0.642/mAP50:95 0.434。训练阶段日志0.463小时，不等于任务完整耗时。曲线整体改善、尚不能断言收敛；预测图有重复类别框和错分。

下一项实验用YOLOv8s，沿用同一train/val划分与上述全部训练设置，使用各自官方COCO预训练，不续训baseline。区别主要为模型规模和对应预训练权重；不能声称只改变一个张量或预训练完全相同。更大模型不保证更好。比较验证效果、恶性检出、模型参数与计算成本，之后在验证集选择操作阈值，最终才评估两模型test。

学校终端从仓库根目录执行：

```bash
cd ~/comp3710-a3
git pull --ff-only
bash recognition/YOLO_s4905914/submit_comparison.sh
```

先在login节点下载约22.6MB官方YOLOv8s权重，校验SHA256，再sbatch提交GPU任务；login节点不训练。校验值为本次从官方发布URL下载文件的实际哈希，不是数字签名或独立的官方校验声明。已有不符文件会保留并报错，不覆盖。不要重复提交。

日志：yolo_comparison_JOBID.out/.err。结果：outputs/comparison/yolov8s_JOBID/comparison.json，train/results.csv、曲线、best.pt、last.pt。资源时限2小时并非预估耗时，训练预算仍30轮。图片级划分存在已知及未知病灶交叉，不能报告病灶/患者独立；不确定类仅1训练样本，不支持该类held-out成绩。当前对比程序已准备，学校尚未运行。

训练config与baseline一致性有单元测试，YOLOv8s初始权重哈希和data.yaml哈希会校验；data.yaml哈希不证明其中引用的所有图片/标签字节完全不变。不要修改已装配数据。后续报告保存实际args.yaml、权重哈希与输出，不用短跑成绩充当最终对比。

知识点：【补】模型容量与泛化。n表示nano，s表示small，规模增大会增加表达能力及计算需求，也可能在有限数据上增加过拟合风险。公平比较要对齐数据和训练/评估协议，而不是让一个模型多训练再把差异全归因于架构。

官方资料：https://docs.ultralytics.com/models/yolov8/
权重来源：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
