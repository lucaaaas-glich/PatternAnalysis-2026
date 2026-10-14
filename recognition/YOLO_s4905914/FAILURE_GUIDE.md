# 失败案例：先区分错误类型，再讨论视觉原因

2026-10-15：学校已生成5张图，下载后完成AI辅助视觉复核与缓存数值核对。分析及demo讲述见 [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。下面保留生成/下载方法，无需重复运行或覆盖结果。第5个“小像素框恶性漏检”实际定位很好、类别错误，不能解释成没找到小病灶。学生理解仍需练习与确认。

候选从已冻结的test预测中确定，不重新训练/推理，不按test调参数。只读取5张学校照片，生成GT框、n预测、s预测对照。无需GPU或sbatch，使用已安装的独立解释器。

学校终端：

```bash
cd ~/comp3710-a3
git pull --ff-only
.venv-yolo/bin/python recognition/YOLO_s4905914/failure_cases.py --test-root outputs/final_test/frozen_661498 --runtime outputs/yolo_runtime/image_level_3710_v1 --output outputs/failure_cases/test_661498
```

输出cases.json及最多5张PNG，最多显示每模型6个最高分框并标出显示/总数、置信度与GT框IoU。若输出已存在先检查，不删除或覆盖。cases.json的visual_cause暂记pending manual review，不能根据模型失败自动归因为毛发、墨水或低对比度；先下载看图再形成可验证假设。挑选的是典型错误，不代表各类错误的频率。

Windows PowerShell下载（不要在学校终端输入C盘路径）：

```powershell
scp -r "s4905914@rangpur.compute.eait.uq.edu.au:/home/Student/s4905914/comp3710-a3/outputs/failure_cases/test_661498" "C:\Users\cyzwo\Downloads\Pattern Recognition\outputs"
```

候选包括高分错类别、良性误警报、警报但定位不合格、重复框、小像素框恶性漏检；像素框面积不等于物理病灶大小或图像归一化面积。图中人工作为答案的是mask导出的GT框；预测分数不是校准患癌概率。下一步人工/AI辅助视觉审查并记录假设，不改变冻结模型与阈值。
