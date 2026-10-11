# 数据准备：理解与演示

## 本阶段在做什么

输入是 ISIC 的皮肤镜照片及人工分割 mask，输出是 YOLO 使用的病灶框标注和检查图。
这一步没有训练模型，也没有模型预测；图中的绿色框是从人工 mask 算出的 ground truth。

| 项目 | 表示什么 | 在本项目中的作用 |
| --- | --- | --- |
| RGB image | 病灶照片，包含皮肤、病灶、毛发等 | 模型的输入 |
| Binary mask | 白色像素为人工标注的病灶，黑色为背景 | 提供病灶位置的正确答案 |
| Bounding box | 包住全部病灶像素的最小轴对齐矩形 | 用作目标检测监督标签 |
| YOLO label | 类别编号及归一化的框中心、宽、高 | 供后续训练和评估读取 |

分割（segmentation）回答“哪些像素属于病灶”；检测（detection）回答“病灶在哪里”；
分类（classification）回答“它属于什么类别”。当前类别只有 lesion，类别编号为 0。
编号 0 不代表良性、不代表无病灶；也不是 confidence score。

## mask 如何变成框

横向是 x，纵向是 y，左上角为原点。找到所有非零 mask 像素的最小/最大 x、y。
代码把右边和下边记录为最后一个病灶像素位置 + 1，保证单像素病灶仍有正面积。
一个 mask 中不连通的区域在本阶段仍合成一个框，不会擅自拆成多病灶。
空 mask 会报错，而不会自动当作健康图片。

YOLO 每行的格式是 `class center_x center_y width height`，后四项均为 0 到 1 的比例。

例如照片宽 100、高 80，病灶覆盖 x=10…29，y=20…35：

- 边界为 (10, 20, 30, 36)，框中心为 (20, 28)，宽 20、高 16。
- 除以照片宽、高后，标注为 `0 0.20 0.35 0.20 0.20`。
- 长方框会包含部分背景，丢失精确轮廓。这是把分割标注转换成检测标注的代价。

## 打开结果

从仓库根目录查看 `data/inspection/isic2018_val/previews/`。
每张预览依次显示原图、人工 mask、从 mask 导出的框。
检查框是否与照片中的病灶对应、是否包住全部白色像素、是否出现明显尺寸或配对错误。
视觉检查之外，程序检查配对数量、尺寸、二值 mask 和归一化范围。

运行（PyCharm 解释器应选择已验证的 Python 3.12）：

```powershell
py -3.12 recognition/YOLO_s4905914/dataset.py --images data/raw/ISIC2018_Task1-2_Validation_Input --masks data/raw/ISIC2018_Task1_Validation_GroundTruth --output data/inspection/isic2018_val
```

`summary.json` 给出有效配对数量与框面积范围；`annotations.csv` 记录每张图的坐标；
`labels/` 中每张图对应一个 TXT 标注文件。原始数据和生成结果都在 Git 忽略的 data 目录。

## demo 中你可以这样讲

“我使用 ISIC 的皮肤镜照片及人工分割标注，把每个病灶 mask 转成最小外接矩形，
并用照片尺寸将坐标归一化，得到 YOLO 的单类别检测标签。
我先检查照片与 mask 的对应关系、尺寸和空标注，再用对照图验证转换结果。
当前展示的是 ground truth 数据准备结果，后续才会训练模型并比较预测框与真实框。”

## 当前边界与下一步

本阶段只检查 ISIC2018 官方验证集。没有创建训练/验证/测试划分，没有性能分数。
图像 ID 不是 patient ID，随机按图片划分不能证明 patient independence。
下一阶段先核实作业说明、患者元数据和数据重叠，再确定正式实验划分。
用于调参的是 validation，最终 test 应保留用于最后评估。

正式说明已经核对：允许使用预训练 YOLOv7 或更新版本，但提到了分类表现。
所以目前的单类别 lesion 仅用于验证定位标签流程，后续需要检查 diagnosis/class labels。
正式训练按学生选择安排在 Rangpur A100，通过 Slurm 提交；当前数据处理在 Windows CPU 执行。
登录节点负责编辑和提交任务，A100 计算节点才执行训练。
中期 check-off 要带一页 feasibility review；提供的 draft 明确说这一咨询不要求现场 demo、幻灯片或口试。
若导师另行安排 demo，以该安排为准，本讲解仍用于帮助你解释项目。

这些知识与课程中的监督学习、图像表示、数据泄漏和泛化有关。
尚未核对本学期期末评分蓝图，因此这里不声称具体考试占比。

## 数据来源

- https://challenge.isic-archive.com/data/ （2018 Task 1 验证图像与 ground truth）
- https://docs.ultralytics.com/datasets/detect/ （YOLO 标签格式）
- Codella et al., Skin Lesion Analysis Toward Melanoma Detection 2018, https://arxiv.org/abs/1902.03368

代码和说明使用 AI 协助编写；学生需要核对实际结果并能解释以上流程。
