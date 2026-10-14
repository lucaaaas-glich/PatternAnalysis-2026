# Rangpur A100：操作与理解

## YOLOv8n两轮流程检查（学校执行尚待确认）

安装已由学生日志确认：Ultralytics8.3.221、torch2.14.0+cu130、torchvision0.29.0+cu130。不要重复运行setup脚本。

```bash
cd ~/comp3710-a3
git pull --ff-only
bash recognition/YOLO_s4905914/submit_smoke.sh
```

入口先在登录节点获取官方预训练权重（不训练），再提交A100 GPU任务；以Submitted batch job编号为准，不重复提交。用sacct查询最终状态，日志yolo_smoke_JOBID.out/.err；成功时outputs/smoke/yolov8n_JOBID/smoke.json记录completed，train/weights/last.pt记录最后权重，train/results.csv记录损失和验证过程。两epoch、320像素、batch8，不是正式baseline成绩。测试集合不使用。39项本地测试及Bash语法检查通过，不表示GPU已跑通。

## YOLO运行目录装配（2026-10-14；学校实际执行待确认）

本地已打包 `outputs/splits/image_level_3710_v1.zip`，仅含划分清单、诊断框标签和摘要，不含照片或原始患者元数据。Windows上传该包到 `~/comp3710-a3/`。学校终端从仓库根目录拉取新代码、激活torch后执行：

```bash
unzip -n image_level_3710_v1.zip -d outputs/splits
python recognition/YOLO_s4905914/assemble_yolo.py --split-root outputs/splits/image_level_3710_v1 --output outputs/yolo_runtime/image_level_3710_v1 --acknowledge-image-level-split
```

`unzip -n`不覆盖既有文件；装配拒绝已存在的输出目录。若输出存在先查看结果，不要删除重跑。先全量检查manifest、图片文件存在/名称、标签类别/框范围、图片ID不跨集合，再用符号链接指向学校共享照片、复制小型标签，写data.yaml与assembly.json。不解码照片、不训练、不请求GPU；当前操作是轻量文件检查。不宣称病灶独立。四项新增本地测试通过，全套37项；Windows测试模拟符号链接调用，真实学校Linux链接仍待执行验证。

最新结果（2026-10-13）：官方标签准备任务655099已确认COMPLETED，耗时11分07秒，退出码0:0；2594/2594组通过、2594标签、0问题。结果已下载，本地CSV/标签一致性检查通过，6张预览经AI辅助复核。无需再次提交本任务；该任务不训练模型。患者信息、诊断映射和独立划分仍待核实。后文历史提交记录保留当时状态。

## 已恢复的流程

### 下一步：元数据发现（尚未在学校执行）

在已登录的学校终端，从仓库根目录执行：

```bash
git pull --ff-only
source ~/miniconda3/bin/activate
conda activate torch
python recognition/YOLO_s4905914/inventory_metadata.py --root /home/groups/comp3710 --max-depth 3 --max-entries 20000
```

这是有界轻量文件名/表头发现，不解码图片、不训练，不需要提交GPU作业。仅输出候选路径及CSV/TSV第一行，不输出患者行；其他候选文件只列名称。若扫描截断或遇权限问题，报告会明确指出，需缩小到相关目录再检查。找到文件不代表已证明患者独立或诊断映射。不要把image_id或lesion_id未经验证当成patient_id。

Assignment 3 使用独立远程目录 `/home/Student/s4905914/comp3710-a3`。
已复用 Demo2 的 `comp3710` account/partition 和 `$HOME/miniconda3` 中的 `torch` 环境。
参考脚本：https://github.com/lucaaaas-glich/comp3710/blob/main/lab2-pattern-recognition/train_oasis_vae_a100.sh

2026-10-12（布里斯班）实测：Slurm 作业 `652217` 在 `a100-3` 运行 45 秒，
状态 `COMPLETED`，退出码 `0:0`。Python `3.11.16`，PyTorch `2.14.0+cu130`，
PyTorch CUDA `13.0`，GPU `NVIDIA A100-PCIE-40GB`（40960 MiB）。
`cuda_available=true`、`gpu_matmul_passed=true`、`course_dataset_exists=true`。
这次只执行基础环境验证，未进行 YOLO 训练。

## 连接与短 GPU 验证

校外先连接 UQ VPN，在本地终端执行 SSH；密码只输入登录提示，不放进项目文件。

```text
ssh s4905914@rangpur.compute.eait.uq.edu.au
cd ~/comp3710-a3
git pull --ff-only
sbatch recognition/YOLO_s4905914/check_a100.sh
```

`sbatch` 返回一个 job ID。使用实际编号查看状态和日志：

```text
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
cat lesion_gpu_check_JOB_ID.out
cat lesion_gpu_check_JOB_ID.err
```

## demo 中需要会解释

- SSH 登录后在共享登录节点：用于编辑、获取代码和提交作业。
- Slurm 是资源调度器，`sbatch` 提交任务，未必立即开始计算。
- `PENDING` 表示排队，`RUNNING` 表示运行，`COMPLETED` 表示完成。
- GPU 必须在计算节点验证，登录节点没有 CUDA 不代表训练环境坏了。
- 日志中的节点名和 GPU 型号是实际执行设备的证据。
- `40 GB` 是 GPU 显存，用于模型、梯度、优化器状态和中间激活；与 CPU 内存不同。
- GPU 验证成功只说明基础环境可用，不说明模型已经训练成功或有任何检测准确率。
- 作业结束后 GPU 由 Slurm 释放，不需要保留一个空闲交互会话。

## 本次发现的 Slurm 配置差异

`sinfo`/`scontrol` 显示课程 A100 节点的 Slurm `RealMemory=1`，
因此显式请求 `--mem=16G` 会被拒绝。沿用 Demo2 脚本不显式申请内存的方式。
该调度配置值不代表 GPU 显存只有 1 MB。
脚本固定 Unix 换行，避免 Windows 的 CRLF 导致 Slurm 解析错误。

## 学校数据目录

```text
/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2
/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2
/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input
```

目录名称已检查，尚未核实实际图片尺寸、mask 配对、诊断类别或患者信息。
不能只根据 `_x2` 猜测缩放方式。
该目录目前没有发现患者/诊断 CSV，或测试集 mask 目录。
正式划分和评估前需要解决这些信息缺口。

## 下一步：完整数据审计（不训练、不占用GPU）

`audit_data.py` 检查照片/mask配对、重复文件编号、可读性、尺寸和有效二值病灶标注；仅列出候选CSV的文件名与表头，不读取患者记录、不推断类别、不创建数据划分。报告只能写到原始输入目录之外，不覆盖已有报告。

2026-10-12：新增审计的5项测试与原标注转换5项测试全部通过；现有本地100组官方validation审计通过。学生自己的终端已执行学校审计job653933；代理未接管该SSH连接。结果见下方记录。

登录后先拉取本次真实提交，再运行。此处目录名来自先前清单，仍需审计确认内容。

```bash
cd ~/comp3710-a3
git pull --ff-only
git branch --show-current
sbatch recognition/YOLO_s4905914/audit_school_data.sh
```

分支应为 `topic-recognition`。完整审计逐张解码，使用CPU调度任务，不在共享登录节点连续执行、不申请GPU。课程account/partition已接受本次CPU-only请求，job653933实际执行；节点名a100-8不意味着程序使用CUDA，当前审计代码仅使用CPU。

`sbatch` 返回任务编号。用真实编号替换 `JOB_ID`：

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
cat school_data_audit_JOB_ID.out
cat school_data_audit_JOB_ID.err
```

报告路径为 `outputs/data_audit/school_training_JOB_ID.json`，每次任务独立保存。退出码0表示文件配对检查通过；1表示检查发现异常。即使通过，也不表示患者独立性或诊断类别已验证。下一步要根据CSV实际字段、官方来源或教师说明核实这些信息。

### job653933 的实际结果与未决问题

- 学生提供的完整终端记录：FAILED，ExitCode1:0，Elapsed00:03:51；输出包含完整审计JSON。这是程序检查发现异常后主动返回1，不是运行中断或模型训练失败。
- 照片2594张、mask2594张；缺失/孤立/重复编号均0；2594对均可读取且尺寸匹配。
- 全部2594项报 `Expected a binary mask with values 0/1 or 0/255`。`valid_pairs=0` 是未通过本程序所有规则的数量，不代表没有配对、没有病灶或全部文件损坏。
- 原始mask的实际像素模式、dtype和灰度分布待少量抽样核实；尚未证明非二值化由缩放/插值导致，不能仅凭 `_x2` 命名推断，也不能自动对所有文件阈值化。
- 审计根目录没有发现CSV候选；这不排除其他位置存在患者/诊断元数据。
- 查看日志时SSH再次Connection reset；审计在此之前已经结束，断线不是该数据检查失败的原因。断线后必须重新登录；Windows PowerShell不能直接调用学校的sbatch。
- 终端只复制命令，不复制提示符和欢迎语。今后查看摘要与少量示例，不再要求打印全部重复异常。

## 官方二值mask的正式全量标注准备（job655099已提交，结果待核对）

小样本参照后，正式来源改用官方原始0/255 mask，按同ID学校照片尺寸做最近邻缩放。不是对学校灰度mask自动选128，也不改学校共享目录。尺寸检查要求原mask与照片约为2倍关系，允许奇数尺寸舍入造成的1个原始像素差；若不符合，暂停并报告，不自动拉伸任意尺寸。这个尺寸规则不等于已证明学校所有图像未裁剪，仍需看代表性预览。

在学校SSH终端、作业仓库内执行：

```bash
cd ~/comp3710-a3
git pull --ff-only
bash recognition/YOLO_s4905914/submit_official_labels.sh
```

启动脚本自动激活已验证的torch环境，因此不要求本次交互窗口预先有Python。先下载/校验约26MB官方mask ZIP到被Git忽略的 `data/reference/`，不下载训练照片。SHA256固定为 `99f8b2bb3c4d6af483362010715f7e7d5d122d9f6c02cac0e0d15bef77c7604c`；已有不符的文件不会覆盖。然后 `sbatch` 提交CPU任务，不请求GPU，最长30分钟。当前24项本地测试和Bash语法检查通过。2026-10-13学生终端显示校验通过并提交job655099；实际最终状态、耗时和导出标签数仍待查询。不要因为SSH断开而重复提交。

拿到真实JOB_ID后：

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
cat official_label_prep_JOB_ID.out
cat official_label_prep_JOB_ID.err
```

日志只打印摘要和最多5个异常示例。结果在 `outputs/official_labels/school_nn_JOB_ID/`：

- 全部2594组验证通过后才生成 `labels/` 中的2594份YOLO TXT、`annotations.csv`、`summary.json`，以及6张预览（面积最小/最大加固定种子随机抽样）。
- 原始mask必须是L/uint8、非空、只有0/255；匹配编号、照片可读性、2倍尺寸规则、最近邻后的二值性/非空和归一化框均检查。
- 任意一组失败，不导出任何训练标签；失败目录保留摘要及 `issues.json`。不要删除原始数据、把空mask自动当健康病例，或覆盖旧的任务输出。
- `class 0 = lesion`，不是良性/恶性；不创建train/val/test，不训练、不报告模型成绩。患者元数据、诊断范围和最终划分仍需解决。
