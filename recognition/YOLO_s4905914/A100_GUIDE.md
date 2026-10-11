# Rangpur A100：操作与理解

## 已恢复的流程

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
