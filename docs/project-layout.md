# 当前项目目录

| 路径 | 职责 |
| --- | --- |
| `data/raw/` 或外部数据磁盘 | 完整 ORE release 与 COCO2017，保持原始数据 |
| `data/owod/<benchmark>/split_manifest.json` | Git 跟踪的 schema-3 协议、类别顺序、完整作者清单与来源哈希 |
| `data/owod/<benchmark>/stage_*/` | 四任务生成标注，Git 忽略 |
| `datasets/` | 数据 loader、增强、采样代码 |
| `models/` | Detector、GNN、LoRA、蒸馏和其他研究组件 |
| `tools/owod/manage_data.py` | 推荐数据入口：plan/check/prepare/verify |
| `tools/owod/run_benchmark.py` | 标准四任务、七训练阶段启动器 |
| `tools/owod/tests/` | 当前协议、数据管理、模型组件和历史报告测试 |
| `tools/analysis/` | 已有结果的读取、绘图与汇总，不启动实验 |
| `tools/windows/build_extension.ps1`、`tools/setup_linux.sh` | 平台环境与 CUDA 扩展构建 |
| `tools/run_dist_launch.sh`、`tools/run_dist_slurm.sh` | 通用分布式执行工具 |
| `baselines/` | 已删除；外部源码不再作为本项目运行时依赖 |
| `pretrained/` | 预训练权重，实验记录其来源与哈希 |
| `exps/` | Checkpoint、日志、指标与运行计划 |
| `output/` | 新生成的图表和导出产物，Git 忽略 |
| `tmp/` | 临时工作文件，Git 忽略，不作为正式协议来源 |
| `docs/` | 当前数据、协议和项目说明 |
| `util/` | 共享运行时工具 |

正式数据目录：

```text
data/owod/
  m-owodb/
    split_manifest.json
    stage_0/ ... stage_3/
  s-owodb/
    split_manifest.json
    stage_0/ ... stage_3/
```

正式实验输出示例：

```text
exps/owod/author-release/
  m-owodb/seed_42/
    benchmark_plan.json
    stage_0/increment/
    stage_1/increment/
    stage_1/finetune/
    stage_2/increment/
    stage_2/finetune/
    stage_3/increment/
    stage_3/finetune/
  s-owodb/seed_42/ ...
```

M/S 使用独立数据和 checkpoint 链。已有旧结果保留，但不能作为新协议实验
的初始化或成绩证据。旧 setting 文件已删除，逐文件清单见
[project-cleanup.json](project-cleanup.json)。

源码、原始数据、生成数据、预训练、实验结果分别存放。不再使用 COCO-only
shared-M 适配、随机划分或旧 pilot 目录作为数据协议来源。
