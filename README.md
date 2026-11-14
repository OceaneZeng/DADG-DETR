# Tree-DETR

当前 OWOD 实验只采用两套作者协议：**M-OWODB（ORE，CVPR 2021）**和
**S-OWODB（OW-DETR，CVPR 2022 附录 A.2）**。更新于 2026-10-08。

[协议来源与评估定义](docs/owod-benchmarks.md) ·
[数据准备操作指南](docs/data-preparation.md) · [项目目录](docs/project-layout.md) ·
[可选的本地 paper-baseline](docs/paper-baselines.md)。

- M 的新增类别数为 20/20/20/20；S 为 19/21/20/20。
- 固定作者类别顺序和 ImageSets；禁止自定义或随机重划分。
- 当前增量训练后单独进行作者清单上的均衡微调；禁止未来类别标签泄漏。
- 使用作者 VOC 评测约定计算 AP50、U-Recall、WI@0.8、A-OSE。
- 仅接受经过哈希和清单验证的 schema-3 manifest；旧 pilot 和 unverified 入口停用。

```bash
python tools/owod/manage_data.py plan
python tools/owod/download_data.py --benchmark both
python tools/owod/manage_data.py check --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco
python tools/owod/manage_data.py prepare --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod
python tools/owod/manage_data.py verify --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod
python tools/owod/run_benchmark.py --help
```

数据准备入口不会改动原始图像，训练入口先完整检查数据与 checkpoint 来源。
原始数据路径在配置中指定；`check` 未通过时不得开始准备或训练。

本项目 detector 与研究方法仍属于本地实现；采用作者 benchmark 不代表复现了
ORE 或 OW-DETR 方法的论文数值。方法训练 recipe 必须明确记录。

新增研究方法：[图风险引导的选择性 adapter](docs/graph-routed-adaptation.md)。
使用 `run_benchmark.py --method graph-adapter` 接入官方七训练阶段和阶段间
GNN 校准。GNN score 参与 query 分支门控与梯度更新，增量阶段冻结旧分支；
默认 `--method control` 保留基础 detector 对照。

[未知发现与类别转化方法](docs/owod-discovery-transition-design.md) 已接入
`--method discovery`；`discovery-scratch` 为关闭匿名表示迁移的消融。
完整数据下载、作者划分、三组任务链与 AP 对照入口：

```bash
python tools/owod/run_discovery_study.py --benchmark both
```

机制与 CUDA 检查已通过，正式 AP 尚未测得。ORE 作者数据入口当前为空时，
脚本记录 M 来源不可用并继续 S，不创建替代划分。运行状态和日志保存在
`output/discovery/full-study/`；数据/权重均断点续传，不加入 Git。

旧 setting 的启动器、配置及说明文件已删除，清理记录见
[project-cleanup.json](docs/project-cleanup.json)。已有日志和 checkpoint 保留。
已有绘图代码位于 `tools/analysis/`，仅分析历史结果。

Git 只保存源码、运行配置、两份协议 manifest、必要说明和许可证。
PDF、图片、论文文稿、压缩包、权重、日志与临时输出均忽略；不要使用
`git add -f` 将这些生成产物加入仓库。

## 双 RTX 3090 服务器实验

当前 S Task 2 后续实验见[新类学习诊断与 hard/soft 配对对照](docs/sowodb-task2-routing-experiment.md)。
Git同步后在服务器运行 `tmux new -s s-task2 'bash tools/owod/run_sowodb_task2_routing.sh'`，
入口先做同权重诊断，再从同一完整Task1分别新建v7两组、各三轮，固定总batch2并自动恢复。

详见 [GPU 0/1 环境、数据迁移、自检与完整消融](docs/server-3090.md)。
`python tools/owod/run_server_experiment.py` 核验计划；加 `--execute` 后依次运行
硬路由、软路由及受限定位更新对照。
