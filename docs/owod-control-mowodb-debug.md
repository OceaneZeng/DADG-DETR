# M-OWODB Task 1 纯检测诊断对照

这次只验证：在相同作者 M 数据、DINO 骨干和基础检测配方下，去掉图路由与发现监督后，
Deformable DETR 是否仍会失去对图片内容的响应。它是内部 supervised control，
不是 ORE / OW-DETR 模型复现，也不是完整 OWOD 成绩。依据见
[v5–v8 退化分析](owod-discovery-v8-analysis-20261110.md)。

## 固定预算与归因

- 全量作者 M Task 1，`seed=42`，GPU 0、1，每卡 batch 2，总 batch 4。
- DINO ResNet-50，100 queries，AdamW，lr=2e-4，backbone lr=2e-5，activation checkpointing。
- 仍登记 `epochs=50`、`lr_drop=50`，仅执行到第 **3 轮自动暂停**；不进入 Task 2。
- 从骨干预训练开始，不载入 v5–v8 权重。每轮按作者口径评估 Known / Current AP50。
- 每轮保留 `checkpoint0000.pth`–`checkpoint0002.pth`，包含 optimizer / scheduler。
  `checkpoint.pth` 指向最新状态，阶段只生成 `training_paused.json`，不能作为已完成 Task 1。

预计训练与评估合计约 1–1.5 小时，沿用此前服务器吞吐估计；纯检测耗时需实际记录。
保持原 batch、优化日程和默认运算路径，未切换 adapter compute、risk 模式或数据清单。
暂停后通过独立进程读回三份权重，输入诊断不消耗训练进程的随机状态，也不修改权重。

## 最短启动命令

先将本地改动通过 Git 同步到服务器。以下在 Linux / Bash 执行，复用既有环境：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new-session -d -s m-control 'bash tools/owod/run_control_mowodb_3090.sh'
tail -n 50 -F output/tree-detr-m-control-3090-logs/console.log
```

启动器先执行 `manage_data.py verify`，校验失败即退出。可选变体为 `standard`（默认）或
`lr-quarter`，开关为 `--dry-run`、`--resume`；已有计划或权重时禁止误启动覆盖。这里的 `--resume` 仅恢复同一 control
到 3 轮预算，不会自动扩大训练预算。运行前核对 GPU 0、1 没有新作业占用。

## 输出与决策

```text
output/tree-detr-m-control-3090-logs/console.log
output/tree-detr-m-control-3090/m-owodb/seed_42/
  benchmark_plan.json
  stage_0/increment/
    run_config.json
    train.log
    metrics.jsonl
    checkpoint.pth
    checkpoint0000.pth
    checkpoint0001.pth
    checkpoint0002.pth
    training_paused.json
    input_diagnostics/<时间戳>/summary.json
```

每轮作者 AP、分类/框 loss 来自 `metrics.jsonl`。control 没有 unknown 检测分支，
其 U-Recall 不是本次验收目标，不能与发现方法的未知召回直接作为方法优劣比较。

训练结束后，`diagnose_control_inputs.py` 自动对固定 **256 张**训练图做只读探针，索引
均匀覆盖已释放 Task 1 训练集，并额外读取各索引的下一张真实图用于像素替换。
`--images 4` 保留此前索引 0、7、15、23 的小探针，当前启动器固定使用 256。
分别将真实图换成标准化均值颜色、另一张真实训练图；保持尺寸、mask、位置编码条件，
比较原始 boxes / logits 及六层 encoder `norm2` 的输出。另一张真实图只在诊断中
resize；不改变训练或作者评估。encoder 每 64 个空间 token 采样，与此前分析一致。
探针核验 manifest、训练标注、类别顺序、DINO 哈希及每轮权重前后哈希。
每次诊断写入新时间戳目录，记录均值、中位数、最大值和低于 1e-6 的比例；框差、
logits 差及 encoder 末层差直接打印到同一 console 日志。

固定 3 轮后停止并分析，不按测试集最佳轮次选权重：

- 若 control 也逐步失去图片响应且 AP 低迷，优先检查基础检测训练与 encoder。
- 若 control 保持图片响应并学到已知检测，再从同一初始状态逐项加入图/发现监督。
- 若结果尚不充分，先分析三份保留权重，再决定是否追加预算；不自动开始全链。

`1e-7` 量级是此前退化权重的观测，不是单独判定新模型成败的通用阈值。
输入差异、loss 和局部覆盖都不能替代作者全量 AP；3 轮结果也不是充分训练后的最终成绩。

## 现有结果之后的学习率对照

首轮 control 实跑 1 小时 22 分钟，Known AP50 为 0.5010% → 0.2193% → 0.2318%。
扩大的只读审计也确认第 2 轮大范围失去像素响应，见
[本次分析](owod-control-mowodb-analysis-20261110.md)。下一轮仅测试学习率缩放，不叠加 warmup、
梯度裁剪或新损失修改：

| 参数 | standard | lr-quarter |
| --- | ---: | ---: |
| 主学习率 | 2e-4 | 5e-5 |
| backbone 学习率 | 2e-5 | 5e-6 |
| sampling offsets / reference points 学习率 | 2e-5 | 5e-6 |
| 作者数据、双卡、每卡 batch | 全量 M / 2 / 2 | 相同 |
| epochs / lr_drop / 执行预算 | 50 / 50 / 3 | 相同 |

这会改变优化轨迹，必须从 DINO 和相同 seed 重新开始，不能更改旧 control 的学习率后续训。
`--control-lr-scale 0.25` 只允许全量 M Task 1 control，缩放值进入不可变计划，实际 lr
进入权重及恢复校验；默认值 1 保持旧计划兼容。配方来源明确标记项目学习率诊断，
不能称为作者原始优化超参数。该变体没有正式 AP 改善的实测证据。

先通过 Git 同步代码，然后在相同 Linux 环境执行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new-session -d -s m-control-lrq 'bash tools/owod/run_control_mowodb_3090.sh lr-quarter'
tail -n 50 -F output/tree-detr-m-control-lr-quarter-3090-logs/console.log
```

新结果目录为 `output/tree-detr-m-control-lr-quarter-3090/`，日志单独保存。
训练预计约 1–1.5 小时，256 图逐轮探针另需几分钟，以实际耗时为准。3 轮后固定暂停：
若输入响应保持且 Known AP 出现持续学习，再讨论延长预算；若仍普遍退化，保留证据，
继续检查 encoder 更新和基础实现。较低学习率可能收敛更慢，3 轮 AP 未超过原组不能
单独证明它无效；同样，输入响应恢复也不能单独证明检测 AP 已修复。

## 本地验证与边界

2026-11-10：control 新入口及原 v7 暂停/恢复、strict setting、v8 相关回归共 54 passed。
覆盖实际训练循环的 3 轮暂停、每轮编号权重、optimizer / scheduler 恢复、拒绝混用
发现权重和未完成阶段转移、脚本数据失败即停及输入诊断不改权重。
另用本机 CUDA、实际 DINO + 6 层 Deformable DETR 对一次性合成图片检查探针执行，
结果保存在忽略目录 `output/analysis/control-mowodb-20261110/cuda-smoke.json`。
这些本地验证只覆盖入口与计算路径；首组服务器实跑结果单列在本次分析中，
不能把机制检查通过解释为检测效果改善。

后续学习率入口与 256 图诊断的直接相关回归共 **66 passed**（4 条已有 scheduler 弃用提示）。
新增覆盖默认计划兼容、仅学习率变化、错配权重拒绝恢复、两种变体独立结果目录和 256 图
统计。这个检查结果不属于服务器优化变体的训练成绩。
