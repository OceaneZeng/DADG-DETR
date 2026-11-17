# S-OWODB Task 2：soft 路由下的定位容量对照

2026-11-17新增显式 `--epochs 3` 快速诊断选项，独立计划/输出，只在第3轮末评估；
无参命令仍使用本页的8轮计划。快速运行、Git同步及未知检测诊断顺序见
[3轮快速检查](sowodb-task2-fast-check.md)。旧8轮恢复须使用原提交，不能改计划接成3轮。

## 依据与边界

2026-11-16，已完成 hard/soft 八轮诊断：Current AP50 为 2.5996%/3.9701%，
Previous AP50 为 63.3823%/63.2711%。本次通过 SSH 只读取得同一服务器 main
提交 `2002056ed73809a4c8ba93523e3fdbf996a36f4f` 保存的 learning probe 与参数审计。

训练图诊断固定为每个当前类前12张作者训练图，去重后187张、553个新类GT。
它是类覆盖的局部诊断，不能当作测试召回或AP。

| 第8轮训练图计数 | hard | soft |
|---|---:|---:|
| 原始框 IoU≥0.5 覆盖 |144/553 (26.04%)|172/553 (31.10%)|
| 实际 joint Top50 正确类覆盖 |80/553 (14.47%)|91/553 (16.46%)|
| 去 unknown 竞争的 known_only |82/553 (14.83%)|93/553 (16.82%)|
| classifier_only |85/553 (15.37%)|97/553 (17.54%)|

soft 中去掉 unknown 竞争只多覆盖2个GT，尚不足以把 Top50 unknown 竞争视为主要瓶颈。
原始框与正确类覆盖均低，说明定位和表示适配值得优先检查；不证明仅解冻框头就能解决。
hard 两个插入层的正确类别路由为83/553、98/553，null为373/553、334/553；
soft正确类别的实际gate均为正。hard类别80（toaster）的两个up矩阵仍为零，
soft全部42个up矩阵非零。两组旧分类行变化均为零；这些是更新证据，不是AP证据。

## 本次复用的实现与固定配置

复用已有 `configure_trainable(..., localization='bbox-last')`，不引入新模块或损失。
它开放现有 box MLP 最后坐标映射的 weight/bias；backbone、基础decoder和框头前两层
仍冻结。标准 Deformable DETR 默认共享框头，本次非 refine 配置只额外开放
`4×256+4=1028`个唯一参数；不新增推理参数。其余新类分类行、类别adapter/router、
discovery路径、旧类蒸馏和GNN风险门控沿用当前v7。

| 项目 | frozen | bbox-last |
|---|---|---|
| 类别路由 |soft|soft|
| 坐标映射 |冻结|更新最后一层|
| 起点 |完整v5 Task1第50轮及对应GNN|必须相同的实际state SHA256|
| 数据/评分 |完整作者S Task2 / hierarchical / Top50|相同|
| 训练 |seed42，两张3090，每卡batch1，总batch2|相同|
| 优化 |AdamW，lr2e-4，masked classifier零decay|相同|
| 日程 |8轮increment，无balanced finetune|相同|
| 评估 |初始化、固定第3/8轮完整作者测试集|相同|

这是定位参数可训练范围的工程对照，不是论文新机制或已证明的效果改进。
新增可训练容量是处理因素的一部分，不能归因为纯路由/GNN收益。
本次尝试读取OW-DETR固定提交的官方main.py时网络超时，未用原文为该冻结策略背书；
作者数据协议和AP口径继续以已核验manifest为准。

两组均重新从同一完整Task1建立Task2，不接续已有8轮诊断权重。新suite、新输出目录
和初始化证据与旧hard/soft完全分开；新代码也改变源码哈希，旧诊断计划不会被强行续接。
resume核验路由、定位策略、组名、整套计划、源码/输入/扩展哈希和每rank随机状态。
评估中断与第3/8轮随机流保护复用已有实现。正式入口继续拒绝诊断checkpoint。

## Git同步后的服务器入口

本地 Windows / PowerShell，当前 main 分支仅提交本次文件；不包含用户已有的
AGENTS.md、research/selection.md 改动：

```powershell
cd C:\programlearning\Tree-DETR
git add -- tools/owod/run_sowodb_task2_routing.py tools/owod/run_sowodb_task2_routing.sh tools/owod/run_sowodb_task2_localization.py tools/owod/run_sowodb_task2_localization.sh tools/owod/train_task2_diagnostic.py tools/owod/task2_learning_probe.py tools/owod/task2_routing_cuda_check.py tools/owod/tests/test_task2_routing_experiment.py tools/owod/tests/test_task2_localization.py docs/sowodb-task2-routing-experiment.md docs/sowodb-task2-localization-experiment.md docs/owod-results-analysis-20261116.md
git commit -m "Add paired S Task2 localization diagnostic and OWOD result analysis"
git push origin main
```

本地源码经Git同步服务器后，Linux/Bash仓库根目录、既有tree-detr环境执行。
GPU0/1须为可用的两张3090；不在运行中的checkout切分支或改源码。

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git pull --ff-only origin main
bash tools/owod/run_sowodb_task2_localization.sh --check
tmux new -s s-task2-loc8 'bash tools/owod/run_sowodb_task2_localization.sh'
```

`--check`仅核验数据及父节点；正式启动先运行两步合成数据双卡检查，分别验证
soft/frozen、soft/bbox-last的坐标更新、旧分类行保持、反向、NCCL、EMA和随机状态。
同一命令自动恢复自己的未完成运行；不与旧结果共用输出，不自动开始其他Task。

根目录 `output/sowodb-task2-v7-localization-8epochs-2gpu/`：

- `console.log`：总日志，恢复追加。
- `results/diagnostic_plan.json`：独立计划与输入/源码身份。
- `results/frozen/`、`results/bbox-last/`：初始化、逐轮checkpoint/统计、固定第3/8轮指标与覆盖。
- `parameter_audit.json`：旧分类行、adapter变化、bbox整体及隐藏层最大变化。
- `results/comparison.json`：固定第8轮，`frozen`、`bbox_last`、差值及第3→8轮变化。

```bash
tail -f output/sowodb-task2-v7-localization-8epochs-2gpu/console.log
cat output/sowodb-task2-v7-localization-8epochs-2gpu/results/comparison.json
```

## 预算、停止条件与判断

前次hard/soft纯训练为13.46/17.96小时；soft第6/7轮出现临时减速，第8轮恢复。
本次两组各8轮，预留约28–38小时含评估/初始化，以首轮实测修正；非精确ETA。
GPU竞争会扩大耗时，应保持运行条件稳定，不中途改batch、卡数或学习率。
输入/初始化不一致、非有限loss/梯度、OOM或设备故障时停止并保留现有成果。
第3轮低AP本身不改变固定8轮预算；不从测试集选最好epoch或调整阈值。

先检查bbox-last的bbox审计非零、隐藏层为零，frozen的全部bbox变化为零。
再联合比较原始框/正确类覆盖与末轮Current AP50，并报告Previous/Known、U-Recall、
unknown precision/数量、WI、A-OSE。定位覆盖改善但Current仍低，下一步调查分类/表示；
Current提高但Previous或unknown恶化，则不能称整体改善。
单seed短increment不能证明收敛或完整任务收益；后续仍需独立increment→作者balanced finetune。

本次没有直接改服务器代码，也没有启动这两组长训练。验证结果另见本次交付记录。

## 本次验证与证据归档

本地现有tree-detr环境：35项直接相关检查通过，覆盖两种定位策略的真实小型
Deformable DETR CUDA反向、坐标层更新/冻结、旧分类行保持、state重载、配对身份与
跨组恢复拒绝，以及固定第3/8轮评估和随机状态保持。7条提示来自既有torchvision/meshgrid。
1项已知Windows双进程Gloo检查未重跑；本次新增路径尚未在目标Linux双3090上运行。
没有运行新八轮实验，也没有优化后的AP结果。

忽略目录 `output/analysis/owod-20261116/` 保存16份服务器原始JSON及逐文件SHA256
的合并证据 `s-task2-server-evidence.json`，以及20轮M附件指标
`m-task1-attachment-metrics.json`。这些生成证据不加入Git。
