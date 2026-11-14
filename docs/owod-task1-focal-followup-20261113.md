# M-OWODB Task 1：同轮诊断与 focal 评分对照

2026-11-14 更新。用户已完成 focal Task 1 的 8 轮暂停实验。本页先前的服务器
第 5 轮只读诊断和实现说明仍保留；新增正式日志结果见下方。接下来可启动独立的
focal 七阶段全量链。

推荐下一组固定为：**γ=1、focal 评分、排序权重 1、主 lr=5e-5，从 DINO 独立
训练，先完成 8 轮即暂停。** 原 combined γ=0.1 保留暂停状态。依据是当前 M
训练图上的评分精度差距，不继续扫描 γ，也不更换数据协议或发现模块。

## 新 focal 8 轮正式结果

日志 `C:/Users/23642/Downloads/console.log` 的 SHA256 为
`2b51af64611f0dceefea71d01e478d72f128d239dd08737ec04f3231528f19a3`，记录的配置为
`variant=focal`、`scoring=focal`、γ=1、rank coefficient=1、主 lr=5e-5、seed 42、
双 3090、每卡 batch 2，正常完成 8/50 轮并保留 checkpoint。第 8 轮使用作者 M
VOC 积分 AP50：

| Task 1 第 8 轮 | focal | 原 ranking | 原 combined |
| --- | ---: | ---: | ---: |
| Known / Current AP50 (%) | **42.1399** | 36.0015 | 34.0246 |
| U-Recall (%) | 9.1338 | 8.2890 | 8.9494 |
| Unknown Precision (%) | 1.0220 | 0.9263 | 0.9780 |
| WI@recall=0.8 | 0.079876 | 0.069896 | 0.061788 |
| A-OSE | 25,992 | 24,530 | 21,424 |
| H-Score (%) | 15.0134 | 13.4754 | 14.1713 |

focal 相对原 ranking 同轮 AP 提高 **6.1383 个百分点**，U-Recall 提高 0.8448
个百分点，Unknown Precision 提高 0.0957 个百分点；WI 增加 0.009980，A-OSE
增加 1,462，未知误检侧的改善并不一致。相对 combined，AP 提高 8.1152 个百分点，
但 WI/A-OSE 同样更高。该结果支持 focal 进入正式全量验证，不能据第 8 轮替代固定
第 50 轮，也不把 AP 改善归因于某一个损失项。

## 当前全量决策

正式全量实验选择 focal，运行 Task 1 increment、Task 2–4 increment 及各自 balanced
finetune，共 7 个训练阶段和 3 个 GNN 校准阶段。`run_discovery_mowodb_3090_full.sh`
现在接受 `ranking`（默认）或 `focal`。由于已有 focal 8 轮目录是 `--task1-only`
计划，不能用 `--resume` 把它改成七阶段计划；正式全量 focal 链使用新目录
`output/tree-detr-m-v8-focal-lr-quarter-3090-full`，从 DINO 重新开始，保留原
`...-task1` 作为 8 轮对照。具体启动命令见全量训练说明。

## 已完成的真实权重诊断

两份输入均为 Task 1 第 5 轮 `checkpoint0004.pth`，每份 641,496,263 字节：

- ranking：`output/tree-detr-m-v8-ranking-lr-quarter-3090-full/m-owodb/seed_42/stage_0/increment/`。
- combined：`output/tree-detr-m-v8-combined-lr-quarter-3090-task1/m-owodb/seed_42/stage_0/increment/`。

原 ranking 没有第 8 轮编号文件，因此不用第 10 轮替代第 8 轮进行机制对照。
服务器诊断时 HEAD 为 `ed1b5969b065805d6161e18deea0ff7fe7e6f302`。抽查训练、模型、
诊断等 8 份源码，与本地统一换行后的 SHA256 一致；这核验了诊断时的源码，不能
追溯证明旧训练进程使用的全部文件逐字一致。服务器本轮未修改源码或更新权重。

两次固定 seed 42，从相同合法 Task 1 训练标注按 source-ID 哈希选取 256 图，
共 741 个已释放 GT。两份报告的图像 ID 顺序完全一致。未读取测试图或未来类标注
来选候选、改阈值或选择评分。GPU 0 空闲时通过 UUID 使用单张 3090，未占用其他
作业的卡；没有 optimizer、EMA 更新或原型更新，输入哈希与原型 buffer 前后不变。

| 训练子集诊断 | ranking γ=1 | combined γ=0.1 |
| --- | ---: | ---: |
| 任意原始框覆盖 IoU>0.5 (%) | 85.0202 | 85.5601 |
| 类别 argmax 正确的框覆盖 (%) | 80.9717 | 81.7814 |
| 实际 hierarchical 联合 Top50 覆盖 (%) | 82.7260 | 81.2416 |
| hierarchical 仅已知 Top50 覆盖 (%) | 83.6707 | 82.7260 |
| 匹配 GT 的已知/未知排序违反率 (%) | 1.7544 | 3.5088 |
| 实际 Top50 中 unknown 比例 (%) | 33.2734 | 36.3594 |
| 有支持的匿名槽数 | 6 | 9 |
| 诊断耗时，含加载与报告（秒） | 41.08 | 37.01 |

γ=0.1 的原始框和正确类别覆盖略高，但实际输出覆盖降低、已知/未知排序违反
更多，支持进一步检查评分与误检排序。覆盖允许一个 query 覆盖多个 GT，也不惩罚
重复预测，不能据此判断一对一 AP。Task 1 尚无增量类别 RiskRoutedStage，本轮
不把 Task 2 路由或继承问题当成 Task 1 退化的已证实原因。

## 同一前向输出的一对一评分审计

随后对相同 256 张训练图补做评分精度审计。每张图只前向一次，在同一组 raw
outputs 上切换评分，保持联合 Top50。复用项目作者 VOC 积分 AP50 evaluator，
保留原始 `voc_bbox`、difficult、分数/坐标量化与一对一匹配；复制 spec 仅用于所选
训练图，不改正式 manifest。训练标注没有完整未知 GT，因此只计算 Known AP，
不输出训练子集 U-Recall、WI 或 A-OSE。

| 同权重反事实评分 | ranking 训练子集 Known AP50 (%) | combined 训练子集 Known AP50 (%) |
| --- | ---: | ---: |
| 原 hierarchical | 49.3160 | 46.6075 |
| **已有 focal** | **60.7337** | **55.7025** |
| 纯 known-sigmoid，内部检测参照 | 60.1850 | 54.6946 |

ranking 的 focal 相比同权重 hierarchical 高 11.4177 个百分点；combined 高
9.0951 个百分点。两次评分审计分别耗时 22.29、20.19 秒。ranking 的 12,800 个
联合输出中，hierarchical 有 4,259 个 unknown，focal 有 5,092 个；该训练子集
差距并非通过清空 unknown 获得，但不能据此认定其真实未知召回提高。

审计还检查了去 novelty 门控、sigmoid 叠加 novelty 门控和带隐式背景的 softmax。
去掉 hierarchical 的 novelty 门控仍只有 48.4728% / 45.9584%；评分归一化值得
关注。其余探索值留存在原始证据，未增加到训练实现或启动选项。复用已有 focal
足以构造一个可归因的对照。历史 S 协议上的 focal 结果没有显示普遍收益，不能
从当前 M 子集推广出跨协议有效结论。

早期评分审计数值来自训练图和已训练的 hierarchical 权重，**不是正式测试成绩，
也不预测重新训练 focal 后将获得相同增益**。新 focal 日志已提供正式 Task 1 第
8 轮结果，但仍需完成完整七阶段验证。

## 实际配置与损失对应

已有实现位于 `models/graph_local/discovery.py`：`DiscoveryPostProcess` 和
`known_unknown_log_ratio`。本次复用，不新增背景头或评分公式。

```text
o = sigmoid(objectness_logit), n = sigmoid(novelty_logit)
hierarchical: S_k = o * (1 - n) * softmax(z_known)[k]
focal:        S_k = o * sigmoid(z_k)
两者未知:    S_u = o * n

focal 匹配已知 GT 的 log(S_u / S_y)
  = logsigmoid(novelty_logit) - logsigmoid(z_y)
L_rank = mean_matched_GT softplus(log(S_u / S_y))，权重 1
```

类别密集 focal 监督使用独立 sigmoid，hierarchical 将这些 logits 再做类间
softmax，会改变跨 query 的置信度排序。新对照使用已有独立类评分，并使排序损失
对应实际分数。**它同时影响联合 Top50 和排序损失，不是仅改评估后处理。**
匹配、普通 GT 分类/框监督、GNN、adapter、原型和 EMA 的结构及其余配方不变。

| 低学习率启动模式 | 评分 | γ | 排序权重 | 用途 |
| --- | --- | ---: | ---: | --- |
| `ranking` | hierarchical | 1 | 1 | 参照配置 |
| `combined`，保留原默认 | hierarchical | 0.1 | 1 | 已完成 8 轮，AP 未改善 |
| **`focal`，本次推荐显式指定** | **focal** | **1** | **1** | 新独立对照 |

固定作者 M-OWODB、seed 42、DINO ResNet-50、100 queries、每卡 batch 2、双 3090
GPU 0/1、activation checkpointing、lr=5e-5 / backbone lr=5e-6、50 轮日程、
lr_drop=50、balanced-bce objectness、count novelty 归一化、3 轮 pseudo ramp。
`focal` 指评分选项，**不是** `--discovery-objectness-loss focal`。

方法仍为 v8；scoring 和 γ 已进入计划和 checkpoint 校验。新组从 DINO 初始化，
不从现有 ranking/combined 权重初始化，不将旧权重改参数 resume。固定独立目录
隔离结果；8 轮之后同组继续到 50 轮不改变注册的优化日程。`--full` 仍仅 Task 1。

## Git 同步与服务器启动

本地 Windows / PowerShell，仓库根目录。本轮没有自动 commit 或 push，使用 Git
同步下列源码、测试和必要 Markdown，`output/` 下的原始诊断证据保持忽略：

```powershell
cd C:\programlearning\Tree-DETR
git add -- tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh tools/owod/probe_discovery_checkpoint.py tools/owod/scoring_probe.py tools/owod/tests/test_task1_low_lr_launcher.py tools/owod/tests/test_discovery_checkpoint_probe.py tools/owod/tests/test_scoring_probe.py docs/owod-task1-focal-followup-20261113.md docs/owod-task1-low-lr-optimization.md docs/owod-task1-combined-epoch8-analysis-20261113.md docs/owod-discovery-v8-debug.md
git commit -m "Add low-LR Task 1 focal scoring comparison and train score audit"
git push origin mowodb
```

服务器 Linux / Bash，沿用已有 `tree-detr` 环境。先确认 GPU 0、1 空闲，再启动
一组，避免与旧训练争用：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
git pull --ff-only origin mowodb
nvidia-smi
bash tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh focal --dry-run
tmux new-session -d -s tree-m-t1-focal 'export PATH=/home/top/disks/new-hdd/conda_envs/tree-detr/bin:$PATH; bash tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh focal'
tail -n 50 -F output/tree-detr-m-v8-focal-lr-quarter-3090-task1-logs/console.log
```

dry-run 应显示仅一个 Task 1 命令，scoring=focal、γ=1、rank=1、50 轮日程和
8 轮暂停。启动器先核验 M 数据；核验/训练失败返回非零状态并保留日志。已有计划
或权重但没有 `--resume` 会拒绝启动。

结果：`output/tree-detr-m-v8-focal-lr-quarter-3090-task1/m-owodb/seed_42/stage_0/increment/`。
其中有 `train.log`、`metrics.jsonl`、`run_config.json`、暂停点 `checkpoint0007.pth`。
console 独立保存在同名 `-logs/console.log`；原两组目录不变。Ctrl+C 退出 tail
不会停止 tmux 训练，`tmux attach-session -t tree-m-t1-focal` 可查看终端。

意外中断时，确认原进程已退出、GPU 空闲，使用同一模式恢复至 8 轮：

```bash
tmux new-session -d -s tree-m-t1-focal-resume 'export PATH=/home/top/disks/new-hdd/conda_envs/tree-detr/bin:$PATH; bash tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh focal --resume'
tail -n 50 -F output/tree-detr-m-v8-focal-lr-quarter-3090-task1-logs/console.log
```

只有固定 8 轮验收支持继续时才追加 `--full`，沿同组计划完成 Task 1。不要裸调用
此启动器：为兼容旧行为，省略模式仍运行 combined。

## 可重复运行的训练图评分诊断

`probe_discovery_checkpoint.py --score-audit` 在原有覆盖、路由、候选和原型检查中
复用同一 forward，额外调用现有 hierarchical / focal postprocess 和纯 known
sigmoid 内部参照。结果在 `score_audit.metrics.*.train_known_ap50`，附带诊断范围。
默认不开此项，原 Task 1/2 probe 行为保留；该选项只允许 Task 1。Task 2 increment
没有完整旧类训练标签，不能将漏标旧类当成完整 Known AP 的误检。

服务器仓库根、训练停止后，在空闲单卡上执行。以下是新组 8 轮的诊断及独立日志：

```bash
set -o pipefail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/owod/probe_discovery_checkpoint.py \
  --checkpoint output/tree-detr-m-v8-focal-lr-quarter-3090-task1/m-owodb/seed_42/stage_0/increment/checkpoint0007.pth \
  --manifest data/owod/m-owodb/split_manifest.json --images 256 --seed 42 --score-audit \
  --output output/tree-detr-m-v8-focal-lr-quarter-3090-task1-logs/task1-epoch8-probe.json \
  2>&1 | tee -a output/tree-detr-m-v8-focal-lr-quarter-3090-task1-logs/task1-epoch8-probe.log
```

输出拒绝覆盖已有 JSON，重复诊断需另给文件名。同轮比较应选实际存在的相同编号
权重、相同样本设置。报告的训练子集 AP 仍不替代正式成绩。

## 预算、停止条件与正式验收

- 新组先投入 8 轮，预估 **4.5–6 小时**，参考 combined 实测 4:40:14。完成即
  暂停，不连续启动别的消融或自动进入 Task 2。后续完整 50 轮暂预留约 28–35
  小时，早期改善不证明末轮改善。
- 固定第 8 轮 Known/Current AP50 对比旧 ranking 的 **36.0015%**、combined 的
  **34.0246%**。同时报告 U-Recall、WI@0.8、A-OSE、Unknown Precision 和输出数；
  旧 ranking 对应为 8.2890%、0.069896、24,530。不能用训练子集 +11.4177 点
  代替这个验收，也不按测试集最高轮次挑 checkpoint。
- 若 AP 改善且未知发现仍有效，才支持继续同组至固定 50 轮。如果 AP 未改善，
  或改善伴随明显的未知发现损失，保留暂停状态，依据报告决定下一因素。遇输入
  校验失败、非有限 loss、unknown 占满输出且 AP 退化，停止并保留日志、有效权重。
  首轮 warmup 无未知输出本身不算失败。
- 已有纯检测 control 第 8 轮 41.8286% 是能力参照，含训练与评分因素差异。正式
  末轮还需对比 ranking 的 56.0979% AP、4.0523% U-Recall，早期结果不能证明后期
  未知召回下降已经解决，也不能证明 Task 2 继承和旧类保持改善。

## 证据位置

服务器原始报告、console：`output/analysis/m-v8-paired-epoch5-20261113-01/`，包含
`ranking-epoch5.json/.log`、`combined-epoch5.json/.log`、`paired-summary.json`、
`score-precision-audit.json`。本地取回副本与协调输出位于
`output/analysis/task1-followup-20261113/`，原始证据保持 Git 忽略。

关键输入 SHA256：

```text
manifest 9488bc9fd0c266958b567bfd2c1c6e8a0258d85452aca19e171123e4d8a3d1c3
Task1 increment annotations 6ae6ae952a54ac1cf010c25440c31ff27d8c101d213a9573de8c101c7987a44d
ranking checkpoint f4eb0130387fd5922b96a37e2fcac65a78758c41e1bdc7a97daaa7c4ff96be84
combined checkpoint 903a97fadab678f4d7a19e366eb4f1ebccbb41460076e2f4935f50e4d7c37e3c
```

本地工具验证使用合成输入和小模型，真实服务器的第 5 轮诊断使用完整权重和合法
训练图；新 focal 日志为用户服务器的正式第 8 轮评估。两者都不替代完整七阶段
结果。

本地直接相关检查 **72 passed**：新三模式与旧默认行为、独立输出和失败保护、
跨评分/γ 续训拒绝、真实 CUDA Task 1/2 checkpoint/EMA/路由前向、评分审计的一对一
重复 FP/difficult、完整样本要求与未来标签拒绝，以及原 v8、四任务链、评分竞争、
排序梯度和作者评估回归。5 条提示来自已有 torchvision 参数和 torch meshgrid API。
Bash 语法、两个入口 help 和 `git diff --check` 通过。本地没有正式 M manifest
及本次完整权重，未在本地进行正式 AP 评估；服务器只做了上文列出的只读诊断。
