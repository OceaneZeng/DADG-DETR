# M-OWODB：Task 1 低学习率与发现梯度对照

2026-11-13。用户要求继续优化，并优先提高 Task 1 AP。本次复用 v8 已实现的发现
输入梯度缩放，增加与当前低学习率匹配的单因素入口，以及适用于正式 Task 1/2
权重的只读诊断。**尚未运行这组服务器实验，没有新的 AP 提升结论。**

## 当前证据与本轮假设

用户附件 `C:/Users/23642/Downloads/consolev8-lrquarter.log` 共 6,671 行、
4,771,458 字节，SHA256：
`6f25d8118d74b4469b71b521b604bf7fd4fa43ee31e88b8fe7e2d7afa849ae3d`。
其中 61 次作者评估分别对应 Task 1 的 50 轮和 Task 2 increment 的前 11 轮。
以下 AP/Recall/Precision 已乘 100，WI 保留原值；使用 M 作者 VOC 积分 AP50。

| 观测点 | Known AP50 (%) | U-Recall (%) | Unknown Precision (%) | Unknown 输出数 | WI@0.8 | A-OSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task 1，第 8 轮 | 36.0015 | 8.2890 | 0.9263 | 208,680 | 0.069896 | 24,530 |
| Task 1，第 40 轮 | 56.3605 | 14.0823 | 1.6796 | 195,521 | 0.063879 | 23,297 |
| Task 1，第 50 轮 | 56.0979 | 4.0523 | 0.2394 | 394,762 | 0.060818 | 8,083 |
| Task 2 increment，第 11 轮 | 22.7009 | 4.1925 | 0.4179 | 168,235 | 0.026869 | 10,855 |

来源行号依次为 838、3654、4534、6582。Task 2 第 11 轮 Previous AP50 为
37.1403%，Current AP50 为 8.2614%；该阶段尚未完成，不能当作 Task 2 最终成绩。
Task 2 初始化记录为 0/20 新类成功继承匿名快照，只能说明本次没有执行有效继承，
不能据此确定原型漂移、路由或定位哪一项是低 AP 的主要原因。

同为低学习率的[纯检测 control 第 8 轮](owod-control-lr-quarter-epoch8-analysis-20261111.md)
Known AP50 为 41.8286%，比完整方法同轮高 5.8271 个百分点。control 使用
known-sigmoid 评分且没有发现分支，这个差距含训练与评分等因素，不能全归为梯度
干扰。因此本轮同时观察原始框覆盖、正确类别覆盖和联合 Top50，验证下列假设：

> 在主 lr=5e-5 下减弱发现监督对共享 query 表示及伪框路径的回传，可能改善已知
> 检测，同时保留 GNN 和发现分支的学习能力。

Task 1 后期 AP 大致持平而未知召回显著下降，说明仅提高早期已知 AP 不够。
第 40/45/50 轮用于观察固定区间的变化，不从作者测试集挑选最好轮次替代第 50 轮。
未知标注不完整，未匹配到未知 GT 的预测不能全部认定为背景。

## 固定实验及续训边界

新入口：`tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh`。

| 模式 | 主 lr / backbone lr | 排序权重 | 发现输入梯度 γ | 训练范围 |
| --- | --- | ---: | ---: | --- |
| `ranking`，可选的同轮参照 | 5e-5 / 5e-6 | 1 | 1 | 仅 Task 1 |
| `combined`，本轮默认 | 5e-5 / 5e-6 | 1 | 0.1 | 仅 Task 1 |

两组都从相同 DINO ResNet-50 重新训练，固定作者 M 数据、seed 42、100 queries、
物理 GPU 0/1、每卡 batch 2、activation checkpointing、50 轮日程与 lr_drop=50。
hierarchical、balanced-bce、count、3 轮 pseudo ramp 及其余训练设置沿用原低学习率
ranking 配方。默认完成 8 轮后暂停；`--full` 仍只完成 Task 1，不自动进入 Task 2。
保留现有完整链及其权重和日志。

γ 的[实现范围](owod-discovery-v8-debug.md#模块与训练位置)是 affinity GNN 的 query
输入回传及伪框一致性的 student boxes 回传。前向数值不变，GNN 参数完整更新；
已知 GT 检测损失、排序损失的已知分类路径及 discovery adapter 的检测训练路径
保持原梯度。因此 γ=0.1 不等于把全部发现损失或全部参数更新乘 0.1。

该参数已有版本与配置校验。不能把旧 γ=1 权重改成 γ=0.1 后继续 resume，也不能
把旧完整链的运行计划换为 Task 1 计划。新组必须从 DINO 开始；仅在同一模式、
同一目录内使用 `--resume`。8 轮暂停改为 50 轮执行不会修改注册的优化日程。

先使用现有完整链的同轮 AP 日志作参照；如需排除两次运行的源码或运行环境差异，
再运行独立 `ranking` 参照。这会额外占用一组约 4.5–6 小时的 8 轮预算，不默认
连续启动两组。不扫描 γ、阈值或损失权重，不利用未来类别标注筛选候选或调参。

## Git 同步与新实验启动

本地 Windows / PowerShell，执行目录为仓库根。本次只提交以下源码与必要文档，
不把 `output/`、日志和权重加入 Git；当前分支为 `mowodb`：

```powershell
cd C:\programlearning\Tree-DETR
git add -- tools/owod/probe_discovery_checkpoint.py tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh tools/owod/tests/test_discovery_checkpoint_probe.py tools/owod/tests/test_task1_low_lr_launcher.py docs/owod-task1-low-lr-optimization.md docs/owod-discovery-v8-debug.md
git commit -m "Add low-LR Task 1 discovery comparison and checkpoint probes"
git push origin mowodb
```

服务器 Linux / Bash，沿用已有环境和已注册的 M 数据。先确认物理 GPU 0、1 空闲，
不要与原训练并行占用同一 GPU：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
git pull --ff-only origin mowodb
nvidia-smi
bash tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh combined --dry-run
tmux new-session -d -s tree-m-t1-low-lr 'bash tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh combined'
tail -n 50 -F output/tree-detr-m-v8-combined-lr-quarter-3090-task1-logs/console.log
```

dry-run 应只打印一个 Task 1 训练命令，lr=5e-5、backbone lr=5e-6、γ=0.1、
epochs=50、lr_drop=50、pause_after_epochs=8。数据核验或训练失败将保留日志并
返回失败状态。已有运行而未指定 `--resume` 时拒绝启动，防止覆盖。

结果目录：

```text
output/tree-detr-m-v8-combined-lr-quarter-3090-task1/m-owodb/seed_42/stage_0/increment/
```

阶段日志为 `train.log`、`metrics.jsonl`，配置为 `run_config.json`。8 轮暂停点为
`checkpoint0007.pth`。退出 tail 的 Ctrl+C 不会停止 tmux 内训练；查看终端可执行
`tmux attach-session -t tree-m-t1-low-lr`，按 Ctrl+B、松开后 D 离开。

意外中断后，在原 tmux 已结束且 GPU 空闲时，仍使用 `combined --resume` 恢复到
第 8 轮。诊断支持继续后，沿同组继续至 50 轮：

```bash
tmux new-session -d -s tree-m-t1-low-lr-resume 'bash tools/owod/run_discovery_mowodb_3090_task1_low_lr.sh combined --resume --full'
tail -n 50 -F output/tree-detr-m-v8-combined-lr-quarter-3090-task1-logs/console.log
```

## 旧权重的只读诊断

`tools/owod/probe_discovery_checkpoint.py` 支持正式 Task 1/2 的 increment 和 Task 2
finetune。核验 manifest、阶段类别顺序、作者评测口径及标注哈希，从对应阶段的
**训练标注**按 source-ID 哈希固定采样，并优先覆盖已释放类别。不同权重使用相同
seed、样本数和 phase 时选样一致；finetune 使用其作者均衡训练标注。

输入保存规则为每 5 轮及暂停点额外保留编号权重，编号从 0 开始。因此第 40、45、
50 轮分别是 `checkpoint0039.pth`、`checkpoint0044.pth`、`checkpoint0049.pth`；
Task 2 前 11 轮中可用的最近常规编号是第 10 轮 `checkpoint0009.pth`。
先核对实际文件存在。运行中的 `checkpoint.pth` 会被替换，诊断应使用编号文件。
旧完整链未必保存了第 8 轮编号，不把第 10 轮当第 8 轮作同轮机制对照。

在服务器仓库根、同一 Conda 环境、单张空闲物理 GPU 0 上运行。训练暂停或结束后
再运行，避免争用训练 GPU。以下循环保存每次诊断的独立报告和 console：

```bash
set -o pipefail
t1=output/tree-detr-m-v8-ranking-lr-quarter-3090-full/m-owodb/seed_42/stage_0/increment
t2=output/tree-detr-m-v8-ranking-lr-quarter-3090-full/m-owodb/seed_42/stage_1/increment
diag=output/analysis/m-v8-low-lr-checkpoint-probes
mkdir -p "$diag"
ls -lh "$t1"/checkpoint0039.pth "$t1"/checkpoint0044.pth "$t1"/checkpoint0049.pth "$t2"/checkpoint0009.pth
for epoch in 39 44 49; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python -u tools/owod/probe_discovery_checkpoint.py \
    --checkpoint "$t1/checkpoint00${epoch}.pth" \
    --manifest data/owod/m-owodb/split_manifest.json --images 256 --seed 42 \
    --output "$diag/task1-epoch$((epoch + 1)).json" \
    2>&1 | tee -a "$diag/task1-epoch$((epoch + 1)).log" || break
done
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/owod/probe_discovery_checkpoint.py \
  --checkpoint "$t2/checkpoint0009.pth" \
  --manifest data/owod/m-owodb/split_manifest.json --images 256 --seed 42 \
  --output "$diag/task2-epoch10.json" 2>&1 | tee -a "$diag/task2-epoch10.log"
```

报告拒绝覆盖已有 JSON。重跑时另给新文件名，保留原报告。新 combined 第 8 轮
暂停后的诊断使用：

```bash
set -o pipefail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/owod/probe_discovery_checkpoint.py \
  --checkpoint output/tree-detr-m-v8-combined-lr-quarter-3090-task1/m-owodb/seed_42/stage_0/increment/checkpoint0007.pth \
  --manifest data/owod/m-owodb/split_manifest.json --images 256 --seed 42 \
  --output output/tree-detr-m-v8-combined-lr-quarter-3090-task1-logs/task1-epoch8-probe.json \
  2>&1 | tee -a output/tree-detr-m-v8-combined-lr-quarter-3090-task1-logs/task1-epoch8-probe.log
```

报告的含义与下一步：

| 结果字段 | 回答的问题 | 如何决定下一步 |
| --- | --- | --- |
| `supervision`、`per_class` | 是否存在 IoU>0.5 的框、类别 argmax 是否正确、各评分 Top50/300 是否保留该 GT | 框覆盖低才优先考虑定位；框尚在而类别低，优先检查分类学习 |
| `score_competition` | 原始框、known-only 与实际联合 Top50 的覆盖差距；匹配 GT 的已知/未知排序 | 原始覆盖相近而联合输出恶化，支持进一步调查评分竞争；反事实评分只用于诊断 |
| `routing`、`routing_observations` | 新类匹配 query 的正确/null argmax、正确分支概率及实际有效 gate | null 多且 gate 为 0 才支持 hard 路由抑制假设；soft 路由若后续尝试，应独立成组 |
| `anonymous_memory` | 匿名槽支持、支持集中度、快照 BA 范数、当前类均值的支持/affinity 门槛 | 区分无支持与低 affinity；当前观测不能还原历史 promotion 输入，不直接降低门槛 |
| `recorded_promotion` | checkpoint 当时保存的实际继承记录 | 与当前 affinity 分开核对，不把当前可继承等同于当初已继承 |
| `ema_candidates` | 使用真实保存 EMA，在固定训练图翻转视图下重建的候选框及权重 | 可检查是否反复选择局部区域；没有完整未知标注，不能自动认定候选正确/错误 |

工具无 optimizer、无 EMA 更新、无原型更新，检查前后原型 buffer 与输入文件哈希；
路由在检测器清理临时 logits 之前采集。非零 gate 不证明对应 adapter 参数实际更新。
固定 eval 变换和翻转视图不回放历史随机增强 batch。覆盖允许一个 query 覆盖多个
GT，不是实际一对一 AP、U-Recall 或外部方法成绩。

## 时间预算、停止条件与验收

- 只读诊断用于定位瓶颈，每份固定 256 图，暂估 5–15 分钟，四份合计约 20–60
  分钟，尚无当前完整权重的实测吞吐。完成固定样本即停止；输入哈希变化、权重
  不匹配或非有限输出时停止排查，不覆盖旧成果、不启动优化器。
- 新 combined 的 8 轮暂估 4.5–6 小时。原日志 Task 1 全部 50 轮实际为
  28:13:41（含评估），因此全程暂预留约 28–35 小时；到 8 轮必须先暂停复核。
- 固定第 8 轮比较 Known AP50 与旧 ranking 同轮 36.0015%，同时核对框、分类和
  联合 Top50 覆盖。control 的 41.8286% 仅作检测能力参照。如果 AP 改善来自未知
  输出几乎消失，不能称为完整 OWOD 改善；同时报告 U-Recall、WI、A-OSE。
- 若早期检测改善且未知发现仍有效，可沿同一配置继续到固定第 50 轮；如果多轮
  已知检测持续退化、unknown 占满输出、非有限 loss 或配置核验失败，保留日志与
  有效权重后正常停止。首轮因 warmup 没有未知输出不单独算失败。
- 第 8 轮改善不证明第 40→50 轮的后期退化已解决。末轮必须实际评估 Known AP50
  对比 56.0979%，并联合报告 U-Recall 对比 4.0523%、WI、A-OSE；不按测试集最好
  轮次选权重，不据测试集未来标签调阈值。Task 1 改善也不证明 Task 2 改善。

本次只增加 Task 2 诊断能力，尚未改变其 hard routing、frozen 定位、继承门槛或
原型更新方式。根据报告再选择一个独立因素，避免同时更换多个机制而无法归因。

本地相关验证：59 passed。包括真实 CUDA Deformable DETR 的 checkpoint/EMA 加载、
训练图前向、临时路由采集、候选重建、输入变更拒绝、模型状态不变，以及两组参数
差异、旧成果保护、错误退出、恢复校验和旧 v8/全任务链回归。3 条提示来自既有
torchvision 预训练参数与 torch meshgrid API。CUDA 集成验证使用合成测试输入和
小模型，只证明工具可执行；本地没有本次正式 M manifest 和完整权重，未运行新 AP。
