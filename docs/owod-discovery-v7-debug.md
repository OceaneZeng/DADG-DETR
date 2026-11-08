# Discovery v7：M-OWODB 双 3090 损失对照

这是针对 v6 Task 1 退化的实验性修改，不是已经验证的 AP 提升。保持 Deformable
DETR、GNN、发现 adapter、原型/快照、类别转化和旧类保持路径；M 作者划分、图像
身份、类别顺序、manifest 核验、均衡记忆、VOC 评测与联合 Top50 均沿用现有协议。

## 诊断依据与修改范围

2026-11-08 的 v6 第 5 轮权重在确定性训练前缀 32 张图、88 个已知 GT 上：
原始框 IoU>0.5 覆盖 60 个，仅已知 Top50 覆盖 39 个，联合 Top50 覆盖 15 个；
1,600 个输出中 unknown 为 1,469 个。匹配 GT 的 query 平均 objectness 为 0.573，
639 个未匹配且重叠已知 GT 的槽位为 0.581。匹配 query 分类 argmax 正确率为
39.77%。这些是定位与评分诊断，不是一对一 AP，不代表整个训练/测试分布。

该证据支持优先检查槽位选择能力与伪未知损失强度，不能唯一归因于候选错误或 GNN。
单改 sigmoid 联合评分的覆盖仍为 15 个，故本轮保持评分规则不变。

### 1. objectness focal loss

`--discovery-objectness-loss focal` 在 `DiscoveryCriterion` 中替换 objectness
一项损失。匹配 GT 和入选 pseudo 仍是正槽位，其余槽位仍是负例；pseudo 保留
候选权重和 v6 ramp，已知 GT 权重为 1，未选中槽位权重为 1。

复用项目现有 sigmoid focal loss，alpha 沿用 `--focal_alpha`（默认 0.25），
gamma 固定为 2。对全部 query 的损失求和，除以各 rank 正槽位数的平均值，
分母至少为 1；不再对正负两组分别求均值后各占一半。不用置信度和作为分母。
DDP 的梯度平均与这个归一化配合，包括某个 rank 没有正例的情况。

目的是让高分负例获得更大的相对影响，并保留未知正例学习。未被 pseudo 选中的
真实未知仍可能受到负监督，抑制增强可能降低 U-Recall，必须联合检查未知精度和
召回。Focal loss 不保证概率校准，也不保证检测 AP 改善。

### 2. novelty 的绝对置信加权

`--discovery-pseudo-normalization count` 只修改 novelty BCE 的组内归一化。
原伪未知正组为 `sum(w * BCE) / sum(w)`；新方案为
`sum(w * BCE) / number_of_supervised_pseudo`，再乘原 ramp 系数。
已知 GT 负组仍按相同规则求均值；两组均存在时仍各占一半，仅存在一组时不额外
除以二。未选中 query 的 novelty 权重为 0，不计入分母，不当作已知物体负例。

低置信候选会降低整组贡献，统一缩小置信度不再被分母抵消。该诊断候选平均权重
约 0.205，是代理质量分数而非候选正确概率。预期风险是未知学习变慢。
objectness、router、一致性、原型更新等不因这个开关额外改变。

两项设置独立；保持 warmup=1、ramp=3、TopK=5、候选阈值 0.65/0.5、原型更新
规则和已知检测监督。没有增加其他教师视图、调候选阈值、扩大网络或移除图消息。

依据：[OW-DETR（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.html)
的 foreground/unknown 监督及作者 focal 实现；
[Soft Teacher（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html)
的软加权和候选可靠性原则。应用到本项目图头上的上述归一化与实验设置属于本项目
假设，不声称论文给出了这些 OWOD 超参数，也不作为创新性或效果证据。

## 版本与恢复

- 两个新开关默认 `balanced-bce` / `weight-sum`，旧 v5/v6 数值路径保留；旧参数
  中缺少这两项时按默认值读取。默认计划不添加新命令参数，旧计划可按原入口恢复。
- 任一开关非默认时登记为 v7；checkpoint 和运行配置记录两项设置。v7 内部也
  禁止切换 objectness/confidence/combined 继续训练；focal alpha 同样锁定。
- 每个新方案从同一 DINO ResNet-50 骨干重新训练 Task 1，不能接 v6 权重续入 v7。
- 物理 GPU 0,1，每卡 batch 2、全局 batch 4、seed 42、activation checkpointing
  与 v6 相同；不同时改卡数、batch、学习率和评分来归因损失改动。

## 最简单的验证命令

前提：先正常中断旧训练、保留权重和日志，使用 Git 把本次本地修改提交并同步到
服务器的 `mowodb` 分支；GPU 0,1 空闲，沿用现有 `tree-detr` Conda 环境。
不要在同一仓库仍有训练进程时更新代码。

Linux / Bash，在新服务器执行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new -s tree-m-v7 'bash tools/owod/run_discovery_mowodb_3090_v7.sh'
```

默认只验证 objectness 单项，训练全量 Task 1 数据并完成每轮作者全测试集评估。
计划仍为 50 轮、lr_drop=50，8 轮后自动暂停。保留 optimizer、scheduler、EMA、
原型和编号权重 `checkpoint0007.pth`；只写 `training_paused.json`，不写阶段
完成标记，不启动 Task 2。8 轮中途结果不能作为完成阶段或正式最终成绩。

约 31–32 分钟/轮，8 轮约 4.2 小时；以实际吞吐为准。比较训练图的 objectness
正负例差距、raw/known_actual/joint 覆盖和分类准确率；固定轮次报告作者 Known
AP50、U-Recall、WI、A-OSE 及 unknown 数量/precision。若连续诊断点再次出现已知
联合覆盖下降且 unknown 占比上升，可正常中断止损；不按测试集最好轮次选权重。

查看默认实验日志：

```bash
tail -n 50 -f output/tree-detr-m-v7-objectness-3090-logs/console.log
```

模式与独立路径：

| 参数 | 改动 | 结果根目录 | 控制台日志目录 |
| --- | --- | --- | --- |
| 默认 / `objectness` | 仅 objectness focal | `output/tree-detr-m-v7-objectness-3090` | `output/tree-detr-m-v7-objectness-3090-logs` |
| `confidence` | 仅 novelty count | `output/tree-detr-m-v7-confidence-3090` | `output/tree-detr-m-v7-confidence-3090-logs` |
| `combined` | 两项组合 | `output/tree-detr-m-v7-combined-3090` | `output/tree-detr-m-v7-combined-3090-logs` |

每个结果根目录下都有 `m-owodb/seed_42/benchmark_plan.json`，阶段位于
`m-owodb/seed_42/stage_0/increment/`，含 `train.log`、`metrics.jsonl`、
`run_config.json` 和权重。先分别判断单项作用，再决定是否运行组合；逐组使用 0,1
两张卡，不并发抢占。后续模式也在 tmux 内执行：

```bash
bash tools/owod/run_discovery_mowodb_3090_v7.sh confidence
bash tools/owod/run_discovery_mowodb_3090_v7.sh combined
```

`--dry-run` 会核验数据并打印实际命令，不启动训练；`--resume` 在相同方案下继续
到 8 轮暂停点。确认诊断值得继续后，可移除执行预算，按原 50 轮日程恢复：

```bash
bash tools/owod/run_discovery_mowodb_3090_v7.sh --resume --full
```

`--full` 仅表示跑完当前 Task 1 的 50 轮，仍不自动进入 Task 2。暂停预算属于执行
控制，不进入 benchmark plan，不改变 optimizer 日程；恢复只沿用同一方案、数据、
硬件与训练参数。此版本的 launcher 用于 Task 1 排障，不是四任务正式成绩入口。

第 8 轮暂停后，用空闲的物理 GPU 0 做只读诊断，结果写独立新 JSON：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/owod/diagnose_discovery_task1.py \
  --checkpoint output/tree-detr-m-v7-objectness-3090/m-owodb/seed_42/stage_0/increment/checkpoint0007.pth \
  --manifest data/owod/m-owodb/split_manifest.json --images 32 \
  --output output/tree-detr-m-v7-objectness-3090-logs/task1-epoch8-diagnostic.json
```

## 验证边界

本次本地 120 项相关检查通过，覆盖损失/梯度、空监督与未来标签隔离、v5/v6 兼容、
v7 版本拒绝、真实 Deformable DETR 小模型 CUDA 前反向及状态恢复、真实两进程
Gloo 正例计数归一化，以及 Bash 三种模式/数据核验失败退出。Bash 语法与 CLI
参数检查通过。实际 `main` 训练循环用一次性小模型验证了第 8 轮暂停、恢复至
第 50 轮、日志连续和阶段完成标记；该测试的 optimizer、EMA 和模型状态与
不中断运行一致，不代表真实数据增强的恢复过程也逐位一致。

本地没有这次服务器 M 全量训练的 manifest/权重，尚未运行新服务器双 3090 的
全量实验，无法在本地证明 AP 提升。服务器实验结果、候选真实准确性及后续任务
保持能力仍需实际评估。
