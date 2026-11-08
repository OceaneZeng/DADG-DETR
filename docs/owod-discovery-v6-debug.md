# Discovery v6：M-OWODB 双 3090 渐进伪未知监督

本版本是针对 2026-11-08 新服务器 Task 1 日志的实验性优化。v5 在完成第 1/2/3
轮时，作者口径 Known AP50 为 1.2215% / 0.6763% / 0.5280%；后两轮
U-Recall 为 5.0386% / 8.7307%，unknown precision 为 0.3241% / 0.5015%。
日志尚未完成 50 轮，不能据此给出收敛或最终 AP 结论。

## 问题与改动

v5 的 novelty BCE 将伪未知正例、已知负例分别按置信权重求均值，再对有效组求平均。
即使只有一个伪未知正例，该组也能占一半损失。将全部 pseudo 权重同时乘小常数，
会在该组的分子和分母中抵消。候选数量减少不等于候选训练影响同比减小。

v6 只引入一项训练因素：伪未知损失贡献在 warmup 后逐步增强。
`--discovery-pseudo-ramp-epochs 3` 启用 v6；默认 `0` 仍为原 v5。
保持 `discovery_warmup=1`，每个训练 phase 内的权重如下：

| phase 内 epoch（从 0 开始） | pseudo 系数 λ | 说明 |
| --- | --- | --- |
| 0 | 0 | 沿用 v5 warmup，不生成伪未知正例 |
| 1 | 1/3 | 第一轮伪未知学习 |
| 2 | 2/3 | 第二轮伪未知学习 |
| 3 及之后 | 1 | 恢复完整伪未知损失贡献 |

objectness、novelty BCE 和 discovery router 的损失分子，仅在 pseudo 槽位乘 λ；
归一化分母继续使用未缩放的权重。一致性损失只来自 pseudo，因此整体乘 λ。
在同批、同输出、同候选条件下，已知 GT 和未选中槽位的梯度不被该系数缩放。
已知分类及其辅助层、框监督、affinity anchor、旧类蒸馏沿用原计算。
λ 达到 1 后，同输入的损失与 v5 一致，但已经改变的优化轨迹不会自动恢复为 v5。

候选筛选、TopK=5、阈值 0.65/0.5、hard/frozen/hierarchical、联合 Top50、
EMA 更新、匿名原型及快照更新规则均保持原设置。λ 不缩放原型支持量。
每个后续 increment / balanced finetune 都从其 phase 内 epoch 重新计算日程。
系数由 epoch 推导，无需另存计数器；同配置 resume 会恢复正确位置。

伪标签学习、物体性及 novelty 的相关原始工作见
[OW-DETR（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.html)；
教师伪标签可靠性及软加权的相关先例见
[Soft Teacher（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html)。
当前改动是本项目的训练稳定性假设，不是这些论文给出的 OWOD 超参数，不作为创新性
或效果证据。它可能缓解早期错误候选的影响，也可能拖慢未知学习；尚无全量 AP 结论。

## 版本与续训

- v5 默认参数、原启动器和旧运行计划保留。旧 v5 checkpoint 缺少新字段时按 0 处理。
- ramp 大于 0 时，checkpoint 记录 `discovery_version=6` 和具体 ramp 轮数。
- 运行计划包含 ramp 参数；阶段链与 resume 同时校验版本及 ramp 轮数。
  不能用 v5 权重续入 v6，不能在同一目录中改 ramp 后恢复。
- v6 从同一 DINO ResNet-50 骨干重新训练 Task 1。M 作者 ImageSets、20/20/20/20
  类顺序、schema-3 manifest、输入哈希、均衡记忆与 VOC 评测保持原协议。
- 两张 3090 使用物理 GPU 0,1，每卡 batch 2、全局 batch 4、seed 42。
  不更改卡数、batch、学习率或训练预算来同时比较本次监督因素。

## 新服务器启动

前提：本次源码已经通过 Git 同步到服务器，沿用原 `tree-detr` 环境，GPU 0,1 已空闲。
旧训练应先正常中断，保留原输出目录及权重；不要在运行中的同一仓库更新代码。
脚本会核验 M 数据，核验失败时应通过数据管理工具补齐真实输入。

Linux / Bash，在仓库根目录执行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new -s tree-m-v6
```

在 tmux 中执行：

```bash
bash tools/owod/run_discovery_mowodb_3090.sh --dry-run \
  && bash tools/owod/run_discovery_mowodb_3090.sh
```

脚本采用完整四任务七阶段计划：每个 increment 50 轮，Task 2–4 各 balanced finetune
20 轮；Task 1 完成后会自动继续后续任务。`--dry-run` 只核验和打印计划。
输出和日志与 v5 分开：

- 日志：`output/tree-detr-m-v6-3090-logs/console.log`。
- 计划：`output/tree-detr-m-v6-3090-full/m-owodb/seed_42/benchmark_plan.json`。
- Task 1：`output/tree-detr-m-v6-3090-full/m-owodb/seed_42/stage_0/increment/`。
- 每阶段保留 `train.log`、`metrics.jsonl`、`run_config.json` 及 checkpoint。

查看及中断后恢复：

```bash
tail -n 50 -f output/tree-detr-m-v6-3090-logs/console.log
# 恢复时仍在已激活环境、仓库根目录的 tmux 内执行：
bash tools/owod/run_discovery_mowodb_3090.sh --resume
```

## 诊断与实验预算

训练日志新增归约后的 `discovery_pseudo_scale`、`discovery_pseudo_per_image` 和
`discovery_pseudo_weight_sum`。后两项分别是每图候选数和每个 rank 批次的权重和；
与原 `discovery_pseudo_count` 相同，跨 rank 取均值。权重和除以候选数可以汇总候选
平均置信权重，不将各 rank 的不同候选量误写成全局总数。
系数只控制正伪标签贡献，warmup 仍训练已知物体的 novelty 负例。

优先用不再被覆盖的第 5 轮编号 checkpoint 做只读诊断。先确认选用的 GPU 空闲，
下例仍使用本次允许的物理 GPU 0；不要与训练抢占同一 GPU：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/owod/diagnose_discovery_task1.py \
  --checkpoint output/tree-detr-m-v6-3090-full/m-owodb/seed_42/stage_0/increment/checkpoint0004.pth \
  --manifest data/owod/m-owodb/split_manifest.json --images 32 \
  --output output/tree-detr-m-v6-3090-logs/task1-epoch5-diagnostic.json
```

工具同样支持现存 v5 checkpoint；替换为其真实路径并使用独立 JSON 名称即可。
JSON 不覆盖已有文件，加载后的模型不执行优化器或原型更新。它记录：

1. `raw` 原始框覆盖、`known_actual` 按 checkpoint 原评分仅保留已知的覆盖、
   `joint` 原评分下已知/未知联合输出覆盖；另外保留 hierarchical / sigmoid 参考。
   三者用于分辨框覆盖不足、已知评分排序不足及未知竞争；不是一对一 AP。
2. 已匹配、未匹配但重叠已知 GT、未匹配且不重叠三组的 objectness、novelty、
   已知 softmax 峰值及实际已知分数分位数。最后一组不能直接命名为背景。
3. 使用 checkpoint 保存的 EMA 教师，在确定性训练视图上重新生成候选，记录框、
   显著性、一致性、语义软权重和当前分数。不是原增强批次的候选重放，
   warmup 内的探测候选也不代表当时参与了训练；不报告未知候选正确率。
4. 匿名原型支持量、达到支持门槛的槽数、最大槽位支持占比；匿名槽不等于真实类别。
5. checkpoint、manifest、训练标注哈希，检查诊断前后输入保持一致。

首轮诊断建议固定 6–8 轮预算，至少观察 warmup 后数轮。按现有日志每轮训练加评估
约 31–32 分钟，单组约 3–4.3 小时，实际时间以服务器为准。达到预算时正常中断并
保留权重日志；同一计划可以 resume，中途权重不能作为完成阶段进入 Task 2。
若原始框覆盖仍低，先排查基础训练；若原始框和已知输出覆盖较高但联合覆盖低，
再独立检验评分；若候选以重复框、物体局部或纹理为主，再独立优化候选质量。
避免同时叠加这些改动而失去归因。

短测和训练图诊断用于机制排障。正式比较使用固定末轮的 Previous / Current /
Known AP50、U-Recall、WI@0.8、A-OSE，并联合观察未知输出精度与数量；Task 4
省略未知指标。不依据作者测试集挑阈值或最好 epoch，不以 loss/机制测试通过宣称提升。
