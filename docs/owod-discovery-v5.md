# Discovery v5：检测槽位监督与条件 novelty

## 诊断依据（2026-10-24）

v4作者口径Known AP50在第8轮到第9轮从43.64%降到15.75%，而验证分类错误率、
bbox loss和GIoU loss都下降。第9轮unknown占240250/247600个输出，precision仅1.08%。
这不等于基础检测器完全退化；固定Top50里的known/unknown竞争需要单独检查。

在同一训练数据集前32张图、159个已知GT上，以相同确定性验证变换做只读推理：

| 已完成轮数 | 全部原始框覆盖 | 联合Top50正确类别覆盖 | 仅已知hierarchical Top50覆盖 | 仅已知sigmoid Top50覆盖 |
|---|---:|---:|---:|---:|
| 5 | 119 | 54 | 114 | 112 |
| 9 | 118 | 33 | 117 | 117 |

第9轮3200个query中，752个未匹配query与已知GT的IoU>0.3，平均objectness为0.969，
平均novelty为0.506。另2289个未匹配且不与已知GT重叠的query平均objectness为0.760、
novelty为0.729；这一组可能包含真实未知，不能称作全是背景。
无EMA的旧可信背景规则仅选中19个query，两视图规则会进一步限制负例。
已知类别的负样本学习不会自动约束另一个独立objectness头；高objectness和偏高novelty
让未选中的query继续以unknown身份占据输出。

检查的checkpoint SHA256：
- 第5轮：`3f1e1eb6adde255d28d0191b67ffb297af77913875b285ac8000a38b49108f91`
- 第9轮：`239b673291813b2826bdccbe71fb95c113e0834ba3ee8cd7b90baad500d9de22`

两份权重在诊断前后的哈希均未变化。上述覆盖不是AP/U-Recall，不是一对一匹配评价，
也不代表全测试集性能。未读未来GT，未创建新划分，未用测试集选择评分参数。

## v5修改及边界

### 2026-11-08：unknown threshold 的评分修正

原实现只把 `unknown_threshold` 写入 `unknown_mask`，却在共享 Top-K 前仍保留所有
`objectness * novelty` unknown 分数；作者评估器只按 `label=91` 计数，因此低于阈值的
候选仍会挤占 known 结果并被算作 unknown。现在在 `DiscoveryPostProcess` 合并 Top-K 前
屏蔽低于阈值的 unknown 候选，并在候选不足 `max_detections` 时丢弃屏蔽占位项。评估器
同时对外部 prediction 做同一阈值的防御性检查。

这只改变 post-process/evaluation，不改变模型权重、hard routing、frozen localization、
训练损失或数据协议；已有 v5 checkpoint 可以只读重评估。修正前后的正式指标不能直接
混写，重评估结果必须记录对应提交和阈值。

1. 保持v4的全部query已知类focal监督。
2. objectness改成检测槽位是否被选中的监督：Hungarian匹配GT及去重后的可信pseudo
   为正，其余槽位为负。pseudo保留其置信权重。负例不再依赖极稀疏的低显著性背景掩码。
3. router与上述选择一致：已知/匿名/未选中三种目标，全部query参与。
4. novelty表示在已选中物体条件下是否未知，只在已知GT及pseudo上训练。
   未选中的query不再被作为“已知物体”来训练novelty。

这是项目监督方案的改变，不只是无行为变化的bug修复。未被pseudo挖掘覆盖的真实未知
可能成为槽位负例，损害U-Recall；训练需要联合检查Known AP50、U-Recall、WI和A-OSE。
不能只因已知AP提高就认定方法成功。真实AP稳定性及增量保持效果尚未验证。

不修改评分公式、Top50、学习率、warmup、伪标签阈值、数据协议或类别顺序。
不关闭未知检测，不添加类别配额，不用更大TopK提升正式结果。
原型、EMA和迁移机制保留；新版本号5阻止v4/v3权重直接续训或初始化后续正式阶段。

训练日志使用跨rank归约后的discovery统计，并显示新字段：
- `discovery_known_supervised_fraction`：应为1。
- `discovery_objectness_supervised_fraction`：通常为1。
- `discovery_novelty_supervised_fraction`：已知匹配+可信pseudo的比例。
- `discovery_proposal_negative_fraction`：未选中检测槽位比例。
- `discovery_background_fraction`：保留旧显著性背景规则的诊断值，不再决定objectness负例。
- `discovery_ignored_fraction`：v5仅表示novelty忽略的未选中槽位比例。

v4的新统计原来仅以`*_unscaled`保存，没有接入直接命名的训练日志；v5补齐统一记录，
旧的pseudo/ignored直接命名统计也改为使用归约结果，避免只读本rank数值。

## 启动与恢复

服务器仍由用户更新和启动。先在旧训练tmux中Ctrl+C退出，确认旧训练进程已结束，
再拉取本地已提交推送的改动，不删除control/v3/v4输出。

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
git pull --ff-only origin main
python tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod \
  && bash tools/owod/run_discovery_full.sh --dry-run \
  && bash tools/owod/run_discovery_full.sh
```

独立输出为`output/tree-detr-s-v5-full`，日志为`output/tree-detr-s-v5-logs/console.log`。
从同一DINO骨干预训练重新开始正式全量Task1，随后完整Task2至Task4增量和平衡微调。
两张3090、每卡batch2、seed42，每个increment 50轮，Task2至Task4各finetune 20轮，
阶段内不降学习率；训练配方命名不代表复现OW-DETR模型结果。
中断后用`bash tools/owod/run_discovery_full.sh --resume`恢复同一v5链。
不能把旧v4 checkpoint复制到v5目录，不允许绕过manifest/阶段/版本校验。

首轮检查监督覆盖、有限梯度；pseudo从warmup后开始，首轮U-Recall为0不作为失败依据。
warmup后关注unknown是否继续几乎占满输出、已知类指标是否仍大幅跳变；同时检查未知召回。
这些是训练异常检查，不使用正式测试集挑选最好epoch/阈值，最终结果按固定末轮报告。

## 重复评分诊断

扩展的`tools/owod/diagnose_discovery_task1.py`在一次前向里同时计算联合输出、仅已知输出，
并报告匹配/未匹配且重叠已知GT/未匹配不重叠组的objectness、novelty等均值及10/50/90分位数。
先确认GPU空闲；示例使用PCI顺序的3090卡2：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python tools/owod/diagnose_discovery_task1.py \
  --checkpoint output/tree-detr-s-v4-full/s-owodb/seed_42/stage_0/increment/checkpoint0004.pth \
  --manifest data/owod/s-owodb/split_manifest.json --images 32
```

只读历史模型推理可使用新诊断工具，不代表允许跨版本训练。诊断不输出权重、不更新模型，
检查checkpoint前后哈希、manifest和annotation哈希；优先使用不再被训练覆盖的编号checkpoint。
