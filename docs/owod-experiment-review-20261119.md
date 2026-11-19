# Tree-DETR 实验复盘与下一步取舍

截至 2026-11-19，M 的全量 Task 1 已经学到已知检测和部分未知发现。真正卡住研究主张的是 Task 2：新类拿到真实标注后，常规增量路径学得弱，而历史实例监督覆盖窄，尚未在这个基础上产生可辨认的新类收益。优先改善有标注的新类学习，再决定匿名实例迁移是否值得作为主要贡献。

## 已经回答的问题

AP、召回和精度均为百分数。M 用作者 VOC 积分 AP50，S 用作者 VOC2007 11 点 AP50。短程实验的训练集和日程在表中单列；同一测试清单不能抵消训练预算差异。

| 实验 | 固定结果 | 对下一步的含义 |
| --- | --- | --- |
| 早期 S，图消息、原型与快照继承 | v3 短测中，图模型 99.56% 的原型支持集中在一个槽；19/21 个新类从同一槽继承，继承和 scratch 微调末轮 Current AP50 都为 0。较晚的 S v5 父模型在固定 512 张训练图事后核对中，对 629 个未来 Task 2 物体，原始 query 覆盖 127 个，pseudo 覆盖 5 个，匹配成熟槽的 pseudo 覆盖 0 个。 | 匹配数量和跨视图一致性都无法证明匿名槽有类别含义。旧方案在不同运行中分别出现了大量同槽继承和完全无继承。 |
| M，早期 v5–v8 与纯监督 control | 原学习率下，纯监督 control 第 2 轮也出现像素响应退化，第 3 轮 Known AP50 为 0.23。学习率减到四分之一的 control 第 3 轮为 29.26，第 8 轮为 41.83。 | 早期接近零的 AP 混有基础检测优化失败。以降学习率后的检测器作为参照。 |
| M，低学习率 Task 1 评分与发现梯度 | 双 3090 完整训练清单第 8 轮，ranking 为 36.00，combined 为 34.02，focal 为 42.14 Known AP50。同一 ranking 权重在 256 张训练图上仅改推理评分，Known AP50 从 49.32 到 60.73。独立单 A40 focal 第 50 轮为 58.50 Known AP50、12.56 U-Recall、1.56 unknown precision，共输出 188,261 个 unknown。 | 在 M 上保留低学习率和 focal。Task 1 已有发现信号，误检仍多。双卡短程与单卡末轮分别解释。 |
| 较长训练的 Task 2 | M v8 ranking 的 Task 1 末轮 Known AP50 为 56.10，Task 2 完整增量清单第 12 轮 Previous / Current / Known 为 37.77 / 8.43 / 23.10。S v5 的 Task 1 末轮为 64.17，Task 2 增量第 8 轮为 14.00 / 3.64 / 8.56。另一组同父 S 增量第 8 轮，hard → soft 使 Current 从 2.60 到 3.97，Previous 从 63.38 到 63.27。 | 新类学习困难跨多种设置出现；父模型、训练量、适配策略会明显改变结果。S 支持保留 soft 作对照，两套协议均缺少这里所需的完整末轮比较。 |
| M v10 实例迁移 | 共同父模型 Task 1 用 6,144 图训练 8 轮，Known AP50 为 22.93。Task 2 用相同 1,536 图增量 3 轮，再用完整作者均衡记忆 1,743 图微调 1 轮。微调后 matched / off / shuffled 的 Current 为 **1.503 / 1.518 / 1.454**，Previous 为 23.92 / 23.84 / 23.77，Known 为 12.71 / 12.68 / 12.61。三组 U-Recall 均为 0.054。 | matched 高于 shuffled、略低于 off。增量末轮也没有同时超过两个对照。单 seed 不足以把百分之几分之一个点的差值解释为稳定收益或伤害。 |

历史依据见[原型与快照继承](owod-discovery-transition-design.md)、[早期 M 输入退化](owod-discovery-v8-analysis-20261110.md)、[学习率对照](owod-control-lr-quarter-epoch8-analysis-20261111.md)、[focal 同轮对照](owod-task1-focal-followup-20261113.md)和[实例迁移设计](owod-instance-transfer-design.md)。单 A40 末轮来自本机下载的原始 `train.log2`。较长训练的阶段点来自双 3090 服务器对应方法、协议和阶段的 `metrics.jsonl`。

## 为什么 v10 检索成功，Current AP50 仍没有提高

1. **上游候选只覆盖了部分未来物体。** M v10 的固定留出切片有 229 个后来成为新类的 GT，选中的 encoder proposals 覆盖 59 个，全部 decoder queries 覆盖 92 个，359 个 pseudo 合计覆盖 27 个。三项是同一批 GT 的覆盖计数；decoder 同时使用 learned queries。这里观察的是阶段末候选，不能据此还原整段训练历史。
2. **进入训练的新类证据分布很偏。** 在 4,096 张父阶段图像上提取的库有 32 个占用槽，保留来自 242 张图的 249 个代表。Task 2 的 3,988 个新类 GT 中，1,390 个通过定位筛选，304 个被接受，覆盖 10/20 类。其中类别 ID 8 和 22 合占 257/304，即 84.5%。打乱组 304 个目标全部跨槽置换。按实现重算，各实例原目标与置换目标的余弦相似度中位数为 0.019，说明对照确实改变了目标向量；没有据此验证历史代表的真实类别。
3. **训练信号只在特定实例出现时生效。** 代码按图像、物体及已释放类别 ID 查询冻结目标，在 GT 匹配 query 上做余弦对齐。增量时 matched 平均每卡每批对齐 0.379 个实例；均衡清单的 3,249 个新类 GT 中仅有 10 个命中，实际平均每卡每批 0.0115 个。目标相似度描述对齐方向，本身不要求正确类别胜过错类和重复框。即使部分匹配正确，对大多数类及微调阶段的监督仍很弱。[目标构建和对齐实现](../models/graph_local/instance_transfer.py)
4. **关闭历史对齐后，新类检测仍然弱。** M off 仍保留 soft 路由、风险门控、发现分支和旧类蒸馏。每个新类的 rank-8 adapter 接在最后两个 decoder 层，输入 query 的路由概率和旧类风险证据调节其残差。独立测试图的几何诊断中，新类 GT 原始框覆盖率为 37.68%（1,871/4,966），旧类为 62.20%。有合格框的新类 GT 中，正确类候选最优排名的中位数是第 78。前者说明不少物体得不到合格框，后者说明有框时也有类别排序损失。不能从这些数字单独确定冻结、路由、样本量各自承担多少。[路由与更新范围](../models/graph_local/routed_adapter.py)

## 11 月 19 日的配对结果排除了哪些捷径

下表使用 M v10 off 微调末轮作为参照。Top-300 与移除 objectness 的结果由相同权重、相同前向输出复评。几何覆盖按独立图片和有效目标计算，作者指标保留原作者清单及匹配规则，两者分母不同。

| 变化 | 直接观察 | 判断 |
| --- | --- | --- |
| 微调延续 pseudo 日程 | unknown TP 从 9 到 80，U-Recall 从 0.054 到 0.477，precision 从 0.412 到 0.602，预测数从 2,184 到 13,287。WI 从 0.03764 到 0.03632，A-OSE 从 20,146 到 19,069。Current 从 1.518 到 1.514。 | 日程重置确实影响这个单轮微调的未知发现表现；后续检验新类学习时固定同一日程。 |
| 只开放 bbox 最后一层 | 微调末轮 Current 从 1.518 到 1.446，Previous 从 23.84 到 22.90。 | 开放坐标输出层没有解决新类学习。更早的 S 小集诊断显示开放完整 decoder 和 bbox 能提高部分训练集指标、也损伤旧类留出表现；单纯放宽冻结同样有代价。 |
| 输出预算 50 → 300 | Current 从 1.518 到 1.550，新类 TP 从 806 到 1,386，FP 从 323,865 到 2,150,517。A-OSE 从 20,146 到 105,815。 | 扩容能捞回真阳性，但伴随大量错报，AP 只增加 0.032 个百分点。保持联合 Top-50 进行后续配对。 |
| Top-50 下去掉已知分数中的 objectness 乘子 | Current 从 1.518 到 1.256，正确类候选最优排名的中位数从 78 滑到 93。 | 当前权重上，objectness 对新类排序有正作用，简单移除会恶化结果。 |

本次从双 3090 服务器只读核对，六份原始小型报告在本地忽略目录 [output/analysis/owod-review-20261119](../output/analysis/owod-review-20261119)。在新环境可查看服务器 `output/m-instance-{expanded,schedule-pair,localization-pair,ranking-probe,objectness-probe}/comparison.json` 及 `output/sowodb-task2-v7-routing-8epochs-2gpu/results/comparison.json`。

## 下一笔 GPU 预算

**优先做一个能改变决策的 M Task 2 对照。** 复用同一 v10 诊断父模型和已完成的 off 对照，新开独立实验，只额外开放最后一个 decoder block 的 cross-attention 参数，使用现有联合损失更新。保留 bbox 头冻结，其他可训练参数、分类行、soft 路由、GNN、蒸馏、发现日程、数据、batch、学习率和 Top-50 均与 off 相同。该选择检验新类是否需要更新从图像取特征的位置和内容。现有 bbox-last 对照只改坐标输出，未回答这个问题。新增参数用同配置对应的学习率组；记录可训练参数、显存和耗时。该诊断组已接入下方启动器，服务器 AP 待测。

仍在 3 轮增量和 1 轮均衡微调两个固定点计分。基于已有 off 训练日志，双 3090 的增量及微调作业内分别记录约 13 和 9 分钟。新组含初始化、完整评估及覆盖审计建议预留 **0.5–1.5 小时**，首轮实测后修正。单独建立运行计划，保留现有权重；新的训练范围会改变优化轨迹，诊断链仅用于同父比较。

- 若新类原始框覆盖和 Current AP 都上升，查看新类逐类 AP、正确类排名，以及 Previous、Known、U-Recall、precision、WI、A-OSE 的代价。可预先约定一个工程筛选门槛，例如微调末轮 Current 至少提高 1 个百分点、Previous 至多下降 1 个点，再换 seed 复核。门槛是预算决定，不是统计显著性的判据。
- 若只改善训练图或框覆盖，留出新类的正确类别排名和 AP 没有改善，停止扩大这组。按已释放类别核对正例被哪些旧类、新类或重复框压过，再决定是否修改有监督的分类目标。若新旧两端一起明显变差，保留冻结参照。
- 若两组仍在低 AP 区间，先做同父、同数据预算的常规有监督增量检测参照，量出弱父模型与短日程下本来能学到多少。若这一参照也学不动，先改善父模型或有效训练量，再评估新机制。更换训练量要重建同预算对照。

**迁移线保留一个明确的研究问题。** 用已释放类别模拟匿名检索，在与检索库图像分离的诊断集上报告每类的同类检索精度与覆盖；正式历史库从不接触未来标签。若历史目标有区分度，再回到 matched / shuffled / off，报告命中类与未命中类的增量及微调末轮 AP，检查真实对应能否同时胜过两种对照。若监督仍集中在少数类、正确对应也没有可重复的收益，就把匿名槽降为辅助记忆，把方法重点放在可测的未知定位与有标注的新类适配。四任务七阶段训练留到短程因果对照出现明确信号后。

外部参照可用 OW-DETR 原论文 Table 1，M Task 1 Known 为 59.2，Task 2 完整日程的 Previous / Current / Known 为 53.6 / 33.5 / 42.9。项目内的 v10 子集短程与之预算不同，应另表展示。现有材料未验证新方案的完整四任务或多 seed 优势。[OW-DETR 原论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.pdf)

## 双 3090 上的三模块配对实验

本轮共用已完成的 v10 Task 1 父模型、GNN、检索缓存、seed 42 和训练清单。三个增量组各训练 3 轮。每组各自的第 3 轮 checkpoint 分出两个 1 轮的作者均衡微调，共 9 个训练阶段。各组提前固定，跑完全部对照后解释结果。根据已有 off 组耗时，双 3090 空闲时整套先预留 2–4 小时，以首组实测修正。

| 研究问题 | 配对方式 | 看哪些结果 |
| --- | --- | --- |
| 新类更新读取的图像特征能否改善，同时保住旧类 | 新的 cross-attention off 对照历史 frozen off，比较相同的增量或重置日程微调末轮 | 新类原始框覆盖、正确类候选排名、Current / Previous / Known AP50。初筛以重置日程微调为主，Current 至少增加 1 个百分点且 Previous 至多下降 1 个点；若通过再换 seed 复核，同时检查未知指标。 |
| 未知实例被命名之后，它的历史表示有无用处 | 在相同更新范围、相同阶段和日程下比较 matched、off、shuffled | 看是否真的有对齐和跨槽打乱，再看 matched 相对两个对照的 Current AP 及旧类代价。按训练时有无检索命中分组报告各类的作者 AP50。 |
| 新类适配、旧类保持期间能否持续发现其余未知物体 | 每个迁移组从相同的增量末轮权重分别重置或延续伪未知监督日程 | 比较 U-Recall、unknown precision、TP、预测数、WI、A-OSE，以及 Previous、Current 的变化。Task 1 的伪框覆盖来自原父模型，作为共同起点。 |

新增的 230,272 个可训练参数属于最后一层 decoder 的 sampling offsets、attention weights、value projection 和 output projection。AdamW 沿用原参数分组，sampling offsets 用主学习率的 0.1 倍。增量主学习率为 5e-5，微调为 5e-6。该改变会影响优化轨迹。各阶段在独立目录训练，同一计划中断后可从对应 checkpoint 恢复。

Git 同步到服务器后，在 Linux / Bash 的仓库根目录执行。需要原 v10 expanded pilot、两个历史 off 末轮及其数据文件，以及与该 pilot 一致的双 RTX 3090 和 tree-detr 环境。先确认卡 0、1 可用。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
bash tools/owod/run_instance_pilot.sh --cross-attention-study --plan
tmux new-session -d -s m-cross-attn 'CUDA_VISIBLE_DEVICES=0,1 bash tools/owod/run_instance_pilot.sh --cross-attention-study'
tail -F output/m-instance-cross-attention/console.log
```

`progress.json` 在各阶段评估后更新，完成后生成 `comparison.json`。各组的 `train.log`、`metrics.jsonl`、`coverage.json` 和 checkpoint 保存在 `output/m-instance-cross-attention/<arm>/<phase>/`。逐类 AP 用完整作者测试清单的 VOC 积分 AP50。框覆盖、最佳 IoU 处的分类和正确类别候选的中位排名来自独立的固定 128 张留出图。恢复时重跑同一启动命令。

## 更充分的 Task 1 父模型与三种 Task 2 适配范围

前一轮开放末层 cross-attention 的 Current AP50 提高了 0.671 个百分点，Previous 下降 0.273 个百分点，留出图上的原始框覆盖从 96/229 降到 93/229。真实历史匹配在均衡微调后仍略低于关闭迁移组。下一轮把时间用在区分父模型训练不足和新类适配容量不足。

从同一份 DINO 权重训练一个 v10 Task 1 父模型。诊断训练从 6,144 图 × 8 轮扩大到 12,288 图 × 12 轮，约为此前三倍的图像训练次数。Task 2 使用同一批 1,536 张增量图训练 3 轮，再用完整作者均衡记忆训练 1 轮。固定同一 128 图留出集、seed 42、双 3090、每卡 batch 2、focal Top-50 和学习率。该计划创建独立诊断链。

| 组 | Task 2 更新范围 |
| --- | --- |
| `frozen` | 现有分类行、适配器与发现分支，基础 decoder 和 bbox 头冻结 |
| `cross-attn-last` | 在 frozen 基础上，开放最后一层 decoder cross-attention |
| `decoder-bbox` | 在 frozen 基础上，开放基础 decoder 和完整 bbox MLP，作为有标注条件下的密集适配参照 |

三组共用新父模型校准出的 GNN。增量阶段共用新父模型作为蒸馏教师，微调阶段各自使用本组的增量末轮权重。三组均关闭历史对齐、继续微调阶段的伪未知监督，并沿用相同分类行掩码和联合损失。额外开放的参数沿用 AdamW 分组，sampling offsets 仍按 0.1 倍学习率更新。密集适配参照包含与其余组相同的发现和保持机制，比较的变量是检测器可训练范围。

同步 Git 后，在服务器 Linux/Bash 仓库根目录启动。确认 GPU 0、1 空闲。根据旧父模型每轮约 10 分钟的训练日志，整套先预留 **5–6 小时**，以新任务前两轮日志修正。启动会先核验 M 数据和双卡 CUDA 计算路径。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
git pull --ff-only origin mowodb &&
tmux new-session -d -s m-stronger-parent 'CUDA_VISIBLE_DEVICES=0,1 bash tools/owod/run_instance_pilot.sh --stronger-parent-study'
tail -F output/m-instance-stronger-parent/console.log
```

`output/m-instance-stronger-parent/progress.json` 记录新父模型和已完成的阶段。全部作业完成后查看同目录的 `comparison.json`。同一提交和输入下，中断后重跑启动命令可从各阶段已保存的 checkpoint 恢复。

先核对新父模型的 Known AP50 和 pseudo 覆盖，再读三组均衡微调末轮。对照标准沿用 Current AP50 至少提高 1 个百分点、Previous AP50 下降不超过 1 个点，并检查 U-Recall、precision、A-OSE 和 WI。若密集适配提升，同时出现旧类或未知检测退化，下一轮针对这一代价重做适配约束。若各组仍然接近旧实验的低 Current AP50，用原始框覆盖和正确类排名决定优先查定位还是训练日程。正式结论需要完整协议与重复种子。
