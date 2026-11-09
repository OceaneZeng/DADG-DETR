# OWOD 未知发现与类别转化：已实现的三模块

更新：2026-10-10。已完成 S 的短程真实图像公平对照和 Task 1→2 继承对照，
均评估完整作者测试集。启动机制已修复，但没有证实 AP/H-Score 增益；
完整四任务、作者训练日程的正式 M/S 结果尚未测得。

2026-11-09 的 M Task 1 退化与实验性 v8 扩展见
[评分约束与发现梯度对照](owod-discovery-v8-debug.md)。默认 v5–v7 行为保留；
新版本的额外训练目标、梯度边界及恢复限制以该版本说明为准。

## 三模块与实际数据流

| 模块 | 实际实现 | 监督与作用 |
| --- | --- | --- |
| 关系图与未知发现 | 物体性/novelty 头、已知/匿名原型、query→原型 affinity 消息聚合 | 已释放 GT、双视图候选及一致性，学习物体与背景的区别 |
| 发现 adapter 与类别转化 | 共享 discovery adapter、三态 router、匿名原型快照、新类 adapter bank | 未命名时获得发现梯度；释放标注后关联历史匿名表示并迁移有效残差映射 |
| 联合学习与旧知识保持 | 检测/发现/路由/一致性损失、冻结旧分支、GNN risk、旧类教师蒸馏、官方均衡微调 | 学习当前新类并约束旧类损失，不增加额外回放图片 |

Affinity 与 interference 是两种不同语义。新增 affinity 图学习表示关联；
原有 interference GNN 仍以合法训练样本上的旧类损失增加为监督。
没有把 harm score 改名成可迁移性。

## 模块一：候选、未知分数与匿名状态

代码：`models/graph_local/discovery_candidates.py`、`discovery_memory.py`、
`discovery.py`。v2 未知推理分数为 `objectness × novelty`；已知类别分数为
`objectness × (1 − novelty) × softmax(known_logits)`。两者分配同一物体概率质量，
不再用未校准的 focal logit 最大值重复否决 unknown。
Task 4 关闭 unknown 类别输出。

候选同时满足合法框、排除匹配/重叠 GT、双视图高区域响应、
翻转视图框/特征一致性；去除重复框并最多选 5 个。flip 仅翻转有效图片区域，
不改变 padding。v2 用本批已释放 GT 匹配分数的 median/MAD 校准相对已知置信度，
与已知原型距离共同生成带下限的软权重，用于排序和监督强度，不作为硬拒绝门槛。
所有筛选证据 detached，不读取未释放的标签；软权重下限不保证候选正确。

前 1 epoch warmup 不生成 unknown 正样本；低响应区可以提供背景监督。
背景需要双视图低响应，不再要求低已知置信度，允许对高置信错误背景施加负监督。
模糊 query 的已知类负梯度被忽略。匹配 GT 始终保留正监督，包括辅助层。

v3 对 EMA 启动去除随机初始化偏差：第 n 次更新衰减为
`min(0.999, (n−1)/n)`，首次复制优化后的学生参数，随后渐进到固定 EMA。
更新计数随 EMA checkpoint 保存；不能将旧版本 checkpoint 静默续入新规则。
此项是项目方法修改，不是声称 ORE/OW-DETR 给出的超参数。

内存记录已知原型和最多 16 个匿名原型的中心、加权支持数与方差。
匿名 ID 无语义名称，内存满时拒绝低相似度强制合并。只在训练优化步骤后
更新状态；DDP 汇总证据、广播状态。评估不更新内存，criterion 过滤未来 GT。

## 模块二：未知→已知的具体过程

Discovery adapter 从 Task 1 就收到物体性、novelty 和一致性梯度。
其 router 区分背景、已知物体和匿名物体。当前实现是**共享可训练 discovery
adapter + 每匿名原型冻结快照**，并非每个匿名类别有独立训练分支。

新任务开始，使用上一任务表示和本任务已经释放的 GT，抽取新类特征和
pre-FFN router 输入中心。与达到支持阈值的匿名原型通过学习到的 affinity
比较；低匹配置信度时正常从零学习。默认最小支持 3、继承阈值 0.8。

高置信匹配把历史快照的有效 `BA` 映射加权后做截断 SVD，初始化新增类别
adapter；不分别平均 A/B。正式类别含义仍由本任务 GT 的检测损失学习。
新类 router 从当前 GT 的 pre-FFN 中心初始化，避免冷启动全选 null。
迁移组和 scratch 组使用相同 router 初始化，只比较残差继承。

混合匿名原型继续保留，不无条件把整个 cluster 归类或删除。当前尚未实现
混合原型拆分、随时间淘汰/遗忘统计，或者独立匿名分支的稳定性控制；这些
属于后续消融，不能在图中画成已经完成的功能。

## 模块三：联合目标与检查点

目标包含 known detection、objectness、pseudo unknown、discovery routing、
跨视图特征/框一致性、已知 anchor affinity，以及旧类保持。
这些候选是代理监督，伪框不是真实 unknown GT。

增量阶段只更新当前类别 bank/分类行和 discovery 路径，旧类别 adapter
冻结；均衡微调开放已知 bank。旧类教师来自实际完成的前一阶段 checkpoint。
框头在适配阶段冻结，但 adapter 对 query 表示的更新仍能影响框预测；是否
开放定位路径需要单独验证。新方法默认开启旧类蒸馏，可独立消融。

匿名内存、快照、adapter bank、转化记录及 EMA 都进入 checkpoint。
严格 resume 绑定 manifest、方法版本、图消息开关、方法配置和阶段链，缺失 EMA 状态时拒绝恢复。
该方法属于项目研究实现，不是 ORE/OW-DETR 方法数值复现。

## 不变的数据与评估边界

M 采用 ORE 20/20/20/20；S 采用 OW-DETR 19/21/20/20，均使用固定作者
ImageSets、四任务、Task 2–4 单独均衡微调，总计七个训练作业。
不读取测试 unknown GT 来训练、选候选或调阈值；不额外增加旧图片预算。
匿名原型不要求同一实例在相邻任务重复出现。

报告作者 AP50 Previous/Current/Known，以及 U-Recall、WI@0.8、A-OSE。
比较最后训练 epoch，不按作者测试成绩挑选最好 epoch。Task 4 省略 unknown
指标。候选数量、ignore 比例、转化覆盖率属于项目诊断。

## 复现实验与当前证据

`run_benchmark.py --method` 支持 `control`、`graph-adapter`、`discovery`、
`discovery-no-message`、`discovery-scratch`。scratch 关闭匿名快照继承，其余设置相同。
no-message 用 query 自环替换跨原型消息，保留相同投影、update MLP、物体性/novelty
头、affinity 监督和参数量。它消融消息聚合，不声称移除所有图相关计算。
三组 discovery launcher 在后续任务均启用同样的旧类教师蒸馏。
`validate_discovery.py compare` 检查真实保存的 checkpoint 参数、数据/权重
哈希、阶段完成情况和最终作者指标；缺数据或不成对的运行无法生成 AP 结论。

完整入口：

```bash
python tools/owod/run_discovery_study.py --benchmark both --proxy http://127.0.0.1:10808
```

自动下载→按作者清单准备/核验→依次运行 discovery、discovery-no-message、scratch
完整任务链→生成两个配对报告。单 GPU、batch 2、seed 42 是本机对照设置；
50 epoch 增量、20 epoch 微调、DINO 与学习率采用 OW-DETR recipe。
硬件和 batch 设置不等于论文多 GPU 实现，不宣称复现其模型结果。
状态见 `output/discovery/full-study-v3/study_status.json`，同版本中断后同命令续跑。
旧版本研究目录与 checkpoint 保留，不能续入 v3。

M 的作者 Drive 入口当前为空，社区备份链接失效；脚本明确记录来源不可用，
继续 S。没有用单独 COCO 数据或随机划分替代 M。

80 项相关测试及真实 CUDA 前向/反向、增量 router 和状态恢复通过。
下载的 DINO 权重已由实际 backbone loader 验证。完整架构在 800×1333、
batch 2 人工输入上完成两次 AdamW 更新和 EMA/内存更新，显存峰值约
7.52 GiB；这验证计算路径与本机资源可执行，不是实际数据训练或 AP 证据。
受控机制测试 100 步：loss 4.8711→0.3976，候选物体性 0.4025→0.9990，
背景物体性 0.4331→0.000138，novelty 0.7313→0.9996；完成一次原型继承。
30 步新类学习后迁移/scratch loss 为 2.6549/2.6651。这些是机制数据，**完整
四任务 AP 尚未测得，不据此声称新类 AP 或真实未知发现提高**。

### 本机短程真实指标检查

用户于 2026-10-09 明确要求先在本机简单验证，再迁移服务器做完整实验。
第一轮历史检查 `tools/owod/local_discovery_check.py` 使用已核验的 S manifest 和全部作者
测试图片，默认两组各 300 次更新、batch 2、seed 42，保留官方图像变换、
完整模型及 DINO 预训练。两组检测器初始权重完全相同，消耗同一训练排列
前缀。没有生成替代划分；没有用测试未来标签选择训练图片、阈值或最好 checkpoint。
三个 100 步窗口用于诊断 warmup，不能当作三个完成的官方 epoch。

```bash
python tools/owod/local_discovery_check.py --output output/discovery/local-check-300
```

结果写入 `comparison.json`，状态写入 `status.json`。AP50、U-Recall、H-Score
保存为 0–1；乘 100 为百分点。H-Score 是已知 AP50 与 U-Recall 的调和平均。
该检查只测 Task 1 发现能力，既不证明未知→已知快照继承有效，也不证明
已知类保持有效；控制与 discovery 还存在容量及输出评分差异。
所有状态明确标记 `official_training_result=false`，诊断权重不保存正式
训练 lineage 参数，不能初始化正式任务链。短程增益不能替代四任务正式结果。

真实 portrait/landscape 三批次、实际 loader/AdamW/EMA 检查已通过。
encoder activation checkpointing 将该检查显存峰值降至约 4.71 GiB，
同样设置用于本机对照；它不改变模型参数或官方图片清单。

2026-10-09 的真实运行已完成两组 300 步及全部 4,952 张测试图片。
以下 AP/Recall/H-Score 数值已乘 100：

| 指标 | control | discovery |
| --- | --- | --- |
| Known/Current AP50 | 0 | 0.02175 |
| U-Recall | 3.59568 | 0 |
| H-Score | 0 | 0 |
| A-OSE（计数） | 0 | 1379 |
| WI（原始比值） | 0 | 0.00557442 |

**未证实有效 OWOD 提升。** control 的 247,600 个预测全部被标成 unknown，
使其 AP/WI/A-OSE 退化；discovery 没有输出 unknown。这轮数据不能单独证明
GNN 有效，也不能否定充分训练后的方法。不是收敛实验，不能作为论文结果。

训练三个窗口伪标签数量均为 0，16 个匿名原型槽都没有支持量。
`inspect_discovery_startup.py` 在本轮前六张合法训练图片上，用确定性原始
val resize 和训练后自身的翻转预测检查启动门控；没有使用测试集或未来 GT。
每张有 80–99 个有效、未被已知 GT 排除的 query，但已知概率门控后全部为 0。
该 probe 的自身教师不是实际训练 EMA，因此只能定位训练后门控现象，不能
重构每个训练 step 的确切 EMA 状态。

```bash
python tools/owod/inspect_discovery_startup.py --directory output/discovery/local-check-300
```

### v3 公平消融与 Task 1→2 短测

当前入口：`tools/owod/local_transition_check.py`。默认 Task 1 两组各 300 步，
Task 2 两组各 150 步增量 + 60 步独立均衡微调；每组用全部 4,952 张 S 测试图。
Task 1 图消息/自环模型加载同一完整初始 state（包含新增头），校验 state SHA256。
Task 2 共用本次图消息 Task 1 父状态和相同的新类标注、顺序、增强及旧类教师。
只用每个新类最多 4 张合法训练图片计算转化特征，不根据测试成绩选预算。
两组均关闭 interference risk（beta=0），记录为 disabled，绝不充当学得的 GNN 分数。
因此这个短测只检查匿名继承，不验证完整风险控制器。FT 使用作者 t2_ft 图片，
单独 loader/optimizer；没有自定义旧类回放。

```bash
python tools/owod/local_transition_check.py --output output/discovery/local-transition-v3
```

仅允许本次同 manifest、同方法版本、Task 1 increment graph 诊断状态进入 Task 2。
这些权重不含正式 lineage，正式训练入口仍拒绝加载。全测试集按宽高比排序降低
padding；图片、变换、每图输出数和评估定义保持一致。`comparison.json` 保存
graph_vs_self_loop 与 transfer_vs_scratch；候选/原型支持量和继承记录单独保存。
旧 v1 历史结果保留，不初始化此轮或正式实验。

v2 启动诊断（`output/discovery/local-transition-v2`）完成图模型 300 步后，
候选与匿名原型均为 0，因此在全测试完成前停止，不生成 AP 对照结论。
同一模型在六张合法训练图上的自身翻转教师却产生 10 个候选，提示实际 EMA
启动滞后。固定 0.999 的 EMA 在 300 步后仍保留约 74% 随机初始化权重。
v3 因此修正 EMA 启动；未依据测试指标调整阈值。

### 2026-10-10 完成的 v3 实测结果

结果目录：`output/discovery/local-transition-v3/`。四个模型各自覆盖全部
4,952 张 S 测试图片。下表 AP、U-Recall、H-Score 均为百分制；WI 是原始比值，
A-OSE 是计数。Task 2 指标取完成独立均衡微调后的状态。

| 任务/方法 | Previous AP50 | Current AP50 | Known AP50 | U-Recall | H-Score | WI@0.8 | A-OSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 自环 | — | 0.043497 | 0.043497 | 7.828376 | 0.086514 | 0.009648 | 293 |
| T1 图消息 | — | 0.007782 | 0.007782 | 9.144512 | 0.015550 | 0.006452 | 109 |
| T2 scratch | 0.059345 | 0 | 0.028189 | 7.815140 | 0.056175 | 0.005089 | 39 |
| T2 继承 | 0.056767 | 0 | 0.026965 | 7.806691 | 0.053743 | 0.005073 | 39 |

图消息提高 U-Recall 约 1.316 个百分点，但 Known AP50 和 H-Score 下降。
WI/A-OSE 下降不能单独解释为成功：已知预测减少也可能降低这些错误计数。
T2 继承没有提高新类 AP，Known AP50 和 H-Score 也略低于 scratch。
所有 AP 都处于接近零的退化区间，不能宣称充分训练后一定无效或有效。

机制证据与局限：

- T1 图消息累计 950 个伪候选，自环 514 个，修复了候选始终为零的问题。
- 图模型 3 个匿名槽有支持量，但 99.56% 加权支持集中在槽 0，只有该槽达到
  支持阈值 3。原型区分不足，不能将槽数视为发现类别数。
- T2 用全部 21 个已释放新类的训练特征关联；19 类发生继承，全部来自槽 0。
  90.48% 的继承覆盖率不是正确语义转化率。
- 增量结束后，两组新类分类行实际变化，旧分类行和 bbox head 参数变化均为 0。
  两组仅有相同的 12/21 类 adapter 出现可测的有效残差更新。继承后的非零
  残差不能算作训练更新；审计比较的是相对于继承初始映射的变化。
- T2 每组 247,600 个保留预测中仅 152 个标为新类，均未贡献新类 AP；约
  96.6% 标为 unknown。未知预测精度约 0.386%，存在严重评分/定位质量问题。
- 本次 T2 风险抑制实际 beta=0，见 checkpoint 的 spec 和参数审计；初版运行
  `task2_args.json` 中沿用的 CLI 默认 beta=2 未被使用，后续脚本已将其同步为 0。

证据文件：`comparison.json`、各组 `*_metrics.json`、`graph_memory_audit.json`、
`transfer_promotion.json`、`transition_audit.json`、`task2_prediction_audit.json`。
31 项相关测试通过；参数审计另有回归测试，防止把继承值误记为训练更新。

**当前结论：计算路径和启动可行，检测收益尚不成立。** 下一轮优先做以下
修改/验证，不根据本次测试集结果直接挑阈值：

1. 在服务器先取得非退化的已知检测基础，再做相同训练预算的消融。当前只训练
   300 步的检测头不足以评估高质量迁移，不能用极小 AP 差值证明创新。
2. 增强区域特征和原型区分：比较 DINO 区域特征与 decoder query 特征，加入
   合法训练监督下的表示约束；对多个新类匹配同一混合原型的情形拒绝或降低继承。
3. 单独排查 unknown 分数挤占 top-50 与框定位不足，使用训练数据做评分校准；
   在相同保持约束下消融受限 bbox/decoder 解冻，避免只更新分类行和小 adapter。
4. 检查新类 router 的覆盖和空路由，避免 9/21 类 adapter 在短程增量中没有
   实际更新。上述因素应逐项消融，不能同时改动后把收益统称为 GNN 效果。

## 评分与路由优化实验（2026-10-10）

新增显式方法开关，默认仍保留原方法，避免未验证改动静默进入正式任务链：

- `--discovery-scoring focal`：已知分数为 `objectness × sigmoid(class_logit)`，
  unknown 仍为 `objectness × sigmoid(novelty_logit)`。不再将 focal 分类头改作
  softmax 条件分布，也不二次使用 novelty 抑制已知分类。这些是独立分数，
  不要求所有类别概率之和等于 1；不声称已经完成概率校准。
- `--graph-routing soft`：保留相同 adapter、router 和 risk 结构，使用所有
  非空类别路由概率加权残差，不进行硬 top-1 截断。训练和推理一致，不使用
  GT 决定推理分支；当前没有宣称实现软到硬的退火。

两个开关绑定正式 checkpoint lineage，禁止续训时静默切换。作者四任务、
ImageSets、类别顺序和评估定义保持不变。这些改动是项目方法，不是论文协议。

```bash
python tools/owod/local_transition_check.py --study optimization --steps 900 --task2-steps 300 --finetune-steps 120 --output output/discovery/local-optimization-v3
```

实验预先固定预算与比较：Task 1 从 DINO 初始化训练 900 步，对同一模型
同时评估 hierarchical/focal 评分；Task 2 两组从本次相同父状态出发，各做
300 步增量和独立 120 步作者 balanced FT，比较 hard/soft 路由。两组均关闭
快照继承与风险抑制，以隔离路由作用，使用相同旧类教师。每个最终模型均做
完整 4,952 张 S 测试，保存两种评分结果；known-only 仅用于诊断评分挤占，
不能作为 OWOD 指标或参与挑选最优方法。额外统计所有 query 对已知 GT 的
定位覆盖率；该归一化 IoU 诊断不是作者 AP 的替代。

不能把本轮 900 步结果与上一轮 300 步结果直接比较后归因于新方法。
本轮有效对照是相同权重的评分比较，以及相同父模型/预算的路由比较。

### 优化实验实测结果

已完成 `output/discovery/local-optimization-v3`，每个模型覆盖完整 4,952 张
作者测试图。下表为百分制；所有 Task 2 结果均为独立 balanced FT 后。

| 任务/路由/评分 | Known AP50 | Current AP50 | U-Recall | H-Score |
| --- | ---: | ---: | ---: | ---: |
| T1 / 原 hierarchical | 0.512545 | 0.512545 | 5.612003 | 0.939303 |
| T1 / 独立 focal | 0.146092 | 0.146092 | 4.438010 | 0.282873 |
| T2 / hard / hierarchical | 0.056390 | 0 | 5.069280 | 0.111539 |
| T2 / soft / hierarchical | 0.060342 | 0.008955 | 5.348091 | 0.119337 |
| T2 / hard / focal | 0.057464 | 0 | 3.844204 | 0.113235 |
| T2 / soft / focal | 0.087377 | 0.005772 | 3.945590 | 0.170968 |

**评分假设未获支持。** T1 同权重下，独立 focal 评分降低三个目标指标；
T2 soft+focal 对比原 hard+hierarchical 虽提高 Known AP/H-Score，但 U-Recall
下降约 1.124 个百分点，不能宣称组合方案三个指标同时改善。保留原评分作为
下一轮主要参照，不将 focal 改为默认。known-only 诊断基于 focal 评分，
T1 Known AP50 为 0.115005；它不是原 hierarchical 评分的已知类上限。

**软路由出现小幅正向信号。** 在两种固定评分下，soft 对比 hard 均提高三个
目标指标。原 hierarchical 下增量分别约为 +0.003952、+0.278810、+0.007798
个百分点；独立 focal 下分别为 +0.029913、+0.101386、+0.057733 个百分点。
原评分的配对差值在运行完成后从预先保存的两种评分结果补算，标为补充分析；
最初设定的 focal 主对照与组合对照原样保留，未重新训练或调整测试阈值。

其他指标并非都改善：hierarchical 下 A-OSE 从 171 增至 209、WI 从 0.007443
增至 0.008412；focal 下 A-OSE 从 428 增至 468、WI 从 0.005564 增至 0.005780。
软路由的新类 AP 大于 0，但绝对值仍很低，不能把这些数字描述为强检测效果。

参数审计确认增量阶段实际更新的新类 adapter 从 hard 的 17/21 类增加到
soft 的 21/21 类；两组新类分类行有变化、旧分类行和 bbox head 参数保持不变。
这支持软路由缓解梯度饥饿，不能单独证明 AP 的因果机制或充分训练后仍有收益。
训练没有发生匿名快照继承，两组风险 beta=0；此次没有验证完整 GNN risk 控制。

框覆盖诊断：T1 所有 100 个 query 在 IoU≥0.5 下覆盖 4,621/17,786（约 25.98%）
已知 GT，包含类别预测正确的覆盖为 3,067/17,786（约 17.24%）。T2 hard/soft
覆盖分别为 5,469/24,945 和 5,486/24,945（约 21.92%/21.99%）。这是归一化
IoU 的全-query 诊断，不是作者 AP，也没有用于更改本轮阈值。

结论限定为：**本机单种子短测中，软路由带来三个目标指标的小幅配对提升；
独立评分和组合优化未全面成功，尚未证明稳定有效。** 下一轮优先保留原评分、
继续对照软路由，单独检验受限定位路径更新和更充分的检测基础训练；不同时
叠加原型重构、评分校准等多个改动。需要完整日程、多种子和四任务验证。

实现与验证：新增评分/路由开关、checkpoint 配置约束、全量评分对照及框覆盖
诊断；36 项相关测试通过。完整数值见 `comparison.json`、各组 `*_metrics.json`
和 `transition_audit.json`。旧实验与原始数据均保留，未启动正式长训练。

## 后续排查

1. U-Recall 低且训练候选近乎为空：先排查跨视图匹配和区域响应启动，再比较
   候选策略；只能依据合法训练诊断调整，不能用测试 future GT 调阈值。
2. U-Recall 上升但 WI/A-OSE、known AP 恶化：减少错误伪正样本自增强，增加
   候选质量/背景约束，单独校准 objectness，而非继续增加候选数量。
3. Discovery 强于旧适配但 transfer 不胜 scratch：验证转化覆盖率、原型漂移，
   比较更保守的迁移权重/拒绝策略、在线中心淘汰和多原型表示。
4. 新类 AP 低且框损失停滞：在相同预算下消融最后 decoder 或 bbox head 的
   受限解冻，保留旧类蒸馏；先确定定位瓶颈再增加图复杂度。
5. 只有 Previous AP 提升：将结论限定为保持作用；继续做发现/迁移/risk/
   蒸馏分离消融，并重复 seed，不能把总体 mAP 提高等同于完整 OWOD 能力。
