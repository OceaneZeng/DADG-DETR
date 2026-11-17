# S-OWODB：未知候选负监督优化与各3轮验收

更新：2026-11-17。适用于本地 main 的 S-OWODB v7 Task2 实验性分叉。
本次已实施训练监督优化，入口为 `tools/owod/run_sowodb_task2_unknown.sh`。
默认顺序执行 baseline、protected，各3个epoch，共6个epoch；不是3次独立随机种子实验。
已有routing/localization对照保留，但不作为这次优化的验收入口。

## 修改的具体问题和训练位置

原策略把GT匹配和每图最多5个未知pseudo以外的全部query作为objectness/router负例。
真实未知目标可能因显著性不足、或pseudo名额已满而被当作背景压低。
本次假设：对有独立双视图证据的少量候选暂缓这种负监督，有望减轻未知召回下降。
附件只能证明有这一风险，不能证明它就是低召回的主要原因；本次训练检验该假设。

训练流程为：学生预测与EMA翻转视图 → 原有pseudo挖掘 → 额外候选保护mask
→ objectness BCE与discovery router CE的query权重 → 其余原有训练路径。

| 路径 | baseline | protected |
|---|---|---|
| GT与原有未知pseudo正监督 | 保留 | 保留且选取规则相同 |
| 已知类别主层和辅助层focal监督 | 全query | 全query |
| 额外候选的objectness/router负监督 | 权重1 | 满足下述条件者权重0 |
| novelty、跨视图一致性与原型记忆 | 原有选中目标 | 仍只用原有选中目标 |
| 重复框、已释放GT重叠槽、低证据/背景槽 | 负监督 | 保留负监督 |
| 参数容量、soft路由、冻结定位、评分与阈值 | 相同 | 相同 |

保护条件固定为：有效框且不与已释放GT/旧类teacher补全框IoU>0.3重叠；
学生和EMA对应框的显著性都≥0.5，IoU×特征余弦一致性达到原pseudo门槛（默认0.5）；
按原quality排序，每图额外最多`discovery_top_k`个（默认5），
与pseudo及已保护代表框IoU≤0.7。正pseudo仍用原显著性门槛0.65。
0.5是本项目事先固定的实验取值，不是原论文给定值，不按测试标签调参。
无teacher、warmup内及评估时不启用保护。第一轮仍按原warmup训练，后两轮实际检验改动。
保护不是增加正标签：候选不获得novelty正监督，不进入原型记忆，也不自动命名为真实类别。

最多保护5个/300个query，不恢复v4的广泛忽略背景设定。
它可能提升召回，也可能保留更多假框、改变已知输出并使WI/A-OSE变差；不能预先宣称同时改善。
评分互斥、框定位解冻、输出Top50和unknown阈值均不在这次优化因素中。

## 原始论文依据与结论边界

已核对本地原文：OW-DETR（CVPR2022）Sec.2.3–2.5的attention pseudo与foreground学习，
以及PROB（CVPR2023）Sec.4.1 Eq.1–5的类别/objectness解耦及matched-query概率目标。
两者支持检查无标注未知物体被背景监督的问题；本次有限候选保护是项目实验性设计，
不是PROB实现，也没有得到论文新颖性或实际指标收益证据。
原文位置与前期日志分析见[未知检测优先](owod-unknown-recall-priority.md)。

## 配对约束与恢复

两组从同一个完整、非诊断的v5 Task1第50轮checkpoint和对应GNN重新建立Task2。
两组实际模型与memory的初始化SHA256必须完全一致；固定完整作者S数据、seed42、
双RTX3090、每卡batch1、总batch2、原增强、学习率和1轮warmup。
第3轮末用完整作者测试集评估，采用VOC2007 11点AP50；不取测试最优轮次。
新实验不依赖历史v7诊断权重，不从旧Task2权重继续训练。

默认父文件：

- `output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth`
- `output/tree-detr-s-v5-full/s-owodb/seed_42/gnn/stage_0/gnn_stage0.pt`
- `data/owod/s-owodb/split_manifest.json`

manifest、父权重、GNN、源码、GPU选择、预算及策略写入独立plan；checkpoint保存策略和RNG。
跨策略、跨预算、变更源码的恢复会被拒绝。诊断checkpoint不能进入正式七阶段链。
本次训练逻辑改变会使优化轨迹分叉；比较限定于同起点同预算，不能直接对照旧8轮末成绩。
旧实验需要继续恢复时保留原checkout/提交；不要在正在运行的checkout中拉取新代码。
不覆盖旧权重、日志、数据或结果。

## 先在本地经Git同步

Windows / PowerShell，目录 `C:\programlearning\Tree-DETR`。
下面仅列本次文件，不包含用户已有AGENTS.md、research/selection.md和M论文比较文档改动。

```powershell
cd C:\programlearning\Tree-DETR
git add -- engine.py main.py models/graph_local/discovery.py models/graph_local/discovery_candidates.py `
  models/graph_local/tests/test_discovery_protection.py tools/owod/train_task2_diagnostic.py `
  tools/owod/strict_setting.py tools/owod/run_sowodb_task2_routing.py `
  tools/owod/run_sowodb_task2_routing.sh tools/owod/task2_routing_cuda_check.py `
  tools/owod/run_sowodb_task2_unknown.py tools/owod/run_sowodb_task2_unknown.sh `
  tools/owod/tests/test_task2_unknown_experiment.py docs/sowodb-task2-fast-check.md `
  docs/owod-unknown-recall-priority.md
git diff --cached --stat
git commit -m "Add bounded unknown candidate protection and paired three-epoch validation"
git push origin main
```

当前实现基于main提交`f2aefbd`继续修改，命令按该分支提供；同步前核对实际分支。
本次助手没有替用户提交、推送或更改服务器代码。

## 服务器启动

Linux / Bash，目录 `/home/top/disks/new-hdd/zhy/Tree-DETR`，沿用既有tree-detr环境。
先确认旧实验不再使用这个checkout，GPU0/1确实空闲；launcher会记录实际提交、分支和GPU信息。

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git branch --show-current
git status --short
git pull --ff-only origin main
git rev-parse HEAD
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
nvidia-smi
unset SOWODB_TASK2_OUTPUT
bash tools/owod/run_sowodb_task2_unknown.sh --check
tmux new -s s-task2-unknown3 'SOWODB_GPUS=0,1 bash tools/owod/run_sowodb_task2_unknown.sh'
```

`--check`核验完整数据及父checkpoint/GNN，不训练；输出应包含`epochs: 3`、
`evaluation_epochs: [3]`、`baseline/protected`和对应`dense/cross-view-protected`策略。
实际启动先运行双卡CUDA/NCCL检查，再顺序训练两组。
非默认空闲GPU可设置`SOWODB_GPUS`，计划会固定并核验选择；仍须两张3090。
不要同时用相同GPU跑旧定位对照。

默认输出：`output/sowodb-task2-v7-unknown-3epochs-2gpu/`。

```bash
tail -f output/sowodb-task2-v7-unknown-3epochs-2gpu/console.log
cat output/sowodb-task2-v7-unknown-3epochs-2gpu/results/comparison.json
```

`results/baseline/`和`results/protected/`分别存run_config、初始化哈希、初始评估、
每轮训练统计、checkpoint、第3轮完整指标、learning probe及参数审计。
`comparison.json`汇总`protected_minus_baseline`、`initial`、`change_from_initial`和`training_activity`。
AP/recall/precision使用[0,1]比例，差值乘100才是百分点；A-OSE和unknown_predictions是计数。
`training_activity`中的protected_count先按进程平均、再按训练步平均；本次每卡batch1，
可理解为每图平均受保护query数。
protected_fraction是query比例，不是实际未知GT召回。
评估时没有teacher，故评估中的保护计数为0属于预期；检查训练epoch JSON的计数。

中断后在同一源码、输入、预算及GPU选择下重用同一命令，自动恢复自己的计划；
若会话仍在，先`tmux attach -t s-task2-unknown3`，不要另开第二份。
launcher用文件锁阻止同输出并发写入。首次运行结果目录必须为空，不能复用旧实验结果。

## 验收与耗时

固定比较第3轮末：U-Recall应上升，WI@recall=0.8与A-OSE期望下降，
同时列出Previous/Current/Known AP50、unknown precision和预测数量，不能只看loss或pseudo数量。
特别检查各自相对初始值的变化；优于baseline不等于训练带来未知召回净增益。
若召回上升但precision、AP或开放集错误明显恶化，判为有代价的权衡，需要继续修正；
若保护计数一直为0，不能据此评价保护机制收益，应先核对候选门槛与双视图证据。
单seed、短程increment只能筛方向，不证明完整Task2或论文收益。

按附件每轮约1.7–2小时，两组6轮约10–12小时纯训练，包含完整数据核验、初始化与评估
预留12–16小时，以首轮实测修正；3epoch仍使用完整作者数据，不是数分钟冒烟。
输入/初始化不一致、非有限loss/梯度、OOM或设备故障立即停止并保留成果。
按固定第3轮完成后再决定：方向成立且其余指标可接受才扩展预算与作者balanced finetune；
若无收益或代价过大，保留本次结果，转查评分冲突、候选定位等独立因素，不中途改阈值挑结果。

## 本地验证范围

新增17项检查通过：保护/未保护槽的实际梯度、双视图与warmup门控、重复和背景抑制、
未来标签隔离、保护上限、配对策略/预算/目录、跨策略恢复拒绝，以及真实小型
Deformable DETR CUDA更新、旧分类行/定位冻结、EMA和checkpoint策略重载。
相关回归另98项通过；1项既有Windows双进程Gloo检查未运行。
目标Linux双3090/NCCL路径由服务器启动器检查，本地未执行该双卡检查。
本地缺少本次完整Task1父权重和对应GNN，尚未启动真实两组3轮训练，没有优化后实际AP成绩。
