# 未知检测优先：U-Recall、WI 与 A-OSE 的优化依据

更新：2026-11-17。本页讨论 main 分支 S-OWODB v5/v7 的未知检测路径。
`mowodb` 分支的 v8 focal/v9 background-negative 是不同实现，不按版本数字混用。
当前已实施有限双视图候选保护：对每图最多5个独立、有双视图证据的额外候选，
撤除objectness/router负监督，保留GT/pseudo正例、密集已知分类及重复/背景负监督。
专用入口为`tools/owod/run_sowodb_task2_unknown.sh`，默认原策略和优化策略各3个epoch。
实施细节、配对约束与完整启动命令见[3轮优化验收](sowodb-task2-fast-check.md)。
尚未获得本次优化后的真实AP/U-Recall成绩。下文保留前期分析与可选只读工具说明，
只读诊断不再是启动本次优化对照的前置要求。

## 为什么调整优先级

附件 `console.log1`，SHA256
`7964315a39a99a2c9b8e14477e20ef142c2861adf56f013d75b304ae9c0ebaef`，
记录 main `34af21ef`，S Task2 v7 soft/frozen 与 soft/bbox-last，各计划8轮increment。
固定第3轮的U-Recall为2.0446% / 2.7205%，unknown precision为1.0109% / 1.1052%，
WI为0.038962 / 0.039846，A-OSE为8230 / 8085。
初始化U-Recall已经为2.8303%，因此解冻组相对冻结组改善，不等于训练获得未知召回净增益。
附件结束时bbox-last还在第4轮中途，最终配对结果缺失。

此前的路由和定位对照主要针对Current AP偏低。这些对照仍有诊断价值，但不能代替
未知检测的直接验收。接下来优先解释未知目标在哪里被丢弃，以及哪些已知输出产生开放集错误。
保留旧类/新类AP作为约束，同时检查unknown precision和预测数量。

## 代码中可核实的两个问题

1. `models/graph_local/discovery.py:proposal_supervision`把GT匹配及可信pseudo设为正，
   其余query全部作为objectness负例；discovery router也学习未选中/null。
   `mine_candidates`需要几何有效、排除已释放GT、双视图显著性、IoU与特征一致性、去重，
   每图最多5个pseudo。未通过这些门槛的真实未知目标可能受到负监督。
   这是可核实的监督冲突风险；目前没有逐实例证据证明它是召回下降的主要原因。
2. `DiscoveryPostProcess`将query×类别展平后取共享Top50；同一query可以输出多个类别，
   包括unknown与known。作者评估独立统计unknown匹配和known→unknown错误，
   已输出unknown不会抵消同框的已知类错误。局部构造检查能复现这个现象，
   但其在实际checkpoint中的占比需要统计，不能从A-OSE总数直接反推。

也不能把所有未匹配query直接忽略。v4历史证据显示objectness/novelty过高，
unknown几乎占满Top50，Known AP急跌；v5密集槽位负监督正是为处理该问题。
已有可信背景novelty负监督开关，以及M分支独立v9对照，不能重复包装成新机制。

## 先看淘汰发生在哪一层

新增 `tools/analysis/audit_unknown_detection.py`，只读取现有完整checkpoint：

| 统计 | 输入输出及含义 |
|---|---|
| `raw_box_coverage` | 全部原始query框→被任意框覆盖的真实unknown GT数 |
| `objectness_gate_coverage` | 只保留objectness≥当前unknown阈值的框 |
| `novelty_gate_coverage` | 单独只保留novelty≥当前unknown阈值的框 |
| `joint_score_gate_coverage` | 保留objectness×novelty≥当前阈值的框 |
| `emitted_unknown_coverage` | 实际postprocess输出的unknown框覆盖 |
| `unknown_tp_local/duplicate/on_known/other` | unknown输出按一对一局部匹配分解，other不等于确定背景 |
| `known_on_unknown` | 已知输出与unknown GT重叠，按作者A-OSE的预测计数口径 |
| `known_on_unknown_same_box_as_unknown` | 上一项中同时输出完全相同unknown框的错误数 |

假设未知阈值为τ，因两个因子均在[0,1]，能通过乘积阈值的query必须同时满足
`objectness≥τ`与`novelty≥τ`，但两个条件同时满足也不保证乘积≥τ。
两个单独门控统计不是互斥归因；几何、objectness、novelty也不能据此完全分离因果。
比较joint gate与实际输出，才能直接检查共享Top-K丢失。

覆盖采用作者的预测坐标量化、inclusive IoU与严格IoU>0.5。
任意框覆盖不是一对一召回，多个GT可被同一个框覆盖，因而不能当作正式U-Recall。
正式指标只在`--images 0`完整作者测试集运行时，复用现有author evaluator计算；
保留作者测试序列的重复项。局部错误分组的分数并列排序可能与全局author排序不同。
默认256张为作者序列去重后的确定性前缀，仅做诊断，不创建新benchmark或报告局部AP。

全标签只在模型前向后用于统计，不进入criterion、候选挖掘、EMA、原型或优化器。
工具检查checkpoint及模型状态前后哈希、schema-3标注、协议/类别/评估定义，
历史S manifest只接受现有方法逐字节恢复旧SHA256的证明；不改正式续训校验。
输出记录实际checkpoint、方法版本、评分、路由、定位、代码提交和源码哈希。

## 如何由诊断决定实现

| 观测 | 优先验证的方向 | 验收与主要风险 |
|---|---|---|
| raw覆盖已低 | 改善类别无关候选/表示；评估独立于已知分类器的区域证据 | raw覆盖和最终U-Recall均提高；额外候选可能带来错误框 |
| raw覆盖较高，objectness门后明显减少 | 候选感知负监督：保留GT/pseudo正例、可信背景和重复槽负例，谨慎处理模糊槽 | 需要实测未知假负例是否减少；必须继续压制重复/背景，防止重演v4 |
| objectness通过，novelty/乘积门后减少 | 检查条件novelty的正例质量与校准 | 同时报告Known AP、unknown precision；不从测试unknown标签调阈值 |
| 乘积通过而实际输出覆盖下降 | 检查共享Top-K竞争及已知/未知分数可比性 | 固定正式Top50；诊断分数替换不当作正式改进 |
| 已知误报中大量同框unknown输出 | 独立验证query层的known/unknown决策与重复输出抑制 | 要测Known/Current AP是否损失；同框计数只表明可检查范围 |

本次已选择单独验证候选负监督修正；objectness/novelty淘汰和同框开放集错误统计仍可辅助解释结果。
保持候选正例门槛、评分、输出阈值、batch和定位策略一致，避免混合因素。
需要学习unknown候选和discovery表示，
不能用旧类保持或回放替代未知学习。

## 原始论文核对与设计边界

- OW-DETR（CVPR2022）Sec.2.3–2.5：attention-driven pseudo unknown与foreground objectness。
  当前项目采用更严格的双视图候选门槛；论文机制不能证明本项目候选覆盖足够。
- PROB（CVPR2023）Sec.4.1、Eq.1–5：分离类别与objectness，用查询分布的概率建模及
  matched-query正向似然目标，降低未知无标注条件下的背景判定依赖。
  Supplementary A.3明确显示objectness温度、U-Recall、AP与WI/A-OSE存在权衡，
  温度τ不是本项目unknown输出阈值，不能照搬其数值。
- CAT（2023）Sec.3.2–3.3：定位与识别级联，以及模型驱动与selective-search输入证据融合。
  说明只依赖已知监督形成的候选可能有覆盖偏差；它是完整不同方法，不能将局部接入视作复现。

本地原文见`tmp/pdfs/ow-detr.pdf`、`prob.pdf`、`cat.pdf`及其文本提取。
上述论文已经研究了objectness解耦、伪标签和候选补充。
候选感知负监督或决策互斥属于待验证的项目设计，不能仅因接入GNN/adapter声称创新或有效。
需要进一步用图关系处理什么额外问题，应在具体机制确定后单独论证。

## 服务器只读运行

代码先经Git同步。当前长训练正在运行时，保留其checkout不变；
可等本轮结束后同步，或用用户已有的独立checkout运行诊断。
下列命令适用于本次S/main v7，不能用这个main实现读取M/v8/v9权重。

服务器Linux/Bash，仓库根目录，既有tree-detr环境，先核对提交并确认选择的GPU空闲。
`checkpoint.pth`在训练中会覆盖，使用下面冻结组已完成的文件，且不要同时重启该组。

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
git branch --show-current
git rev-parse HEAD
nvidia-smi
set -o pipefail
mkdir -p output/analysis/s-task2-unknown-audit
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -u tools/analysis/audit_unknown_detection.py \
  --checkpoint output/sowodb-task2-v7-localization-8epochs-2gpu/results/frozen/checkpoint.pth \
  --manifest data/owod/s-owodb/split_manifest.json \
  --images 256 --num-workers 2 \
  --output output/analysis/s-task2-unknown-audit/frozen-e8-prefix.json \
  2>&1 | tee output/analysis/s-task2-unknown-audit/frozen-e8-prefix.log
```

GPU2只在附件启动时枚举为3090，空闲状态必须以实际运行前为准。
这不是训练，不需要额外GNN重建或新Conda环境。256张预计数分钟，连同加载和哈希
预留5–15分钟，以工具每50张打印的实测耗时修正。输入/状态哈希变化、OOM、非有限输出立即停止。
若成功且要确认完整指标，固定相同checkpoint用`--images 0`并换用新的JSON和日志名；
全S测试集单卡预留约15–45分钟，实际由前缀吞吐修正。较长全量诊断放入tmux。
不扫描阈值或选择最优测试epoch；本工具没有阈值覆盖参数。

本次用独立3轮训练对照检验未知召回、预测精度/数量、WI/A-OSE及已知AP的联合变化。
方向成立且代价可接受后才扩展长预算。正式收益仍需要同起点、同容量/预算的固定末轮比较，
以及完整increment→作者balanced finetune；本地没有启动服务器长训练。

## 本次验证

本地既有tree-detr环境运行新增诊断检查及作者评估相关检查：14项通过，1项跳过
（既有外部OW-DETR源码对照，仓库已移除该外部checkout）。覆盖门控分解、同框双输出与
实际author evaluator的A-OSE对应、一对一/任意覆盖区别、困难目标、错误类型、
严格IoU与量化、作者序列重复项、版本/类别拒绝和旧manifest精确还原。CLI帮助与语法检查通过。
本地有S标注目录，但没有本次目标checkpoint；没有运行真实checkpoint的端到端诊断，
也没有获得新训练或评估成绩。服务器的实时进度、GPU空闲及最终bbox-last结果未读取。
