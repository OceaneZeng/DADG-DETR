# Discovery v8：已知／未知评分约束与发现梯度对照

2026-11-13 后续：γ=0.1 的 8 轮 AP 未改善。服务器完成相同训练图的第 5 轮权重
诊断后，新增低学习率 Task 1 启动器的显式 `focal` 模式（γ=1、排序权重 1），见
[同轮诊断与 focal 评分对照](owod-task1-focal-followup-20261113.md)。原入口默认
不变。下文历史模式表中的评分均为 hierarchical；新评分和对应排序损失使用
已有实现，仍登记 v8，禁止跨模式续训。

2026-11-13：当前低学习率 ranking 日志的复核、Task 1 AP 优化对照和正式 Task 1/2
只读权重诊断，见[低学习率与发现梯度对照](owod-task1-low-lr-optimization.md)。

2026-11-11：用户启动的 M-OWODB 低学习率四任务全量链见
[全量启动与 Task 2 新类观察](owod-discovery-mowodb-full.md)。本页旧启动器的
`--full` 仍只完成 Task 1；新的全任务入口使用独立目录，不接续旧诊断权重。

教师计算精简和 `--profile-runtime` 的日志位置、恢复边界，以及独立的新链计算／
风险消融，见[运行效率说明](owod-runtime-optimization-v3.md)。

这是针对 2026-11-09 M-OWODB Task 1 退化的实验性实现，尚无服务器 AP 提升结论。
保留 Deformable DETR、affinity GNN、discovery adapter、匿名原型、快照继承与旧类保持。
作者 ImageSets、类别顺序、schema-3 manifest／输入哈希、联合 Top50 与作者 VOC 积分
AP 评估继续沿用当前 M 配置。

## 原因与解决办法

v7 objectness 单项完成 8 轮，Known AP50 从第 1 轮的 0.8556% 降到第 3 轮的
0.001721%，第 5 轮 512,300 个输出全部是 unknown，AP 为 0；第 8 轮为 0.02792%。
第 2→3 轮验证分类损失由 1.1355 升到 1.3529，bbox 损失由 1.0025 升到 1.2953。
因此需要同时检查评分竞争和检测学习，不能仅凭低 AP 判定框已全部丢失。

| 问题 | 解决办法 | 实现与边界 |
| --- | --- | --- |
| `weight-sum` 抵消伪未知置信度的整体缩小 | 复用 v7 `count` novelty 归一化 | 所有 v8 对照均使用；不改变 v7 旧配置的定义 |
| 同一 query 的 unknown 分数容易超过真实已知类别 | 加入已知 GT 的实际评分排序损失 | 只用已释放 GT，不给已知类别输出配额，不调整测试阈值 |
| 发现监督可能干扰共享检测表示和框 | 可配置发现分支输入梯度缩放 | GNN 参数仍正常学习；梯度干扰是否是首因尚未证实 |
| 缺少训练中的直接证据 | 保存匹配已知目标的分数、novelty、objectness 和排序违反计数 | 机制统计不替代完整作者 AP／U-Recall |

候选真实准确率尚未核验，本轮不同时修改候选阈值。学习率、batch、卡数、warmup、
ramp 和原型更新沿用同一配方。旧 v7 objectness 单项未获改善证据，v8 各模式以
v7 **confidence 单项**为共同参照。

## 模块与训练位置

对于 Hungarian 匹配到已释放 GT 类别 y 的 query，原 hierarchical 评分为：

```text
S_y = sigmoid(a_o) * (1 - sigmoid(a_n)) * softmax(z_known)[y]
S_u = sigmoid(a_o) * sigmoid(a_n)
log(S_u / S_y) = a_n - log_softmax(z_known)[y]
L_rank = mean_GT softplus(log(S_u / S_y))
```

`--discovery-known-rank-coef` 控制此损失的总权重，默认 0 保留旧计算，实验值为 1。
在 log 空间计算，避免饱和概率下溢。分母使用全局匹配 GT 数量，DDP 保持与拼接
所有 rank 的单进程目标一致，覆盖部分 rank 无 GT、全局仅一个 GT 和全局无 GT。

输入为最后层分类 logits、novelty logits、已知 GT 匹配索引；损失加入现有联合
目标。同时降低匹配已知槽位的 novelty，提高正确类别条件概率。原密集 sigmoid
focal 类别监督、辅助层和框监督保留。伪未知／未匹配槽位不参与此项，不要求真实
未知物体排在其他位置的已知目标之后。若明确使用原 `focal` 评分，按相应实际分数
的 log 比值计算；旧启动器仍用 hierarchical，低学习率入口显式 `focal` 模式除外。
Task 4 无 unknown 输出，此损失为 0。

`--discovery-feature-grad-scale γ` 默认为 1，实验值为 0.1，范围为 [0, 1]：

```text
f_graph = stop_gradient(f) + γ * (f - stop_gradient(f))
```

f 是最后 decoder 的 query 表示，送入 affinity GNN 后生成 objectness、novelty、
affinity 和 embedding。前向数值不变，GNN 参数获得完整梯度；沿此入口回到 query
的梯度乘 γ，包括对应发现损失和排序损失的 novelty 路径。原型观测仍用原 query
及候选权重。伪框一致性中的 student boxes 同样缩放，以限制此项对框路径的影响。

标准 GT 框损失、密集已知分类及排序损失的已知分类路径保留原梯度；router 原有
detach、discovery adapter 的检测训练路径和候选生成保持。缩放只定义该分支裁剪
前的输入梯度，总梯度经过全局裁剪和 AdamW 后，更新不保证恰好变成 γ 倍。小 γ
可能拖慢未知发现和后续适配，需要联合检查 U-Recall、未知精度及后续任务 AP。

## 模式、版本与可比性

| 入口／模式 | novelty 归一化 | 排序权重 | 梯度系数 | 方法版本 |
| --- | --- | --- | --- | --- |
| 原 v7 启动器 `confidence` | count | 0 | 1 | 7，共同参照 |
| v8 默认／`ranking` | count | 1 | 1 | 8，排序单项 |
| v8 `guarded` | count | 0 | 0.1 | 8，梯度单项 |
| v8 `combined` | count | 1 | 0.1 | 8，组合 |

以上均用原 `balanced-bce` objectness，不叠加 v7 objectness focal。因此 v8 和旧
v7 objectness 日志之间有多个训练因素差异，不能把 AP 差值全部归为排序损失。
权重 1 和系数 0.1 是预先登记的项目实验设定，不声称来自作者论文，不通过作者
测试集选择数值。单项通过才考虑组合，不默认连续启动所有模式。

新选项进入运行计划、配置、checkpoint、恢复与阶段链校验。旧 checkpoint 缺少
新字段按 0／1 读取，v5–v7 原训练损失和计划命令保留。任一新选项生效则登记 v8，
禁止从 v7 续入或在 v8 模式间切换续训。每组从同一 DINO 骨干重新训练，保留旧成果。

## 新服务器启动

先通过 Git 同步本地修改，沿用现有 Conda 环境，确认物理 GPU 0、1 空闲。
Linux / Bash：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new -s tree-m-v8 'bash tools/owod/run_discovery_mowodb_3090_v8.sh'
```

默认排序对照：每卡 batch 2、全局 batch 4、seed 42、activation checkpointing；
50 轮计划、lr_drop=50，8 轮后自动暂停，不启动 Task 2。参考本次日志每轮约
31–32 分钟，单组约 4.2 小时；达到预算即停，不按作者测试集最好轮次选权重。

日志和结果：

```text
output/tree-detr-m-v8-ranking-3090-logs/console.log
output/tree-detr-m-v8-ranking-3090/m-owodb/seed_42/stage_0/increment/
```

阶段含 `train.log`、`metrics.jsonl`、`run_config.json`、EMA／原型和编号权重。
查看日志：

```bash
tail -n 50 -f output/tree-detr-m-v8-ranking-3090-logs/console.log
```

若尚未运行共同参照，在 tmux 中单独执行原入口：

```bash
bash tools/owod/run_discovery_mowodb_3090_v7.sh confidence
```

其他模式同样在 tmux 内逐组运行，每组各占一份预算：

```bash
bash tools/owod/run_discovery_mowodb_3090_v8.sh guarded
bash tools/owod/run_discovery_mowodb_3090_v8.sh combined
```

相应目录将 `ranking` 换成模式名。`--dry-run` 核验数据并打印真实命令；`--resume`
沿同方案恢复到 8 轮暂停点。诊断支持继续时，可恢复同方案原 50 轮预算：

```bash
bash tools/owod/run_discovery_mowodb_3090_v8.sh --resume --full
```

发现已有 checkpoint 而未指定 `--resume` 时退出，防止覆盖成果。暂停预算不进入
运行计划，不改变优化日程；`--full` 仍只完成 Task 1。

## 诊断与停止判断

`metrics.jsonl` 新增 `discovery_known_count`、`discovery_known_rank_violations`、
`discovery_known_score_sum`、`discovery_known_unknown_score_sum`、
`discovery_known_objectness_sum`、`discovery_known_novelty_sum`、
`discovery_unselected_count`、`discovery_unselected_objectness_sum`，经现有跨 rank
归约。训练字段实际以 `train_discovery_` 开头，验证字段以 `test_discovery_` 开头并带
`_unscaled` 后缀。用同一窗口内的和／计数之比得到加权均值和违反率，避免空 rank／
不同目标数量导致简单平均偏差。console 显示排序损失、匹配数和违反数，其余随
结构化日志保存。

第 8 轮暂停后，在空闲 GPU 0 执行只读诊断，预计数分钟：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/owod/diagnose_discovery_task1.py \
  --checkpoint output/tree-detr-m-v8-ranking-3090/m-owodb/seed_42/stage_0/increment/checkpoint0007.pth \
  --manifest data/owod/m-owodb/split_manifest.json --images 32 \
  --output output/tree-detr-m-v8-ranking-3090-logs/task1-epoch8-diagnostic.json
```

工具新增实际 GT 类别分数的 `matched_known_ranking`，保留 raw／known_actual／joint
覆盖、匹配与未匹配组、EMA 候选和匿名原型。固定训练图前缀不代表总体准确率，
单 query 排序改善也不保证全图 Top50 改善。

若固定诊断点继续显示 joint 覆盖下降、unknown 占满输出或原始框与分类持续恶化，
可保存状态后正常停止。排序改善但检测仍恶化时，再检查梯度模式；候选主要是局部
或纹理时，再独立改进候选。正式结论使用固定末轮作者 AP50／U-Recall／WI／A-OSE。

## 本地验证边界

检查实际评分比值和梯度、饱和 logits／空目标、未来标签隔离、未知正例学习保留、
图前向与图参数梯度、特征／伪框梯度缩放、旧版本兼容、版本和模式校验、真实 CUDA
前反向及增量恢复、DDP 归一化数学边界、Bash 数据失败退出和已有权重保护。
归一化边界使用确定性 collective 替身比对全局目标与梯度；真实双进程测试已保留，
但本机 Windows Gloo 在进程组初始化时超时，尚未验证真实通信。未改服务器 NCCL
配置。运行结果见本次交付说明。本地缺少这次服务器 M 全量 manifest 和权重，未运行
新全量 AP。
