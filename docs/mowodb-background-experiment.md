# M-OWODB：可信背景 novelty 负监督对照

本入口在 `mowodb` 分支运行完整作者 Task 1 的八轮配对诊断，比较 focal v8
与新增可信背景负监督的 v9。目的：检查未知输出过多、precision 很低，是否部分来自
novelty 仅在少量前景 query 上受监督。它是工程改进假设，尚无本次 AP 提升证据，
不作为新颖性结论。

## 实际改动

输入仍是学生视图输出、EMA 翻转视图输出，以及当期已释放的 GT。沿用
`discovery_candidates.py` 的候选与背景集合：排除已匹配 GT 和与已知 GT 重叠的区域；
两视图有 query 匹配，且各自框内显著性均低于 0.15，才可进入背景集合。
没有教师、warmup 第一轮及评估时不启用新增背景监督。低显著性是启发式证据，
不保证区域内确实没有未知物体。

v8 的 novelty 权重仅覆盖已知 GT query（目标 0）和伪未知 query（目标 1，权重为
confidence）。v9 将可信背景 query 加入目标 0 的集合，权重为 1；不确定的 unmatched
query 继续忽略。实现是 `DiscoveryCriterion.forward` 中仅改变 `loss_unknown` 的
`novelty_weights`。沿用正、负组分别取均值再平均的 BCE，因此新增背景与已知 GT
共同分担负组的归一化；伪未知的 count normalization、confidence 和三轮 ramp 保留。

没有新增模型参数。候选阈值、objectness、已知分类、router、原型证据、GNN、adapter
及评分保持相同。未知分数仍为 `objectness × sigmoid(novelty)`，已知分数仍为
`objectness × sigmoid(class_logit)`，共同参与 Top-50。

## 固定配置与预算

| 项目 | reference | background |
|---|---|---|
| 版本 | v8 focal | v9 focal |
| 新增背景 novelty 负监督 | 关闭 | 开启 |
| 数据 | 作者 M-OWODB Task 1，完整训练与测试清单 | 相同 |
| 初始化 | DINO ResNet-50，seed 42，实际初始化权重 SHA256 | 必须完全一致 |
| GPU | nvidia-smi 物理编号 0、1，两张 RTX 3090 | 相同 |
| batch | 每卡 2，总计 4 | 相同 |
| 学习率 | 主干 5e-6，其余 5e-5 | 相同 |
| novelty / objectness | count / balanced-bce | 相同 |
| ranking / feature gradient scale | 1 / 1 | 相同 |
| routing / localization | hard / frozen | 相同 |
| 日程 | 注册 50 轮、lr_drop=50，执行完第 8 轮暂停 | 相同 |

完整数据协议不等于复现 OW-DETR 模型或论文优化设置：这里是项目方法、低学习率，
作者 M 的 VOC integral AP50 评测。第 8 轮没有学习率衰减，也没有进行 Task 2 或均衡微调。

按此前双 3090 的 M 日志估算，每组约 4.5–6 小时，两组顺序运行约 **9–12 小时**；
以首轮实际训练与评估耗时修正。正常情况下两组均执行到第 8 轮；数据/来源校验失败、
非有限 loss、OOM 或设备错误时停止，修复前不切卡、不改 batch、不混接旧权重。

## 运行

服务器：`top@toplab.billzhou233.moe`，Linux/Bash，仓库根目录
`~/disks/new-hdd/zhy/Tree-DETR`。沿用既有环境
`/home/top/disks/new-hdd/conda_envs/tree-detr/bin/python`，不需要手动激活 Conda。
驱动显示 CUDA 13.2 不代表需要切换 PyTorch CUDA runtime；既有环境为 PyTorch 2.4.1 / CUDA 12.1。

**先等待当前 S-OWODB 作业结束、物理 GPU 0/1 空闲，再在这份服务器 checkout 切分支。**
不能在旧进程运行中切换其工作目录的源码。启动器会拒绝 GPU 0/1 上任何 compute 进程，
不会停止旧作业或自动换用其他 GPU。

本地改动已提交并推送到 `origin/mowodb` 后，在服务器普通 Bash 中执行：

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR && git switch mowodb && git pull --ff-only origin mowodb && tmux new -s m-bg8 'bash tools/owod/run_mowodb_background_pair.sh'
```

它先完整核验 schema-3 数据、清单和原始输入哈希，核对两组计划，然后顺序运行参照组和
实验组。实际使用两张物理 GPU 的 UUID，避免 CUDA 排序改变物理编号含义。仅本地扩展
导入失败时，才使用既有环境 CUDA toolkit 为 sm_86 重建忽略的编译产物。

只检查数据与计划，不占用 GPU：

```bash
bash tools/owod/run_mowodb_background_pair.sh --check
```

同一输出目录有 `flock` 保护。中断后在 tmux 中重新运行同一脚本即可：已有计划必须与
当前数据和源码内容一致，自动恢复 model、optimizer、scheduler、EMA；已完成的第八轮
结果经校验后跳过。checkpoint 已保存但末轮评估中断时，恢复后补完评估，不重复训练。
旧训练器没有保存各 rank 的 RNG，因此中途续训不保证等同于从未中断的优化轨迹；
汇总记录各组的 resume 调用数量，若有中断，结论须注明此限制，关键收益再用连续运行验证。

这些权重带有配对诊断标记，不能初始化正式七阶段链；也不能与 A40、旧 v8 或其他版本
权重混接。后续正式验证应使用独立输出，从相同预训练设置重新开始完整日程。

## 日志与判断

总目录为 `output/mowodb-task1-background-8epochs-2gpu/`：

- `console.log`：总日志。
- `paired_plan.json`：两组计划与源码内容哈希。
- `initialization.json`：DDP 同步后的实际初始化权重证据。
- `reference/`、`background/`：各自完整 benchmark 目录、checkpoint、`train.log`、`metrics.jsonl`。
- `comparison.json`：固定第 8 轮对照和差值，JSON 中 AP/recall/precision 为 0–1 分数，控制台转为百分数。

```bash
tail -f output/mowodb-task1-background-8epochs-2gpu/console.log
```

按 Ctrl+B 后再按 D 离开 tmux，训练继续。再次进入用 `tmux attach -t m-bg8`。

只比较固定末轮的 Current/Known AP50、U-Recall、unknown precision、unknown predictions、
WI@recall=0.8、A-OSE，H-Score 作为项目附加指标。Task 1 没有 Previous AP，记录 null；
缺失指标拒绝汇总，不填零，也不选择历史最佳轮替换末轮。

优先考虑进入完整日程验证的结果是：Known AP50 下降不超过 0.5 个百分点，U-Recall
不下降，且 unknown precision 提高。0.5 是本次工程筛查容差，不是显著性检验。如果
只是未知预测数量减少而 U-Recall 同时明显降低，不能称为发现能力改善；应先检查
背景误选，再独立设计背景权重或候选质量对照。八轮结果不证明 50 轮后期退化已解决，
不能作为完整 M-OWODB 的论文成绩。

## 本次验证范围

本地 127 项相关检查通过（最后的汇总来源校验另经 13 项回归通过），覆盖：
新背景梯度、伪未知 confidence/ramp、模糊区域忽略、warmup、缺教师、
未来 GT 隔离、默认兼容、版本/计划/续训校验、固定轮汇总、启动器失败退出、
初始化审计及末轮评估中断恢复；另有实际小型 Deformable DETR 的 CUDA 前后向、
增量 adapter 路径与 state 重载。

目标服务器完成了实际 M 数据/DINO 的只读计划检查，以及不分配 CUDA 的双进程 CPU
DDP 初始化一致性、RNG 不变和错误广播检查。GPU 占用检查实际识别到旧 S 作业并拒绝。
本次没有占用服务器训练 GPU 做双卡训练测试，完整八轮和 AP 结果尚未产生。
