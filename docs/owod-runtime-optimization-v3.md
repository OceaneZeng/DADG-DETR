# OWOD 运行优化 v3：教师、批量 adapter 与控制器消融

2026-11-10。本次保留 Deformable DETR、未知发现 affinity 图、discovery adapter、
匿名原型／快照继承、旧类蒸馏和作者均衡微调。默认不删除 interference GNN。
服务器实际吞吐和正式 AP 尚未测量；本机组件计时不能替代双 3090 的训练耗时。
本次运行实现版本为 3，与 discovery 的方法版本独立；此前的服务器短测与采用范围
保留在[运行优化 v2](owod-runtime-optimization-v2.md)，不作为本次新改动的服务器证据。

## 默认生效的改动

- 旧类教师只返回最后层 logits／boxes，跳过发现图头和额外预测头。
  decoder 中影响最终框的迭代 refinement、类别 adapter 和 discovery adapter 继续执行。
- EMA 教师跳过无用的辅助预测，仍计算 query 特征、区域响应和 affinity 图 embedding，
  保留原双视图候选和一致性损失。图头没有整体删除，两种教师没有合并。
- 校准数据集一次建立类别→图片索引，保持原图片顺序、每类前缀和重复实例计数规则，
  避免每次 source-target probe 都扫描整套标注。不缓存或改变训练增强。
- `calibration_summary.json` 新增 `timing_seconds` 和 `cache_hits`，区分模型加载、
  sketch、基础损失、经验 probe、GNN 拟合，以及三类缓存命中数量。CUDA 在阶段边界
  同步计时；不改变 probe 的数值、顺序、监督、图片预算或 GNN 超参数。

旧版本默认运行计划和 state_dict 布局保持可用。教师路径不改变所消费的前向数值、
训练目标、EMA 规则或数据协议；原有同方案 checkpoint 可按原命令 resume。
冻结 backbone 的跨模型共享、降低教师频率和训练稀疏路由未在本次引入。

## 可选的批量 adapter 计算

`--graph-adapter-compute batched` 将同一 stage 的低秩 A/B 分别堆叠，批量完成
`B(Ax)`，减少逐分支小矩阵调用。保留独立参数／checkpoint 键、全部阶段的贡献、
hard／soft 门控，以及 DDP 的零梯度和未选分支 AdamW 跳过规则。没有展开完整 BA，
也没有把全部历史阶段压成一个全局路由。

默认 `loop` 保留旧运算路径；旧 checkpoint 缺字段时解释为 `loop`。
`batched` 数学目标相同，但矩阵计算顺序可能产生浮点差异。它进入计划、checkpoint
及阶段链校验，只用于预先选择的新运行，禁止已有链中途切换。
小矩阵批量化的实际收益依赖 GPU 和 bank 大小，先测组件，再测整步时间。

## 独立的无干扰控制器消融

`--graph-risk-mode disabled` 将有效 beta 设为 0，在完整四任务计划中：

1. 不安排三次 interference GNN 校准，也不要求 GNN checkpoint。
2. 新任务不提取控制器的梯度 sketch，不加载或推断 interference GNN。
3. adapter 前向跳过冻结旧类证据和风险矩阵计算。
4. 保留新类转化特征提取、router 初始化、快照继承、蒸馏和均衡微调。

该模式保留同样的 bank 参数布局和全零风险占位 buffer，供容量与恢复审计；
旧类证据层不执行。配置标记 `disabled_no_interference_controller`，没有学得的
风险分数，不能将全零占位解释为 GNN 预测。未知发现的 AffinityGraph 仍正常训练。

默认 `learned` 保留原链。旧 checkpoint 缺字段时按 `learned` 处理，模式在 Task 1
就登记并贯穿全部阶段，不能把旧 Task 1 权重接入 disabled 链，也不能绕过校验。
不能配合共享 GNN parent。已有 learned 链的 `beta=0` 数值消融仍可恢复，但仍按原
计划校准；如需真正省去校准，必须新建 disabled 链。

只关风险控制不代表更高 AP。比较时保持 compute、发现损失、数据、seed、训练日程
等相同，使用固定末轮 Previous／Current／Known AP50、U-Recall、WI、A-OSE；
Task 4 不报未知指标。一次只比较一个因素，不能同时切换 compute 和 risk 来归因。

Linux / Bash，仓库根目录、已激活 `tree-detr` 环境，M manifest 和 DINO 权重已核验：

```bash
python tools/owod/run_benchmark.py \
  --manifest data/owod/m-owodb/split_manifest.json \
  --recipe ow-detr-paper --method discovery \
  --backbone-weights pretrained/dino_resnet50.pth \
  --output-dir output/tree-detr-m-no-risk \
  --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
  --seed 42 --activation-checkpointing \
  --graph-routing hard --graph-localization frozen \
  --discovery-scoring hierarchical --discovery-pseudo-ramp-epochs 3 \
  --discovery-pseudo-normalization count --discovery-known-rank-coef 1.0 \
  --graph-risk-mode disabled --dry-run
```

这仅打印新链计划，不启动训练。GPU 0,1 是当前双 3090 配方，执行前核对实际空闲卡。
该示例沿用 v8 ranking 的发现目标、compute=loop，输出和 learned 参照分开。
若另测 batched，用相同 learned 参照的新结果目录，只添加
`--graph-adapter-compute batched`，不同时关闭 risk。暂不自动启动完整链。

## 对真实训练做有限窗口 profiling

`--profile-runtime-steps 50 --profile-runtime-warmup 20` 仅记录每个作业首次执行 epoch
中的 50 个 batch，之前预热 20 个 batch；随后释放 profiler，继续原训练计划。
它不改变 epoch 数、学习率、恢复位置或正式阶段完成标记，也不按测试成绩挑选权重。
这两个参数不进入不可变运行计划，可在同配置 resume 时临时开启。

窗口必须放得进一个 epoch。若首次执行 epoch 仍在 discovery warmup，EMA 教师
前向按原规则跳过；要测 warmup 后路径，应从已有 warmup 后 checkpoint 正常恢复。

已有 v8 ranking 结果、需继续该方案时，先通过 Git 同步代码。Linux / Bash：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new -s tree-m-v8-profile 'bash tools/owod/run_discovery_mowodb_3090_v8.sh --resume --profile-runtime'
```

原入口仍在第 8 轮暂停；若已完成 8 轮，没有新训练 batch，因而不产生 trace。
只有需要按原 50 轮计划继续同方案时，才按 v8 文档同时使用 `--full`。
此处没有新增一次性训练权重或替代数据，也不切换已有实验的 compute／risk。

日志与结果：

```text
output/tree-detr-m-v8-ranking-3090-logs/console.log
output/tree-detr-m-v8-ranking-3090/m-owodb/seed_42/stage_0/increment/
  profiles/epochNNNN-rankR-唯一后缀/
    summary.json
    operators.txt
    trace.json
```

```bash
tail -n 50 -f output/tree-detr-m-v8-ranking-3090-logs/console.log
```

每个 rank 独立保存，重复检查不覆盖旧 trace。`summary.json` 区分学生、旧类教师、
EMA 教师、候选、criterion、蒸馏、反向、优化器、原型更新、EMA 更新和取下一批数据。
同时记录 affinity 图和类别 adapter 的嵌套作用域。`cuda_events_recorded` 为 false
时未采集到 CUDA kernel 事件，不能根据 CPU 作用域或零 device totals 判断 GPU 开销。
CUDA trace 用于区分 CPU 处理、设备计算、拷贝、同步和通信；各作用域嵌套，
CPU／device totals 不能相加解释为整步占比。profiler 本身有额外开销，正式吞吐要
结合 profiler 关闭后的训练日志判断。不会因耗时超过估算而自动停止正式训练。

## 本地检查范围

相关回归共 **157 passed**（23 条已有弃用提示），日志保存在被忽略的
`output/runtime-optimization-20261110/verification.log`。

覆盖选择性教师输出与辅助头调用、候选、loss、梯度、AdamW、EMA／原型状态；
hard／soft／null 路由的 loop 与 batched 对比；disabled 链七个阶段与恢复校验；
默认旧 checkpoint／计划兼容、类别索引缓存、有限 profiling 窗口和随机状态。
真实 CUDA smoke 使用一次性小输入，仅检查计算路径，不作为真实检测效果证据。
服务器 NCCL、完整校准耗时、完整作者 AP 和服务器吞吐仍需实测。

2026-11-10 本机 RTX 5060 Laptop、torch 2.7.1+cu128 的一次性 bank 组件计时：
batch=2、query=100、dim=256、rank=8、两插入层，预热 10 次后计时 30 次，
CUDA 在窗口前后同步。3 个 stage 共 60 类时，loop／batched 前向约
9.96／2.98 ms，含反向、活跃分支检查和 AdamW 的模拟步骤约 35.26／12.67 ms。
这没有运行图像 backbone、检测器、真实数据或双卡通信；不是服务器端到端提速。
该次 batched 训练窗口峰值分配约 60.31 MiB，loop 约 58.47 MiB，也不代表完整模型
显存。一次计时不作稳定性能保证，生成报告位于被忽略的
`output/runtime-optimization-20261110/microbench.json`。

本机 CUPTI 初始化返回 `CUPTI_ERROR_INVALID_DEVICE`，因此 CUDA kernel trace
未采集成功；同步 wall-clock 组件计时独立于 CUPTI。输出已标记
`cuda_events_recorded=false`，没有将缺失 trace 当作 GPU 耗时为零或服务器验证通过。
