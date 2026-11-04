# Discovery v6 调试记录（2026-11-04）

状态：实验性修正，尚无全量 AP 改善结论。默认仍为 v5，旧启动器和旧运行计划保持可恢复。
新方法通过 `--discovery-policy v6` 显式选择，版本为 6；正式链不能使用 v5 checkpoint。
旧权重可作为下述独立诊断的起点。诊断器不保存任何权重，不能产生正式阶段父节点。

## 已确认的失效

服务器 main 提交为 `024180d8c7d1270c1c8163b1c5f35ed914dc95c2`。
S-OWODB seed42：Task 1 第50轮与 Task 2 第5、8轮的匿名中心、计数、方差和
adapter 快照完全相同；已知新类计数正常增加。16个匿名槽已满，三个槽支持权重不足1。
v5对不相似且无空槽的候选直接拒绝，长期弱槽无法回收。
Task 2 的21类 promotion 全部 `inherited=false`，历史未知到新类的继承未发生。

Task 2 increment 只有当前类标注。核对原始训练图像30、36、49、64中的已释放旧类GT，
12个pseudo中3个与旧类IoU>0.5，另7个覆盖旧类的query接受objectness负目标。
这是局部失效复现，不是全量错误率。未使用未来类GT训练。

## 三项待验证修正

1. **固定容量的回收**：每次训练证据观察推进一个持久时钟。与现有原型匹配时照常更新；
   无匹配且满容量时，只在至少32次观察未接收证据的槽中选择支持最弱者，清空后建立新原型。
   不强行合并无关特征，不增加容量；每槽32次观察内最多被替换一次。
   时钟与last_seen随checkpoint和DDP缓冲区同步，替换后正常刷新adapter快照。
2. **router三组平衡**：背景、已知、匿名各自按有效权重归一化，再平均非空组，
   避免大量背景槽稀释匿名路由梯度。objectness的原有平衡BCE不变。
3. **旧类目标一致**：复用已有冻结Task teacher的输出，以sigmoid分数>=0.5、每图最多20框、
   类内NMS=0.7、当前GT排斥IoU=0.5补充旧类伪目标；随后统一匹配、已知监督、
   unknown候选排斥和原型语义。无新增teacher模型或前向；通常增量运行原本已为蒸馏计算该输出。
   单独关闭蒸馏的v6运行仍需要此teacher，不得把这类运行说成零额外teacher开销。

这些参数是本项目实验设计，不是作者论文设置。原型/候选容量、模型宽度、batch和正式数据协议不变。
评分与冻结定位暂不调整，以保持归因清晰。未盲目降低promotion阈值，也未保证新原型一定可迁移。
回收可能遗忘稀有未知簇，平衡router可能增加误检，teacher补标可能传播错误；必须联合评估
Known AP50、U-Recall、unknown precision、WI、A-OSE及实际promotion记录。

## 已运行验证

本地 Windows，既有 `tree-detr` 环境，RTX 5060 Laptop GPU（约8 GiB）。
使用服务器原始Task 1/2 checkpoint的只读副本，以及作者训练清单的确定性4图前缀，
保留图像/manifest/checkpoint哈希。每策略40次更新、单图batch、384短边、验证变换、
固定seed42、新AdamW lr=2e-5。重复训练流只是调试，不是正式划分或论文日程。
Task 2复原当前类参数冻结、adapter路由损失和旧类蒸馏，并核对冻结teacher的父节点哈希。

| 诊断 | v5 | v6 |
|---|---:|---:|
| Task1峰值allocated显存 MiB | 964.75 | 962.27 |
| Task1匿名中心变化 | 否 | 是 |
| Task2峰值allocated显存 MiB | 594.02 | 590.77 |
| Task2每步旧类正证据均值 | 0 | 1 |
| Task2匿名中心变化 | 否 | 是 |
| Task2稳定段live显存 MiB | 503.77–504.80 | 501.15–502.55 |

两组参数量相同，未发生OOM、非有限loss/梯度或持续显存增长。进程RAM采样均低于3.1 GiB；
Windows工作集受系统回收影响，不将其差异解释为内存优化收益。增加的持久记忆仅为16个last_seen
和一个时钟，共136字节int64缓冲区（每模型），不新增可训练参数。
资源护栏为每策略180秒、allocated显存6000 MiB、进程RAM8192 MiB；超限记录失败并退出。
异常退出不改动父权重。它是独立诊断器，不为训练主进程提供绝对的OOM保证。

报告位于忽略目录：
`output/discovery-v6-debug/paired-40-r4.json`、`task2-paired-40.json`。
早期`paired-40.json`加载失败；`paired-40-r2.json`意外更换了BN类型且未过warmup，不能作为证据。
调试加载器已改为只跳过冗余预训练初始化，保留原有FrozenBN结构；回归测试检查逐张量恢复。

## 重复本地诊断

执行目录为仓库根目录，使用既有tree-detr Python。fixture只包含已释放训练类别的xyxy像素框，
每条图像记录含相对文件路径及SHA256，顶层含manifest_sha256；样本文件保存在同目录。

```powershell
& C:/Users/23642/miniconda3/envs/tree-detr/python.exe tools/owod/debug_discovery_revision.py `
  --checkpoint output/discovery-v6-debug/task2-v5.pth `
  --old-teacher-checkpoint output/discovery-v6-debug/task1-v5.pth `
  --fixture output/discovery-v6-debug/task2-fixture.json `
  --output output/discovery-v6-debug/task2-paired-repeat.json --steps 40
```

程序stdout为策略摘要，详细测量保存在独立JSON。已有报告不覆盖，不保存诊断checkpoint。
相关测试包括原型回收/恢复、非有限证据、router背景数量不变性、旧类目标保护、
v5兼容、真实CUDA检测器增量反向/权重恢复、跨主机预训练路径与FrozenBN一致性。

## 尚未完成的效果验收

### 服务器下一步入口

补充验证：本地Task2真实checkpoint在800短边、batch2下完成每策略40步，
v5/v6峰值allocated显存分别1594.36/1589.96 MiB，用时45.11/45.86秒，原权重哈希不变。
双进程Gloo测试确认匿名中心、计数、方差、时钟及last_seen在40次同步后完全一致。
这些仍不能替代Linux双卡NCCL实测，且未使用训练随机增强或原优化器状态。

代码通过Git同步到服务器main后，在Linux/Bash既有tree-detr环境执行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new -s discovery-v6-debug
bash tools/owod/run_discovery_debug.sh
```

该入口锁定已核对的物理3090卡1、2的UUID；先验证CUDA可用且每卡至少空闲14 GiB，
再验证S-OWODB数据。随后读取现有Task2第8轮与其真实Task1父权重，
直接从作者increment训练标注读取16张图像，不依赖本地fixture文件。
使用800短边、两卡每卡batch2；两种策略各40步，仍为新AdamW lr=2e-5的独立诊断。
每策略300秒，整个torchrun最多15分钟；每卡allocated显存上限12000 MiB，
每rank进程RAM上限8192 MiB。资源检查是步间护栏，不能保证算子内部绝不OOM；
若失败，torchrun终止其他rank，保留错误报告，不继续正式训练。

每次自动建立独立目录`output/discovery-v6-server-debug/run-XXXXXXXX/`，包含
`console.log`、`paired.rank0.json`和`paired.rank1.json`。控制台首先打印实际目录。
另一个终端查看：

```bash
ls -td output/discovery-v6-server-debug/run-*/ | head -n 1
tail -f output/discovery-v6-server-debug/run-XXXXXXXX/console.log
```

将XXXXXXXX替换为本次实际目录。成功必须两rank均completed、resource_guard_passed、
checkpoint_unchanged和old_teacher_unchanged为true；只通过资源检查不自动批准正式全量训练。
下一步仍需针对旧类误入unknown的变化、原型支持和可迁移性做效果对照，不能只看loss。

这些证据说明机制恢复及小规模运行稳定，不能证明未知检测有效、全量AP提升或方案最优。
未验证服务器800短边、双卡每卡batch2的峰值资源，也未验证Linux NCCL下的新状态同步；
本地CUDA不等于服务器CUDA。正式v6需要同一版本从DINO骨干重新完成Task1到Task4。
服务器源码未修改、未启动正式训练，旧v5输出与原始checkpoint均保留。
