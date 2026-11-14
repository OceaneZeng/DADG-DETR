# M-OWODB 全任务训练：v8 ranking / focal 与低学习率配方

2026-11-14：focal Task 1 第 8 轮 Known AP50=42.1399%，高于 ranking 同轮
36.0015%。用户要求继续进行 M-OWODB 全量实验，重点观察 Task 2 新类学习。
本入口运行完整作者 M 数据及四任务七阶段；支持 v8 ranking 或 focal 评分与原有数据/阶段校验，
将所有检测训练阶段的学习率缩小至原配方的 1/4。低学习率的前 8 轮 control 证据见
[结果分析](owod-control-lr-quarter-epoch8-analysis-20261111.md)。

这是项目完整方法的一条独立运行链。M 数据协议来自 ORE；50/20 轮日程沿用
OW-DETR Appendix D，学习率属于本项目稳定性适配，不能称为 OW-DETR 原始优化设定
或外部方法复现。完整方法及 Task 2 的实际 AP 尚未验证。

## 固定配置与已有成果

- 作者 M-OWODB：每个任务新释放 20 类，schema-3 manifest 与输入哈希核验。
- DINO ResNet-50、100 queries、seed 42、单张 NVIDIA A40（默认物理 GPU 0）、batch 4、
  全局 batch 4、
  activation checkpointing；沿用既有 Linux `tree-detr` 环境。
- v8 ranking：hierarchical 评分；v8 focal：已有 focal 评分；两者均使用 balanced-bce objectness、count novelty 归一化、
  3 轮 pseudo ramp、排序权重 1、发现输入梯度系数 1。
- hard routing、frozen 定位策略、图风险控制、discovery adapter、原型与快照继承、
  旧类蒸馏和作者均衡微调保持当前实现。
- 增量主 lr=5e-5，backbone/采样偏移/参考点 lr=5e-6；均衡微调主 lr=5e-6，
  对应基础参数组 lr=5e-7。Task 2 之后部分基础参数冻结，冻结参数不参与更新。
- interference GNN 校准是独立目标，其 probe/GNN 优化器不受检测学习率缩放影响。

本次根据服务器实际硬件改为单卡 A40：双 3090、每卡 batch 2 改为单卡 batch 4，
有效 batch 仍为 4。这样保持每次优化更新的样本数不变，但 DDP 梯度归约、样本顺序、
单步耗时和浮点累积顺序会变化，因此这是新的 A40 全量运行，不能与此前双 3090
暂停权重直接续训，也不应把两种硬件下的逐轮数值当成严格复现实验。若使用另一张
A40，启动前设置 `GPU_INDEX=1`；同一条链恢复时保持 GPU 选择和其他配置不变。

本链从相同 DINO 骨干初始化，重新完成 Task 1 的 50 轮，再进入 Task 2；不载入此前
8 轮 control、旧 v8 诊断权重或其他版本。学习率改变了优化轨迹，属于新运行，不能
以旧高学习率 checkpoint 续训。现有诊断目录、权重及结果保留。

新增 `run_benchmark.py --discovery-lr-scale 0.25` 把各 increment/finetune 的实际 lr
写入命令、配置和 checkpoint，缩放及来源进入不可变计划。默认 1 不改变旧计划。
恢复仍需同一计划与实际 lr；方法版本保持 8，因为本次没有改变模型结构或损失实现。
原 `--control-lr-scale` 继续仅用于 control，不混用两种选项。

## 执行顺序

| Task | 代码目录 | 数据用途 | epochs | 主学习率 |
| --- | --- | --- | ---: | ---: |
| 1 | `stage_0/increment` | 16,551 张作者训练图 | 50 | 5e-5 |
| 2 | `stage_1/increment` | 45,520 张作者增量训练图 | 50 | 5e-5 |
| 2 | `stage_1/finetune` | 1,743 张作者均衡记忆图 | 20 | 5e-6 |
| 3 | `stage_2/increment` | 39,402 张作者增量训练图 | 50 | 5e-5 |
| 3 | `stage_2/finetune` | 2,361 张作者均衡记忆图 | 20 | 5e-6 |
| 4 | `stage_3/increment` | 40,260 张作者增量训练图 | 50 | 5e-5 |
| 4 | `stage_3/finetune` | 2,749 张作者均衡记忆图 | 20 | 5e-6 |

合计 260 个训练 epoch，Task 2–4 开始前各有一次前序任务的 interference GNN 校准。
上述图数于 2026-11-11 从服务器已注册数据只读核对；不改变作者清单或训练预算。
每个训练 epoch 使用完整作者测试清单评估，不以少量诊断图代替 AP。
阶段只接受完成前一阶段的对应权重，校准产物也绑定源 checkpoint 与 manifest。

旧 `run_discovery_mowodb_3090_v8.sh --full` 仍仅完成 Task 1；本次必须使用下述新入口。
新入口不会在第 8 轮暂停，会完成整条任务链。训练故障时返回错误码并保留日志，
不要绕过阶段/版本/哈希校验强行进入 Task 2。

本次 focal Task 1 的 8 轮结果已经高于 ranking 同轮，但该暂停权重属于 Task 1-only
计划，不能直接改成全链计划或跨目录续训 Task 2。正式全量 focal 链使用独立的
`tree-detr-m-v8-focal-lr-quarter-a40-full` 目录，从 DINO 重新开始 Task 1，保证
七个阶段共用同一份完整计划和评分配置。原 `...-task1` 权重和日志保留为优化对照。

## 启动和恢复

先在本地 Windows 仓库提交并推送本次源码和文档，用户通过 Git 同步服务器。
不把本地 `output/analysis/`、日志、图像或权重加入 Git。服务器 Linux / Bash：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
git pull --ff-only origin mowodb
nvidia-smi
bash tools/owod/run_discovery_mowodb_3090_full.sh focal --dry-run
```

确认选定的单张 A40 没有其他训练；默认使用物理 GPU 0，也可以用 `GPU_INDEX=1` 选择
另一张 A40。检查 dry-run 共打印 7 个训练阶段和 3 个校准命令，
增量/微调主 lr 分别为 5e-5/5e-6，Task 2 路径为 `stage_1`。启动：

```bash
tmux new-session -d -s tree-m-full-focal 'bash tools/owod/run_discovery_mowodb_3090_full.sh focal'
tail -n 50 -F output/tree-detr-m-v8-focal-lr-quarter-a40-full-logs/console.log
```

`Ctrl+C` 退出这里的 tail 不会停止 tmux 内训练。进入训练终端：

```bash
tmux attach-session -t tree-m-full-focal
```

在 tmux 中按 Ctrl+B，松开后按 D 离开。中断后，同一 tmux 训练已结束、GPU 空闲时恢复：

```bash
tmux new-session -d -s tree-m-full-focal-resume 'bash tools/owod/run_discovery_mowodb_3090_full.sh focal --resume'
tail -n 50 -F output/tree-detr-m-v8-focal-lr-quarter-a40-full-logs/console.log
```

恢复跳过已有有效完成标记的训练阶段，重新核验/复用校准缓存，并接续当前阶段的
optimizer / scheduler / EMA 等状态。始终追加同一份 console 与阶段日志。
不要同时启动初次命令和恢复命令。入口拒绝新启动覆盖已有计划，并拒绝没有计划的恢复。

## Task 2 的新类学习如何观察

Task 2 开始时，代码使用已释放的新类 GT 初始化当前 router，按支持与 affinity 条件
尝试匿名快照继承；继承条件不满足时允许从零学习。increment 更新当前 20 类的分类行、
新增类别 adapter/router 及 discovery 路径，旧类别分支冻结，保留旧类教师蒸馏。
finetune 使用作者均衡数据，开放全部已知类别 bank/分类行。
这些是实现路径，是否学会新类必须由实际 AP 判断。

重点查看 `stage_1/increment` 和 `stage_1/finetune`，分别记录每轮：

- Current AP50：Task 2 新释放的 20 类，是新类学习的主要证据。
- Previous AP50：Task 1 旧 20 类，观察新类学习的遗忘代价。
- Known AP50：当前全部 40 类，不能单独用于判断新类是否学会。
- U-Recall、WI@recall=0.8、A-OSE：检查剩余 40 个未释放类别的未知发现/误识别。

运行中从仓库根目录执行以下只读查看命令；尚未进入 Task 2 时会明确提示：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('output/tree-detr-m-v8-focal-lr-quarter-a40-full/m-owodb/seed_42/stage_1')
for phase in ('increment', 'finetune'):
    path = root / phase / 'metrics.jsonl'
    print('\nTask 2', phase)
    if not path.is_file():
        print('not started')
        continue
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    for r in rows[-10:]:
        print('epoch', r['epoch'] + 1, {
            k: round(100 * r['test_owod_' + k], 3)
            for k in ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall')
            if 'test_owod_' + k in r})
PY
```

AP/Recall 在保存文件中为 0–1，命令乘 100；M 使用作者 VOC 积分 AP50，不使用 COCO
101 点 AP50。上面显示最近 10 轮仅用于观察，不按测试集挑选最好权重代替固定末轮。
Task 2 的 increment 和 finetune 末轮分别报告，避免混淆两阶段作用。

如果 Current AP 持续提高，同时 Previous AP 未严重崩溃，才支持当前链具备有用的新类
学习能力。若只有 Previous / Known 看起来较高而 Current 接近 0，仍需排查新类学习。
`graph_adapter_config.json` 中 `specs[].promotion` 和 router 初始化记录可以核对转化
是否执行；发生继承或新类 AP 大于 0，均不能单独证明匿名快照迁移带来了增益，
此项因果结论需要相同配方的 scratch 对照。

## 时间预算、停止条件与验证边界

先前完整 v8 Task 1 前 8 轮用时 4:14:42，当前低学习率 control 每轮约 36–38 分钟。
按这些实际吞吐，Task 1 的 50 轮暂预留约 27–35 小时。首次 Task 2 指标还需等待
GNN 校准、当前 adapter/router 初始化和 Task 2 首轮；这些没有本次全量实测耗时。
Task 2 增量图片数约为 Task 1 的 2.75 倍，同时冻结与教师计算改变每步开销，
不把 Task 1 每轮耗时直接套给 Task 2。完整链需按多天预算准备，在进入 Task 2 后
用实际迭代速度更新估计；没有承诺精确完成时间。

用户已选择直接运行全量任务链，不再强制增加一轮 8-epoch 前置诊断。仍需关注
Task 1 的基本学习和 Task 2 的 Current AP；若非有限 loss、OOM、阶段校验失败，
保存日志和已产生的 checkpoint 后定位，不更改配置强行恢复。若持续出现已知 AP
崩塌、unknown 几乎占满输出或新类 AP 长期接近 0，正常停止并分析对应权重，避免
用后续任务掩盖失败。首轮未知召回为 0 可能来自 warmup，不能单独作为失败条件。

本地直接相关回归：75 passed，4 条既有 scheduler 弃用提示。覆盖完整七阶段 lr
缩放和顺序、前序权重与三次校准衔接、旧默认计划兼容、恢复拒绝学习率错配、启动器
数据失败即停、日志及已有成果保护，以及原 control/v7/v8/strict setting 相关回归。
另通过 SSH stdin 在服务器内存中执行本地 runner 的只读 dry-run，使用真实 M 数据
生成并核对完整计划；没有修改服务器源码/计划，也没有启动训练。
实际 full-chain 训练、Task 2 吞吐、AP、增量保持与完整方法的低学习率稳定性均未验证。
