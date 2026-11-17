# S-OWODB：未知检测诊断与3轮快速检查

更新：2026-11-17。适用于main的S-OWODB v7 Task2诊断，不适用于mowodb分支v8/v9。

## 当前应该先跑什么

当前新增加的是未知检测只读诊断工具，还没有修改训练损失或未知评分规则。
已有附件包含frozen、bbox-last两组第3轮结果；只把相同配置改为3轮重跑，不构成新优化对照。
建议先对已完成的frozen第8轮checkpoint运行256张只读诊断，确定objectness、novelty、
框覆盖、Top-K或同框已知/未知输出中哪一环节值得修正，再用3轮比较下一次独立改动。

显式 `--epochs 3` 已接入现有routing/localization启动器，便于需要重跑时快速检查。
它是新建的短程increment计划，不把现有8轮计划改成3轮，不接续旧诊断权重。

## Git同步

本地Windows / PowerShell，仓库根目录。只提交本轮及上一轮的诊断相关文件，
不包含用户已有的AGENTS.md、research/selection.md及M论文比较文档改动。

```powershell
cd C:\programlearning\Tree-DETR
git add -- tools/owod/run_sowodb_task2_routing.py tools/owod/run_sowodb_task2_routing.sh `
  tools/analysis/audit_unknown_detection.py tools/owod/tests/test_task2_short_budget.py `
  tools/owod/tests/test_unknown_detection_audit.py docs/owod-unknown-recall-priority.md `
  docs/sowodb-task2-fast-check.md docs/sowodb-task2-routing-experiment.md `
  docs/sowodb-task2-localization-experiment.md
git diff --cached --stat
git commit -m "Add isolated three-epoch S Task2 diagnostics and unknown audit"
git push origin main
```

若旧8轮实验还在运行，不要在它的checkout中拉取新代码。可以先让旧运行结束，
或在其tmux中Ctrl+C正常退出并保留全部checkpoint/日志，再同步。
修改启动器改变了诊断计划绑定的Python源码哈希，旧8轮即使预算不变也不能用新源码恢复；
附件对应旧提交为`34af21ef3cc2b02531f50cf543d8cc194dbea36c`，恢复仍须匹配该运行原有身份。

服务器Linux / Bash，确认该checkout为main，且旧运行已结束或正常退出后执行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git branch --show-current
git rev-parse HEAD
git status --short
git pull --ff-only origin main
nvidia-smi
```

## 推荐：先跑未知检测只读诊断

既有tree-detr环境；GPU2只在附件中确认是3090，运行前必须确认实际空闲。
已完成frozen组的`checkpoint.pth`应保持不再被训练覆盖；工具结束前会重复核验哈希。

```bash
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
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

256张诊断预留5–15分钟，以每50张打印的实测吞吐修正。JSON若已存在会拒绝覆盖，
需要重新诊断时换用新的JSON和日志名。该前缀只定位问题，不报告正式AP/U-Recall。
具体字段解释见[未知检测优先](owod-unknown-recall-priority.md)。

## 需要重新跑定位对照时：各3轮

使用完整v5 Task1第50轮及对应GNN重新建立两组，保持soft路由、冻结/末层定位对照、
作者完整S Task2清单、seed42、每卡batch1、双3090总batch2、原学习率和评分。
这仍是已有定位容量对照，不能称作新增未知学习机制的效果验证。

确认GPU0/1均可用，再在服务器仓库根目录执行：

```bash
bash tools/owod/run_sowodb_task2_localization.sh --epochs 3 --check
tmux new -s s-task2-loc3 'SOWODB_GPUS=0,1 bash tools/owod/run_sowodb_task2_localization.sh --epochs 3'
```

`--check`只核验完整数据/父权重/GNN/历史参考，输出应为`epochs: 3`、
`evaluation_epochs: [3]`以及新的`...localization-3epochs-2gpu/results`目录。
实际启动先检查两卡、扩展和两步CUDA/NCCL路径，再训练。
如果此前设置过`SOWODB_TASK2_OUTPUT`，应检查输出是否指向专用新目录；不要复用旧8轮目录。

新输出：`output/sowodb-task2-v7-localization-3epochs-2gpu/`。

```bash
tail -f output/sowodb-task2-v7-localization-3epochs-2gpu/console.log
cat output/sowodb-task2-v7-localization-3epochs-2gpu/results/comparison.json
```

`results/frozen/`与`results/bbox-last/`分别保存初始指标、第3轮完整作者评估及训练图probe、
参数审计和checkpoint。`comparison.json`是本次第3轮末结果，`interim=false`表示
本次短预算结束，不表示正式Task2完成。没有第8轮文件或3→8轮差值。

中断后在同一源码/输入/GPU/预算下执行相同命令，自动恢复自己的3轮计划；
tmux会话仍在时先`tmux attach -t s-task2-loc3`，不另启第二份。
8轮计划不能续成3轮，3轮计划也不能直接扩展为8轮；需建立独立计划。

需要hard/soft路由快速对照时改用：

```bash
bash tools/owod/run_sowodb_task2_routing.sh --epochs 3 --check
tmux new -s s-task2-route3 'SOWODB_GPUS=0,1 bash tools/owod/run_sowodb_task2_routing.sh --epochs 3'
```

它使用单独的`output/sowodb-task2-v7-routing-3epochs-2gpu/`。两类对照不要同时占用同一组GPU。

## 3轮能验收什么

缩短预算不改变前三轮的优化器、学习率、batch、增强或warmup日程；
第1轮为discovery warmup，实际匿名未知学习只有后两轮。已释放的新类监督从第1轮参与。
因此可以检查明显退化和候选/监督方向，并比较相同预算末轮指标；不证明充分训练或完整任务收益。
改变源码后须从同一有效Task1起点独立建立配对，不能将不同父权重/预算的结果当作单因素收益。

按附件每轮约1.7–2小时，3轮约5–6小时/组，两组约10–12小时纯训练；
包括数据核验、初始化和完整评估，预留12–16小时，以首轮实际耗时修正。
输入/初始化不一致、非有限loss/梯度、OOM或设备故障停止，保留已有成果。
不因中途测试AP低就修改本次固定预算或评分阈值。

看第3轮末Previous/Current/Known AP50、U-Recall、unknown precision/数量、WI@0.8、A-OSE，
不以loss下降或pseudo数量增加替代效果。正式新链仍需要完整increment与作者balanced finetune。

本地验证：55项相关检查通过，包括3/8轮身份、输出、评估及报告隔离、跨预算恢复拒绝，
以及真实小型Deformable DETR CUDA更新和state重载。1项既有Windows双进程Gloo检查未重跑；
7条提示为既有torchvision/meshgrid提示。Bash语法和CLI帮助检查通过。
未在目标Linux双3090运行新增3轮入口，也未启动真实3轮长训练或取得优化后的AP。
