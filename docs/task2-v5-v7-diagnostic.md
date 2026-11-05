# Task 2 三轮效果对照

这是独立诊断，非正式成绩。复用完整v5 Task1第50轮和对应GNN，分别新建Task2，
比较v5与v7；不读取失败的Task2权重，不重跑Task1，不启动Task3/4。
v7保留v6的旧类补标与三组平衡router，增加受约束回收，不改变评分和定位冻结。

## 输入、机制与预算

- 作者S-OWODB全量Task2 increment清单，保留原图像身份、类别顺序、schema-3核验。
- 单张3090、batch2、seed42、普通训练增强；每组3轮increment，不做balanced finetune。
  作者均衡记忆只用于既有风险图构造；这次未完成完整Task2日程。
- AdamW分组沿用main逻辑：lr=2e-4、backbone lr=2e-5、weight_decay=1e-4，
  受掩码分类器零decay。三轮内LR保持不变，不加载Task1优化器；保留父配置warmup。
- 每组初始化及固定第三轮末使用完整作者评估集和既有VOC AP模式，记录Previous/
  Current/Known AP50、U-Recall、unknown precision/数量、WI@0.8、A-OSE。
  不用末轮前的测试指标选epoch或评分阈值。三轮结果仅用于观察明显失败和决定是否扩大训练。
- v7未知候选仍使用原匹配阈值；支持不足3的槽，32次观察未匹配后可回收；
  支持达到3的槽保护至1024次未匹配观察。全局每32次观察最多回收一个槽。
  v7时钟仅在提供有效匿名证据的观察中推进，known-only warmup不消耗匿名保护期。
  last_seen、clock、last_recycle和recycle_count持久保存。
  这些是项目假设，非论文参数；保护过长可能拒收新簇，回收仍可能丢失稀有知识。
- v5行为不变，v6原实现不变；新增v7在本诊断入口显式启用。
  现有v5 Task1记忆的初始promotion对两组相同，不验证v7从Task1开始学习后的转化能力。

单卡总batch2与之前双卡总batch4优化轨迹不同，本轮两组设置相同，不作为正式续训。
每轮保存独立诊断checkpoint、EMA、优化器、随机状态及loader状态；中断从最近完整轮恢复，
未完成轮重跑。权重顶层diagnostic_only=true，正式入口继续拒绝它们；不绕过正式版本校验。
同版本同输入的--resume会校验计划、源文件哈希和父权重身份，不允许改训练实现后接着续训。

## 运行

本地Windows/PowerShell：将本次提交通过Git推送。服务器Linux/Bash，既有tree-detr环境：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git pull --ff-only origin main
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
tmux new -s task2-v5-v7
bash tools/owod/run_task2_diagnostic.sh
```

入口锁定物理GPU2的UUID，启动前核对单卡CUDA与至少14 GiB空闲显存，并运行数据verify。
输出位于`output/task2-v5-v7-diagnostic-3epochs/`，console.log自动追加，
results/v5及results/v7分别保存run_config.json、initial_metrics.json、epoch_*.json、
checkpoint.pth和final_metrics.json；两组完成后生成results/comparison.json。
首次启动拒绝已有结果计划；恢复使用：

```bash
bash tools/owod/run_task2_diagnostic.sh --resume
```

另一个服务器终端：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
tail -f output/task2-v5-v7-diagnostic-3epochs/console.log
cat output/task2-v5-v7-diagnostic-3epochs/results/comparison.json
```

首轮epoch_0.json的seconds反映真实全量训练耗时；两组训练粗估为其6倍，另加两次初始化、
共四次完整评估，以及v7补标额外耗时。不能用重复32图诊断速度外推。
遇到非有限loss/梯度、OOM或来源校验失败立即停止，保留日志和上一完整轮checkpoint。
不要通过删校验、修改manifest或从诊断权重接正式链来恢复。
