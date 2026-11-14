# S-OWODB Task 2 新类学习诊断与路由对照

这是独立诊断，采用作者完整 S-OWODB 数据和 VOC2007 11 点 AP50，非正式四任务成绩。
沿用已有完整 v5 Task 1 第50轮及其 GNN，两张3090、每卡 batch1、总 batch2、seed42。
默认只比较 v7 hard/frozen 与 v7 soft/frozen，各3轮 increment，不做 balanced finetune。
本页的 v7 是 main 的旧类补标、发现路由平衡和受约束原型回收，不是 M 分支的 objectness focal v7。

## 实验依据与顺序

服务器旧双卡对照 `output/task2-v5-v7-diagnostic-3epochs-2gpu/results/comparison.json`
已经完整结束。v5/v7 的 Previous AP50 为21.83%/59.80%，Current AP50 为2.48%/2.88%，
Known AP50 为11.67%/29.92%。该结果支持优先定位新类瓶颈，尚不支持启动 v7 正式链。

入口依次完成：

1. 核验当前数据、Task1、GNN、历史v7诊断来源，记录独立运行计划。
2. 只读历史v7第三轮：在当前实现下完整评估作者测试集，并测量已释放Task2训练标注上的
   框覆盖、评分候选竞争、匹配query的正确路由/null路由及实际gate。诊断不更新权重。
3. 从同一完整Task1独立新建 hard/frozen、soft/frozen 两组，各3轮，初始化权重和原型哈希必须相同。
4. 每组初始化、固定第三轮末评估完整作者测试集，另记训练标注诊断和参数变化。
   只在第三轮结束后作配对比较，不选最优测试epoch或更改评分阈值。

训练标注诊断按当前21类分别取作者训练序列中最先出现的12张图像，再去重并保序，最多252张。
它只保证检查各类有输入，不产生新benchmark划分，不是代表性召回或AP。
框覆盖使用归一化query框IoU≥0.5；`joint`遵循当前hierarchical评分、unknown阈值和Top50，
`known_only`只移除unknown候选竞争，`classifier_only`还移除objectness/novelty并换用sigmoid。
后两者只是同一forward上的诊断覆盖，不能作为正式AP或用于改动本次训练的评分。
路由统计使用已标注新类的Hungarian匹配，不加入teacher补标，因而不是训练全部匹配的复刻。
参数变化和非零adapter只能说明更新发生，不能代替有效检测证据。

除类别adapter的hard/soft路由外两组保持一致：hierarchical评分、冻结定位、v7发现机制、
旧类蒸馏、完整作者清单、初始化、增强、warmup1轮、AdamW lr=2e-4、backbone lr=2e-5、
weight_decay=1e-4、masked classifier零decay，三轮内无学习率衰减。
相同初始权重不保证两种路由的优化轨迹相同；对照测量的就是该差异。
仍使用v5 Task1的同一匿名记忆，不检验v7从Task1学习后能否改善unknown到新类的转化。
定位消融和学习率消融暂不启动。

## 原始manifest的严格等价核验

历史Task1和GNN记录manifest `13a1f3b3…`；当前Git内便携schema-3 manifest为`0060c5f7…`。
`task2_parent_provenance.py`从当前manifest恢复历史Linux路径，并仅去掉末尾新增的
`splits`/`split_hashes`字段，按原序列化规则逐字节重建旧文件。
重建SHA-256必须与父checkpoint记录完全相同，16个阶段文件仍按当前完整作者清单和哈希核验，
GNN仍核验其原始Task1权重、训练标注、类别、层数及无验证标签来源。

严格父节点验证使用单独的参数副本及已证明的原始身份；新诊断训练、计划与恢复记录当前身份
和完整证明。不会修改Task1/GNN/checkpoint中的Namespace，也不修改正式入口的校验。
该能力仅用于本独立诊断，输出checkpoint顶层仍为`diagnostic_only=true`，正式入口继续拒绝。
任意标注、评测、类别或源哈希改变且不能精确重建原hash时停止，不提供忽略校验选项。

## 启动与恢复

本地代码经Git同步到服务器后，服务器Linux/Bash执行：

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
git pull --ff-only origin main
tmux new -s s-task2 'bash tools/owod/run_sowodb_task2_routing.sh'
```

脚本自动使用已核实的 `/home/top/disks/new-hdd/conda_envs/tree-detr/bin/python`，无需另建环境。
已核实该环境为PyTorch2.4.1/CUDA12.1，Conda内有匹配的nvcc和GCC12。脚本沿用它们，
不切换系统CUDA13.x/GCC13。若清理后仅剩不兼容的旧egg扩展，会以3090的8.6架构
重建项目内的忽略产物`.so`并验证导入，不安装/替换PyTorch或修改环境包。
每次训练入口还先运行两步随机小输入的双进程hard/soft CUDA检查，覆盖前后向、NCCL、EMA、
原型同步及每rank随机状态恢复，不保存权重，也不把此检查视为AP验证。
默认PCI顺序GPU0、1，要求恰好两张3090且各自至少14 GiB空闲；2026-11-14读取服务器时这两张卡空闲。
GPU和空闲状态以启动时检查为准。确需更换GPU，在新计划首次启动前设置`SOWODB_GPUS=…`；
恢复绑定同一GPU选择、双进程、总batch、父节点、参考权重、manifest及Python源码哈希。
PyTorch/torchvision/CUDA版本和实际扩展二进制哈希也写入恢复身份；编译产物改变时拒绝续训。
这次总batch2沿用已完成诊断，比历史正式S运行的总batch4小，不能混作相同优化轨迹。

仅核验输入、不占用GPU训练：

```bash
bash tools/owod/run_sowodb_task2_routing.sh --check
```

首次运行新目录；再次执行同一脚本会在计划完全一致时自动恢复，跳过已完成的参考诊断和组别，
从最近完整轮checkpoint恢复优化器、EMA及每rank随机/loader状态，未完成轮重跑。
源码或输入改变时拒绝续跑；不要删计划强接旧checkpoint。
同时运行同一目录会由`flock`拒绝。`Ctrl+B`再按`D`离开tmux，`tmux attach -t s-task2`返回。
若原tmux会话已结束，用相同启动命令重新建立；仍存在时先attach，不另启第二份。

## 输出与预算

根目录 `output/sowodb-task2-v7-routing-3epochs-2gpu/`：

- `console.log`：来源、完整训练/评估日志，恢复时追加。
- `results/diagnostic_plan.json`、`parent_provenance.json`：计划及manifest等价证明。
- `results/reference/`：旧权重当前完整评估、训练标注诊断及不改权重证明。
- `results/hard/`、`results/soft/`：初始化身份、初始/末轮指标、训练标注诊断、
  每轮训练统计、checkpoint、最终参数审计。
- `results/comparison.json`：固定第三轮的Previous/Current/Known AP50、U-Recall、
  unknown precision/数量、WI@0.8、A-OSE和soft减hard差值。AP/recall/precision为0–1值。

```bash
tail -f output/sowodb-task2-v7-routing-3epochs-2gpu/console.log
cat output/sowodb-task2-v7-routing-3epochs-2gpu/results/comparison.json
```

历史双卡三轮训练约4.8–4.9小时/组。两组约10小时纯训练，连同初始化、参考诊断和评估预留12–16小时，
实际以首轮`epoch_0.json`的`seconds`修正。出现OOM、非有限loss/梯度、来源失败或初始化不一致立即停。
三轮末若soft改善Current且Previous/Known和未知指标可接受，再决定更长预算；框覆盖低时单独设计
`bbox-last`定位消融。最终验收必须完成独立的Task2 increment＋作者balanced finetune和正式新链，
本诊断权重不能作为正式链父节点，不会自动开始Task3/4。

## 本次验证范围（2026-11-14）

本地30项相关检查通过，覆盖历史manifest逐字节还原、内容变化拒绝、正式入口仍拒绝跨manifest及
诊断checkpoint、路由/评分覆盖计数、配对身份、旧分类行保护、真实检测器CUDA反向及权重恢复。
既有一项Windows双进程Gloo检查超时；在目标Linux服务器同一项检查通过。
另在服务器两张3090上运行了本次新增的hard/soft随机输入检查，前后向、NCCL、EMA、原型同步
及每rank随机状态恢复均通过；真实v7权重完成一张训练图的诊断，模型权重状态哈希未改变。
Task1/GNN及16份阶段文件已在服务器只读核验；扩展已用现有CUDA12.1/GCC12编译并通过前向。
这些检查不证明AP提升。本次完整参考评估、三轮hard/soft训练及其端到端恢复仍待用户启动。
