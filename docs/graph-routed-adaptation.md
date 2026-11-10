# 图风险引导的选择性适配

当前实现把 GNN 从回放选择器扩展为 **adapter 的前向与更新控制器**。
这是项目研究方法，使用 M-OWODB / S-OWODB 作者数据协议；不是 ORE 或
OW-DETR 原方法，也未证明论文创新性或实验提升。

可选批量计算及完整跳过控制器的 `--graph-risk-mode disabled` 见
[运行效率说明](owod-runtime-optimization-v3.md)。后者是独立消融，不改变默认 learned 链。

## 三个模块

1. Class-Interference GNN：用类别梯度 sketch 预测有向干扰矩阵。
2. Persistent Class Adapter Bank：从 Task 2 起，为新类在末两层 decoder
   FFN 旁新增低秩分支；旧分支保留在 checkpoint 和推理中，不再合并。
3. Risk-Conditioned Query Routing：query 自身选择类别分支或 null，GNN
   风险与冻结旧类证据共同调节分支输出及梯度。

旧阶段共享 LoRA 入口仍用于历史算法对照和校准 probe；标准启动器通过
`--method graph-adapter` 使用上述新实现。

## 前向与学习

设当前阶段新类集合为 N，旧类集合为 O。每个插入层、每个增量阶段有
独立 router，输出 N 类及 null 的 softmax 概率 p。只启用 argmax 对应分支；
null 胜出时该阶段不产生更新。推理不接收 GT、不输出未来类别名。

GNN 输出 H[c,o] 表示“更新新类 c 对旧类 o 的相对损失干扰”，不是语义
相似度，也不是校准的遗忘概率。冻结的上一阶段 classifier 快照对当前
query 产生旧类证据 q[o]。实际门控为：

```text
r[c] = max_o H[c,o] * sigmoid(old_classifier(query)[o])
g[c] = p[c] * 1[argmax(p) == c] * exp(-beta * r[c])
FFN residual += sum_stage sum_c g[c] * B[c] A[c] query
```

同层各阶段 adapter 使用相同输入，互不串接。跨层 query 仍可能受新分支
影响，因此冻结旧参数不等于保证旧类输出不变或完全不遗忘。

检测损失学习分类与定位；附加 router loss 使用现有 Hungarian 匹配：
该分支所负责类别的匹配 query 学习对应 route，其他类别及未匹配 query
学习 null。正样本和 null 各自取均值后平均，防止空 query 数量压倒正样本。
GT 只用于损失。router 使用 detach 的 query，路由监督不反向改写上游特征；
检测损失仍可经被选 adapter 反向传播。初始 B=0，router 可先从 null 学习
识别新类，随后激活其分支，避免依赖人工给定 query 类别。

未选分支虽然参与张量计算以兼容 DDP，但 backward 后会将其梯度设为
None；该步不执行 AdamW 衰减或已有动量更新。分布式训练先同步全局分支
活跃标记：任一 rank 选择的分支在所有 rank 一致更新。

## 阶段与数据来源

| 阶段 | 可更新参数 | 图控制器来源 |
| --- | --- | --- |
| Task 1 | 原始 detector 密集训练 | 无旧类，bootstrap |
| Task 2–4 increment | 当前新类 adapter、当前 router、新类 classifier 行 | 已完成上一任务的 GNN |
| Task 2–4 balanced finetune | 所有已建立 adapter/router、已见 classifier 行 | 沿用本任务已保存图，分支不重复增长 |

Task 1 后 backbone、基础 FFN、box head 保持冻结；新类定位通过 decoder
adapter 学习。官方均衡微调阶段同样保持这些基础参数冻结。此参数策略属于
项目方法设计，不能称作作者原模型训练配置。classifier 的 masked row 使用
零 weight decay（Task 1 保持原始密集优化器）；每个阶段新建优化器，防止
旧行动量跨阶段改变旧参数。训练日志记录 `loss_adapter_route` 和
`adapter_active_fraction`（可训练 bank 的非 null query 比例）。

标准启动器在 Task 1 结束及 Task 2/3 均衡微调结束后校准 GNN。校准仅用
已完成任务的训练标注和临时共享 LoRA probe 的类别损失增加，不访问未来类
或验证集标签。它估计的是共享更新的风险代理，并非直接测得 class adapter
的因果干扰；这项近似必须通过消融和真实任务验证。

进入下一 increment 时，在冻结 detector 上重新提取梯度 sketch：新类来自
当前官方 increment 清单，旧类来自当前官方 balanced memory 清单。
这是控制器特征统计，不改变 detector 增量训练 loader、图片预算或采样比例。
GNN checkpoint 必须匹配上一阶段、原 detector checkpoint SHA256、manifest
SHA256、梯度层范围，且完整完成经验监督校准。旧 pilot GNN 被拒绝。

完整 bank 布局、风险矩阵、源文件哈希保存在 checkpoint args 和实验目录的
`graph_adapter_config.json`。恢复时严格重建布局；同阶段微调复用分支，断点
恢复复用已保存图。checkpoint 链不能切换方法或修改研究超参数。

## 启动

先按 [数据准备指南](data-preparation.md) 完成 prepare/verify。示例沿用
OW-DETR 的训练 schedule；权重与 manifest 路径换成已验证的实际文件：

```bash
python tools/owod/run_benchmark.py \
  --manifest data/owod/s-owodb/split_manifest.json \
  --output-dir exps/owod/graph-adapter \
  --recipe ow-detr-paper --backbone-weights pretrained/dino_resnet50.pth \
  --method graph-adapter --dry-run
```

去掉 `--dry-run` 执行；中断后加 `--resume`。M 使用独立 manifest 和输出链，
若选 `ore-paper` 仍须明确提供该 recipe 未规定的独立微调 epoch 数。
默认 `--method control` 保留基础模型对照，禁止与新方法共用输出目录。

方法默认超参数：rank=8、末两层、beta=2、router loss coefficient=1、
每类最多 12 张训练图片用于 sketch。GNN 校准默认 probe 3 步、lr=1e-4、
每类最多 12 张，GNN 400 epochs；完整值见启动计划与校准 run_config。
这些都是研究配置，不是 M/S 数据协议规定的参数。beta=0 可用于关闭图风险
门控的消融；应与同 bank/router 的 beta>0 对照。

需要比较：基础 control、beta=0 的路由 bank、有图门控的 bank，并记录
Previous/Current/Known AP50、WI、A-OSE、U-Recall、参数量及路由活跃率。
特别检查 router 长期停留 null、风险过高抑制新类学习，以及校准 score
跨阶段泛化失败。这次实现不更改原有 unknown scoring，也不等于无标注
发现或命名新类。“自主”仅指按 query 与图风险选择适配路径及更新范围。

当前验证覆盖 CPU 上的分数影响、选择性梯度、AdamW 动量跳过、旧参数冻结、
路由监督启动、真实 decoder FFN/criterion 接入、bank 增长与 state_dict 重建。
尚未运行真实 M/S 完整训练或多 GPU 性能实验。
