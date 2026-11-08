# Discovery：受信任 background 的 novelty 负监督

## 状态与目的

这是 discovery v5 的显式实验开关，不改变 `graph-routing=hard` 或
`graph-localization=frozen`，默认关闭。它针对全量 S-OWODB 诊断中发现的评分—监督错位：
novelty 只在匹配到的已知目标和高置信 pseudo 上训练，但推理时会对所有 query 计算
`objectness * novelty`。因此大量未被选择的 query 可能进入 unknown Top-K。

开关只把 `mine_candidates` 已经标记为低显著性、且满足现有视图一致性约束的
`background` query 作为 `novelty=0` 的负样本。普通 unmatched query 继续忽略，避免把
潜在未知目标全部当成背景。`objectness`、router、伪标签挖掘、阈值、Top-K、数据协议和
作者 ImageSets 均保持不变。

该监督改变了训练轨迹，不能把开启和关闭的 checkpoint 混接。旧 checkpoint 缺失该字段时
按关闭处理；严格续训校验会拒绝在同一链中切换开关。需要比较时从同一 DINO 骨干和同一
作者 manifest 建立独立链，并同时报告 Known/Previous/Current AP50、U-Recall、unknown
precision、unknown predictions、WI 和 A-OSE。低 loss、更多 pseudo 或更高 novelty
监督覆盖都不能单独证明未知学习成功。

## 启用方式

在仓库根目录、Linux 服务器的 `tree-detr` 环境中，通过 benchmark 计划显式开启：

```bash
python tools/owod/run_benchmark.py \
  --manifest data/owod/s-owodb/split_manifest.json \
  --recipe ow-detr-paper --method discovery \
  --backbone-weights pretrained/dino_resnet50.pth \
  --output-dir output/tree-detr-s-v5-bgneg-full \
  --gpus 1,2 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
  --seed 42 --activation-checkpointing \
  --graph-routing hard --graph-localization frozen \
  --discovery-scoring hierarchical \
  --discovery-trusted-background-negative \
  --dry-run
```

先检查 dry-run 中的每个阶段都包含
`--discovery-trusted-background-negative`，再去掉 `--dry-run` 启动正式独立链。运行前仍需
执行 `manage_data.py verify`，并按对应输出目录保存日志；不要把该计划的阶段接到旧 v5/v6
结果上。

