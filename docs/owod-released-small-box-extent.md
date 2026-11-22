# 已释放小目标的近邻宽高监督

这是 discovery v10 的 Task 1 诊断选项。它使用 [M-OWODB 和 S-OWODB 作者协议](owod-benchmarks.md) 的训练来源，在每个协议内比较原模型与额外宽高监督。

候选组选取原图标注框面积小于 32² 像素的已释放目标。COCO 数据加载器保留原框面积，裁剪后按相同索引移除消失的目标。模型用现有的原生锚点距离为每个小目标找最近四个有效位置。同一位置有多个小目标时，选距离最近者。每个位置的编码器特征仍由原 offset 层输出框，新增项仅计算预测宽高和变换后可见框宽高的对数差，以两轴 Smooth L1 求和，再除以批内已释放框计数并乘以 0.1。原中心及最近邻全框 L1、16 近邻质量损失、查询预算和评分照常运行。日志单独记录 `loss_small_box_extent`。该选项只允许已核验的 Task 1 诊断训练，恢复时检查开关与 checkpoint 一致。

每协议两组都从相同的 DINO ResNet-50 权重启动，使用同一份 12,288 张训练图、seed 42、双 3090 每卡 batch 2 和三个 epoch。每协议两份 1,024 图验证切片来自其作者 Task 1–2 训练图，排除两组训练、正式测试及计划列出的旧留出。每片先锁定两个小代理尺寸档各 192 张图，再补足已知类覆盖和剩余图像。验证标注不进入训练。仅对末轮用原生 focal Top-50 评估，报告诊断性 Current／Known AP50、U-Recall、WI@0.8 和 A-OSE。另按小目标尺寸统计密集框、预筛、选中框和 decoder 覆盖，以及实测未知召回。

在 Linux 服务器的本项目仓库根目录和 `zhy` 里，双卡编号以实际可用的同型号 GPU 为准。先写出完整计划，再启动训练。

```bash
CUDA_VISIBLE_DEVICES=1,2 bash tools/owod/run_instance_pilot.sh --task1-m-s-released-small-extent --plan-only
CUDA_VISIBLE_DEVICES=1,2 bash tools/owod/run_instance_pilot.sh --task1-m-s-released-small-extent
tail -f output/task1-m-s-released-small-extent/console.log
```

运行根目录为 `output/task1-m-s-released-small-extent/`。其中 `plan.json` 锁定启动命令、输入哈希和验收门槛，每个协议的 `control/task1`、`extent/task1` 保留训练配置、指标和权重。末轮评估按协议与训练组选出独立报告，四组齐备后写出 `comparison.json`。两协议两片必须同时通过几何与实测召回门槛，并满足已知 AP50 和开放集代价约束，才能进入独立种子及 Task 2 复核。
