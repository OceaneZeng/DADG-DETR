# M-OWODB lr-quarter control：早期检测学习与输入响应明显改善

核验日期：2026-11-11。附件：`C:/Users/23642/Downloads/console.log`。
已只读核对服务器的运行配置、不可变计划、三轮完整指标及 256 图逐图诊断；
没有修改服务器源码或启动新的训练。本次本地只补充分析文档。

**结论：所有参数组学习率降到原来的 1/4 后，三轮 Known AP50 达到 29.2592%，
原 control 同轮为 0.2318%；此前第 2 轮的大范围像素响应退化没有在这三轮中重现。**
这为基础优化步长过大的假设提供了有力证据，但尚未验证完整发现方法或后续增量任务。

## 实际运行与可比性

远端当前 HEAD 为 `48c38dc70d92b8d5ad49a2069c6f94b6cf247347`，与本地一致。
两份 `run_config.json` 逐项比较，仅 `lr`、`lr_backbone` 和 `output_dir` 不同。
主训练循环、检测器和 transformer 的源码哈希与原 control 审计记录一致；
本地与服务器本轮关键源码在统一 CRLF/LF 后哈希相同。
服务器另有 baseline 子目录及未跟踪构建/清理文件，不属于这次 control 的配置差异。

| 配置 | 原 control | lr-quarter |
| --- | --- | --- |
| 主学习率 | 2e-4 | 5e-5 |
| backbone 学习率 | 2e-5 | 5e-6 |
| sampling offsets / reference points 学习率 | 2e-5 | 5e-6 |
| 数据 / 初始化 / seed | 完整作者 M Task 1 / DINO R50 / 42 | 相同 |
| GPU / 每卡 batch / 全局 batch | 0,1 / 2 / 4 | 相同 |
| 架构、loss、clip norm、重计算 | Deformable DETR control / 原损失 / 0.1 / 开启 | 相同 |
| epochs / lr_drop / 执行预算 | 50 / 50 / 3 | 相同 |

训练仍为 16,551 图、47,223 个已知标注；每轮评估作者完整测试清单，
包含 10,246 条目、9,119 个独立图像，保留作者清单重复规则。
256 图只是训练侧输入探针，没有缩小 AP 评估数据。

- manifest SHA256：`9488bc9fd0c266958b567bfd2c1c6e8a0258d85452aca19e171123e4d8a3d1c3`。
- DINO SHA256：`156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c`。
- 原 control 的运行和审计见[前次分析](owod-control-mowodb-analysis-20261110.md)。

## 全量作者 AP 与学习轨迹

以下 AP 已乘 100，使用 M/ORE 的 VOC 积分 AP50；不是 COCO 101 点 AP50。
Task 1 的 Current AP50 = Known AP50，没有 Previous AP。
loss 为逐轮平均、加权的末层损失，总 loss 另包含辅助层。

| 完成轮数 | 原 Known AP50 (%) | 新 Known AP50 (%) | 新训练分类 loss | 新验证分类 loss | 新验证 bbox loss | 新训练分类错误率 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5010 | 14.3524 | 0.9216 | 0.7366 | 0.8456 | 35.18 |
| 2 | 0.2193 | 19.2857 | 0.7361 | 0.7238 | 0.7690 | 18.56 |
| 3 | 0.2318 | 29.2592 | 0.6756 | 0.6591 | 0.6764 | 14.71 |

第 3 轮相对原 control 提高 **29.0274 个百分点**。新验证总 loss 为
15.3785 → 14.3087 → 13.1703，验证 GIoU loss 为 0.9847 → 0.9243 → 0.8523。
分类、定位和完整 AP 同时改善，支持模型确实开始学习图像内容。
三轮内没有明显验证性能退化证据；尚不能外推到 50 轮收敛水平。

平均裁剪前梯度范数仍为 163.77、175.83、177.61，原组为 177.20、238.40、175.59。
两组始终启用 clip norm=0.1，因此大 `grad_norm` 不能解释成未裁剪更新，
也不能单凭该数值判定训练是否成功。本次变化主要是 AdamW 更新的学习率。

## 256 图响应：中位数和逐图比例均改善

固定 256 个索引均匀覆盖完整已释放训练清单；本轮每张图都分别做均值颜色替换、
真实换图两种对照。尺寸、mask 和几何条件保持，eval 模式，不更新 optimizer。
下表框差是所有 query 的归一化坐标平均绝对差，再取逐图中位数，不能当成 IoU 或 AP。

| 完成轮数 | 均值颜色框差中位数 | 真实换图框差中位数 | 真实换图 encoder 末层差中位数 | 两种对照各自框差 <1e-6 |
| --- | ---: | ---: | ---: | --- |
| 1 | 0.08504 | 0.09252 | 0.77801 | 均为 0/256 |
| 2 | 0.08998 | 0.10743 | 0.80517 | 均为 0/256 |
| 3 | 0.09621 | 0.11476 | 0.76096 | 均为 0/256 |

两种对照的 logits、六层 encoder 响应也都没有低于 1e-6 的逐图案例。
原 control 第 2 轮的 256 图审计中，均值颜色对照 123/128、真实换图 124/128
接近数值误差；新组的中位数及最小框差均明显高于该范围，并非少数样本抬高平均值。
旧扩大审计每种对照各做 128 图，新组每种各 256 图，比例的分母应明确区分。

## 未知指标与结论边界

这个 control 只输出 known 类别，没有 unknown 预测分支。因此三轮
`unknown_predictions=0`、U-Recall=0、H-Score=0 符合入口定义，不能用它衡量发现模块。

| 完成轮数 | A-OSE | WI@recall=0.8 |
| --- | ---: | ---: |
| 1 | 26,352 | 0.04769 |
| 2 | 34,276 | 0.05554 |
| 3 | 37,380 | 0.06359 |

A-OSE / WI 仍在上升，说明已知检测改善没有同时解决未知误识别。
A-OSE 统计预测级开放集错误，不是去重的未知 GT 数，允许超过 23,320 个未知 GT。
第 3 轮原组 A-OSE=20,299、WI=0.03977；原组较低 Known AP 也不能据此称为更好的 OWOD。

此次应调整排查优先级：先固定基础检测优化稳定性，再验证图路由和发现监督。
原 control 在没有这些模块时也会退化，本轮仅学习率变化便明显改善；
此前 v6/v7/v8 的低 AP 因而不能全部归因于未知损失。
仍未证明学习率是全部版本的唯一根因、5e-5 是最优值，或完整 OWOD 方法已经修复。
这是单 seed、Task 1 前三轮的内部 control，不是外部方法复现或最终成绩。

## 推荐的下一步：同一低学习率 control 续到固定第 8 轮

验证目标是这套配方能否继续学习且保持像素响应。保持全部优化/数据设置和
50 轮日程，加载本次 optimizer / scheduler，只把执行暂停预算从 3 增至 8。
这不是在旧高学习率权重上临时降学习率。恢复训练不承诺与不中断训练逐位相同，
但不改变登记配方，可用于继续观察本次运行。

本轮训练和评估共 **1:51:03**：每轮训练约 28 分钟、评估约 8 分钟。
原组为 1:22:00，训练和评估都更慢；计算图配置没有变化，目前没有证据把耗时增加
归因于降低学习率本身。按本轮吞吐，追加 5 轮、恢复前评估和 256 图诊断约需
**3–3.5 小时**，实际取决于服务器负载。

固定第 8 轮暂停，不用作者测试集挑最佳轮替代阶段结束权重。
若到预算末 AP 总体继续提高且输入响应正常，下一步再让完整方法从相同 DINO
初始化、相同低学习率配方独立起跑，检验发现路径的增量影响。
若出现明显退化，则依据保留的逐轮权重定位首次退化阶段，再决定下一次改动。
若发生非有限 loss 或训练错误，保留日志和已有权重后诊断。

现有短脚本预算固定 3；`lr-quarter --resume` **不会增加训练轮数**。
下面使用已经存在的底层入口，无需改脚本。已在远端只读生成 8 轮预算的计划，
逐项确认它与保存的 `benchmark_plan.json` 完全相同；没有实际启动训练或验证一次真实续训。

在 Linux / Bash 打开 tmux，先确认 GPU 0、1 可用：

```bash
tmux new-session -s m-control-lrq8
```

在 tmux 内执行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail
{
  python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod &&
  python -u tools/owod/run_benchmark.py \
    --manifest data/owod/m-owodb/split_manifest.json \
    --recipe ow-detr-paper --method control --task1-only \
    --backbone-weights pretrained/dino_resnet50.pth \
    --output-dir output/tree-detr-m-control-lr-quarter-3090 \
    --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
    --seed 42 --activation-checkpointing --control-lr-scale 0.25 \
    --pause-after-epochs 8 --resume &&
  python -u tools/owod/diagnose_control_inputs.py \
    --run-dir output/tree-detr-m-control-lr-quarter-3090/m-owodb/seed_42/stage_0/increment \
    --completed-epochs 8 --images 256
} 2>&1 | tee -a output/tree-detr-m-control-lr-quarter-3090-logs/console.log
```

这是同一次运行的追加记录；旧的逐轮权重保留，`checkpoint.pth` 更新为最新状态。
结果仍在 `output/tree-detr-m-control-lr-quarter-3090/m-owodb/seed_42/stage_0/increment/`。
`Ctrl+B` 然后 `D` 离开 tmux，另一个终端从仓库根目录查看：

```bash
tail -n 50 -F output/tree-detr-m-control-lr-quarter-3090-logs/console.log
```

## 证据保存与本次验证范围

本地忽略目录 `output/analysis/m-control-lr-quarter-20261111/` 保存：

- `server-audit.json`：两组配置、计划、完整 metrics、输入响应逐图报告及源码哈希。
- `provenance.json`：附件哈希、远端提交及原 control 证据引用。
- `continuation-check.json`：8 轮执行预算的不可变计划比较与只读 dry-run、两组耗时。

本轮没有修改训练业务代码，没有重跑模型测试或新增 GPU 实验。
续训指令验证仅覆盖计划生成及与现有计划相等，不代表真实续训已经完成。
