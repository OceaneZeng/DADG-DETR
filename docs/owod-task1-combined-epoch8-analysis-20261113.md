# M-OWODB combined：第 8 轮结果分析

后续更新：已完成真实权重的同轮训练图诊断，并准备 γ=1、focal 评分的新独立
8 轮对照，见[后续实验](owod-task1-focal-followup-20261113.md)。下文保留读取附件
当时的分析状态；新组尚无正式成绩。

核验日期：2026-11-13。用户新附件 `C:/Users/23642/Downloads/console.log` 为
580,075 字节、811 行，SHA256：
`e13977261cbab64a8d0435f162b7fe8edbee0d404dfbadd5f6527e33b7759d98`。
对照附件为 `consolev8-lrquarter.log`，SHA256：
`6f25d8118d74b4469b71b521b604bf7fd4fa43ee31e88b8fe7e2d7afa849ae3d`。
本节分析仅读取本地附件及 Git 历史；后续服务器只读模型诊断另见上方更新。

**结论：combined（γ=0.1）第 8 轮 Known AP50 为 34.0246%，比 ranking（γ=1）
同轮低 1.9769 个百分点。未知召回及误识别指标有所改善，但当前结果不支持把该组
作为提高 Task 1 AP 的方案。建议保留第 8 轮暂停状态，先做同轮只读诊断，不直接
投入剩余 42 轮。尚不能据前 8 轮断言固定第 50 轮的结果。**

## 运行身份与可比范围

- 新组日志记录：M-OWODB Task 1 / stage 0 / increment，v8 combined，γ=0.1、
  排序权重 1；主 lr=5e-5、backbone lr=5e-6，seed 42，双 GPU 0/1、每卡 batch 2。
- 新旧日志中的 manifest SHA256 均为
  `9488bc9fd0c266958b567bfd2c1c6e8a0258d85452aca19e171123e4d8a3d1c3`，
  DINO SHA256 均为 `156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c`。
  作者 M 数据核验通过；hierarchical、balanced-bce、count、3 轮 pseudo ramp 相同。
- 核对日志配置中 `owod_evaluation` 之前的全部参数，差异只有 γ、输出目录和
  `pause_after_epochs`。新日志的巨大 Namespace 行被警告输出打断，不能把它当成
  完整可解析的配置；评测口径另结合日志指标及现有协议说明核对。
- 两个提交 `dc477ca`→`ed1b596` 只增加启动器、诊断、测试和文档，已提交的训练、
  模型、数据和评测核心源码没有变化。但两份服务器日志均标记工作区有未提交内容，
  本次未核验远端实际文件哈希，不能保证进程所用源码逐字一致，也没有多 seed 结果。
- 新日志中只有 8 次训练后作者评估，无恢复重复评估；耗时 4:40:14。末尾明确
  `completed_epochs=8/50`、保留权重、阶段未完成，属于正常预算暂停，没有报错。

## 同轮 AP 与未知召回

全部使用作者 VOC 积分 AP50，AP/Recall 已乘 100；不采用控制台 COCO 101 点 AP50。
Task 1 的 Current AP50 = Known AP50，没有 Previous AP。

| 完成轮数 | ranking AP50 (%) | combined AP50 (%) | AP 差值（百分点） | ranking U-Recall (%) | combined U-Recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.8402 | 11.1181 | +1.2779 | 0.0000 | 0.0000 |
| 2 | 18.1665 | 15.5077 | -2.6587 | 0.6861 | 0.7762 |
| 3 | 23.1876 | 22.8867 | -0.3009 | 4.5540 | 4.4254 |
| 4 | 25.3102 | 25.3412 | +0.0310 | 6.6209 | 7.1655 |
| 5 | 29.2728 | 27.9143 | -1.3585 | 7.1312 | 6.5995 |
| 6 | 32.0940 | 29.2121 | -2.8819 | 10.0729 | 7.6458 |
| 7 | 34.5920 | 31.5266 | -3.0654 | 9.1767 | 9.1381 |
| 8 | 36.0015 | 34.0246 | -1.9769 | 8.2890 | 8.9494 |

combined 的 AP 持续上升，没有重现早期 AP 崩溃；但第 5–8 轮连续落后，不能只把
末轮差距解释为单点波动。末轮未知召回略高，前 8 轮并没有稳定的 U-Recall 优势。

| 第 8 轮指标 | ranking γ=1 | combined γ=0.1 | 变化 |
| --- | ---: | ---: | --- |
| Known AP50 (%) ↑ | 36.0015 | 34.0246 | -1.9769 个百分点 |
| U-Recall (%) ↑ | 8.2890 | 8.9494 | +0.6604 个百分点 |
| Unknown Precision (%) ↑ | 0.9263 | 0.9780 | +0.0517 个百分点 |
| WI@recall=0.8 ↓ | 0.069896 | 0.061788 | 降低约 11.60% |
| A-OSE ↓ | 24,530 | 21,424 | 少 3,106，降低约 12.66% |
| Unknown 输出数 | 208,680 | 213,404 | 多 4,724 |

未知发现仍有效，改善不是通过清空 unknown 输出获得的；但这几项开放集指标
不能抵消用户关心的已知 AP 下降。WI 和 A-OSE 也会随已知分类行为改变，不能仅
根据两项降低便认定整体方法提升。Unknown Precision 依赖已有未知标注，不把
其余预测全部认定为背景。

[历史纯检测 control 第 8 轮](owod-control-lr-quarter-epoch8-analysis-20261111.md)
AP50 为 41.8286%。新 combined 与其相差 -7.8040 个百分点，比原 ranking 的
-5.8271 更大。control 没有发现分支且评分不同，只用于提示检测能力差距，不能
把它当成与完整方法仅有 γ 差别的外部 baseline。

## 损失提供的有限线索

以下为控制台 `Averaged stats` 中带括号的全轮平均，分类/框/GIoU 为加权末层 loss。
总验证 loss 另含辅助层；不把只打印末个值的 `class_error`、`grad_norm` 当全轮均值。

| 第 8 轮 | ranking γ=1 | combined γ=0.1 |
| --- | ---: | ---: |
| 验证分类 loss | 0.5618 | 0.5847 |
| 验证 bbox loss | 0.5215 | 0.5239 |
| 验证 GIoU loss | 0.7335 | 0.7189 |
| 验证总 loss | 11.5028 | 11.6813 |
| 训练排序 loss | 0.0661 | 0.0762 |
| 训练匹配已知 query 的排序违反率，约 | 2.45% | 2.77% |
| 训练伪未知候选数/图 | 2.1358 | 1.6485 |

排序违反率按全轮 `discovery_known_rank_violations / discovery_known_count` 计算，
控制台均值已四舍五入，故仅报近似值。

这组没有显示明确的整体定位改善：bbox 近似，GIoU 略好，分类 loss 和已知/未知
排序约束稍差。优先检查分类及实际评分保留率更合理，但 loss 不能判定 AP 瓶颈。
伪候选变少既可能减少噪声，也可能丢失有用证据，不能单独认定为收益。现有结果
不支持“降低发现回传就能保护检测 AP”的假设；也没有证据说明 Task 2 路由是
这次 Task 1 退化的原因，Task 1 尚未添加增量类别的 RiskRoutedStage。

## 下一步：先比较相同第 5 轮的训练图诊断

原完整 ranking 链没有常规保存第 8 轮编号权重，而两组均应保留第 5 轮的
`checkpoint0004.pth`。先核对文件存在，再固定相同 256 张训练图比较两组。
不拿旧第 10 轮与新第 8 轮做同轮因果对照。

目的：区分原始定位不足、正确类别学习不足和联合 Top50 竞争。两次暂估合计
10–30 分钟，尚无这两份正式权重的诊断实测吞吐。完成固定样本即停止；文件不匹配、
输入哈希变化或非有限输出时退出排查。使用单张空闲 GPU，不运行优化器或修改权重。

服务器 Linux / Bash，沿用当前环境。以下命令不启动训练，分别记录报告和日志：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
set -o pipefail
diag=output/analysis/m-v8-ranking-combined-epoch5
mkdir -p "$diag"
for run in ranking-lr-quarter-3090-full combined-lr-quarter-3090-task1; do
  tag=${run%%-*}
  checkpoint="output/tree-detr-m-v8-${run}/m-owodb/seed_42/stage_0/increment/checkpoint0004.pth"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python -u tools/owod/probe_discovery_checkpoint.py \
    --checkpoint "$checkpoint" \
    --manifest data/owod/m-owodb/split_manifest.json --images 256 --seed 42 \
    --output "$diag/${tag}-epoch5.json" \
    2>&1 | tee -a "$diag/${tag}-epoch5.log" || break
done
```

报告拒绝覆盖已有 JSON；重跑使用新输出名。可再按[实验说明](owod-task1-low-lr-optimization.md)
读取新 combined 第 8 轮，但其作用是观察当前状态，不能替代两组同轮比较。

- 原始框覆盖相近、正确类别覆盖下降：优先调查已知类别学习及联合目标的影响。
- 原始框与正确类别覆盖相近、实际联合 Top50 下降：优先调查 objectness/novelty
  门控和评分竞争。反事实评分仅供诊断，不直接改正式评测口径。
- 原始框覆盖也下降：再调查共享表示和定位学习，不能继续假定定位已获保护。

当前建议是暂不直接 `--resume --full`，保留所有权重和日志。该建议基于预先固定
8 轮预算下 AP 未改善，避免额外约一天的训练成本；不表示 γ=0.1 的 50 轮结果
已经被测出。不要把现有 combined 权重改回 γ=1 强行恢复，也不立即追加一轮 γ 扫描。

本次只新增分析记录和结果入口，未修改训练或评测实现，未运行新的模型实验。
