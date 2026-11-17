# M-OWODB 论文与本项目结果对照

整理日期：2026-11-16。

## 口径

- M-OWODB 使用 ORE 作者的 superclass-mixed 划分，四个 Task 各新增 20 类。
- OW-DETR、PROB 的论文表直接标为 M-OWODB；CAT 的 Table 1 标为 OWOD split，指同一套 ORE/M-OWODB 划分。
- 论文列出的 AP 是作者表中的 known-class mAP@0.5，拆成 Previous / Current / Both（本表把 Both 记为 Known）。Task 1 没有 Previous，Current=Known。
- 本项目 manifest 的评估配置为 VOC inclusive IoU、IoU>0.5、`ap_mode=integral`；论文原始 OW-DETR `OWEvaluator` 默认是 VOC07 11-point AP。因而下表适合做同一数据协议下的参考，不应当作严格同评测实现的复现对比。
- 本项目 `v8 focal` 目前只有 Task 1 increment 的结果，没有 balanced finetune，也没有 Task 2–4 结果。`train.log2` 已完成 Task 1 的 50/50 轮。日志目录名含 A40，但日志本身没有 GPU 型号枚举，因此这里按“A40 目录日志”标注，不把显卡型号当作日志已证实事实。

## 已知类 AP / mAP@0.5（%）

| 方法 / 运行 | Task | Previous | Current | Known / Both |
|---|---:|---:|---:|---:|
| OW-DETR（论文） | 1 | — | 59.2 | 59.2 |
|  | 2 | 53.6 | 33.5 | 42.9 |
|  | 3 | 38.3 | 15.8 | 30.8 |
|  | 4 | 31.4 | 17.1 | 27.8 |
| PROB（论文，默认 τ） | 1 | — | 59.5 | 59.5 |
|  | 2 | 55.7 | 32.2 | 44.0 |
|  | 3 | 43.0 | 22.2 | 36.0 |
|  | 4 | 35.7 | 18.9 | 31.5 |
| CAT（论文） | 1 | — | 60.0 | 60.0 |
|  | 2 | 55.5 | 32.7 | 44.1 |
|  | 3 | 42.8 | 18.7 | 34.8 |
|  | 4 | 34.4 | 16.6 | 29.9 |
| 本项目 v8 focal，A40 日志，第 18 轮（Task 1 AP 峰值） | 1 | — | 52.5788 | 52.5788 |
| 本项目 v8 focal，A40 日志，第 20 轮 | 1 | — | 52.4442 | 52.4442 |
| 本项目 v8 focal，A40 日志，第 50 轮（Task 1 完成） | 1 | — | 58.4958 | 58.4958 |

## 未知类指标

论文通常只报告 U-Recall；PROB Supplementary Table 4 额外报告 WI 和 A-OSE。CAT 的论文 Table 1 没有给出 WI/A-OSE。

| 方法 / 运行 | Task | U-Recall (%) | WI@recall=0.8 | A-OSE |
|---|---:|---:|---:|---:|
| OW-DETR（论文） | 1 | 7.5 | 0.0571 | 10,240 |
|  | 2 | 6.2 | 0.0278 | 8,441 |
|  | 3 | 5.7 | 0.0156 | 6,803 |
| PROB（论文，默认 τ） | 1 | 19.4 | 0.0569 | 5,195 |
|  | 2 | 17.4 | 0.0344 | 6,452 |
|  | 3 | 19.6 | 0.0151 | 2,641 |
| PROB（论文，τ=0.5） | 1 | 19.3 | 0.0415 | 1,428 |
|  | 2 | 17.7 | 0.0133 | 562 |
|  | 3 | 19.7 | 0.0082 | 387 |
| CAT（论文） | 1 | 23.7 | — | — |
|  | 2 | 19.1 | — | — |
|  | 3 | 24.4 | — | — |
| 本项目 v8 focal，A40 目录日志，第 18 轮 | 1 | 12.2041 | 0.076312 | 16,170 |
| 本项目 v8 focal，A40 目录日志，第 20 轮 | 1 | 8.2933 | 0.071819 | 12,927 |
| 本项目 v8 focal，A40 目录日志，第 50 轮 | 1 | 12.5600 | 0.069818 | 23,844 |

本项目第 18/20 轮的 unknown precision 分别为 **0.9035% / 0.5646%**；论文表没有给出可直接对应的 unknown precision。Task 4 没有未知类，因此不报告 U-Recall、WI、A-OSE。

## 本项目 Task 1 内部轨迹

以下结果来自 M-OWODB Task 1 increment，便于区分“同协议下的内部诊断”与论文最终四任务结果。8 轮记录来自双 3090 的旧 v8 对照；18/20/50 轮来自新的 v8 focal A40 目录日志。

| 运行 | 完成轮数 | Current / Known AP50 (%) | U-Recall (%) | Unknown precision (%) | Unknown predictions | WI | A-OSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| v8 ranking，γ=1 | 8 | 36.0015 | 8.2890 | 0.9263 | — | 0.069896 | 24,530 |
| v8 combined，γ=0.1 | 8 | 34.0246 | 8.9494 | 0.9780 | — | 0.061788 | 21,424 |
| v8 focal，γ=1 | 8 | 42.1399 | 9.1338 | 1.0220 | 208,411 | 0.079876 | 25,992 |
| v8 focal，A40 | 8 | 41.7655 | 9.6141 | 0.9390 | 238,764 | 0.079887 | 24,722 |
| v8 focal，A40 | 12 | 46.5661 | 11.5480 | 0.9734 | 276,664 | 0.079511 | 19,931 |
| v8 focal，A40 | 16 | 49.8228 | 13.3705 | 0.9666 | 322,570 | 0.068448 | 14,273 |
| v8 focal，A40 | 18 | 52.5788 | 12.2041 | 0.9035 | 314,990 | 0.076312 | 16,170 |
| v8 focal，A40 | 20 | 52.4442 | 8.2933 | 0.5646 | 342,551 | 0.071819 | 12,927 |
| v8 focal，A40 | 50 | 58.4958 | 12.5600 | 1.5558 | 188,261 | 0.069818 | 23,844 |

## 目前能得出的比较

1. 论文结果中，CAT 的 Task 1 U-Recall 和 Current/Known AP 最高；PROB 在 Task 2–4 的 Known/Both AP 最高或接近最高，并且其补充材料的 WI/A-OSE 明显优于 OW-DETR。
2. 本项目 v8 focal 在 Task 1 第 50 轮的 AP 为 58.4958%，距离 OW-DETR 论文 59.2% 还差 0.7042 个百分点；U-Recall 为 12.5600%，高于 OW-DETR 的 7.5%，但低于 PROB 的 19.4% 和 CAT 的 23.7%。
3. 相比同一训练链第 20 轮，Task 1 AP 提高 6.0516 个百分点，U-Recall 提高 4.2667 个百分点，Unknown Precision 提高 0.9912 个百分点；A-OSE 增加 10,917，说明未知预测数量和误识别仍需继续分析。
4. 本项目还没有 Task 2 的 Current AP，因此不能据此判断是否具备新类增量学习能力，也不能计算完整四任务平均结果或声称超过论文方法。

## 来源

- [OW-DETR PDF](../tmp/pdfs/ow-detr.pdf)，Table 1 与 Table 5。
- [PROB PDF](../tmp/pdfs/prob.pdf)，Table 1 与 Supplementary Table 4。
- [CAT PDF](../tmp/pdfs/cat.pdf)，Table 1（OWOD split）。
- [本项目 M-OWODB 结果分析](owod-results-analysis-20261116.md)。
- [M-OWODB 协议说明](owod-benchmarks.md)。
