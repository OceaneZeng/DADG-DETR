# M-OWODB / S-OWODB：当前唯一有效的实验协议

2026-10-15 补充：用户授权先进行独立的小数据方法筛选，再从头启动优胜方法的
正式全量训练。诊断流程见 [owod-screening.md](owod-screening.md)；它不替换
本文的作者协议，不使用正式测试集调参，诊断检查点禁止用于初始化正式训练。

更新日期：2026-10-09。本文取代旧的 DEUS、随机 COCO 20×4、
`pilot_unverified` 和 D1–D4 实验 setting。历史日志保留，旧 setting 文件和启动器已删除；历史汇总工具位于 `tools/analysis/`。

## 原文定义与来源

M-OWODB 来自 Joseph 等《Towards Open World Object Detection》CVPR 2021
§5.1、§5.2 和补充材料 F。原文没有 M/S 这两个名字，也没有定义第二套 S 划分。
S-OWODB 经用户明确确认，补充采用 Gupta 等《OW-DETR》CVPR 2022
附录 A.2、Table 7。两篇原文是协议来源，DEUS 不再决定本项目 setting。

| 项目 | M-OWODB（ORE 原始划分） | S-OWODB（OW-DETR 新划分） |
|---|---|---|
| 原始数据 | VOC2007/2012 + COCO | COCO2017 |
| Task 1–4 新增类别数 | 20 / 20 / 20 / 20 | 19 / 21 / 20 / 20 |
| 累计已知类别数 | 20 / 40 / 60 / 80 | 19 / 40 / 60 / 80 |
| Task 1 | 全部 20 个 VOC 类别 | Animals / Person / Vehicles |
| Task 2 | Outdoor / Accessories / Appliances，以及 truck、剩余动物 | Appliances / Accessories / Outdoor / Furniture |
| Task 3 | Sports / Food | Sports / Food |
| Task 4 | Electronic / Indoor / Kitchen / 剩余 Furniture | Electronic / Indoor / Kitchen |
| 官方训练清单图片数 | 16551 / 45520 / 39402 / 40260 | 89490 / 55870 / 39402 / 38903 |
| 官方微调清单图片数（Task 2–4） | 1743 / 2361 / 2749 | 1679 / 2345 / 2664 |
| 发布测试清单 | all_task_test.txt：10246 行，9119 个唯一图片 | test.txt：4952 个图片 |

80 类的完整顺序和原始稀疏 COCO 类别 ID 见
`tools/owod/benchmark_spec.py`；禁止把类别 ID 的数值顺序当作任务顺序。
M 第一任务包括 bottle、sofa 等，因此相同超类会出现在不同任务；S 将整个超类
一次引入。例如 truck、elephant、bear、zebra、giraffe 在 M 的 Task 2，
在 S 的 Task 1。这是类别/超类协议差异，不等于要求图片在任务间互斥。

`data/owod/{m-owodb,s-owodb}/split_manifest.json` 内嵌作者原始清单，来源为
OW-DETR 官方仓库 commit `3515ff8c36687a4f582ef6a2c83866966b56ce23`；其中
`split_source` 保留上游路径、字节 SHA-256、行数和唯一图片数，`splits` 保留原始
顺序与重复项。`.gitattributes` 固定 manifest 的文本字节，避免换行转换损坏哈希。

官方清单中存在跨任务共用图片，必须保留。M 测试清单也存在重复行，原始作者
评测会受其影响。项目序列化 JSON 时保存唯一图片记录，但在作者评测中恢复
清单原始顺序和重复次数，包括 GT 计数及预测次数；不能悄悄去重后声称复现
原始 ORE 结果。修正重复行的后续基准属于另一个明确命名的协议版本，当前不启用。

## 可见标签、增量学习与均衡微调

每个 Task 的增量训练只含本任务类别标签，未来类不作为标签暴露。
图像中旧类和未来类可以共现。评测使用完整标注：旧类与当前类按真实类别评测，
未来所有类别合并为一个 unknown。Task 4 所有 80 类已知，无 unknown 指标。
当前及之前任务类别共同构成模型的已知输出空间，不能输出尚未引入的类别名。

Task 1 训练后直接评测；Task 2–4 从上一任务的完成模型继续，先训练当前任务，
再用作者发布的 `t2_ft/t3_ft/t4_ft` 均衡集合微调。阶段编号从 0 开始。
`instances_increment_train2017.json` 表示当前增量；
`instances_train2017.json` 在 Task 2–4 表示发布的均衡微调集合，
不能理解为所有历史训练图片的累加。

ORE §5.2/§6.2 的 Nex=50 指均衡集合每类至少 50 个 **实例**，
不是固定每类 50 张图片。OW-DETR 补充材料 D 使用每已知类 50 个 exemplars。
项目使用作者给定清单，验证每类实例覆盖，不重新随机抽样或改成 GNN 附加图片预算。
训练标签不包含未来类；作者 XML 中的 difficult 标记在训练保留、评测按 VOC 规则处理。
S 的 COCO→VOC 坐标整数化和 crowd 处理完全沿用作者 `datasets/coco2voc.py`，
原始 VOC 坐标同时保存为 `voc_bbox`，用于一致的评测。
原始对象 ID 同时保留；不同对象即使整数化后拥有同一个框，仍保留为独立 GT。
重复检查针对同一原始标注的重复记录，不按框位置合并真实对象。

ORE §5.1 还说明每任务留出 1000 张训练图片做 validation，原始 EBUI
会利用带 known/unknown 的 held-out 数据拟合能量分布。该机制是 ORE 方法复现
的一部分；OW-DETR 对比的是不使用这种 unknown 验证监督的 ORE−EBUI。
当前工具不随机生成这些 held-out 图片，也不把测试集当成训练/阈值拟合数据。
复现 ORE 本身必须另行取得作者 held-out 清单和训练流程；本次实现的是 benchmark，
不是 ORE 模型或其 EBUI。选择模型或调参时不得读取测试集未来类标签。

## 指标与作者评测实现

必须报告 Previous / Current / Known mAP@0.5，U-Recall，A-OSE，以及
WI@recall=0.8。Known 是所有已知类别 AP 的宏平均；S 的前两任务类别数不同，
不能用 Previous、Current 两个平均值再简单平均。H-Score 可作为附加指标，
不是 ORE 定义的原始指标。

ORE 原文 WI 为 `P_known / P_known+unknown - 1`，固定 recall=0.8；
本实现按作者程序，逐已知类别选取离 0.8 最近的召回位置，聚合 open-set FP
和对应的 TP+FP。A-OSE 同样遵守作者程序，统计误判未知目标的已知类预测数，
而不是对未知 GT 做简单的“是否被某框覆盖”计数。Unknown 的 TP 必须一对一匹配，
重复预测是 FP，单一预测不能召回多个重叠 GT。两个指标都不再使用旧代码的近似算法。

作者代码还规定 VOC inclusive 坐标 IoU、严格 `IoU > 0.5`、difficult 规则，
预测 score 保留 3 位小数、坐标保留 1 位小数。
`models/owod_benchmark_eval.py` 保留这些约定，读取未经图像缩放破坏的原始标注。
分布式采样补齐的重复图片先去除，再应用发布测试清单的实际重复次数。

原始 ORE 作者评测源
`JosephKJ/OWOD@23890f188cd1a6801c6ac0e3dacd78b8572b8c29`
`detectron2/evaluation/pascal_voc_evaluation.py` 把 `_is_2007` 设为 False，
使用积分 AP；OW-DETR 官方 `OWEvaluator` 默认用 VOC07 11 点 AP。
这两个源实现的差异已记录在 manifest 的 `ap_mode` 中，不会静默替换为 COCO
101 点 AP50。后续 M-OWODB 表格可能采用 OW-DETR 的 VOC07 约定，引用它们时
必须检查评测版本，不能与本项目原始 ORE 约定直接混用。
S 的原始 OW-DETR 模型使用 top-50 detections、DINO ResNet-50 预训练。
模型架构、优化器、预训练、未知类别识别方式都是方法 setting，必须单独记录。

## 数据准备与运行

截至 2026-10-09，本机完整 COCO2017、DINO ResNet-50 权重和 S schema-3
manifest 已准备并核验。S 四任务的训练图片/实例数，以及各任务当前类别的
测试图片/实例数，逐项与 OW-DETR Table 7 相符。S 发布的完整测试清单为
4,952 张；Table 7 的各任务测试图片数是含该任务类别的图片数，不是需要
自行生成的四个互斥测试集合。

VOC 三个原始 TAR 已解压，M 作者清单需要的 21,503 对 VOC 图像/XML 已整理。
M 的 COCO XML/图像合并尚不完整，尚无可训练的 M schema-3 manifest。
M 必须取得 ORE 作者发布的完整 VOC-style
`Annotations/`、`JPEGImages/`，不能仅凭 COCO2017 拼出 VOC Task 1。
作者下载入口：
<https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k>。
S 使用 COCO2017 的 `train2017/`、`val2017/` 及两个 `instances_*.json`。

```bash
python tools/owod/manage_data.py plan
python tools/owod/manage_data.py check --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco
python tools/owod/manage_data.py prepare --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod
python tools/owod/manage_data.py verify --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod --decode-images
```

完整操作指南见 [data-preparation.md](data-preparation.md)。工具不会移动原始图像、生成随机划分或覆盖旧目录。它会完整检查缺失数据，
序列化作者清单，验证类别、标签可见性、标注重复、均衡集合覆盖并生成 schema-3
manifest，记录原始数据与输出 SHA-256。数据管理入口要求完整原始数据，验证成功后发布任务目录；重复运行会验证并复用。
底层转换器支持仅标注转换，但不能把它当成准备就绪的实验数据。

`run_benchmark.py` 按 `Task1 increment → Task2 increment → Task2 finetune
→ Task3 increment → Task3 finetune → Task4 increment → Task4 finetune` 执行。
当前运行的是本项目 Deformable DETR control，不会命名为 ORE/OW-DETR 的真实复现。
`--recipe` 必须显式选择，避免把优化器选择隐含成 benchmark 定义。

```bash
python tools/owod/run_benchmark.py \
  --manifest data/owod/s-owodb/split_manifest.json \
  --recipe ow-detr-paper --backbone-weights /path/to/dino_resnet50_pretrain.pth \
  --output-dir exps/owod/author-release --gpus 0,1 \
  --nproc-per-node 2 --batch-size 2 --seed 42 --dry-run
```

OW-DETR 原文补充 D 的 recipe 为 50 epochs 当前训练 + 20 epochs 均衡微调，
基础 LR=2e-4、微调 LR=2e-5、weight decay=1e-4；AdamW 沿用作者实际实现。
它可用于同一 Deformable DETR 在 M/S 上的对照，但不是原始 ORE 方法训练 recipe。
ORE recipe 记录原文补充 F 的 SGD、LR=0.01、有效 batch=8、每任务 8 epochs。
ORE 原文没有明确给出独立均衡微调的 epoch 数，启动器要求另行显式提供
`--finetune-epochs`，不会把旧的 20/50 epochs 或源码中不同迭代配置偷偷代入。
所以“复现 benchmark”“复现某个方法的全部训练超参数”“复现论文结果”是三个
不同的验收范围，当前不得把本地 control 声称为作者模型复现。

去掉 `--dry-run` 才启动训练。`--resume` 只能使用相同 manifest、相同计划与
对应阶段 checkpoint；每任务均衡微调完成模型才可用于下一任务初始化。
旧 pilot、DEUS schema-1/2 manifest、未验证 override、任意 train/val 替换、
COCO 全类别 detector 初始化、轻量缩放与自定义 replay-exposure 都会被拒绝。
旧的 graph-local、自定义 replay、单阶段 detector-control、本地多方法比较、
COCO-only shared-M 适配及外部实验计划生成文件已删除。研究组件留在 `models/`，
历史日志分析移至 `tools/analysis/`；当前数据与训练分别只有标准管理入口和启动器。
日志记录类别顺序、阶段、原始评测约定、图像根目录和数据/预训练来源。

数据仍缺失时，不能声称已经构建了可训练的完整数据集或取得了真实实验结果。
历史日志/模型无需删除，但不得作为新协议结果或初始化使用。
