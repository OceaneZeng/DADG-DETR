# ORE 官方代码与 M-OWODB 数据审计（2026-10-10）

后续实现：`tools/owod/prepare_ore.py` 已补齐按以下规则重建 COCO 组件的入口。
样例输出与固定版本官方转换器逐字节一致；边界框、crowd、冲突保护和来源变化
测试通过。完整使用命令见 [数据准备](data-preparation.md)，完整 M 数据准备
是否成功仍以实际 prepare/verify 结果为准。

核对源：<https://github.com/JosephKJ/OWOD>，固定 commit
`23890f188cd1a6801c6ac0e3dacd78b8572b8c29`。只读副本保存在忽略目录
`tmp/ore-official`，不纳入本项目 Git。

## COCO XML 可以按作者代码生成

此前将“无法取得作者 Drive XML 压缩包”等同于“无法准备 M”是不完整的。
仓库提供了明确的转换实现：
[coco_annotation_to_voc_style.py](https://github.com/JosephKJ/OWOD/blob/23890f188cd1a6801c6ac0e3dacd78b8572b8c29/datasets/coco_utils/coco_annotation_to_voc_style.py)。
可分别调用 `coco_to_voc_detection` 处理 `instances_train2017.json` 和
`instances_val2017.json`。脚本主入口只写了作者机器上的 val 路径，不能直接
运行主入口就认为 train/val 都完成。

确定的转换语义：

- 遍历 `COCO.imgToAnns`，每张有标注图片写一个 XML；不为无标注图片写 XML。
- 文件名沿用 COCO 图片 stem，类别保留 COCO 原始名称；写入原始宽高、depth=3。
- 所有标注都保留，包括 `iscrowd=1`；每个目标的 `difficult=0`。
- 框为 `int(x+1), int(y+1), int(x+w+1), int(y+h+1)`；不裁剪越界框。
- 不写入 unknown 标签，不按任务过滤 XML；标签可见性由训练 loader 处理。
- 脚本只生成 XML，不下载/复制图片，不生成作者任务清单。

直接调用固定版本原函数的小样例检查已通过：crowd 保留、小数截断、越界不裁剪、
无标注图片不生成 XML。不能用通用转换器默认的 crowd 过滤或四舍五入替代。
这能建立“依据作者源码重建”的来源链，但未取得 Drive 原始包，不能声称与该包
逐字节一致。重建时应另存源 commit、转换器和输入 JSON 哈希。

README 指向的 [数据说明](https://github.com/JosephKJ/OWOD/issues/59#issuecomment-897747744)
给出 `datasets/VOC2007/{Annotations,JPEGImages,ImageSets/Main}` 布局。
这里的 VOC2007 是合并数据目录名，并不表示只包含 VOC2007 数据。
本项目对应 `data/raw/ore/{Annotations,JPEGImages}`，固定清单由
`data/owod/m-owodb/split_manifest.json` 管理。

## 与当前项目的 M 协议比较

比较官方 `datasets/OWOD_imagesets` 与 M manifest 的 `splits`：

| 文件 | 行数 | 图片身份、顺序、重复次数 |
| --- | ---: | --- |
| t1_train | 16551 | 一致 |
| t2_train | 45520 | 一致 |
| t2_ft | 1743 | 一致 |
| t3_train | 39402 | 一致 |
| t3_ft | 2361 | 一致 |
| t4_train | 40260 | 一致 |
| t4_ft | 2749 | 一致 |
| all_task_test | 10246 | 一致，9119 张唯一图片 |

现有清单来自已固定的 OW-DETR 发布版本，本次直接交叉核对 ORE。源 Git blob
除 t4_train 外存在末尾换行差异，不能声称全部字节哈希相同；图片序列完全一致。
不修改既有清单哈希，不重新运行作者的随机 balanced_ft 采样脚本。

四组各 20 类，逐类顺序经 AST 提取官方
`detectron2/data/datasets/pascal_voc.py` 常量验证，与项目完全一致。
六组 VOC/COCO 同义名按作者规则对应，例如 aeroplane/airplane、sofa/couch。

官方 `detectron2/data/build.py` 明确：增量训练仅保留当前新类；FT 保留全部
已知类；测试/验证将未来类映射为 unknown。原始 XML 有未来类标注不等于允许
训练读取这些标签。原始 loader 还删除过滤后无目标的图片。

`run.sh` 明确 Task 1 不做 FT，Task 2–4 各进行 train→ft，因此是四任务、
七个训练阶段。val 阶段拟合 Weibull，属于 ORE 方法的能量校准流程；不能把
该方法步骤混同于所有 OWOD 方法必须使用的训练标签或额外任务。

官方 `detectron2/evaluation/pascal_voc_evaluation.py` 强制 `_is_2007=False`，
AP 使用积分式 VOC AP，不因目录叫 VOC2007 就使用 11 点 AP。保留 inclusive
框 IoU、difficult 约定与 A-OSE/WI；项目报告 WI@0.8、U-Recall 和额外 H-Score。
H-Score 不是 ORE 原始指标。项目的 DINO/DETR、50/20 epoch recipe 不能称作
ORE 模型复现；官方 ORE 是独立架构和按 iteration 的训练配置。

## 原始数据覆盖核查与未完成事项

在本机完整 COCO2017 上对所有固定 M 清单的并集核查：

- 需要 98,459 张 train2017、4,167 张 val2017，共 102,626 张 COCO 图片。
- 所需图片在官方 JSON 中缺失 0，在本机 JPEG 目录缺失 0，无标注图片 0。
- 加上已整理的 21,503 张 VOC 图片，共需 124,129 张唯一图片。

这些结果证明 COCO 官方 JSON 具有重建所需的图片和标注覆盖。M 的 schema-3
manifest 已生成并把作者清单内嵌为协议源；本机 `data/raw/ore` 仍只有 VOC
部分，缺少 COCO 合并后的图像/XML，完整 M 训练前仍需按固定转换器补齐数据，
再运行 `manage_data verify --benchmark m-owodb`。
