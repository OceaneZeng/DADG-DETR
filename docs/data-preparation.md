# 实验数据准备

数据管理入口为 `tools/owod/manage_data.py`。它只按
`data/owod/<protocol>/split_manifest.json` 中嵌入的作者清单选择数据，不生成随机
划分，不根据类别 ID 大小重排任务。两套基准分别准备、分别训练和报告。

## 1. 准备原始数据

| 基准 | 原始输入 | 原始输入目录 |
| --- | --- | --- |
| M-OWODB | ORE 作者发布的 VOC2007/2012 + COCO 合并版 | `Annotations/*.xml`、`JPEGImages/*.jpg` |
| S-OWODB | 完整 COCO2017 | `train2017/`、`val2017/`、`annotations/instances_train2017.json`、`annotations/instances_val2017.json` |

M 可使用 [ORE 作者数据入口](https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k)
的完整发布版，或依据已核验的作者转换代码从 VOC/COCO2017 重建。
2026-10-10 已确认官方仓库包含 COCO JSON→XML 转换器，具体规则和覆盖核查见
[ORE 源码审计](ore-source-audit.md)。重建入口为 `tools/owod/prepare_ore.py`，
下载入口不会自动执行该重建。
COCO2017 单独不足以构造 M 的 VOC Task 1。

S 从 COCO 官方获取
[train2017](https://images.cocodataset.org/zips/train2017.zip)、
[val2017](https://images.cocodataset.org/zips/val2017.zip)、
[2017 标注](https://images.cocodataset.org/annotations/annotations_trainval2017.zip)。
解压后保留官方目录结构。

下载入口现已提供（需要安装 `requirements.txt` 中的 filelock/gdown）：

```bash
python tools/owod/download_data.py --benchmark both
# 本机代理可加：--proxy http://127.0.0.1:10808
```

脚本下载官方 COCO ZIP 与 DINO ResNet-50，支持断点续传、目标锁、来源/哈希
记录与安全解压。COCO 使用同一官方 S3 bucket 的有效 TLS 地址，不关闭证书
校验。M 下载分支仍尝试获取 ORE 合并发布版；2026-10-08 访问作者 Drive 时未
找到可下载文件，失败会明确记录。作者源码重建是另一条可行路径，但下载失败
不能用猜测的转换规则或重新抽样的清单替代。

默认布局如下；原始数据也可以保存在另一块磁盘上，修改配置中的路径即可。

```text
data/
  raw/
    ore/
      Annotations/
      JPEGImages/
    coco/
      annotations/
      train2017/
      val2017/
  owod/                    # 脚本生成的标注与 manifest
    m-owodb/
    s-owodb/
```

数据路径可以直接通过 `--ore-root`、`--coco-root` 和 `--output-root` 传入；相对
路径按项目根目录解析，Windows 绝对路径建议写成 `D:/datasets/coco`。这些路径
必须指向真实原始数据。

已下载的 VOC 原始 TAR 可以这样整理：

```bash
python tools/owod/prepare_voc.py --downloads C:/Users/23642/Downloads
```

脚本分别保留三个 release 的全部原始目录：

```text
data/raw/voc/
  voc2007-trainval/VOCdevkit/VOC2007/
  voc2007-test/VOCdevkit/VOC2007/
  voc2012-trainval/VOCdevkit/VOC2012/
```

随后严格按已固定的 M 作者清单，将所需 21,503 对 VOC 图像/XML 放到
`data/raw/ore/JPEGImages` 和 `data/raw/ore/Annotations`。同盘优先使用硬链接，
跨盘复制并核验哈希；原始 TAR 不移动、不删除，已有不同内容不覆盖。
原始 VOC2012 TAR 包含 17,125 对图像/XML，而作者检测 trainval 清单使用
11,540 张；额外原始文件保留在 release 目录，不进入自定义划分。

`data/raw/ore/voc_source_manifest.json` 记录来源、原始清单哈希及检查结果；
该文件是原始数据来源记录，不是 schema-3 的实验 split manifest。
整理 VOC 不会虚构缺失的作者 COCO XML，也不会把不完整 M 数据标记为可训练。

## 2. 查看和生成划分

服务器已上传六个 VOC/COCO 压缩包时，在项目根目录依次执行：

```bash
mkdir -p data/raw/coco
unzip -n -q data/raw/archives/train2017.zip -d data/raw/coco
unzip -n -q data/raw/archives/val2017.zip -d data/raw/coco
unzip -n -q data/raw/archives/annotations_trainval2017.zip -d data/raw/coco
python tools/owod/prepare_voc.py --downloads data/raw/archives
python tools/owod/prepare_ore.py --coco-root data/raw/coco --ore-root data/raw/ore
python tools/owod/manage_data.py prepare --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod
python tools/owod/manage_data.py verify --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod --decode-images
```

每条命令成功后再执行下一条。已成功解压的包无需重复解压。
`prepare_ore.py` 仅发布固定 M 清单所需的 COCO 图像/XML，保留全部原始
类别和 crowd 目标，严格采用作者坐标转换规则。图像同盘优先硬链接；不同
内容的已有 XML/图片拒绝覆盖。`coco_source_manifest.json` 记录作者 commit、
转换实现/输入 JSON/输出 XML/图片哈希；它不替代正式 schema-3 split manifest。
若已有目录来自其他机器或输入版本，使用全新的输出目录，不手工修改旧 manifest。

在项目根目录运行：

```bash
python tools/owod/manage_data.py plan
python tools/owod/manage_data.py check --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco
python tools/owod/manage_data.py prepare --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod
python tools/owod/manage_data.py verify --benchmark both --ore-root data/raw/ore --coco-root data/raw/coco --output-root data/owod --decode-images
```

- `plan`：核验发布清单哈希，列出每任务的新类、已知类和图片数量，无需原始数据。
- `check`：检查两套基准需要的全部原始图像和标注路径，报告缺失数量及样例。
- `prepare`：转换标注、验证四任务协议后发布数据。转换失败不留下半成品任务目录。
- `verify`：检查全部阶段文件哈希、标签可见性、图片存在性及原始标注来源哈希。
  `--decode-images` 进一步验证图片解码。

只操作一套基准时加 `--benchmark m-owodb` 或 `--benchmark s-owodb`；默认两套。
也可以直接覆盖路径：

```bash
python tools/owod/manage_data.py prepare --benchmark m-owodb --ore-root D:/datasets/ORE --output-root data/owod
python tools/owod/manage_data.py prepare --benchmark s-owodb --coco-root D:/datasets/coco --output-root data/owod
```

所有命令可加 `--report 路径.json` 保存检查报告。缺数据、哈希变化或协议错误时
退出码为 1。再次 `prepare` 会验证并复用相同数据，不会重新抽样；旧协议或半成品
非空目录会被拒绝。原始图像不复制、不移动，两套基准引用配置中的原始图像。

## 3. 每个任务实际使用什么

| 项目 | M | S |
| --- | --- | --- |
| 每任务新增类别 | 20 / 20 / 20 / 20 | 19 / 21 / 20 / 20 |
| 增量训练清单图片数 | 16551 / 45520 / 39402 / 40260 | 89490 / 55870 / 39402 / 38903 |
| Task 2–4 均衡微调清单图片数 | 1743 / 2361 / 2749 | 1679 / 2345 / 2664 |
| 发布测试清单 | 10246 行，9119 张唯一图片 | 4952 张图片 |

每个基准生成 `stage_0` 至 `stage_3`，对应 Task 1–4。每阶段文件为：

| 文件 | 用途与可见标签 |
| --- | --- |
| `instances_increment_train2017.json` | 作者本任务训练图片，仅本任务新类标签 |
| `instances_train2017.json` | Task 1 为首任务训练集；Task 2–4 为作者均衡微调集，含全部已知类标签 |
| `instances_val2017.json` | 同一作者测试图片，仅已知类标注的视图 |
| `instances_val2017_full.json` | 同一作者测试图片、完整 80 类标注，用于开放世界评估 |
| `split_manifest.json`（基准根目录） | 协议、类别顺序、全部作者清单、阶段路径、数据来源与哈希 |

任务间的作者图片重叠保留。M 的作者测试清单重复行也保留于 manifest，评估按
原始次数处理，JSON 图片记录本身保持唯一。未知类标签不进入增量/均衡训练，
测试时所有未来类合并为 unknown；Task 4 不报告 unknown 指标。

文件名里的 `val` 是适配现有 loader 的命名，指作者测试集，不能用来训练、
拟合未知阈值或选择超参数。ORE 的 1000 张 held-out 验证及 EBUI 属于方法复现；
没有作者清单时，本工具不会自行添加这种划分。

## 4. 开始实验

训练使用 `data/owod/<benchmark>/split_manifest.json`。标准训练入口
`tools/owod/run_benchmark.py` 执行 Task 1 增量训练，然后依次执行 Task 2–4
的增量训练和均衡微调；方法 recipe、DINO 预训练和评估约定见
[协议文档](owod-benchmarks.md)。

当前完成的是数据协议与本地 detector 运行支持；没有真实数据准备/训练记录时，
不得报告已复现 ORE 或 OW-DETR 论文数值。
