# M/S 协议原文核对（2026-10-09）

2026-10-10 补充：[ORE 官方源码审计](ore-source-audit.md)确认存在作者 COCO
JSON→XML 转换脚本，当前 M 清单/类别顺序与 ORE 一致，现有原始数据覆盖完整。
下文“XML 尚缺失”描述生成状态，不意味着必须等待 Drive 原始包才能重建。

核对了本机 Zotero 中两篇完整 PDF 的相关正文、附录和表格，并比较了项目
中的作者清单、类别常量、生成标注、标签可见性和评测实现。

| 协议 | 原始来源 | 原文位置 | 每任务新增类别 |
| --- | --- | --- | --- |
| M-OWODB | Joseph 等，Towards Open World Object Detection，CVPR 2021 | PDF 第 6 页 §5.1/Table 1；§5.2；附录 F | 20/20/20/20 |
| S-OWODB | Gupta 等，OW-DETR: Open-world Detection Transformer，CVPR 2022 | PDF 第 9 页 A.2，第 10 页 Table 7 | 19/21/20/20 |

ORE 原文定义的是原始 OWOD 协议，不包含 S，也没有把两套划分命名为 M/S。
M/S 是后续文献的常用简称。这里的来源归属指协议，不主张这两个简称最早
出现在哪篇论文。两套都是四个任务；Task 2–4 的独立均衡微调是训练阶段，
不构成第五至第七个任务。

## 实现位置

- `tools/owod/benchmark_spec.py`：精确类别顺序、原始 COCO ID、指标约定。
- `data/owod/{m-owodb,s-owodb}/split_manifest.json`：固定作者图片身份、顺序、源 commit、原始文件 SHA-256 和行数。
- `tools/owod/prepare_benchmarks.py`：原始标注转项目格式，保留原始 VOC 框；
  增量只暴露当前类，微调只暴露已知类，测试保留未来标注。
- `tools/owod/manage_data.py`：plan/check/prepare/verify 数据入口。
- `tools/owod/protocol.py`、`strict_setting.py`：schema-3、哈希、类别/阶段与 checkpoint 链约束。
- `tools/owod/run_benchmark.py`：四任务顺序及三次独立均衡微调。
- `models/owod_benchmark_eval.py`：作者 AP50、WI@0.8、A-OSE、U-Recall，另加项目 H-Score。

## S 与 Table 7 的实际统计比较

| Task | 训练图片：论文/生成 | 训练实例：论文/生成 | 当前类测试图片：论文/生成 | 当前类测试实例：论文/生成 |
| --- | --- | --- | --- | --- |
| 1 | 89490/89490 | 421243/421243 | 3793/3793 | 17786/17786 |
| 2 | 55870/55870 | 163512/163512 | 2351/2351 | 7159/7159 |
| 3 | 39402/39402 | 114452/114452 | 1642/1642 | 4826/4826 |
| 4 | 38903/38903 | 160794/160794 | 1691/1691 | 7010/7010 |

作者完整测试清单共 4,952 张。Table 7 的测试图片数对应含本任务当前类别的
图片，不是四个互斥测试集合。保留原始完整测试清单，不能按这些数字自行抽样。

四任务类别顺序逐项匹配已核验的 OW-DETR 作者源码
`split_source` 所记录版本中的
`VOC_CLASS_NAMES/T2_CLASS_NAMES/T3_CLASS_NAMES/T4_CLASS_NAMES`，先转换
VOC/COCO 同义名后比较；该源文件 SHA-256 为
`1a3b0bcc7c38e08b3c33b22172f5c1c4c6297242847875f7491c1fd77fa70bc2`。
相关协议、序列化、评测、数据流程及短程指标比较测试：24 项、4 个 subtest 通过。
评测测试还直接调用本地作者 `voc_eval`，核对积分/11 点 AP、A-OSE 与 WI。

S schema-3 manifest SHA-256：
`0060c5f7b080ea417f88295fdcabd7308cc745610319a160ce9787e219d87640`。
M schema-3 manifest SHA-256：
`a5bd4b44dbded3f7f5e62bc3e95e2f69ac9f3eedcead3b3daeb64c32a59daa94`。
实际审计结果保存在忽略目录 `output/discovery/s-table7-audit.json` 与
`output/discovery/s-class-order-audit.json`。

## 结论范围和尚未完成的项目

S 的数据划分、类别顺序、关键表格统计及相关实现检查一致。该核对不等于
复现 OW-DETR 的模型结果，也不证明本项目 GNN/adapter 方法有效。
本机 batch 2、encoder activation checkpointing 与短程更新预算属于项目
资源/可行性设置，不是作者完整训练实验的复现。

M 的类别顺序、作者清单和协议代码已实现并有对应测试，schema-3 manifest
已经把全部作者清单和来源哈希内嵌为唯一协议源。21,503 对必需 VOC 图像/XML
已准备；本机 ORE 仍缺少 102,626 张 COCO 图像/XML，因此 `verify` 会报告数据
缺口，不能把 M 标记为可训练完成。

ORE 的 held-out unknown 验证与 EBUI 是原方法训练机制。项目当前不自行
抽样作者未发布的 held-out 身份，也不使用测试 future GT 调阈值。benchmark
协议实现和 ORE/OW-DETR 模型机制须分别归属。H-Score 是项目附加指标，
不是 ORE 原文指标。
