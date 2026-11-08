# 可选的本地 paper-baseline 实现

`models/paper_baselines/` 保存的是项目内的受控本地实现，用于在相同的
M-OWODB / S-OWODB 数据协议和评估口径下做内部对照。它们不等同于
ORE、OW-DETR、PROB、OWOBJ、CAT 或 EW-DETR 的作者源码，也不能据此声称
复现论文数值。

## 当前边界

- 标准四任务、七阶段入口是 `tools/owod/run_benchmark.py`；它默认运行
  `control`、`graph-adapter` 或 `discovery`。paper-baseline 不属于正式主方法链。
- `main.py --paper-baseline` 仍可用于显式的本地受控对照，运行时必须记录
  `paper_baseline`、manifest、recipe、预训练来源和 checkpoint 哈希。
- `cat` 还需要先用 `tools/owod/prepare_cat_proposals.py` 生成与已核验
  M-OWODB manifest 绑定的 proposal 数据库。
- 外部作者源码位于 `baselines/`，本目录的实现与那些源码分开维护。

## 方法范围

`ow-detr`、`prob`、`owobj`、`cat` 和 `ew-detr` 是本地适配入口。代码中未明确
给出的原文超参数、数据处理或训练日程不能自行补成“论文设定”；每次运行的
具体选择都应写入 `run_config.json`，并与 control 和项目方法分开统计。

如果后续不再需要这些内部对照，应整体移除 `models/paper_baselines/`、CAT
proposal 生成器、paper checkpoint verifier、对应 parser 参数和测试；不要只
删除其中一个方法而留下半套入口。
