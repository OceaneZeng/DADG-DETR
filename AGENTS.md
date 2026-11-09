# Tree-DETR 项目约定

## 实验协议与数据

- 正式实验只用两套作者协议：M-OWODB 来自 ORE（CVPR 2021），S-OWODB 补充采用 OW-DETR（CVPR 2022）附录 A.2 / Table 7；S 不是 ORE 定义的。以 `configs/owod/` 和 [协议说明](docs/owod-benchmarks.md) 为准。
- 保持作者 ImageSets、图像身份和类别顺序；M 新类数为 20/20/20/20，S 为 19/21/20/20。禁止随机、启发式、S 的 20×4 或 DEUS 替代划分。允许作者清单的跨任务图像重叠，禁止重复标注记录和未来类别监督。
- 四个 Task 对应七个训练阶段：Task 1 increment，随后 Task 2–4 各 increment → balanced finetune；代码 stage 从 0 开始。使用作者发布的均衡记忆清单，不换成历史自定义 replay exposure / budget。
- 数据统一由 `tools/owod/manage_data.py` 的 `plan/check/prepare/verify` 管理，原始源路径与生成目录分别配置。正式训练要求 schema-3 manifest 和输入哈希核验；缺少作者数据时报告缺口，不拼造替代数据，不使用 `allow-unverified`。
- 续训只接同一 manifest、方法版本、运行计划及正确 Task/phase 链。保留有效 checkpoint 和已有成果；调整吞吐、卡数、batch 或训练逻辑前说明对续训兼容性、优化轨迹和 AP 可比性的影响，不绕过校验。诊断权重和旧 pilot 不能初始化正式训练或作为正式成绩。

## 方法与验收

- 检测器沿用 Deformable DETR；方法改造围绕未知目标发现、未知→已知学习及旧类保持，不能仅以抗遗忘替代 OWOD 目标。保持 GNN、adapter、联合目标的作用与实际代码对应，不为解释旧图而悄悄换回历史方法。
- 论文、架构图和实现必须一致；说明模块输入输出、衔接及训练位置。具体模块名称、参数冻结、路由与评分以当前方法文档和运行配置为准，不能把旧三模块名称或某次诊断设定当成永久架构。按需阅读 [图路由适配](docs/graph-routed-adaptation.md)、[发现与转化](docs/owod-discovery-transition-design.md) 和对应版本说明。
- 外部对照选择真实 OWOD 方法，与本方法核对相同的数据协议和评测口径；内部 control、消融、本地方法重实现及作者源码复现分别标明。复现数据协议不等于复现 ORE/OW-DETR 模型或论文数值；架构、优化器、预训练、微调日程及未知监督单独归因，原文未给的超参数不自行补成“论文设定”。
- 效果验收使用实际评估的 Previous / Current / Known AP50，以及有未知类时的 U-Recall、WI@recall=0.8、A-OSE；H-Score 是项目附加指标，Task 4 不报未知指标。明确作者 VOC 评测约定及 AP 版本，不能换用 COCO 101 点 AP50。机制测试、loss、局部覆盖或小数据拟合通过，不能证明全量 AP 提升。

## 本地与服务器

- 代码在本地仓库修改，用户通过 Git 同步并启动服务器训练；不要直接改服务器代码或用实验压缩包代替 Git。读取服务器日志或运行已授权的相关测试，仍须区分本地文件与远端实际运行配置。
- 本地是 Windows / PowerShell，训练服务器是 Linux / Bash + Conda。命令注明所在环境；Bash 续行用 `\`，PowerShell 用反引号，避免复制出 `\_`、`\--` 等 Markdown 转义。优先使用已有 `tree-detr` 环境，不因 baseline 接入另建或改换环境。
- Linux 环境与 CUDA 扩展按 `environment.yml`、`tools/setup_linux.sh`；Windows 构建按 `tools/windows/build_extension.ps1`。两端 GPU 架构不同，不能直接共用编译产物或照搬 CUDA 架构值。GPU 编号及卡数按实际服务器和本次运行配置核对。
- 每个 baseline、主方法和诊断实验都保存可追踪的训练日志。给启动指令时一并给出本次日志、结果目录和查看方式；服务器长训练使用 tmux。沿用当前启动器的输出路径和日志格式，续训继续对应日志，不把不同运行混为一份结果。

## 当前入口与目录边界

从仓库根目录、已激活的项目 Python 环境核对数据和训练参数：

```text
python tools/owod/manage_data.py plan
python tools/owod/manage_data.py --help
python tools/owod/run_benchmark.py --help
```

数据操作详见 [数据指南](docs/data-preparation.md)，目录职责见 [项目目录](docs/project-layout.md)。正式全量启动/恢复按 [discovery v5](docs/owod-discovery-v5.md) 核对；实验性渐进监督 v6 见 [调试记录](docs/owod-discovery-v6-debug.md)，双 3090 的 v7 损失对照见 [v7 调试记录](docs/owod-discovery-v7-debug.md)，已知／未知评分约束、发现梯度对照及 8 轮暂停入口见 [v8 调试记录](docs/owod-discovery-v8-debug.md)，不要从聊天旧命令直接启动。

- `baselines/` 集中保存外部源码；`tools/analysis/` 只读历史日志、绘图和汇总，不启动旧实验。不要恢复已删除的 pilot 启动器、旧 setting 或论文导出文件。
- Git 保留源码、运行配置、固定作者 ImageSets、必要 Markdown 和许可证；生成文档、图像、压缩包、权重和日志保持忽略，不 force-add。清理时保留原始数据、权重、指标日志、外部源码版本及无关用户改动；论文压缩包不随意解压进实验仓库。

本次规则的原始消息证据、覆盖限制和阶段状态见 [research/selection.md](research/selection.md)。通用沟通、自主范围与检查范围沿用全局 AGENTS.md，不在此重复。
