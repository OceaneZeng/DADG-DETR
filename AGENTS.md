# Tree-DETR 项目约定

## 实验协议与数据

- 正式实验只用两套作者协议：M-OWODB 来自 ORE（CVPR 2021），S-OWODB 补充采用 OW-DETR（CVPR 2022）附录 A.2 / Table 7；S 不是 ORE 定义的。以 `configs/owod/` 和 [协议说明](docs/owod-benchmarks.md) 为准。
- 保持作者 ImageSets、图像身份和类别顺序；M 新类数为 20/20/20/20，S 为 19/21/20/20。禁止随机、启发式、S 的 20×4 或 DEUS 替代划分。允许作者清单的跨任务图像重叠，禁止重复标注记录和未来类别监督。
- 四个 Task 对应七个训练阶段：Task 1 increment，随后 Task 2–4 各 increment → balanced finetune；代码 stage 从 0 开始。使用作者发布的均衡记忆清单，不换成历史自定义 replay exposure / budget。
- 数据统一由 `tools/owod/manage_data.py` 的 `plan/check/prepare/verify` 管理，原始源路径与生成目录分别配置。正式训练要求 schema-3 manifest 和输入哈希核验；缺少作者数据时报告缺口，不拼造替代数据，不使用 `allow-unverified`。
- 续训只接同一 manifest、方法版本、运行计划及正确 Task/phase 链。保留有效 checkpoint 和已有成果；调整吞吐、卡数、batch 或训练逻辑前说明对续训兼容性、优化轨迹和 AP 可比性的影响，不绕过校验。诊断权重和旧 pilot 不能初始化正式训练或作为正式成绩。

## 方法与验收

- 检测器沿用 Deformable DETR。方法主线是：①从当前训练图像挖掘未知候选并学习未知表示；②通过 discovery adapter 学习未知目标，标注释放后将可匹配的历史知识转化为新类适配；③控制增量更新干扰并保持旧知识。第一、二模块服务于主动发现和学习 unknown，不能缩减成历史表示读取或旧类抗遗忘，也不能用回放代替未知学习。作者规定的均衡记忆仍保留；这里的“主动学习”不表示新增人工查询标注流程。
- 论文、架构图和实现必须一致；说明模块输入输出、衔接及训练位置。具体模块名称、参数冻结、路由与评分以当前方法文档和运行配置为准，不能把旧三模块名称或某次诊断设定当成永久架构。按需阅读 [图路由适配](docs/graph-routed-adaptation.md)、[发现与转化](docs/owod-discovery-transition-design.md) 和对应版本说明。
- 解释 discovery 方法时，区分 query–prototype affinity 与类别更新干扰风险；双适配器指共享发现路径与持久类别路径。未知原型的内部编号不等于真实类别，快照继承不等于自动命名。保持 GNN、adapter、联合目标与代码对应，不为解释旧图换回历史方法。
- 方法写作先说明具体研究问题如何引出所选机制；不能只列功能收益，或用 OWOD 共性困难代替关系建模的理由。区分已有工作、本文设计假设和实验结论；问题定义描述任务，不堆入作者清单、文件或实现合规措辞。参考相关论文的对应章节，而非只替换连接词。
- 提出或实质改造机制时，用相关原始论文核对可行性与新颖性，说明具体未解决问题和自己的设计依据；已有技术的拼接、实现接通和设计收益不直接作为创新或效果证据。
- 外部对照选择真实 OWOD 方法，与本方法核对相同的数据协议和评测口径；内部 control、消融、本地方法重实现及作者源码复现分别标明。复现数据协议不等于复现 ORE/OW-DETR 模型或论文数值；架构、优化器、预训练、微调日程及未知监督单独归因，原文未给的超参数不自行补成“论文设定”。
- 效果验收使用实际评估的 Previous / Current / Known AP50，以及有未知类时的 U-Recall、WI@recall=0.8、A-OSE；H-Score 是项目附加指标，Task 4 不报未知指标。明确作者 VOC 评测约定及 AP 版本，不能换用 COCO 101 点 AP50。机制测试、loss、局部覆盖或小数据拟合通过，不能证明全量 AP 提升。
- 分析“是否学到 unknown”时，联合检查未知召回、预测数量/precision 与已知类表现；伪标签或非零发现损失仅说明监督参与，不能证明有效发现或整体改善。阶段中途观测与固定末轮成绩分开，缺失指标不填零。

## 本地与服务器

- 代码在本地仓库修改，用户通过 Git 同步并启动服务器训练；不要直接改服务器代码或用实验压缩包代替 Git。读取服务器日志或运行已授权的相关测试，仍须区分本地文件与远端实际运行配置。
- 排障、解释方法或给启动指令时，核对目标运行的 Git 分支/提交、M/S 协议、方法版本和运行计划；分支名不决定协议，本地默认也不代表服务器正在运行的设置。正式入口与实验性修正分开，不能把诊断中加载旧权重当成正式跨版本续训兼容。
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

数据准备时查 [数据指南](docs/data-preparation.md)，整理文件时查 [项目目录](docs/project-layout.md)。全量启动/恢复按对应版本文档与脚本核对；当前正式入口见 [discovery v5](docs/owod-discovery-v5.md)，实验性 v6 的范围与限制见 [调试记录](docs/owod-discovery-v6-debug.md)。不要从聊天旧命令直接启动。

- `baselines/` 集中保存外部源码；`tools/analysis/` 只读历史日志、绘图和汇总，不启动旧实验。不要恢复已删除的 pilot 启动器、旧 setting 或论文导出文件。
- Git 保留源码、运行配置、固定作者 ImageSets、必要 Markdown 和许可证；生成文档、图像、压缩包、权重和日志保持忽略，不 force-add。清理时保留原始数据、权重、指标日志、外部源码版本及无关用户改动；论文压缩包不随意解压进实验仓库。

规则证据、历史取舍与阶段快照见 [research/selection.md](research/selection.md)，其中运行状态不作为实时配置。通用沟通、自主范围与检查范围沿用全局 AGENTS.md，不在此重复。
