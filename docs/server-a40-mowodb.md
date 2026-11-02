# A40 GPU 0：M-OWODB discovery v5 全量实验

本分支为独立的 M-OWODB 正式任务链。使用作者固定 ImageSets，20/20/20/20
新类及发布的均衡记忆，四任务共七个训练阶段；不接续 S-OWODB、control、旧版本
或诊断 checkpoint。保留已有实验目录。

## 运行配置

- 第一张 A40：PCI 顺序 GPU 0，单进程，batch 4，workers 4，seed 42。
- 完整 discovery v5，hard / frozen / hierarchical，activation checkpointing。
- DINO ResNet-50 骨干；increment 50 轮、学习率 2e-4；Task 2–4 各平衡微调
  20 轮、学习率 2e-5；阶段内不降学习率。任务间校准 GNN，Task 2 起旧类蒸馏。
- `ow-detr-paper` 表示采用仓库中的训练配方，不是 ORE 或 OW-DETR 模型复现。
- 总 batch 与旧双卡每卡 2 相同，但单卡采样、浮点计算及优化轨迹可能不同，
  不保证 AP 一致。M 与 S 是不同协议，结果分别报告。
- batch 4 尚未在目标 A40 实测；若 OOM，保留日志后重新规划配置及独立输出，
  不修改运行计划强行续训，也不默默改为 batch 2。

## 环境与数据准备（新服务器 Linux / Bash）

以下命令均在新服务器的仓库根目录执行。先进入你实际 clone 的目录；不要照搬旧服务器路径。

```bash
git fetch origin
git switch --track origin/mowodb
# 若本地已有 mowodb 分支，改用 git switch mowodb
git pull --ff-only origin mowodb
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.6 bash tools/setup_linux.sh
conda activate tree-detr
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

优先复用 `tree-detr` 环境；setup 会更新该环境至 environment.yml 的依赖。
若已有环境仅以绝对路径管理，先激活该匹配环境，再按脚本的检查和构建步骤执行，
不要把环境路径传给当前使用 `--name` 的 setup 脚本。
保留 PyTorch 2.4.1 / torchvision 0.19.1 / CUDA runtime 与 toolkit 12.1，
不要依据 nvidia-smi 显示的 CUDA 13.0 替换项目 toolkit。A40 使用 8.6 编译，
扩展必须在此服务器构建，不复制 Windows 编译产物。

将下方 ORE_ROOT 替换为真实作者数据根目录，包含 JPEGImages 和 Annotations。
若尚无 ORE 合并数据，先按 [数据指南](data-preparation.md) 使用
`download_data.py --benchmark m-owodb --m-source reconstruct` 下载 VOC/COCO，
再依次运行 prepare_voc.py 和 prepare_ore.py 完成作者规则转换。
准备过程引用原始图片；路径必须在训练期间保持有效。迁移旧 manifest 后不要手工改路径，
应从真实原始数据重新 prepare，生成本机 schema-3 manifest。

```bash
ORE_ROOT=/absolute/path/to/ORE
python tools/owod/manage_data.py plan
python tools/owod/manage_data.py check --benchmark m-owodb --ore-root "$ORE_ROOT"
python tools/owod/manage_data.py prepare --benchmark m-owodb --ore-root "$ORE_ROOT" --output-root data/owod
python tools/owod/manage_data.py verify --benchmark m-owodb --ore-root "$ORE_ROOT" --output-root data/owod --decode-images
test -f pretrained/dino_resnet50.pth
```

每条检查成功后再继续；缺作者数据或 DINO 权重时停止并补齐真实来源，不使用替代划分、
allow-unverified 或旧训练权重。DINO 下载和数据来源见 [数据指南](data-preparation.md)。

## 启动、日志与恢复

在仓库根目录、已激活的 tree-detr 环境中运行：

```bash
tmux new -s tree-m-a40
bash tools/owod/run_discovery_mowodb_a40.sh --dry-run \
  && bash tools/owod/run_discovery_mowodb_a40.sh
```

dry-run 也会核验数据并打印计划，不执行训练，不证明 GPU 显存足够。
启动器限制 GPU 0，且只接受 --dry-run / --resume。不会额外启动 M control；
本次直接执行完整方法，已有 S control 不能冒充 M 的对照。

- 控制台：`output/tree-detr-m-v5-a40-logs/console.log`。
- 计划及阶段结果：`output/tree-detr-m-v5-a40-full/m-owodb/seed_42/`。
- 每阶段：`stage_0/increment/`，随后 `stage_1` 至 `stage_3` 的 increment / finetune。
  每阶段保留 metrics.jsonl、run_config.json、checkpoint 及训练日志。

Ctrl+B、D 脱离 tmux。另一个终端在仓库根目录查看：

```bash
tail -n 50 -f output/tree-detr-m-v5-a40-logs/console.log
```

中断后进入同一仓库和环境，在 tmux 中运行：

```bash
bash tools/owod/run_discovery_mowodb_a40.sh --resume
```

恢复沿用相同数据、方法版本、卡数、batch、路径及计划；不绕过任何校验。
首轮检查有限 loss/梯度和监督覆盖；warmup 后联合检查已知 AP 与未知发现。
正式报告固定末轮的 Previous / Current / Known AP50、U-Recall、WI@0.8、A-OSE，
遵循作者 VOC AP 约定；Task 4 不报未知指标。实现检查不代表效果提升。

本地 Windows 无法验证此服务器的 Conda 安装、CUDA 扩展构建、数据完整性或显存峰值；
以上项目以服务器实际运行结果为准。
