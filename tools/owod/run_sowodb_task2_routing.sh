#!/usr/bin/env bash
# Same-weight measurements, then fresh paired v7 hard/soft Task 2 runs.
set -euo pipefail
if (( $# )) && [[ $# != 1 || $1 != --check ]]; then
  echo 'Accepted option: --check (CPU provenance and data only).' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_bin="${TREE_DETR_PYTHON:-/home/top/disks/new-hdd/conda_envs/tree-detr/bin/python}"
output="${SOWODB_TASK2_OUTPUT:-output/sowodb-task2-v7-routing-8epochs-2gpu}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${SOWODB_GPUS:-0,1}"
export LOCAL_SIZE=2
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$python_bin" ]] || { echo "tree-detr Python unavailable: $python_bin" >&2; exit 1; }
environment_root="$(cd "$(dirname "$python_bin")/.." && pwd)"
# Preserve the existing environment; use its CUDA 12.1 toolkit/compiler rather
# than the unrelated system CUDA 13.x and GCC 13 defaults.
export PATH="$environment_root/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/models/ops${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$environment_root/bin/nvcc" ]]; then
  export CUDA_HOME="$environment_root"
fi
if [[ -x "$environment_root/bin/x86_64-conda-linux-gnu-g++" ]]; then
  export CXX="$environment_root/bin/x86_64-conda-linux-gnu-g++"
  export CC="$environment_root/bin/x86_64-conda-linux-gnu-gcc"
fi
export LD_LIBRARY_PATH="$environment_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$output"
# Refuse two launchers writing the same plan/checkpoint/log concurrently.
exec 9>"$output/run.lock"
flock -n 9 || { echo "This output directory already has a running experiment: $output" >&2; exit 1; }
exec > >(tee -a "$output/console.log") 2>&1
trap 'echo "Exit status: $?; results: $output/results; log: $output/console.log"' EXIT
git rev-parse HEAD
git branch --show-current
git status --short
"$python_bin" tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod
"$python_bin" tools/owod/run_sowodb_task2_routing.py --preflight --output "$output/results"
if [[ ${1:-} == --check ]]; then
  exit 0
fi
[[ -n ${TMUX:-} ]] || { echo 'Run in tmux: tmux new -s s-task2-8e "bash tools/owod/run_sowodb_task2_routing.sh"' >&2; exit 1; }
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.free --format=csv
"$python_bin" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expected exactly two selected CUDA GPUs'
for index in range(2):
    assert '3090' in torch.cuda.get_device_name(index), 'Select two RTX 3090 cards'
    torch.ones(1, device=f'cuda:{index}')
    free, _ = torch.cuda.mem_get_info(index)
    assert free >= 14 * 2**30, f'Selected GPU {index}: less than 14 GiB free'
    print(index, torch.cuda.get_device_name(index), 'free GiB:', round(free / 2**30, 2))
PY
# A previous git clean can remove the in-place .so while leaving a stale egg
# compiled against another PyTorch. Rebuild only this ignored local artifact.
if ! "$python_bin" -c 'import torch; import MultiScaleDeformableAttention as extension; print("CUDA extension:", extension.__file__)'; then
  echo 'Rebuilding the local CUDA extension for the existing PyTorch and RTX 3090.'
  export TORCH_CUDA_ARCH_LIST=8.6
  export MAX_JOBS=4
  "$python_bin" models/ops/setup.py build_ext --inplace --force
  "$python_bin" -c 'import torch; import MultiScaleDeformableAttention as extension; print("CUDA extension:", extension.__file__)'
fi
# Fail on a broken NCCL/extension/backward path before any long data run.
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/owod/task2_routing_cuda_check.py
resume=()
if [[ -f "$output/results/diagnostic_plan.json" ]]; then
  resume=(--resume)
fi
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/owod/run_sowodb_task2_routing.py --output "$output/results" "${resume[@]}"
