#!/usr/bin/env bash
# Fresh focal reference and trusted-background treatment on physical RTX 3090 0/1.
set -euo pipefail
if (( $# )) && [[ $# != 1 || $1 != --check ]]; then
  echo 'Accepted option: --check (CPU data/plan validation only).' >&2; exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_bin="${TREE_DETR_PYTHON:-/home/top/disks/new-hdd/conda_envs/tree-detr/bin/python}"
output="${MOWODB_BACKGROUND_OUTPUT:-output/mowodb-task1-background-8epochs-2gpu}"
[[ -x "$python_bin" ]] || { echo "Missing existing tree-detr Python: $python_bin" >&2; exit 1; }
environment_root="$(cd "$(dirname "$python_bin")/.." && pwd)"
export PATH="$environment_root/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/models/ops${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$environment_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
if [[ -x "$environment_root/bin/nvcc" ]]; then export CUDA_HOME="$environment_root"; fi
if [[ -x "$environment_root/bin/x86_64-conda-linux-gnu-g++" ]]; then
  export CXX="$environment_root/bin/x86_64-conda-linux-gnu-g++"
  export CC="$environment_root/bin/x86_64-conda-linux-gnu-gcc"
fi
mkdir -p "$output"
exec 9>"$output/run.lock"
flock -n 9 || { echo "An experiment is already using $output" >&2; exit 1; }
exec > >(tee -a "$output/console.log") 2>&1
trap 'echo "Exit status: $?; log: $output/console.log; report: $output/comparison.json"' EXIT
git rev-parse HEAD
git branch --show-current
if [[ ${1:-} != --check ]]; then
  [[ -n ${TMUX:-} ]] || { echo 'Run in tmux: tmux new -s m-bg8 "bash tools/owod/run_mowodb_background_pair.sh"' >&2; exit 1; }
  "$python_bin" tools/owod/run_mowodb_background_pair.py --gpu-check
fi
"$python_bin" tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
"$python_bin" tools/owod/run_mowodb_background_pair.py --output "$output" --check
if [[ ${1:-} == --check ]]; then exit 0; fi
if ! "$python_bin" -c 'import torch; import MultiScaleDeformableAttention as extension; print(extension.__file__)'; then
  export TORCH_CUDA_ARCH_LIST=8.6
  export MAX_JOBS=4
  "$python_bin" models/ops/setup.py build_ext --inplace --force
  "$python_bin" -c 'import torch; import MultiScaleDeformableAttention as extension; print(extension.__file__)'
fi
"$python_bin" -u tools/owod/run_mowodb_background_pair.py --output "$output"
