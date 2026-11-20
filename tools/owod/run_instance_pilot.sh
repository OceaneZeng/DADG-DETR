#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_bin="${TREE_DETR_PYTHON:-/home/top/disks/new-hdd/conda_envs/tree-detr/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Set TREE_DETR_PYTHON to the Python executable in the tree-detr environment." >&2
  exit 1
fi
env_dir="$(cd "$(dirname "$python_bin")/.." && pwd)"
export PATH="$env_dir/bin:$PATH"
export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == --decoder-bbox-lr-pair || ${1:-} == --decoder-bbox-lr-three-quarter ]]; then
  dir=output/m-instance-decoder-bbox-lr-half
  lr_args=()
  if [[ $1 == --decoder-bbox-lr-three-quarter ]]; then
    dir=output/m-instance-decoder-bbox-lr-three-quarter
    lr_args+=(--three-quarter)
  fi
  shift
  if [[ ${1:-} == --plan ]]; then
    exec "$python_bin" tools/owod/run_decoder_bbox_lr_pair.py "${lr_args[@]}" "$@"
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_decoder_bbox_lr_pair.py "${lr_args[@]}" "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --stronger-parent-study ]]; then
  shift
  if [[ ${1:-} == --plan ]]; then
    exec "$python_bin" tools/owod/run_stronger_parent_study.py "$@"
  fi
  dir=output/m-instance-stronger-parent
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_stronger_parent_study.py "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --cross-attention-study ]]; then
  shift
  if [[ ${1:-} == --plan ]]; then
    exec "$python_bin" tools/owod/run_cross_attention_study.py "$@"
  fi
  dir=output/m-instance-cross-attention
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_cross_attention_study.py "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --schedule-pair ]]; then
  dir=output/m-instance-schedule-pair
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_schedule_pair.py 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --localization-pair ]]; then
  dir=output/m-instance-localization-pair
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_localization_pair.py 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --classifier-pair ]]; then
  dir=output/m-instance-classifier-pair
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_localization_pair.py --classifier-lr 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --ranking-probe ]]; then
  dir=output/m-instance-ranking-probe
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_ranking_probe.py 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --objectness-probe ]]; then
  dir=output/m-instance-objectness-probe
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_ranking_probe.py --objectness 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --plan ]]; then
  exec "$python_bin" tools/owod/run_instance_pilot.py "$@"
fi
mkdir -p output/m-instance-expanded
"$python_bin" -u tools/owod/run_instance_pilot.py "$@" 2>&1 | tee -a output/m-instance-expanded/console.log
