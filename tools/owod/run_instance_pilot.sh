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
if [[ ${1:-} == --s-task2-wi-sources || ${1:-} == --task2-old-only-independent || ${1:-} == --s-task2-current-phase || ${1:-} == --s-task2-increment-giou || ${1:-} == --s-task2-increment-classifier-lr || ${1:-} == --s-task2-current-fp-sources || ${1:-} == --s-task2-current-exposure || ${1:-} == --s-task2-matched-exposure || ${1:-} == --s-task2-unknown-transition || ${1:-} == --s-task2-proposal-stages || ${1:-} == --s-task2-increment-proposal-quality || ${1:-} == --s-task2-finetune-quality-swap ]]; then
  flags=()
  dir=output/s-instance-task2-dedup-wi-sources
  module=tools.owod.probe_s_task2_wi_sources
  if [[ $1 == --task2-old-only-independent ]]; then
    dir=output/instance-task2-old-only-independent
    module=tools.owod.probe_task2_old_only_independent
  elif [[ $1 == --s-task2-current-phase ]]; then
    dir=output/s-instance-task2-current-phase
    module=tools.owod.probe_s_task2_current_phase
  elif [[ $1 == --s-task2-increment-giou ]]; then
    dir=output/s-instance-task2-increment-giou
    module=tools.owod.run_s_task2_increment_giou
  elif [[ $1 == --s-task2-increment-classifier-lr ]]; then
    dir=output/s-instance-task2-increment-classifier-lr
    module=tools.owod.run_s_task2_increment_giou
    flags=(--classifier-lr)
  elif [[ $1 == --s-task2-increment-proposal-quality ]]; then
    dir=output/s-instance-task2-increment-proposal-quality
    module=tools.owod.run_s_task2_increment_giou
    flags=(--proposal-quality)
  elif [[ $1 == --s-task2-finetune-quality-swap ]]; then
    dir=output/s-instance-task2-finetune-quality-swap
    module=tools.owod.probe_s_task2_quality_swap
  elif [[ $1 == --s-task2-current-fp-sources ]]; then
    dir=output/s-instance-task2-current-fp-sources
    module=tools.owod.probe_s_task2_current_fp_sources
  elif [[ $1 == --s-task2-current-exposure ]]; then
    dir=output/s-instance-task2-current-exposure
    module=tools.owod.probe_s_task2_current_exposure
  elif [[ $1 == --s-task2-matched-exposure ]]; then
    dir=output/s-instance-task2-matched-exposure
    module=tools.owod.probe_s_task2_matched_exposure
  elif [[ $1 == --s-task2-unknown-transition ]]; then
    dir=output/s-instance-task2-unknown-transition
    module=tools.owod.probe_s_task2_unknown_transition
  elif [[ $1 == --s-task2-proposal-stages ]]; then
    dir=output/s-instance-task2-proposal-stages
    module=tools.owod.probe_s_task2_proposal_stages
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  if [[ -f "$dir/comparison.json" ]]; then exit 0; fi
  flock -n "$dir/run.lock" "$python_bin" -u -m "$module" "${flags[@]}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-dedup-open-set || ${1:-} == --task2-old-only-refill ]]; then
  dir=output/instance-task2-old-dedup-open-set
  flags=()
  if [[ $1 == --task2-old-only-refill ]]; then
    dir=output/instance-task2-old-only-refill
    flags+=(--old-only)
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m tools.owod.probe_task2_dedup_open_set "${flags[@]}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --s-task1-known-fraction || ${1:-} == --s-task1-input-scale || ${1:-} == --s-task1-proposal-stages || ${1:-} == --s-task1-finest-anchor || ${1:-} == --s-task1-anchor-choice || ${1:-} == --s-task1-released-seed || ${1:-} == --s-task1-small-slot || ${1:-} == --s-task1-crop-refill || ${1:-} == --s-task1-crop-proxy || ${1:-} == --s-task1-crop-timeline || ${1:-} == --s-task1-region-geometry || ${1:-} == --s-task1-region-ranking || ${1:-} == --s-task2-duration || ${1:-} == --s-task2-score-competition || ${1:-} == --s-task2-paired-targets || ${1:-} == --s-task2-voc-precision || ${1:-} == --s-task2-old-dedup || ${1:-} == --s-task2-old-dedup-holdout ]]; then
  study="${1#--s-}"
  shift
  dir="output/s-instance-$study"
  if [[ $study == task2-score-competition ]]; then
    dir+="-replay"
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m "tools.owod.run_s_${study//-/_}" "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-duration-holdout || ${1:-} == --task2-old-dedup || ${1:-} == --task2-distillation-teachers || ${1:-} == --task2-distillation-selection || ${1:-} == --task2-history-retrieval || ${1:-} == --task2-slot-naming || ${1:-} == --task2-slot-features || ${1:-} == --task1-discovery-gates || ${1:-} == --task1-soft-discovery || ${1:-} == --task1-empty-allocation || ${1:-} == --task1-labeled-quality || ${1:-} == --task1-quality-views || ${1:-} == --task1-view-boxes || ${1:-} == --task1-decoder-depth || ${1:-} == --task1-encoder-levels || ${1:-} == --task1-encoder-counts || ${1:-} == --task1-encoder-residual || ${1:-} == --task1-encoder-transfer || ${1:-} == --task1-encoder-neighborhood || ${1:-} == --task1-unknown-budget || ${1:-} == --task1-unknown-factors || ${1:-} == --task1-visible-classes || ${1:-} == --task1-known-queries || ${1:-} == --task1-known-confirmation || ${1:-} == --task1-known-fraction ]]; then
  probe="${1#--}"
  dir="output/m-instance-$probe"
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m "tools.owod.probe_${probe//-/_}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-exposure-pair ]]; then
  dir=output/m-instance-task2-exposure-pair
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m tools.owod.run_task2_exposure_pair 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-annotated-distillation || ${1:-} == --task2-history-alignment ]]; then
  study="${1#--}"
  dir="output/m-instance-$study"
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m "tools.owod.run_${study//-/_}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-six-epochs || ${1:-} == --task2-six-epochs-seed43 ]]; then
  dir=output/m-instance-task2-six-epochs
  seed=42
  if [[ $1 == --task2-six-epochs-seed43 ]]; then
    seed=43
    dir="$dir-seed43"
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m tools.owod.run_task2_duration_pair --seed "$seed" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --decoder-bbox-lr-pair || ${1:-} == --decoder-bbox-lr-three-quarter || ${1:-} == --decoder-bbox-seed43 ]]; then
  script=tools/owod/run_decoder_bbox_lr_pair.py
  dir=output/m-instance-decoder-bbox-lr-half
  lr_args=()
  if [[ $1 == --decoder-bbox-lr-three-quarter ]]; then
    dir=output/m-instance-decoder-bbox-lr-three-quarter
    lr_args+=(--three-quarter)
  elif [[ $1 == --decoder-bbox-seed43 ]]; then
    dir=output/m-instance-decoder-bbox-seed43
    script=tools/owod/run_decoder_bbox_seed_pair.py
  fi
  shift
  if [[ ${1:-} == --plan ]]; then
    exec "$python_bin" "$script" "${lr_args[@]}" "$@"
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u "$script" "${lr_args[@]}" "$@" 2>&1 | tee -a "$dir/console.log"
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
