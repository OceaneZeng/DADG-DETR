#!/usr/bin/env bash
# Internal M-OWODB supervised control: fixed three-epoch diagnostic budget.
set -euo pipefail
variant=standard
if [[ $# -gt 0 && $1 != --* ]]; then
  variant=$1
  shift
fi
optimization_args=()
case "$variant" in
  standard) lr_scale=1; result_dir=output/tree-detr-m-control-3090 ;;
  lr-quarter)
    lr_scale=0.25
    optimization_args=(--control-lr-scale 0.25)
    result_dir=output/tree-detr-m-control-lr-quarter-3090
    ;;
  *) echo 'Variant must be standard (default) or lr-quarter.' >&2; exit 2 ;;
esac
resume=false
dry_run=false
runner_args=()
for arg in "$@"; do
  case "$arg" in
    --resume) resume=true; runner_args+=("$arg") ;;
    --dry-run) dry_run=true; runner_args+=("$arg") ;;
    *) echo 'Accepted options: [standard|lr-quarter] --resume --dry-run. The diagnostic budget is fixed at 3 epochs.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
log_dir="${result_dir}-logs"
stage_dir="$result_dir/m-owodb/seed_42/stage_0/increment"
if [[ $dry_run == false ]]; then
  if [[ $resume == true ]]; then
    for artifact in "$result_dir/m-owodb/seed_42/benchmark_plan.json" "$stage_dir/checkpoint.pth"; do
      if [[ ! -s "$artifact" ]]; then
        echo "Cannot resume control: missing or empty $artifact" >&2
        exit 2
      fi
    done
  elif [[ -e "$result_dir/m-owodb/seed_42/benchmark_plan.json" || -e "$stage_dir/checkpoint.pth" ]]; then
    echo 'Existing control run found; use --resume to preserve it. No old discovery checkpoint is accepted.' >&2
    exit 2
  fi
fi
mkdir -p "$log_dir"
{
  printf 'M-OWODB supervised control variant=%s GPU=0,1 batch_per_rank=2 seed=42 lr_scale=%s pause_after_epochs=3 schedule=50 lr_drop=50 task1_only=true\n' "$variant" "$lr_scale"
  python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
  python -u tools/owod/run_benchmark.py \
    --manifest data/owod/m-owodb/split_manifest.json \
    --recipe ow-detr-paper --method control --task1-only \
    --backbone-weights pretrained/dino_resnet50.pth \
    --output-dir "$result_dir" \
    --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
    --seed 42 --activation-checkpointing \
    --pause-after-epochs 3 "${optimization_args[@]}" "${runner_args[@]}"
  if [[ $dry_run == false ]]; then
    python -u tools/owod/diagnose_control_inputs.py --run-dir "$stage_dir" --completed-epochs 3 --images 256
  fi
} 2>&1 | tee -a "$log_dir/console.log"
