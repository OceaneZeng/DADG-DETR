#!/usr/bin/env bash
# Fixed low-LR Task 1 comparison: ranking vs ranking + discovery gradient guard.
set -euo pipefail
variant=combined
if [[ $# -gt 0 && $1 != --* ]]; then variant=$1; shift; fi
case "$variant" in
  ranking) gradient=1.0 ;;
  combined) gradient=0.1 ;;
  *) echo 'Variant must be ranking or combined (default).' >&2; exit 2 ;;
esac
resume=false
dry_run=false
pause_epochs=8
for arg in "$@"; do
  case "$arg" in
    --resume) resume=true ;;
    --dry-run) dry_run=true ;;
    --full) pause_epochs=0 ;;
    *) echo 'Accepted options: [ranking|combined] --resume --dry-run --full.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
result_dir="output/tree-detr-m-v8-${variant}-lr-quarter-3090-task1"
log_dir="${result_dir}-logs"
plan="$result_dir/m-owodb/seed_42/benchmark_plan.json"
checkpoint="$result_dir/m-owodb/seed_42/stage_0/increment/checkpoint.pth"
runner_args=()
if [[ $resume == true ]]; then runner_args+=(--resume); fi
if [[ $dry_run == true ]]; then runner_args+=(--dry-run); fi
mkdir -p "$log_dir"
{
  printf 'Discovery v8 low-LR Task1 variant=%s rank_coef=1 feature_grad_scale=%s lr=5e-5 GPU=0,1 batch_per_rank=2 seed=42 schedule=50 pause_after_epochs=%s\n' "$variant" "$gradient" "$pause_epochs"
  if [[ $dry_run == false ]]; then
    if [[ $resume == true && ! -s "$plan" ]]; then
      echo "Cannot resume: missing or empty plan $plan" >&2; exit 2
    elif [[ $resume == false && ( -e "$plan" || -e "$checkpoint" ) ]]; then
      echo 'Existing run found; use --resume with the same variant.' >&2; exit 2
    fi
  fi
  python -c 'from models.graph_local.discovery import DISCOVERY_STABILITY_VERSION; assert DISCOVERY_STABILITY_VERSION == 8'
  python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
  python -u tools/owod/run_benchmark.py \
    --manifest data/owod/m-owodb/split_manifest.json \
    --recipe ow-detr-paper --method discovery --task1-only \
    --backbone-weights pretrained/dino_resnet50.pth --output-dir "$result_dir" \
    --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
    --seed 42 --activation-checkpointing --discovery-lr-scale 0.25 \
    --graph-routing hard --graph-localization frozen --discovery-scoring hierarchical \
    --discovery-pseudo-ramp-epochs 3 --discovery-pseudo-normalization count \
    --discovery-known-rank-coef 1.0 --discovery-feature-grad-scale "$gradient" \
    --pause-after-epochs "$pause_epochs" "${runner_args[@]}"
} 2>&1 | tee -a "$log_dir/console.log"
