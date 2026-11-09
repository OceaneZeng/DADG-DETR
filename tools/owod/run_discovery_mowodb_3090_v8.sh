#!/usr/bin/env bash
# Experimental v8: ranking and gradient ablations on full M-OWODB Task 1.
set -euo pipefail
variant=ranking
if [[ $# -gt 0 && $1 != --* ]]; then
  variant=$1
  shift
fi
case "$variant" in
  ranking) stability_args=(--discovery-known-rank-coef 1.0) ;;
  guarded) stability_args=(--discovery-feature-grad-scale 0.1) ;;
  combined) stability_args=(--discovery-known-rank-coef 1.0 --discovery-feature-grad-scale 0.1) ;;
  *) echo 'Variant must be ranking (default), guarded, or combined.' >&2; exit 2 ;;
esac
pause_epochs=8
resume=false
dry_run=false
runner_args=()
for arg in "$@"; do
  case "$arg" in
    --resume) resume=true; runner_args+=("$arg") ;;
    --dry-run) dry_run=true; runner_args+=("$arg") ;;
    --full) pause_epochs=0 ;;
    *) echo 'Accepted options: [ranking|guarded|combined] --resume --dry-run --full.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from models.graph_local.discovery import DISCOVERY_STABILITY_VERSION; assert DISCOVERY_STABILITY_VERSION == 8'
result_dir="output/tree-detr-m-v8-${variant}-3090"
log_dir="${result_dir}-logs"
if [[ -f "$result_dir/m-owodb/seed_42/stage_0/increment/checkpoint.pth" && $resume == false && $dry_run == false ]]; then
  echo 'Existing checkpoint found; use --resume with the same variant to preserve this run.' >&2
  exit 2
fi
mkdir -p "$log_dir"
{
  printf 'Discovery v8 variant=%s GPU=0,1 batch_per_rank=2 pause_after_epochs=%s schedule=50 task1_only=true pseudo_normalization=count objectness=balanced-bce\n' "$variant" "$pause_epochs"
  python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
  python -u tools/owod/run_benchmark.py \
    --manifest data/owod/m-owodb/split_manifest.json \
    --recipe ow-detr-paper --method discovery --task1-only \
    --backbone-weights pretrained/dino_resnet50.pth \
    --output-dir "$result_dir" \
    --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
    --seed 42 --activation-checkpointing \
    --graph-routing hard --graph-localization frozen \
    --discovery-scoring hierarchical --discovery-pseudo-ramp-epochs 3 \
    --discovery-pseudo-normalization count \
    --pause-after-epochs "$pause_epochs" "${stability_args[@]}" "${runner_args[@]}"
} 2>&1 | tee -a "$log_dir/console.log"
