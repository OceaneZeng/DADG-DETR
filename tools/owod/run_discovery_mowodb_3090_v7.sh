#!/usr/bin/env bash
# Experimental v7 loss ablations; full-data Task 1 on physical RTX 3090 GPUs 0,1.
set -euo pipefail
variant=objectness
if [[ $# -gt 0 && $1 != --* ]]; then
  variant=$1
  shift
fi
case "$variant" in
  objectness) loss_args=(--discovery-objectness-loss focal) ;;
  confidence) loss_args=(--discovery-pseudo-normalization count) ;;
  combined) loss_args=(--discovery-objectness-loss focal --discovery-pseudo-normalization count) ;;
  *) echo 'Variant must be objectness (default), confidence, or combined.' >&2; exit 2 ;;
esac
pause_epochs=8
runner_args=()
for arg in "$@"; do
  case "$arg" in
    --resume|--dry-run) runner_args+=("$arg") ;;
    --full) pause_epochs=0 ;;
    *) echo 'Accepted options: [objectness|confidence|combined] --resume --dry-run --full.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from models.graph_local.discovery import DISCOVERY_LOSS_VERSION; assert DISCOVERY_LOSS_VERSION == 7'
log_dir="output/tree-detr-m-v7-${variant}-3090-logs"
mkdir -p "$log_dir"
{
  printf 'Discovery v7 variant=%s GPU=0,1 batch_per_rank=2 pause_after_epochs=%s schedule=50 task1_only=true\n' "$variant" "$pause_epochs"
  python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
  python -u tools/owod/run_benchmark.py \
    --manifest data/owod/m-owodb/split_manifest.json \
    --recipe ow-detr-paper --method discovery --task1-only \
    --backbone-weights pretrained/dino_resnet50.pth \
    --output-dir "output/tree-detr-m-v7-${variant}-3090" \
    --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
    --seed 42 --activation-checkpointing \
    --graph-routing hard --graph-localization frozen \
    --discovery-scoring hierarchical --discovery-pseudo-ramp-epochs 3 \
    --pause-after-epochs "$pause_epochs" "${loss_args[@]}" "${runner_args[@]}"
} 2>&1 | tee -a "$log_dir/console.log"
