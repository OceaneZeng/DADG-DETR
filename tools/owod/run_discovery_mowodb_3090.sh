#!/usr/bin/env bash
# Experimental v6: full M-OWODB chain on physical GPUs 0,1 (two RTX 3090).
set -euo pipefail
for arg in "$@"; do
  case "$arg" in
    --resume|--dry-run) ;;
    *) echo 'Accepted options: --resume --dry-run. Configuration overrides are disabled.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from models.graph_local.discovery import DISCOVERY_RAMP_VERSION; assert DISCOVERY_RAMP_VERSION == 6'
python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
mkdir -p output/tree-detr-m-v6-3090-logs
python -u tools/owod/run_benchmark.py \
  --manifest data/owod/m-owodb/split_manifest.json \
  --recipe ow-detr-paper --method discovery \
  --backbone-weights pretrained/dino_resnet50.pth \
  --output-dir output/tree-detr-m-v6-3090-full \
  --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
  --seed 42 --activation-checkpointing \
  --graph-routing hard --graph-localization frozen \
  --discovery-scoring hierarchical --discovery-pseudo-ramp-epochs 3 "$@" \
  2>&1 | tee -a output/tree-detr-m-v6-3090-logs/console.log
