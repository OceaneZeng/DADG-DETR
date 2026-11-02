#!/usr/bin/env bash
# Full M-OWODB discovery v5 on physical GPU 0; fresh independent task chain.
set -euo pipefail
for arg in "$@"; do
  case "$arg" in
    --resume|--dry-run) ;;
    *) echo 'Accepted options: --resume --dry-run. Configuration overrides are disabled.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python -c 'from models.graph_local.discovery import DISCOVERY_VERSION; assert DISCOVERY_VERSION == 5, "This launcher requires discovery v5"'
# Verify source and annotation hashes on both fresh starts and resume.
python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
mkdir -p output/tree-detr-m-v5-a40-logs
python -u tools/owod/run_benchmark.py \
  --manifest data/owod/m-owodb/split_manifest.json \
  --recipe ow-detr-paper --method discovery \
  --backbone-weights pretrained/dino_resnet50.pth \
  --output-dir output/tree-detr-m-v5-a40-full \
  --gpus 0 --nproc-per-node 1 --batch-size 4 --num-workers 4 \
  --seed 42 --activation-checkpointing \
  --graph-routing hard --graph-localization frozen \
  --discovery-scoring hierarchical "$@" \
  2>&1 | tee -a output/tree-detr-m-v5-a40-logs/console.log
