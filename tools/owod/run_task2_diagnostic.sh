#!/usr/bin/env bash
# Full-author-data, single-3090, v5/v7 paired diagnostic; no formal continuation.
set -euo pipefail
if [[ $# -gt 1 || ( $# == 1 && $1 != --resume ) ]]; then
  echo 'Accepted option: --resume' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-938ce8ba-8aa0-eefb-9a75-09fef406559d
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
output=output/task2-v5-v7-diagnostic-3epochs
mkdir -p "$output"
exec > >(tee -a "$output/console.log") 2>&1
trap 'echo "Exit status: $?; results: output/task2-v5-v7-diagnostic-3epochs"' EXIT
git rev-parse HEAD
python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one CUDA GPU'
assert '3090' in torch.cuda.get_device_name(0), 'Expected selected RTX 3090'
torch.ones(1, device='cuda')
free, _ = torch.cuda.mem_get_info()
assert free >= 14 * 2**30, 'Less than 14 GiB free; stopping'
print(torch.cuda.get_device_name(0), 'free GiB:', round(free / 2**30, 2))
PY
python tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod
python tools/owod/train_task2_diagnostic.py \
  --manifest data/owod/s-owodb/split_manifest.json \
  --checkpoint output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth \
  --gnn output/tree-detr-s-v5-full/s-owodb/seed_42/gnn/stage_0/gnn_stage0.pt \
  --output "$output/results" "$@"
