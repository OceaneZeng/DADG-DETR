#!/usr/bin/env bash
# Full-author-data, two-3090 DDP, v5/v7 paired diagnostic; no formal continuation.
set -euo pipefail
if [[ $# -gt 1 || ( $# == 1 && $1 != --resume ) ]]; then
  echo 'Accepted option: --resume' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Physical GPUs 0 and 1 (RTX 3090), using the server's PCI ordering.
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
output=output/task2-v5-v7-diagnostic-3epochs-2gpu
mkdir -p "$output"
exec > >(tee -a "$output/console.log") 2>&1
trap 'echo "Exit status: $?; results: $output"' EXIT
git rev-parse HEAD
git branch --show-current
git status --short
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.free --format=csv
python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expected two CUDA GPUs (0,1)'
for index in range(2):
    assert '3090' in torch.cuda.get_device_name(index), 'Expected selected RTX 3090'
    torch.ones(1, device=f'cuda:{index}')
    free, _ = torch.cuda.mem_get_info(index)
    assert free >= 14 * 2**30, f'GPU {index}: less than 14 GiB free; stopping'
    print(index, torch.cuda.get_device_name(index), 'free GiB:', round(free / 2**30, 2))
PY
python tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/owod/train_task2_diagnostic.py \
  --manifest data/owod/s-owodb/split_manifest.json \
  --checkpoint output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth \
  --gnn output/tree-detr-s-v5-full/s-owodb/seed_42/gnn/stage_0/gnn_stage0.pt \
  --output "$output/results" "$@"
