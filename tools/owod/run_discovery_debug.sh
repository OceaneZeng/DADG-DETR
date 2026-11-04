#!/usr/bin/env bash
# Bounded Task 2 diagnostics on toplab; no formal training or weight writes.
set -euo pipefail
if (( $# )); then
  echo 'This fixed diagnostic launcher accepts no arguments.' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Physical 3090 cards 1 and 2, verified by UUID; exclude unrelated/broken cards.
export CUDA_VISIBLE_DEVICES=GPU-82ffd094-019f-1f3b-a428-70894db481d2,GPU-938ce8ba-8aa0-eefb-9a75-09fef406559d
mkdir -p output/discovery-v6-server-debug
run_dir=$(mktemp -d output/discovery-v6-server-debug/run-XXXXXXXX)
echo "Diagnostic output: $run_dir"
exec > >(tee -a "$run_dir/console.log") 2>&1
trap 'echo "Exit status: $?; diagnostic log: $run_dir/console.log"' EXIT
git rev-parse HEAD
git status --short
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable; stopping before training'
assert torch.cuda.device_count() == 2, 'Expected the two selected 3090 cards'
for index in range(2):
    assert '3090' in torch.cuda.get_device_name(index), 'Unexpected GPU identity'
    torch.empty(1, device=f'cuda:{index}')
    free, total = torch.cuda.mem_get_info(index)
    assert free >= 14 * 2**30, 'Insufficient free GPU memory; stopping'
    print(index, torch.cuda.get_device_name(index), 'free GiB:', round(free / 2**30, 2))
PY
python tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod
timeout --signal=TERM --kill-after=30s 15m \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/owod/debug_discovery_revision.py \
  --checkpoint output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth \
  --start-task2 \
  --graph-gnn-checkpoint output/tree-detr-s-v5-full/s-owodb/seed_42/gnn/stage_0/gnn_stage0.pt \
  --manifest data/owod/s-owodb/split_manifest.json \
  --images 16 --batch-size 2 --full-resolution --steps 40 \
  --memory-limit-mib 12000 --ram-limit-mib 8192 --seconds-per-policy 300 \
  --output "$run_dir/paired.json"
python - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.loads((root / f'paired.rank{i}.json').read_text()) for i in range(2)]
assert all(r['status'] == 'completed' and r['resource_guard_passed'] and
           r['checkpoint_unchanged'] and r['old_teacher_unchanged'] for r in rows)
print('Both ranks passed resource and checkpoint identity guards.')
print('Diagnostic only: no AP gain or formal-chain compatibility claim.')
PY
