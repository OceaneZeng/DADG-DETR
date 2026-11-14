#!/usr/bin/env bash
# Full M-OWODB discovery v8 chain on one A40: ranking or focal scoring.
set -euo pipefail
variant=ranking
if [[ $# -gt 0 && $1 != --* ]]; then variant=$1; shift; fi
case "$variant" in
  ranking) scoring=hierarchical ;;
  focal) scoring=focal ;;
  *) echo 'Variant must be ranking (default) or focal.' >&2; exit 2 ;;
esac
resume=false
dry_run=false
gpu_index=${GPU_INDEX:-0}
if [[ ! $gpu_index =~ ^[0-9]+$ ]]; then
  echo 'GPU_INDEX must be a nonnegative physical GPU index (default: 0).' >&2
  exit 2
fi
for arg in "$@"; do
  case "$arg" in
    --resume) resume=true ;;
    --dry-run) dry_run=true ;;
    *) echo 'Accepted options: [ranking|focal] --resume --dry-run. The full four-task configuration is fixed.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_index"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
result_dir=output/tree-detr-m-v8-${variant}-lr-quarter-a40-full
log_dir="${result_dir}-logs"
plan="$result_dir/m-owodb/seed_42/benchmark_plan.json"
checkpoint="$result_dir/m-owodb/seed_42/stage_0/increment/checkpoint.pth"
mkdir -p "$log_dir"
{
  printf 'Discovery v8 %s full M-OWODB scoring=%s GPU=%s device=A40 batch_per_rank=4 effective_batch=4 nproc=1 seed=42 lr_scale=0.25 increment_lr=5e-5 finetune_lr=5e-6 tasks=4 training_phases=7 schedule=50/20\n' "$variant" "$scoring" "$gpu_index"
  if [[ $dry_run == false ]]; then
    if [[ $resume == true && ! -s "$plan" ]]; then
      echo "Cannot resume: missing or empty full-chain plan $plan" >&2
      exit 2
    elif [[ $resume == false && ( -e "$plan" || -e "$checkpoint" ) ]]; then
      echo 'Existing full-chain run found; use --resume with the same configuration to preserve it.' >&2
      exit 2
    fi
  fi
  python -c 'from models.graph_local.discovery import DISCOVERY_STABILITY_VERSION; assert DISCOVERY_STABILITY_VERSION == 8'
  python tools/owod/manage_data.py verify --benchmark m-owodb --output-root data/owod
  python -u tools/owod/run_benchmark.py \
    --manifest data/owod/m-owodb/split_manifest.json \
    --recipe ow-detr-paper --method discovery \
    --backbone-weights pretrained/dino_resnet50.pth \
    --output-dir "$result_dir" \
    --gpus "$gpu_index" --nproc-per-node 1 --batch-size 4 --num-workers 4 \
    --seed 42 --activation-checkpointing --discovery-lr-scale 0.25 \
    --graph-routing hard --graph-localization frozen \
    --discovery-scoring "$scoring" --discovery-pseudo-ramp-epochs 3 \
    --discovery-pseudo-normalization count --discovery-known-rank-coef 1.0 "$@"
} 2>&1 | tee -a "$log_dir/console.log"
