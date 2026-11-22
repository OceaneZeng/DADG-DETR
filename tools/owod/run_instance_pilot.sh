#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_bin="${TREE_DETR_PYTHON:-/home/top/disks/new-hdd/conda_envs/tree-detr/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Set TREE_DETR_PYTHON to the Python executable in the tree-detr environment." >&2
  exit 1
fi
env_dir="$(cd "$(dirname "$python_bin")/.." && pwd)"
export PATH="$env_dir/bin:$PATH"
export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == --task2-m-s-previous-hard-negatives || ${1:-} == --task2-m-s-previous-hard-negative-head || ${1:-} == --task2-m-s-previous-hard-negative-misses || ${1:-} == --task2-m-s-previous-hard-negative-scores || ${1:-} == --task2-m-s-previous-competitor-training-probe || ${1:-} == --task2-m-s-current-training-ranks || ${1:-} == --task2-m-s-current-training-geometry || ${1:-} == --task2-m-s-current-shortlist-ranks || ${1:-} == --task2-m-s-current-shortlist-replacement || ${1:-} == --task2-m-s-current-replacement-trace || ${1:-} == --task2-m-s-native-box-blend || ${1:-} == --task2-m-s-current-low-iou-boxes || ${1:-} == --task2-m-s-current-low-iou-box-misses ]]; then
  dir="output/${1#--}"
  module=tools.owod.run_task2_m_s_previous_hard_negatives
  flags=()
  if [[ $1 == --task2-m-s-current-low-iou-boxes ]]; then
    flags=(--low-iou-boxes)
  elif [[ $1 == --task2-m-s-native-box-blend ]]; then
    module=tools.owod.probe_task2_m_s_native_box_blend
  elif [[ $1 == --task2-m-s-previous-hard-negative-head ]]; then
    module=tools.owod.probe_task2_m_s_previous_hard_negative_head
  elif [[ $1 == --task2-m-s-current-shortlist-replacement || $1 == --task2-m-s-current-replacement-trace ]]; then
    module=tools.owod.probe_task2_m_s_shortlist_replacement
    if [[ $1 == --task2-m-s-current-replacement-trace ]]; then
      flags=(--trace-injected)
    fi
  elif [[ $1 == --task2-m-s-previous-competitor-training-probe || $1 == --task2-m-s-current-training-ranks || $1 == --task2-m-s-current-training-geometry || $1 == --task2-m-s-current-shortlist-ranks ]]; then
    module=tools.owod.probe_task2_m_s_previous_competitors
    if [[ $1 == --task2-m-s-current-training-ranks ]]; then
      flags=(--current-ranks)
    elif [[ $1 == --task2-m-s-current-training-geometry ]]; then
      flags=(--current-geometry)
    elif [[ $1 == --task2-m-s-current-shortlist-ranks ]]; then
      flags=(--shortlist-ranks)
    fi
  elif [[ $1 == --task2-m-s-previous-hard-negative-misses || $1 == --task2-m-s-previous-hard-negative-scores || $1 == --task2-m-s-current-low-iou-box-misses ]]; then
    module=tools.owod.probe_task2_m_s_previous_hard_negative_misses
    if [[ $1 == --task2-m-s-previous-hard-negative-scores ]]; then
      flags=(--score-components)
    elif [[ flags=(--score-components) == --task2-m-s-current-low-iou-box-misses ]]; then
      flags=(--low-iou-boxes)
    fi
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  if [[ -f "$dir/comparison.json" ]]; then exit 0; fi
  flock -n "$dir/run.lock" "$python_bin" -u -m "$module" "${flags[@]}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --s-task2-wi-sources || ${1:-} == --task2-old-only-independent || ${1:-} == --s-task2-current-phase || ${1:-} == --s-task2-increment-giou || ${1:-} == --s-task2-increment-classifier-lr || ${1:-} == --s-task2-current-fp-sources || ${1:-} == --s-task2-current-exposure || ${1:-} == --s-task2-matched-exposure || ${1:-} == --s-task2-unknown-transition || ${1:-} == --s-task2-proposal-stages || ${1:-} == --s-task2-increment-proposal-quality || ${1:-} == --s-task2-finetune-quality-swap || ${1:-} == --s-task2-quality-timing || ${1:-} == --s-task2-finetune-half-lr || ${1:-} == --s-task2-step-classifier || ${1:-} == --s-task2-finetune-offset-rate || ${1:-} == --s-task2-step-box || ${1:-} == --s-task2-finetune-old-box-loss || ${1:-} == --s-task2-old-box-competition || ${1:-} == --s-task2-fixed-score-ranking || ${1:-} == --s-task2-finetune-current-positive || ${1:-} == --s-task2-absent-class-errors || ${1:-} == --s-task2-increment-match-coverage || ${1:-} == --s-task2-increment-no-random-crop || ${1:-} == --s-task2-crop-ap-anchors || ${1:-} == --s-task2-crop-ap-independent || ${1:-} == --s-task2-increment-tail-positives || ${1:-} == --s-task2-tail-query-ranking || ${1:-} == --s-task2-tail-affinity || ${1:-} == --s-task2-shortlist-depth || ${1:-} == --s-task2-tail-cell-ranks || ${1:-} == --s-task2-cell-pair-quality || ${1:-} == --s-task2-quality-query-count || ${1:-} == --s-task2-shared-query-refill || ${1:-} == --s-task2-refill-capacity || ${1:-} == --s-task2-refill-allocation || ${1:-} == --s-task2-increment-alternate-images || ${1:-} == --s-task2-alternate-pool-fit || ${1:-} == --s-task2-increment-encoder-ffn || ${1:-} == --s-task2-ffn-target-transfer || ${1:-} == --s-task2-ffn-score-components || ${1:-} == --s-task2-frozen-query-readout || ${1:-} == --s-task2-readout-fixed-slots || ${1:-} == --task2-m-s-refill-score-order || ${1:-} == --task2-m-s-refill-protect-current || ${1:-} == --task2-m-s-refill-protect-current-independent || ${1:-} == --task2-m-s-refill-duration || ${1:-} == --task2-m-s-refill-previous-order || ${1:-} == --task2-m-s-refill-same-class-overlap || ${1:-} == --task2-m-s-refill-same-class-overlap-independent || ${1:-} == --task2-m-s-refill-same-class-overlap-three-epoch || ${1:-} == --task2-m-s-refill-overlap-s-short-order || ${1:-} == --task2-m-s-current-head-restore || ${1:-} == --task2-m-s-decoder-ffn-restore || ${1:-} == --task2-m-s-decoder-ffn-current-logits || ${1:-} == --task2-m-s-decoder-ffn-boxes || ${1:-} == --task2-m-s-decoder-ffn-boxes-current-logits || ${1:-} == --task2-m-s-decoder-ffn-discovery-scores || ${1:-} == --task2-m-s-decoder-ffn-native-role-counts || ${1:-} == --task2-m-s-previous-competition || ${1:-} == --task2-m-s-previous-competition-background || ${1:-} == --task2-m-s-unknown-refill-independent || ${1:-} == --task2-m-s-unknown-refill || ${1:-} == --task2-m-s-unique-query || ${1:-} == --task2-m-s-current-native-order || ${1:-} == --task2-m-s-current-top-class || ${1:-} == --task2-m-s-current-fp-near-target || ${1:-} == --task2-m-s-current-fp-residual || ${1:-} == --task2-m-s-current-fp-open-set || ${1:-} == --task2-m-s-current-fp || ${1:-} == --task2-m-s-current-score || ${1:-} == --task2-m-s-score-margin || ${1:-} == --task2-m-s-shared-current || ${1:-} == --task1-m-s-tiny-shortlist || ${1:-} == --task1-m-s-tiny-extent-cap || ${1:-} == --task1-m-s-geometry-stages || ${1:-} == --task1-m-s-candidate-capacity || ${1:-} == --task1-m-s-consistency-validation || ${1:-} == --s-task1-consistency-budget || ${1:-} == --s-task1-view-consistency || ${1:-} == --s-task1-pseudo-coverage || ${1:-} == --s-task1-novelty-budget || ${1:-} == --task1-m-s-later-sampling || ${1:-} == --task1-m-s-first-layer-forward || ${1:-} == --task1-m-s-first-layer-geometry || ${1:-} == --task1-m-s-query-trace || ${1:-} == --task1-m-s-prescreen-replacement || ${1:-} == --task1-m-s-prescreen-ranks || ${1:-} == --task1-m-s-paired-discovery || ${1:-} == --task1-m-s-released-small-extent || ${1:-} == --task1-m-s-small-box-extent || ${1:-} == --task1-m-s-small-box-geometry ]]; then
  flags=()
  dir=output/s-instance-task2-dedup-wi-sources
  module=tools.owod.probe_s_task2_wi_sources
  if [[ $1 == --task2-old-only-independent ]]; then
    dir=output/instance-task2-old-only-independent
    module=tools.owod.probe_task2_old_only_independent
  elif [[ $1 == --s-task2-current-phase ]]; then
    dir=output/s-instance-task2-current-phase
    module=tools.owod.probe_s_task2_current_phase
  elif [[ $1 == --s-task2-increment-giou ]]; then
    dir=output/s-instance-task2-increment-giou
    module=tools.owod.run_s_task2_increment_giou
  elif [[ $1 == --s-task2-increment-classifier-lr ]]; then
    dir=output/s-instance-task2-increment-classifier-lr
    module=tools.owod.run_s_task2_increment_giou
    flags=(--classifier-lr)
  elif [[ $1 == --s-task2-increment-proposal-quality ]]; then
    dir=output/s-instance-task2-increment-proposal-quality
    module=tools.owod.run_s_task2_increment_giou
    flags=(--proposal-quality)
  elif [[ $1 == --s-task2-finetune-quality-swap ]]; then
    dir=output/s-instance-task2-finetune-quality-swap
    module=tools.owod.probe_s_task2_quality_swap
  elif [[ $1 == --s-task2-step-box ]]; then
    dir=output/s-instance-task2-step-box
    module=tools.owod.probe_s_task2_step_classifier
    flags=(--box-head)
  elif [[ $1 == --s-task2-fixed-score-ranking ]]; then
    dir=output/s-instance-task2-fixed-score-ranking
    module=tools.owod.probe_s_task2_fixed_score_ranking
  elif [[ $1 == --s-task2-old-box-competition ]]; then
    dir=output/s-instance-task2-old-box-competition
    module=tools.owod.probe_s_task2_old_box_competition
  elif [[ $1 == --s-task2-step-classifier ]]; then
    dir=output/s-instance-task2-step-classifier
    module=tools.owod.probe_s_task2_step_classifier
  elif [[ $1 == --s-task2-finetune-offset-rate ]]; then
    dir=output/s-instance-task2-finetune-offset-rate
    module=tools.owod.run_s_task2_finetune_half_lr
    flags=(--restore-offsets)
  elif [[ $1 == --s-task2-finetune-old-box-loss ]]; then
    dir=output/s-instance-task2-finetune-old-box-loss
    module=tools.owod.run_s_task2_finetune_half_lr
    flags=(--double-old-boxes)
  elif [[ $1 == --task1-m-s-released-small-extent ]]; then
    dir=output/task1-m-s-released-small-extent
    module=tools.owod.run_task1_m_s_released_small_extent
    flags=("${@:2}")
  elif [[ $1 == --task1-m-s-small-box-extent ]]; then
    dir=output/task1-m-s-small-box-extent
    module=tools.owod.probe_task1_m_s_small_box_extent
  elif [[ $1 == --task1-m-s-small-box-geometry ]]; then
    dir=output/task1-m-s-small-box-geometry-replay
    module=tools.owod.probe_task1_m_s_small_box_geometry
  elif [[ $1 == --task1-m-s-consistency-validation ]]; then
    dir=output/task1-m-s-consistency-validation
    module=tools.owod.probe_task1_m_s_consistency_validation
  elif [[ $1 == --task1-m-s-candidate-capacity ]]; then
    dir=output/task1-m-s-candidate-capacity
    module=tools.owod.probe_task1_m_s_candidate_capacity
  elif [[ $1 == --task1-m-s-geometry-stages ]]; then
    dir=output/task1-m-s-geometry-stages
    module=tools.owod.probe_task1_m_s_geometry_stages
  elif [[ $1 == --task1-m-s-tiny-extent-cap ]]; then
    dir=output/task1-m-s-tiny-extent-cap
    module=tools.owod.probe_task1_m_s_tiny_extent_cap
  elif [[ $1 == --task1-m-s-tiny-shortlist ]]; then
    dir=output/task1-m-s-tiny-shortlist
    module=tools.owod.probe_task1_m_s_tiny_shortlist
  elif [[ $1 == --task2-m-s-shared-current ]]; then
    dir=output/task2-m-s-shared-current
    module=tools.owod.probe_task2_m_s_shared_current
  elif [[ $1 == --task2-m-s-score-margin ]]; then
    dir=output/task2-m-s-score-margin
    module=tools.owod.probe_task2_m_s_score_margin
  elif [[ $1 == --task2-m-s-current-score ]]; then
    dir=output/task2-m-s-current-score
    module=tools.owod.probe_task2_m_s_current_score
  elif [[ $1 == --task2-m-s-current-fp ]]; then
    dir=output/task2-m-s-current-fp
    module=tools.owod.probe_task2_m_s_current_score
    flags=(--fp-sources)
  elif [[ $1 == --task2-m-s-current-fp-open-set ]]; then
    dir=output/task2-m-s-current-fp-open-set
    module=tools.owod.probe_task2_m_s_current_fp_open_set
  elif [[ $1 == --task2-m-s-current-fp-residual ]]; then
    dir=output/task2-m-s-current-fp-residual
    module=tools.owod.probe_task2_m_s_current_fp_open_set
    flags=(--residual-overlap)
  elif [[ $1 == --task2-m-s-current-fp-near-target ]]; then
    dir=output/task2-m-s-current-fp-near-target
    module=tools.owod.probe_task2_m_s_current_fp_open_set
    flags=(--near-target)
  elif [[ $1 == --task2-m-s-current-top-class ]]; then
    dir=output/task2-m-s-current-top-class
    module=tools.owod.probe_task2_m_s_current_top_class
  elif [[ $1 == --task2-m-s-current-native-order ]]; then
    dir=output/task2-m-s-current-native-order
    module=tools.owod.probe_task2_m_s_current_top_class
    flags=(--native-order)
  elif [[ $1 == --task2-m-s-unique-query ]]; then
    dir=output/task2-m-s-unique-query
    module=tools.owod.probe_task2_m_s_current_top_class
    flags=(--unique-query)
  elif [[ $1 == --task2-m-s-unknown-refill ]]; then
    dir=output/task2-m-s-unknown-refill
    module=tools.owod.probe_task2_m_s_current_top_class
    flags=(--unknown-refill)
  elif [[ $1 == --task2-m-s-unknown-refill-independent ]]; then
    dir=output/task2-m-s-unknown-refill-independent
    module=tools.owod.probe_task2_m_s_unknown_refill_independent
  elif [[ $1 == --task2-m-s-refill-protect-current-independent ]]; then
    dir=output/task2-m-s-refill-protect-current-independent
    module=tools.owod.probe_task2_m_s_unknown_refill_independent
    flags=(--protect-current)
  elif [[ $1 == --task2-m-s-refill-duration ]]; then
    dir=output/task2-m-s-refill-duration
    module=tools.owod.probe_task2_m_s_refill_duration
  elif [[ $1 == --task2-m-s-refill-previous-order ]]; then
    dir=output/task2-m-s-refill-previous-order
    module=tools.owod.probe_task2_m_s_refill_duration
    flags=(--audit-previous)
  elif [[ $1 == --task2-m-s-refill-same-class-overlap ]]; then
    dir=output/task2-m-s-refill-same-class-overlap
    module=tools.owod.probe_task2_m_s_refill_duration
    flags=(--same-class-overlap)
  elif [[ $1 == --task2-m-s-refill-same-class-overlap-independent ]]; then
    dir=output/task2-m-s-refill-same-class-overlap-independent
    module=tools.owod.probe_task2_m_s_unknown_refill_independent
    flags=(--same-class-overlap)
  elif [[ $1 == --task2-m-s-refill-same-class-overlap-three-epoch ]]; then
    dir=output/task2-m-s-refill-same-class-overlap-three-epoch
    module=tools.owod.probe_task2_m_s_unknown_refill_independent
    flags=(--overlap-short)
  elif [[ $1 == --task2-m-s-refill-overlap-s-short-order ]]; then
    dir=output/task2-m-s-refill-overlap-s-short-order
    module=tools.owod.probe_task2_m_s_unknown_refill_independent
    flags=(--audit-overlap-short)
  elif [[ $1 == --task2-m-s-current-head-restore ]]; then
    dir=output/task2-m-s-current-head-restore
    module=tools.owod.probe_task2_m_s_current_head_restore
  elif [[ $1 == --task2-m-s-decoder-ffn-restore ]]; then
    dir=output/task2-m-s-decoder-ffn-restore
    module=tools.owod.probe_task2_m_s_current_head_restore
    flags=(--restore-ffn)
  elif [[ $1 == --task2-m-s-decoder-ffn-current-logits ]]; then
    dir=output/task2-m-s-decoder-ffn-current-logits
    module=tools.owod.probe_task2_m_s_current_head_restore
    flags=(--current-logits)
  elif [[ $1 == --task2-m-s-decoder-ffn-boxes ]]; then
    dir=output/task2-m-s-decoder-ffn-boxes
    module=tools.owod.probe_task2_m_s_current_head_restore
    flags=(--boxes-only)
  elif [[ $1 == --task2-m-s-decoder-ffn-boxes-current-logits ]]; then
    dir=output/task2-m-s-decoder-ffn-boxes-current-logits
    module=tools.owod.probe_task2_m_s_current_head_restore
    flags=(--boxes-current-logits)
  elif [[ $1 == --task2-m-s-decoder-ffn-discovery-scores ]]; then
    dir=output/task2-m-s-decoder-ffn-discovery-scores
    module=tools.owod.probe_task2_m_s_current_head_restore
    flags=(--discovery-scores)
  elif [[ $1 == --task2-m-s-decoder-ffn-native-role-counts ]]; then
    dir=output/task2-m-s-decoder-ffn-native-role-counts
    module=tools.owod.probe_task2_m_s_current_head_restore
    flags=(--fixed-role-counts)
  elif [[ $1 == --task2-m-s-previous-competition ]]; then
    dir=output/task2-m-s-previous-competition
    module=tools.owod.probe_task2_m_s_previous_competition
  elif [[ $1 == --task2-m-s-previous-competition-background ]]; then
    dir=output/task2-m-s-previous-competition-background
    module=tools.owod.probe_task2_m_s_previous_competition
    flags=(--background-only)
  elif [[ $1 == --task2-m-s-refill-score-order ]]; then
    dir=output/task2-m-s-refill-score-order
    module=tools.owod.probe_task2_m_s_refill_score_order
  elif [[ $1 == --task2-m-s-refill-protect-current ]]; then
    dir=output/task2-m-s-refill-protect-current
    module=tools.owod.probe_task2_m_s_refill_score_order
    flags=(--protect-current)
  elif [[ $1 == --s-task1-consistency-budget ]]; then
    dir=output/s-task1-consistency-budget
    module=tools.owod.probe_s_task1_consistency_budget
  elif [[ $1 == --s-task1-view-consistency ]]; then
    dir=output/s-task1-view-consistency
    module=tools.owod.probe_s_task1_view_consistency
  elif [[ $1 == --s-task1-pseudo-coverage ]]; then
    dir=output/s-task1-pseudo-coverage
    module=tools.owod.probe_s_task1_pseudo_coverage
  elif [[ $1 == --s-task1-novelty-budget ]]; then
    dir=output/s-task1-novelty-budget
    module=tools.owod.probe_s_task1_novelty_budget
  elif [[ $1 == --task1-m-s-later-sampling ]]; then
    dir=output/task1-m-s-later-sampling
    module=tools.owod.probe_task1_m_s_later_sampling
  elif [[ $1 == --task1-m-s-first-layer-forward ]]; then
    dir=output/task1-m-s-first-layer-forward
    module=tools.owod.probe_task1_m_s_first_layer_forward
  elif [[ $1 == --task1-m-s-first-layer-geometry ]]; then
    dir=output/task1-m-s-first-layer-geometry
    module=tools.owod.probe_task1_m_s_first_layer_geometry
  elif [[ $1 == --task1-m-s-query-trace ]]; then
    dir=output/task1-m-s-query-trace
    module=tools.owod.probe_task1_m_s_query_trace
  elif [[ $1 == --task1-m-s-prescreen-replacement ]]; then
    dir=output/task1-m-s-prescreen-replacement
    module=tools.owod.probe_task1_m_s_prescreen_replacement
  elif [[ $1 == --task1-m-s-prescreen-ranks ]]; then
    dir=output/task1-m-s-prescreen-ranks
    module=tools.owod.probe_task1_m_s_prescreen
  elif [[ $1 == --task1-m-s-paired-discovery ]]; then
    dir=output/task1-m-s-paired-discovery
    module=tools.owod.probe_task1_paired_discovery
  elif [[ $1 == --s-task2-readout-fixed-slots ]]; then
    dir=output/s-instance-task2-readout-fixed-slots
    module=tools.owod.probe_s_task2_readout_fixed_slots
  elif [[ $1 == --s-task2-frozen-query-readout ]]; then
    dir=output/s-instance-task2-frozen-query-readout
    module=tools.owod.probe_s_task2_frozen_query_readout
  elif [[ $1 == --s-task2-ffn-score-components ]]; then
    dir=output/s-instance-task2-ffn-score-components
    module=tools.owod.probe_s_task2_ffn_score_components
  elif [[ $1 == --s-task2-ffn-target-transfer ]]; then
    dir=output/s-instance-task2-ffn-target-transfer-replay
    module=tools.owod.probe_s_task2_ffn_target_transfer
  elif [[ $1 == --s-task2-increment-encoder-ffn ]]; then
    dir=output/s-instance-task2-increment-encoder-ffn
    module=tools.owod.run_s_task2_increment_giou
    flags=(--encoder-ffn)
  elif [[ $1 == --s-task2-alternate-pool-fit ]]; then
    dir=output/s-instance-task2-alternate-pool-fit
    module=tools.owod.probe_s_task2_alternate_pool_fit
  elif [[ $1 == --s-task2-increment-alternate-images ]]; then
    dir=output/s-instance-task2-increment-alternate-images
    module=tools.owod.run_s_task2_increment_giou
    flags=(--alternate-images)
  elif [[ $1 == --s-task2-refill-allocation ]]; then
    dir=output/s-instance-task2-refill-allocation
    module=tools.owod.probe_s_task2_refill_allocation
  elif [[ $1 == --s-task2-refill-capacity ]]; then
    dir=output/s-instance-task2-refill-capacity
    module=tools.owod.probe_s_task2_refill_capacity
  elif [[ $1 == --s-task2-shared-query-refill ]]; then
    dir=output/s-instance-task2-shared-query-refill
    module=tools.owod.probe_s_task2_shared_query_refill
  elif [[ $1 == --s-task2-quality-query-count ]]; then
    dir=output/s-instance-task2-quality-query-count
    module=tools.owod.probe_s_task2_quality_query_count
  elif [[ $1 == --s-task2-cell-pair-quality ]]; then
    dir=output/s-instance-task2-cell-pair-quality
    module=tools.owod.probe_s_task2_cell_pair_quality
  elif [[ $1 == --s-task2-tail-cell-ranks ]]; then
    dir=output/s-instance-task2-tail-cell-ranks
    module=tools.owod.probe_s_task2_tail_cell_ranks
  elif [[ $1 == --s-task2-shortlist-depth ]]; then
    dir=output/s-instance-task2-shortlist-depth
    module=tools.owod.probe_s_task2_shortlist_depth
  elif [[ $1 == --s-task2-tail-affinity ]]; then
    dir=output/s-instance-task2-tail-affinity
    module=tools.owod.probe_s_task2_tail_query_ranking
    flags=(--affinity)
  elif [[ $1 == --s-task2-tail-query-ranking ]]; then
    dir=output/s-instance-task2-tail-query-ranking
    module=tools.owod.probe_s_task2_tail_query_ranking
  elif [[ $1 == --s-task2-increment-tail-positives ]]; then
    dir=output/s-instance-task2-increment-tail-positives
    module=tools.owod.run_s_task2_increment_giou
    flags=(--tail-positives)
  elif [[ $1 == --s-task2-crop-ap-independent ]]; then
    dir=output/s-instance-task2-crop-ap-independent
    module=tools.owod.probe_s_task2_crop_ap_anchors
    flags=(--independent)
  elif [[ $1 == --s-task2-crop-ap-anchors ]]; then
    dir=output/s-instance-task2-crop-ap-anchors
    module=tools.owod.probe_s_task2_crop_ap_anchors
  elif [[ $1 == --s-task2-increment-no-random-crop ]]; then
    dir=output/s-instance-task2-increment-no-random-crop
    module=tools.owod.run_s_task2_increment_giou
    flags=(--no-random-crop)
  elif [[ $1 == --s-task2-increment-match-coverage ]]; then
    dir=output/s-instance-task2-increment-match-coverage
    module=tools.owod.probe_s_task2_increment_match_coverage
  elif [[ $1 == --s-task2-absent-class-errors ]]; then
    dir=output/s-instance-task2-absent-class-errors
    module=tools.owod.probe_s_task2_absent_class_errors
  elif [[ $1 == --s-task2-finetune-current-positive ]]; then
    dir=output/s-instance-task2-finetune-current-positive
    module=tools.owod.run_s_task2_finetune_half_lr
    flags=(--double-current-positives)
  elif [[ $1 == --s-task2-finetune-half-lr ]]; then
    dir=output/s-instance-task2-finetune-half-lr
    module=tools.owod.run_s_task2_finetune_half_lr
  elif [[ $1 == --s-task2-quality-timing ]]; then
    dir=output/s-instance-task2-quality-timing
    module=tools.owod.probe_s_task2_quality_timing
  elif [[ $1 == --s-task2-current-fp-sources ]]; then
    dir=output/s-instance-task2-current-fp-sources
    module=tools.owod.probe_s_task2_current_fp_sources
  elif [[ $1 == --s-task2-current-exposure ]]; then
    dir=output/s-instance-task2-current-exposure
    module=tools.owod.probe_s_task2_current_exposure
  elif [[ $1 == --s-task2-matched-exposure ]]; then
    dir=output/s-instance-task2-matched-exposure
    module=tools.owod.probe_s_task2_matched_exposure
  elif [[ $1 == --s-task2-unknown-transition ]]; then
    dir=output/s-instance-task2-unknown-transition
    module=tools.owod.probe_s_task2_unknown_transition
  elif [[ $1 == --s-task2-proposal-stages ]]; then
    dir=output/s-instance-task2-proposal-stages
    module=tools.owod.probe_s_task2_proposal_stages
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  if [[ -f "$dir/comparison.json" ]]; then exit 0; fi
  flock -n "$dir/run.lock" "$python_bin" -u -m "$module" "${flags[@]}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-dedup-open-set || ${1:-} == --task2-old-only-refill ]]; then
  dir=output/instance-task2-old-dedup-open-set
  flags=()
  if [[ $1 == --task2-old-only-refill ]]; then
    dir=output/instance-task2-old-only-refill
    flags+=(--old-only)
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m tools.owod.probe_task2_dedup_open_set "${flags[@]}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --s-task1-known-fraction || ${1:-} == --s-task1-input-scale || ${1:-} == --s-task1-proposal-stages || ${1:-} == --s-task1-finest-anchor || ${1:-} == --s-task1-anchor-choice || ${1:-} == --s-task1-released-seed || ${1:-} == --s-task1-small-slot || ${1:-} == --s-task1-crop-refill || ${1:-} == --s-task1-crop-proxy || ${1:-} == --s-task1-crop-timeline || ${1:-} == --s-task1-region-geometry || ${1:-} == --s-task1-region-ranking || ${1:-} == --s-task2-duration || ${1:-} == --s-task2-score-competition || ${1:-} == --s-task2-paired-targets || ${1:-} == --s-task2-voc-precision || ${1:-} == --s-task2-old-dedup || ${1:-} == --s-task2-old-dedup-holdout ]]; then
  study="${1#--s-}"
  shift
  dir="output/s-instance-$study"
  if [[ $study == task2-score-competition ]]; then
    dir+="-replay"
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m "tools.owod.run_s_${study//-/_}" "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-duration-holdout || ${1:-} == --task2-old-dedup || ${1:-} == --task2-distillation-teachers || ${1:-} == --task2-distillation-selection || ${1:-} == --task2-history-retrieval || ${1:-} == --task2-slot-naming || ${1:-} == --task2-slot-features || ${1:-} == --task1-discovery-gates || ${1:-} == --task1-soft-discovery || ${1:-} == --task1-empty-allocation || ${1:-} == --task1-labeled-quality || ${1:-} == --task1-quality-views || ${1:-} == --task1-view-boxes || ${1:-} == --task1-decoder-depth || ${1:-} == --task1-encoder-levels || ${1:-} == --task1-encoder-counts || ${1:-} == --task1-encoder-residual || ${1:-} == --task1-encoder-transfer || ${1:-} == --task1-encoder-neighborhood || ${1:-} == --task1-unknown-budget || ${1:-} == --task1-unknown-factors || ${1:-} == --task1-visible-classes || ${1:-} == --task1-known-queries || ${1:-} == --task1-known-confirmation || ${1:-} == --task1-known-fraction ]]; then
  probe="${1#--}"
  dir="output/m-instance-$probe"
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m "tools.owod.probe_${probe//-/_}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-exposure-pair ]]; then
  dir=output/m-instance-task2-exposure-pair
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m tools.owod.run_task2_exposure_pair 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-annotated-distillation || ${1:-} == --task2-history-alignment ]]; then
  study="${1#--}"
  dir="output/m-instance-$study"
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m "tools.owod.run_${study//-/_}" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --task2-six-epochs || ${1:-} == --task2-six-epochs-seed43 ]]; then
  dir=output/m-instance-task2-six-epochs
  seed=42
  if [[ $1 == --task2-six-epochs-seed43 ]]; then
    seed=43
    dir="$dir-seed43"
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u -m tools.owod.run_task2_duration_pair --seed "$seed" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --decoder-bbox-lr-pair || ${1:-} == --decoder-bbox-lr-three-quarter || ${1:-} == --decoder-bbox-seed43 ]]; then
  script=tools/owod/run_decoder_bbox_lr_pair.py
  dir=output/m-instance-decoder-bbox-lr-half
  lr_args=()
  if [[ $1 == --decoder-bbox-lr-three-quarter ]]; then
    dir=output/m-instance-decoder-bbox-lr-three-quarter
    lr_args+=(--three-quarter)
  elif [[ $1 == --decoder-bbox-seed43 ]]; then
    dir=output/m-instance-decoder-bbox-seed43
    script=tools/owod/run_decoder_bbox_seed_pair.py
  fi
  shift
  if [[ ${1:-} == --plan ]]; then
    exec "$python_bin" "$script" "${lr_args[@]}" "$@"
  fi
  export TMPDIR="$PWD/output/.tmp-owod"
  mkdir -p "$dir"
  mkdir -p -m 700 "$TMPDIR"
  flock -n "$dir/run.lock" "$python_bin" -u "$script" "${lr_args[@]}" "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --stronger-parent-study ]]; then
  shift
  if [[ ${1:-} == --plan ]]; then
    exec "$python_bin" tools/owod/run_stronger_parent_study.py "$@"
  fi
  dir=output/m-instance-stronger-parent
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_stronger_parent_study.py "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --cross-attention-study ]]; then
  shift
  if [[ ${1:-} == --plan ]]; then
    exec "$python_bin" tools/owod/run_cross_attention_study.py "$@"
  fi
  dir=output/m-instance-cross-attention
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_cross_attention_study.py "$@" 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --schedule-pair ]]; then
  dir=output/m-instance-schedule-pair
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_schedule_pair.py 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --localization-pair ]]; then
  dir=output/m-instance-localization-pair
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_localization_pair.py 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --classifier-pair ]]; then
  dir=output/m-instance-classifier-pair
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_localization_pair.py --classifier-lr 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --ranking-probe ]]; then
  dir=output/m-instance-ranking-probe
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_ranking_probe.py 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --objectness-probe ]]; then
  dir=output/m-instance-objectness-probe
  mkdir -p "$dir"
  flock -n "$dir/run.lock" "$python_bin" -u tools/owod/run_ranking_probe.py --objectness 2>&1 | tee -a "$dir/console.log"
  exit
fi
if [[ ${1:-} == --plan ]]; then
  exec "$python_bin" tools/owod/run_instance_pilot.py "$@"
fi
mkdir -p output/m-instance-expanded
"$python_bin" -u tools/owod/run_instance_pilot.py "$@" 2>&1 | tee -a output/m-instance-expanded/console.log
