import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.owod.task2_parent_provenance import (
    legacy_manifest_bytes, manifest_equivalence, legacy_validation_arguments,
)
from tools.owod.task2_learning_probe import measure_batch, merge_measurements, per_class_probe_indices
from tools.owod.run_sowodb_task2_routing import routing_identity, paired_report


ROOT = Path(__file__).resolve().parents[3]


def legacy_parent():
    return SimpleNamespace(
        owod_manifest='/home/top/disks/new-hdd/zhy/Tree-DETR/data/owod/s-owodb/split_manifest.json',
        owod_image_root='/home/top/disks/new-hdd/zhy/Tree-DETR/data/raw/coco',
        owod_manifest_sha256='13a1f3b3082358e690e04d0030f10edd32d5a6e16e9b1a90a6e2faa7c6f8771d')


def test_reproduces_actual_historical_server_manifest_byte_identity_without_editing_inputs():
    path = ROOT / 'data/owod/s-owodb/split_manifest.json'
    manifest = json.loads(path.read_text())
    before = copy.deepcopy(manifest)
    parent = legacy_parent()
    parent.owod_protocol = 's-owodb'
    parent.owod_evaluation = copy.deepcopy(manifest['evaluation'])
    parent.owod_known_class_ids = manifest['stages'][0]['active_classes'].copy()
    saved = copy.deepcopy(vars(parent))
    assert hashlib.sha256(legacy_manifest_bytes(manifest, parent)).hexdigest() == parent.owod_manifest_sha256
    proof = manifest_equivalence(path, parent)
    assert proof['proof'] == 'exact_precanonical_serialization_sha256'
    assert proof['canonical_manifest_sha256'] != proof['original_manifest_sha256']
    assert manifest == before and vars(parent) == saved


@pytest.mark.parametrize('change', ['annotation', 'evaluation', 'order', 'source', 'image_root'])
def test_equivalence_rejects_changed_content_and_wrong_roots(tmp_path, change):
    canonical = ROOT / 'data/owod/s-owodb/split_manifest.json'
    manifest = json.loads(canonical.read_text())
    parent = legacy_parent()
    if change == 'annotation':
        manifest['stages'][1]['sha256']['increment_train'] = 'different'
    elif change == 'evaluation':
        manifest['evaluation']['ap_mode'] = 'integral'
    elif change == 'order':
        manifest['stages'][1]['classes'].reverse()
    elif change == 'source':
        manifest['source_hashes']['../../raw/coco/annotations/instances_train2017.json'] = 'different'
    else:
        manifest['image_root'] = '../../other-coco'
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='original|root'):
        manifest_equivalence(path, parent)


def test_routing_identity_binds_both_fresh_variants_reference_and_gpu_selection(tmp_path):
    files = [tmp_path / name for name in ('manifest', 'parent', 'gnn')]
    for path in files:
        path.write_text('fixture')
    identity = routing_identity(*files, {'proof': 'exact'}, {'checkpoint_sha256': 'old-v7'}, 2, '0,1')
    assert identity['variants'] == ['hard', 'soft'] and identity['policies'] == [7]
    assert identity['batch_size'] == 1 and identity['effective_batch_size'] == 2
    assert identity != routing_identity(*files, {'proof': 'exact'}, {'checkpoint_sha256': 'other'}, 2, '0,1')
    assert identity != routing_identity(*files, {'proof': 'exact'}, {'checkpoint_sha256': 'old-v7'}, 2, '1,2')


def test_verified_diagnostic_fork_preserves_formal_manifest_rejection(tmp_path):
    from tools.owod.protocol import file_sha256
    from tools.owod.strict_setting import validate_checkpoint
    path = ROOT / 'data/owod/s-owodb/split_manifest.json'
    manifest = json.loads(path.read_text())
    old = legacy_parent()
    old.owod_protocol = 's-owodb'
    old.owod_evaluation = manifest['evaluation']
    old.owod_known_class_ids = manifest['stages'][0]['active_classes']
    old.owod_stage, old.owod_phase, old.epochs = 0, 'increment', 50
    old.owod_recipe, old.owod_discovery = 'ow-detr-paper', False
    original = copy.deepcopy(vars(old))
    args = copy.deepcopy(old)
    args.owod_manifest = str(path)
    args.owod_manifest_sha256 = file_sha256(path)
    args.owod_stage = 1
    parent = {'args': old, 'epoch': 49}
    proof = manifest_equivalence(path, old)
    with pytest.raises(ValueError, match='old/custom/different'):
        validate_checkpoint(args, parent)
    initialization = legacy_validation_arguments(args, parent, path, proof)
    assert initialization.owod_manifest_sha256 == old.owod_manifest_sha256
    assert args.owod_manifest_sha256 == proof['canonical_manifest_sha256']
    assert vars(old) == original
    with pytest.raises(ValueError, match='Diagnostic checkpoints'):
        validate_checkpoint(args, dict(parent, diagnostic_only=True))
    with pytest.raises(ValueError, match='proof changed'):
        legacy_validation_arguments(args, parent, path, dict(proof, proof='different'))


def test_probe_reports_gt_routing_and_masks_unknown_before_topk():
    outputs = dict(pred_logits=torch.tensor([[[0., 4., 3.], [0., 1., 3.]]]),
                   pred_boxes=torch.tensor([[[.2, .2, .2, .2], [.8, .8, .1, .1]]]),
                   pred_objectness=torch.full((1, 2), 8.),
                   pred_novelty=torch.tensor([[1., 2.]]))
    targets = [dict(labels=torch.tensor([1]), boxes=outputs['pred_boxes'][0, :1])]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    stage = SimpleNamespace(class_ids=(1,), probe_logits=torch.tensor([[[0., 4.], [4., 0.]]]),
                            last_gates=torch.tensor([[[0.], [.5]]]))
    original = {key: value.clone() for key, value in outputs.items()}
    masked = measure_batch(outputs, targets, indices, [1, 2], [('layer', stage)], threshold=.99, limit=1)
    joint = measure_batch(outputs, targets, indices, [1, 2], [('layer', stage)], threshold=.5, limit=1)
    assert masked['totals']['unknown_predictions'] == 0
    assert joint['totals']['unknown_predictions'] == 1
    row = joint['classes']['1']
    assert row['raw_box_coverage'] == 1 and row['joint_coverage'] == 0
    assert row['known_only_coverage'] == 1
    assert row['layers']['layer'] == dict(matched=1, correct_route=0, null_route=1,
                                        positive_gate=0, gate_sum=0.)
    merged = merge_measurements([joint, joint])
    assert merged['classes']['1']['gt'] == 2
    assert merged['classes']['1']['layers']['layer']['null_route'] == 2
    for key in outputs:
        assert torch.equal(outputs[key], original[key])


def test_probe_covers_each_released_class_without_using_future_labels():
    dataset = SimpleNamespace(ids=[1, 2, 3], coco=SimpleNamespace(imgToAnns={
        1: [{'category_id': 1}], 2: [{'category_id': 1}, {'category_id': 3}],
        3: [{'category_id': 3}]}))
    assert per_class_probe_indices(dataset, [1, 3], per_class=1) == [0, 1]
    with pytest.raises(ValueError, match='no released'):
        per_class_probe_indices(dataset, [5])


def test_short_comparison_reports_actual_ap_units_and_missing_metrics_are_not_zero():
    result = paired_report({'owod_current_ap50': .02}, {'owod_current_ap50': .03})
    assert result['soft_minus_hard']['owod_current_ap50'] == pytest.approx(.01)
    assert 'owod_previous_ap50' not in result['hard']
    assert result['diagnostic_only'] and not result['finetune']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real detector CUDA check')
def test_real_v7_soft_detector_updates_and_keeps_old_classifier_rows(tmp_path):
    from main import get_args_parser
    from models.graph_local.discovery import build
    from models.graph_local.routed_adapter import append_stage, configure_trainable
    from tools.owod.gnn_calibration import load_detector
    from tools.owod.train_task2_diagnostic import make_optimizer
    from tools.owod.task2_learning_probe import state_sha256, parameter_audit
    args = get_args_parser().parse_args([])
    args.owod_discovery = True
    args.owod_known_class_ids = [1, 3]
    args.owod_previous_class_ids = [1]
    args.owod_current_class_ids = [3]
    args.owod_max_detections = 3
    args.discovery_version = 7
    args.backbone = 'resnet18'
    args.hidden_dim = 32
    args.dim_feedforward = 64
    args.enc_layers = args.dec_layers = args.graph_adapter_last_n = 1
    args.num_queries = 3
    args.skip_backbone_initialization = True
    args.graph_adapter_specs = [dict(stage=1, class_ids=[3], old_ids=[1], rank=2, last_n=1,
                                     risk_beta=2., risk=[[.1]], routing='soft')]
    model, criterion, _ = build(args)
    append_stage(model, args.graph_adapter_specs[0])
    model.cuda()
    criterion.cuda().eval()  # No teacher needed for the supervised backward check.
    hooks = configure_trainable(model, [3], [1, 3])
    model.train_discovery()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = make_optimizer(model, args)
    outputs = model([torch.rand(3, 96, 96, device='cuda')])
    targets = [dict(labels=torch.tensor([3], device='cuda'),
                    boxes=torch.tensor([[.5, .5, .3, .3]], device='cuda'))]
    losses = criterion(outputs, targets)
    total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
    optimizer.step()
    audit = parameter_audit(model, before, [3], [1])
    assert audit['previous_max_abs_change'] == 0
    assert audit['current_classifier_row_delta_norm']['3'] > 0
    assert any(value > 0 for value in audit['adapter_up_norms'].values())
    path = tmp_path / 'soft.pth'
    torch.save(dict(diagnostic_only=True, model=model.state_dict(), args=args), path)
    restored, _ = load_detector(args, path, torch.device('cuda'))
    assert state_sha256(model) == state_sha256(restored)
    for hook in hooks:
        hook.remove()
