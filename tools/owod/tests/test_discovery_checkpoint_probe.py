"""Protect train-only sampling, real routed gates and honest promotion diagnostics."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.owod.probe_discovery_checkpoint import (capture_routes, checked_inputs, memory_report,
    routing_rows, select_training_indices, summarize_routes)
from tools.owod.tests.test_discovery_v7 import configured, inputs


def dataset():
    return SimpleNamespace(ids=[10, 20, 30, 40], coco=SimpleNamespace(
        imgs={i: {'owod_source_id': str(i)} for i in [10, 20, 30, 40]},
        imgToAnns={10: [{'category_id': 1}], 20: [{'category_id': 1}],
                   30: [{'category_id': 4}], 40: []}))


def test_sampling_covers_released_classes_and_keeps_identical_checkpoint_samples():
    data = dataset()
    before = copy.deepcopy(data)
    indices = select_training_indices(data, [1, 4], 3, 42)
    assert len(indices) == len(set(indices)) == 3
    assert 2 in indices and any(i in indices for i in (0, 1))
    assert indices == select_training_indices(data, [1, 4], 3, 42)
    assert sorted(select_training_indices(data, [1, 4], 100, 42)) == [0, 1, 2, 3]
    assert data == before
    with pytest.raises(ValueError, match='positive'):
        select_training_indices(data, [1, 4], 0, 42)


def test_hard_and_soft_gate_statistics_distinguish_route_choice_from_gate_mass():
    from models.graph_local.routed_adapter import RiskRoutedStage, routing_outputs
    head = nn.Linear(4, 6)
    spec = dict(class_ids=[2, 4], old_ids=[1], rank=2, risk_beta=2., risk=[[.5], [.2]])
    model = nn.Sequential(RiskRoutedStage(head, spec))
    stage = model[0]
    with torch.no_grad():
        stage.router.weight.zero_()
        stage.router.bias.copy_(torch.tensor([2., 0., 3.]))  # null wins
    x = torch.randn(1, 2, 4)
    targets = [dict(labels=torch.tensor([2, 4]), image_id=torch.tensor(10))]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    with capture_routes(model) as captured:
        model(x)
        routing_outputs(model)  # Real DETR consumes these before returning.
    assert stage.last_logits is None and not stage._forward_hooks
    hard = summarize_routes(routing_rows(captured, indices, targets))
    assert hard['2']['null_argmax_fraction'] == hard['4']['null_argmax_fraction'] == 1
    assert hard['2']['positive_gate_fraction'] == 0
    assert hard['4']['gate_p10_p50_p90'] == [0., 0., 0.]
    stage.routing = 'soft'
    with capture_routes(model) as captured:
        model(x)
        routing_outputs(model)
    soft = summarize_routes(routing_rows(captured, indices, targets))
    assert soft['2']['null_argmax_fraction'] == 1
    assert soft['2']['positive_gate_fraction'] == soft['4']['positive_gate_fraction'] == 1
    assert soft['2']['gate_p10_p50_p90'][1] < soft['2']['mean_probability']
    assert summarize_routes([]) == {}


class CosineAffinity(nn.Module):
    def similarities(self, features, centers):
        return nn.functional.normalize(features, dim=-1) @ nn.functional.normalize(centers, dim=-1).T


def test_memory_audit_distinguishes_support_affinity_and_missing_samples_without_mutation():
    from models.graph_local.discovery_memory import PrototypeMemory
    from models.graph_local.discovery import DiscoveryFFN
    model = nn.Module()
    model.memory = PrototypeMemory(4, capacity=2, known_slots=3)
    model.affinity_graph = CosineAffinity()
    model.branch = DiscoveryFFN(4, 2, 2)
    model.min_support, model.transfer_threshold, model.transfer_enabled = 3., .8, True
    model.memory.counts[3] = 4.
    model.memory.centers[3, 0] = 1.
    before = copy.deepcopy(model.state_dict())
    sums = {1: torch.tensor([2., 0., 0., 0.]), 2: torch.tensor([-1., 0., 0., 0.])}
    report = memory_report(model, sums, {1: 2, 2: 1}, [1, 2, 4])
    classes = report['current_feature_affinity']
    assert classes['1']['reason'] == 'eligible_now' and classes['1']['eligible_slots'] == [0]
    assert classes['2']['reason'] == 'below_affinity_threshold'
    assert classes['4']['reason'] == 'no_sampled_GT'
    assert report['snapshot_effective_norms']['branch'] == [0., 0.]
    assert report['largest_support_fraction'] == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    model.memory.counts.zero_()
    report = memory_report(model, sums, {1: 2, 2: 1}, [1])
    assert report['current_feature_affinity']['1']['reason'] == 'no_supported_slots'
    assert report['largest_support_fraction'] is None
    model.transfer_enabled = False
    assert memory_report(model, sums, {1: 2}, [1])['current_feature_affinity']['1']['reason'] == 'transfer_disabled'


def test_probe_rejects_unverified_stage_classes_and_evaluation_changes(inputs):
    from tools.owod import run_benchmark
    inputs.discovery_pseudo_normalization = 'count'
    inputs.discovery_known_rank_coef = 1.
    args = configured(run_benchmark.build_plan(inputs))
    manifest, annotation = checked_inputs(args, inputs.manifest)
    assert manifest['protocol'] == 'm-owodb' and annotation.name == 'instances_increment_train2017.json'
    for field, value, message in [('owod_screening', 'screen.json', 'full-data'),
        ('paper_comparable_protocol', False, 'full-data'), ('owod_stage', 3, 'Task 1/2'),
        ('owod_manifest_sha256', 'bad', 'manifest mismatch'),
        ('owod_current_class_ids', [999], 'class order'),
        ('owod_evaluation', {'ap_mode': 'voc2007_11point'}, 'evaluation convention')]:
        changed = copy.deepcopy(args)
        setattr(changed, field, value)
        with pytest.raises(ValueError, match=message):
            checked_inputs(changed, inputs.manifest)


def test_cli_refuses_non_json_and_existing_outputs_before_loading(tmp_path):
    from tools.owod.probe_discovery_checkpoint import main
    common = ['--checkpoint', 'missing.pth', '--manifest', 'missing.json', '--output']
    with pytest.raises(SystemExit):
        main(common + [str(tmp_path / 'report.pth')])
    path = tmp_path / 'existing.json'
    path.write_text('preserve this report')
    with pytest.raises(FileExistsError):
        main(common + [str(path)])
    assert path.read_text() == 'preserve this report'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cli_real_cuda_checkpoint_ema_and_consumed_routes_are_read_only(inputs, monkeypatch):
    """Synthetic input verifies execution only, never a formal AP experiment."""
    from PIL import Image
    from datasets.coco import CocoDetection
    import datasets.transforms as transforms
    from models.graph_local.discovery import build, make_ema
    from models.graph_local.routed_adapter import append_stage
    from tools.owod import gnn_calibration, run_benchmark
    from tools.owod.probe_discovery_checkpoint import main
    from tools.owod.protocol import build_official_manifest, file_sha256

    root = inputs.manifest.parent
    for path in root.glob('stage_*/*.json'):
        data = json.loads(path.read_text())
        for annotation in data['annotations']:
            annotation['area'] = 1.
        path.write_text(json.dumps(data))
    manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
    manifest['image_root'] = str(root)
    inputs.manifest.write_text(json.dumps(manifest))
    Image.new('RGB', (64, 64), (100, 140, 180)).save(root / '100001.jpg')
    saved = configured(run_benchmark.build_plan(inputs))
    saved.owod_stage = 1
    saved.owod_known_class_ids = manifest['stages'][1]['active_classes']
    saved.owod_current_class_ids = manifest['stages'][1]['classes']
    saved.no_pretrained_backbone, saved.backbone_weights = True, ''
    saved.hidden_dim, saved.dim_feedforward = 64, 128
    saved.enc_layers, saved.dec_layers, saved.num_queries = 1, 2, 8
    saved.graph_adapter_last_n = 1
    spec = dict(stage=1, class_ids=saved.owod_current_class_ids,
        old_ids=manifest['stages'][0]['classes'], rank=8, last_n=1, risk_beta=2.,
        risk=[[.2] * 20 for _ in range(20)], promotion=[dict(inherited=False)])
    saved.graph_adapter_specs = [spec]
    model, _, _ = build(saved)
    append_stage(model, spec)
    checkpoint = root / 'checkpoint0009.pth'
    torch.save(dict(args=saved, epoch=9, model=model.state_dict(),
                    discovery_ema=make_ema(model).state_dict()), checkpoint)
    before = file_sha256(checkpoint)
    # Keep the real dataset/normalization and detector; reduce only test image size.
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    monkeypatch.setattr(gnn_calibration, 'build_calibration_dataset',
        lambda image_root, annotation: CocoDetection(image_root, annotation, transform, False))
    loaded = []
    original_loader = gnn_calibration.load_detector

    def observe_model(*args):
        detector, criterion = original_loader(*args)
        loaded.append(detector)
        return detector, criterion

    monkeypatch.setattr(gnn_calibration, 'load_detector', observe_model)
    output = root / 'probe.json'
    common = ['--checkpoint', str(checkpoint), '--manifest', str(inputs.manifest),
              '--images', '1', '--device', 'cuda:0', '--output']
    report = main(common + [str(output)])
    assert report == json.loads(output.read_text())
    assert report['task'] == 2 and report['epoch'] == 10
    assert report['optimization_steps'] == 0 and report['ema_candidates']['available']
    assert report['supervision']['gt_count'] == 20
    assert len(report['routing_observations']) == 8  # Matcher selected 8 of 20 GT.
    assert report['recorded_promotion'] == [[dict(inherited=False)]]
    assert file_sha256(checkpoint) == before
    for name, value in loaded[0].state_dict().items():
        assert torch.equal(value.cpu(), model.state_dict()[name])
    assert all(not module._forward_hooks for module in loaded[0].modules())

    # Change an input after validation, as a concurrent write would do.
    from models.matcher import HungarianMatcher
    def change_input(*_args):
        with inputs.manifest.open('a') as handle:
            handle.write('\n')
        return loaded[0], SimpleNamespace(matcher=HungarianMatcher())

    monkeypatch.setattr(gnn_calibration, 'load_detector', change_input)
    rejected = root / 'changed-input.json'
    with pytest.raises(ValueError, match='Input changed during probe'):
        main(common + [str(rejected)])
    assert not rejected.exists()
