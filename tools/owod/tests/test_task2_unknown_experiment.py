import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.owod import run_sowodb_task2_routing as driver
from tools.owod.train_task2_diagnostic import validate_pair_resume, run_policy


def test_single_factor_pair_binds_policy_budget_and_same_initialization(tmp_path, monkeypatch):
    files = [tmp_path / name for name in ('manifest', 'checkpoint', 'gnn')]
    for path in files:
        path.write_text('same inputs')
    inputs = (*files, {'proof': 'exact'}, None, 2, '0,1')
    identity = driver.unknown_identity(*inputs)
    assert identity['epochs'] == 3 and identity['evaluation_epochs'] == [3]
    assert identity['variants'] == ['baseline', 'protected']
    assert identity['routing'] == 'soft' and identity['localization'] == 'frozen'
    assert identity['negative_policies'] == {'baseline': 'dense', 'protected': 'cross-view-protected'}
    assert identity['suite'] != driver.routing_identity(*inputs, epochs=3)['suite']
    assert driver.output_directory('unknown', 3) != driver.output_directory('localization', 3)
    observed = []
    monkeypatch.setattr(driver, 'run_policy', lambda *args, **kwargs: observed.append((args, kwargs)))
    driver.run_paired_variants(SimpleNamespace(epochs=3), identity, {'proof': 'exact'}, 'unknown')
    (base_args, baseline), (opt_args, optimized) = observed
    assert base_args == opt_args  # Same opt, full identity and method revision.
    assert baseline.pop('run_name') == 'baseline' and optimized.pop('run_name') == 'protected'
    assert baseline.pop('negative_policy') == 'dense'
    assert optimized.pop('negative_policy') == 'cross-view-protected'
    assert baseline.pop('initialization_reference') is None
    assert optimized.pop('initialization_reference') == 'baseline'
    assert baseline == optimized  # Routing, localization, parent proof all held fixed.


def test_preflight_defaults_to_three_epochs_and_needs_no_old_diagnostic(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['unknown', '--preflight'])
    monkeypatch.setattr(driver, 'verify_parent_inputs', lambda *args: {'proof': 'fixture'})
    monkeypatch.setattr(driver, 'verify_reference', lambda *args: pytest.fail('Not an input to this pair'))
    monkeypatch.setattr(driver, 'run_policy', lambda *args, **kwargs: pytest.fail('Preflight cannot train'))
    driver.main(comparison='unknown')
    result = json.loads(capsys.readouterr().out)
    assert result['epochs'] == 3 and result['evaluation_epochs'] == [3]
    assert result['variants'] == ['baseline', 'protected']
    assert result['reference'] is None
    assert result['output'] == str(driver.output_directory('unknown', 3).resolve())


@pytest.mark.parametrize('policy,other', [('dense', 'cross-view-protected'), ('cross-view-protected', 'dense')])
def test_cross_policy_resume_is_rejected(policy, other):
    saved = dict(variant='soft', run_name='protected', args=SimpleNamespace(
        graph_routing='soft', graph_localization='frozen', discovery_negative_policy=policy))
    validate_pair_resume(saved, 'soft', 'frozen', 'protected', negative_policy=policy)
    with pytest.raises(ValueError, match='negative policy'):
        validate_pair_resume(saved, 'soft', 'frozen', 'protected', negative_policy=other)


def test_protection_cannot_be_enabled_outside_its_paired_plan():
    opt = SimpleNamespace(rank=0, world_size=1, gpu=0)
    identity = dict(epochs=3, evaluation_epochs=[3])
    with pytest.raises(ValueError, match='explicitly paired'):
        run_policy(opt, identity, 7, variant='soft', provenance={'proof': 'fixture'},
                   run_name='protected', negative_policy='cross-view-protected')
    identity['negative_policies'] = {'protected': 'cross-view-protected'}
    with pytest.raises(ValueError, match='explicitly paired'):
        run_policy(opt, identity, 7, variant='soft', provenance={'proof': 'fixture'},
                   run_name='protected', localization='bbox-last', negative_policy='cross-view-protected')


def test_formal_entry_rejects_protection_before_touching_data():
    from tools.owod.strict_setting import configure_run
    with pytest.raises(ValueError, match='paired Task 2 diagnostic launcher'):
        configure_run(SimpleNamespace(discovery_negative_policy='cross-view-protected'))


@pytest.mark.parametrize('epochs', [3, 8])
def test_unknown_report_preserves_directions_missing_metrics_and_final_epoch(tmp_path, epochs):
    for index, name in enumerate(('baseline', 'protected')):
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'initial_metrics.json').write_text(json.dumps(dict(
            owod_known_ap50=.33, owod_u_recall=.0283)))
        for epoch in range(epochs):
            (directory / f'epoch_{epoch}.json').write_text(json.dumps(dict(train=dict(
                discovery_pseudo_count=2., discovery_protected_count=0. if epoch == 0 else index * 3.,
                discovery_protected_fraction=0. if epoch == 0 else index * .01))))
        for epoch in driver.evaluation_epochs(epochs):
            (directory / f'metrics_epoch_{epoch}.json').write_text(json.dumps(dict(
                owod_known_ap50=.32, owod_u_recall=.02 + .01 * index,
                owod_unknown_precision=.01, owod_unknown_predictions=24000 + 1000 * index,
                owod_wi=.04 - .001 * index, owod_a_ose=8230 - 100 * index)))
    report = driver.write_reports(tmp_path, 'unknown', epochs)
    assert not report['interim'] and not report['finetune'] and report['diagnostic_only']
    assert report['evaluation_epoch'] == report['epochs'] == epochs
    assert report['protected_minus_baseline']['owod_u_recall'] == pytest.approx(.01)
    assert report['protected_minus_baseline']['owod_a_ose'] == -100
    assert report['desired_direction']['owod_wi'] == 'lower'
    assert 'owod_previous_ap50' not in report['baseline']
    assert report['change_from_initial']['baseline']['owod_u_recall'] < 0
    assert report['change_from_initial']['protected']['owod_u_recall'] > 0
    assert report['training_activity']['protected'][0]['discovery_protected_count'] == 0
    assert report['training_activity']['protected'][-1]['discovery_protected_count'] == 3
    assert json.loads((tmp_path / 'comparison.json').read_text()) == report
    assert ('change_from_epoch_3_to_8' in report) == (epochs == 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real detector CUDA backward/reload')
def test_real_cuda_network_protects_controlled_candidate_and_reloads_policy(tmp_path, monkeypatch):
    from main import get_args_parser
    from models.graph_local.discovery import build, make_ema, update_ema
    from models.graph_local.routed_adapter import append_stage, configure_trainable, mask_inactive_adapter_gradients
    from tools.owod.gnn_calibration import load_detector
    from tools.owod.task2_learning_probe import state_sha256, parameter_audit
    from tools.owod.train_task2_diagnostic import make_optimizer
    torch.manual_seed(42)
    args = get_args_parser().parse_args([])
    args.owod_discovery = args.diagnostic_only = True
    args.owod_known_class_ids, args.owod_previous_class_ids, args.owod_current_class_ids = [1, 3], [1], [3]
    args.owod_max_detections = args.num_queries = 3
    args.discovery_version, args.discovery_warmup = 7, 0
    args.discovery_negative_policy = 'cross-view-protected'
    args.backbone, args.hidden_dim, args.dim_feedforward = 'resnet18', 32, 64
    args.enc_layers = args.dec_layers = args.graph_adapter_last_n = 1
    args.skip_backbone_initialization = True
    args.graph_routing, args.graph_localization = 'soft', 'frozen'
    args.graph_adapter_specs = [dict(stage=1, class_ids=[3], old_ids=[1], rank=2, last_n=1,
        risk_beta=2., risk=[[.1]], routing='soft')]
    model, criterion, _ = build(args)
    append_stage(model, args.graph_adapter_specs[0])
    model.cuda(); criterion.cuda().train()
    hooks = configure_trainable(model, [3], [1, 3], localization='frozen')
    model.train_discovery()
    teacher = make_ema(model)
    optimizer = make_optimizer(model, args)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    outputs = model([torch.rand(3, 96, 96, device='cuda')])
    # Control only candidate geometry/saliency so the rare protection branch
    # certainly fires; heads, features, graph, adapters and backward are real.
    boxes = torch.tensor([[[.2, .2, .1, .1], [.5, .5, .1, .1], [.8, .8, .1, .1]]], device='cuda')
    outputs['pred_boxes'] = boxes
    cross_view = dict(pred_boxes=boxes.clone(), query_features=outputs['query_features'].detach(),
                      discovery_embedding=outputs['discovery_embedding'].detach(),
                      attention_feature=outputs['attention_feature'], image_sizes=outputs['image_sizes'],
                      padded_size=outputs['padded_size'])
    cross_view['pred_boxes'][..., 0] = 1 - cross_view['pred_boxes'][..., 0]
    outputs['discovery_teacher'] = cross_view
    outputs['discovery_old_teacher'] = dict(pred_logits=torch.full_like(outputs['pred_logits'], -20.),
                                            pred_boxes=boxes)
    monkeypatch.setattr('models.graph_local.discovery_candidates.saliency_in_boxes',
                        lambda *unused: torch.tensor([[.9, .9, .55]], device='cuda'))
    outputs['pred_objectness'].retain_grad()
    for logits in outputs['discovery_routes']:
        logits.retain_grad()
    targets = [dict(labels=torch.tensor([3], device='cuda'), boxes=boxes[0, :1])]
    losses = criterion(outputs, targets)
    assert losses['discovery_pseudo_count'] == losses['discovery_protected_count'] == 1
    total = sum(value * criterion.weight_dict[key] for key, value in losses.items() if key in criterion.weight_dict)
    assert torch.isfinite(total)
    total.backward()
    assert outputs['pred_objectness'].grad[0, 2] == 0
    assert all(logits.grad[0, 2].count_nonzero() == 0 for logits in outputs['discovery_routes'])
    mask_inactive_adapter_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
    optimizer.step()
    model.observe_training(criterion.evidence)
    update_ema(teacher, model)
    audit = parameter_audit(model, before, [3], [1])
    assert audit['previous_max_abs_change'] == audit['bbox_max_abs_change'] == 0
    assert audit['current_classifier_row_delta_norm']['3'] > 0
    assert any(value > 0 for value in audit['adapter_up_norms'].values())
    path = tmp_path / 'protected.pth'
    torch.save(dict(args=args, model=model.state_dict(), diagnostic_only=True), path)
    restored, restored_criterion = load_detector(args, path, torch.device('cuda'))
    assert restored_criterion.negative_policy == 'cross-view-protected'
    assert state_sha256(restored) == state_sha256(model)
    for hook in hooks:
        hook.remove()
