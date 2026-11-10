import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.deformable_transformer import DeformableTransformerDecoderLayer
from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from models.graph_local.routed_adapter import (
    RiskRoutedStage, append_stage, configure_trainable, install_specs,
    mask_inactive_adapter_gradients, routing_loss, routing_outputs,
)


def spec(classes=(3, 4), old=(1, 2), stage=1, risk=0.0):
    return dict(stage=stage, class_ids=list(classes), old_ids=list(old), rank=2,
                last_n=1, risk_beta=2., risk=[[risk] * len(old) for _ in classes])


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(d_model=4, d_ffn=8, dropout=0, n_heads=1)])
        self.class_embed = nn.ModuleList([nn.Linear(4, 7)])
        self.bbox_embed = nn.Linear(4, 4)

    def forward(self, x):
        x = self.transformer.decoder.layers[0].forward_ffn(x)
        return {'pred_logits': self.class_embed[0](x),
                'pred_boxes': self.bbox_embed(x).sigmoid(),
                'adapter_routes': routing_outputs(self)}


def stage_of(model, index=0):
    return model.transformer.decoder.layers[0].graph_adapter.stages[index]


def select(stage, column):
    with torch.no_grad():
        stage.router.weight.zero_()
        stage.router.bias.fill_(-8.)
        stage.router.bias[column] = 8.
        stage.old_evidence.weight.zero_()
        stage.old_evidence.bias.fill_(8.)


def test_directed_gnn_risk_changes_forward_and_gradients():
    torch.manual_seed(2)
    low = RiskRoutedStage(nn.Linear(4, 7), spec())
    high = copy.deepcopy(low)
    high.risk[0, 1] = 1.  # current class 3 harms old class 2
    for m in (low, high):
        select(m, 0)
        nn.init.ones_(m.adapters[0].up.weight)
    x = torch.ones(1, 2, 4)
    low_result, high_result = low(x), high(x)
    assert torch.allclose(high_result, low_result * torch.exp(torch.tensor(-2.) * torch.sigmoid(torch.tensor(8.))))
    low_result.sum().backward()
    high_result.sum().backward()
    assert high.adapters[0].up.weight.grad.norm() < low.adapters[0].up.weight.grad.norm()
    assert low.adapters[1].up.weight.grad.count_nonzero() == 0


def test_soft_routing_updates_branches_even_when_null_is_selected():
    layout = spec()
    layout['routing'] = 'soft'
    stage = RiskRoutedStage(nn.Linear(4, 7), layout)
    with torch.no_grad():
        stage.router.weight.zero_()
        stage.router.bias[:] = torch.tensor([-1., -1., 1.])
    stage(torch.ones(1, 2, 4)).sum().backward()
    assert stage.last_gates.min() > 0
    assert all(a.up.weight.grad.abs().sum() > 0 for a in stage.adapters)
    restored = RiskRoutedStage(nn.Linear(4, 7), layout)
    restored.load_state_dict(stage.state_dict(), strict=True)
    stage.eval(); restored.eval()
    assert torch.equal(stage(torch.ones(1, 2, 4)), restored(torch.ones(1, 2, 4)))


def test_unselected_branch_ignores_adamw_decay_and_existing_momentum():
    model = Detector()
    append_stage(model, spec())
    configure_trainable(model, [3, 4], [1, 2, 3, 4])
    stage = stage_of(model)
    optimizer = torch.optim.AdamW(stage.adapters.parameters(), lr=.05, weight_decay=.1)
    x = torch.ones(1, 2, 4)
    select(stage, 1)
    stage(x).sum().backward()
    mask_inactive_adapter_gradients(model)
    optimizer.step()  # Establish branch 1 momentum.
    saved = copy.deepcopy(stage.adapters[1].state_dict())
    optimizer.zero_grad(set_to_none=True)
    select(stage, 0)
    stage(x).sum().backward()
    mask_inactive_adapter_gradients(model)
    optimizer.step()
    for name, value in stage.adapters[1].state_dict().items():
        assert torch.equal(value, saved[name])
    assert stage.adapters[0].up.weight.count_nonzero() > 0


def test_stage_growth_freezes_old_banks_and_reconstructs_exact_inference():
    model = Detector()
    specs = [spec(), spec(classes=(5, 6), old=(1, 2, 3, 4), stage=2)]
    install_specs(model, specs)
    handles = configure_trainable(model, [5, 6], [1, 2, 3, 4, 5, 6])
    assert not any(p.requires_grad for p in stage_of(model).parameters())
    assert all(p.requires_grad for p in stage_of(model, 1).adapters.parameters())
    assert not stage_of(model, 1).old_evidence.weight.requires_grad
    select(stage_of(model, 1), 0)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.05, weight_decay=0)
    x = torch.randn(1, 2, 4)
    model(x)['pred_logits'][..., 5].sum().backward()
    mask_inactive_adapter_gradients(model)
    optimizer.step()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[name], p), name
    assert torch.equal(before['class_embed.0.weight'][:5], model.class_embed[0].weight[:5])
    restored = Detector()
    install_specs(restored, specs)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(model(x)['pred_logits'], restored(x)['pred_logits'])
    with pytest.raises(ValueError, match='duplicate'):
        append_stage(model, specs[-1])
    for handle in handles:
        handle.remove()
    configure_trainable(model, [5, 6], [1, 2, 3, 4, 5, 6], finetune=True)
    assert stage_of(model).router.weight.requires_grad
    assert not stage_of(model).old_evidence.weight.requires_grad


def test_router_learns_new_class_from_null_then_activates_adapter():
    stage = RiskRoutedStage(nn.Linear(4, 7), spec())
    select(stage, 2)  # Null route initially dominates.
    # Use moderate bias so a few supervised steps can overcome null.
    with torch.no_grad():
        stage.router.bias[:] = torch.tensor([-1., -1., 1.])
    x = torch.ones(1, 1, 4)
    targets = [{'labels': torch.tensor([3])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    optimizer = torch.optim.SGD(stage.parameters(), lr=.3)
    stage(x)
    assert stage.last_gates.count_nonzero() == 0
    for _ in range(8):
        optimizer.zero_grad()
        value = stage(x)
        loss = routing_loss([(stage.last_logits, stage.class_ids)], targets, indices) - value.sum()
        loss.backward()
        optimizer.step()
    stage(x)
    assert stage.last_gates[..., 0].min() > 0
    assert stage.adapters[0].up.weight.count_nonzero() > 0


def test_detector_criterion_routes_only_matched_seen_labels():
    model = Detector()
    append_stage(model, spec())
    configure_trainable(model, [3, 4], [1, 2, 3, 4])
    criterion = SetCriterion(7, HungarianMatcher(), {}, ['labels', 'boxes'])
    criterion.active_class_ids = [1, 2, 3, 4]
    targets = [{'labels': torch.tensor([3]), 'boxes': torch.tensor([[.5, .5, .2, .2]])}]
    outputs = model(torch.randn(1, 3, 4))
    losses = criterion(outputs, targets)
    assert losses['loss_adapter_route'].isfinite()
    sum(losses[k] for k in ('loss_ce', 'loss_bbox', 'loss_giou', 'loss_adapter_route')).backward()
    assert stage_of(model).router.weight.grad.abs().sum() > 0
    assert stage_of(model).last_logits is None


@pytest.mark.parametrize('risk', [float('nan'), -1., 2.])
def test_invalid_graph_scores_rejected(risk):
    with pytest.raises(ValueError, match='Invalid'):
        RiskRoutedStage(nn.Linear(4, 7), spec(risk=risk))


def test_gnn_controller_refuses_future_or_stale_checkpoint(tmp_path):
    from tools.owod.graph_adaptation import validate_gnn_metadata
    from tools.owod.protocol import file_sha256
    source = tmp_path / 'source.pth'
    source.write_bytes(b'checkpoint identity fixture')
    args = SimpleNamespace(owod_manifest_sha256='manifest', owod_stage=2,
                           owod_previous_class_ids=[1, 2], graph_adapter_last_n=2)
    metadata = dict(supervision='empirical_train_loss_increase', production_ready=True,
                    uses_validation_labels=False, manifest_sha256='manifest', stage=1,
                    checkpoint_sha256=file_sha256(source), training_classes=[1, 2], last_decoder_layers=2)
    validate_gnn_metadata(metadata, args, source)
    for field, value in [('stage', 2), ('checkpoint_sha256', 'stale'),
                         ('uses_validation_labels', True), ('training_classes', [1, 2, 3])]:
        with pytest.raises(ValueError):
            validate_gnn_metadata({**metadata, field: value}, args, source)


def test_controller_builds_correct_directed_slice_from_actual_gnn(tmp_path, monkeypatch):
    from models.graph_local.gnn import ClassInterferenceGNN, save_gnn_checkpoint, compress_gradient_sketches
    from tools.owod.graph_adaptation import build_stage_spec
    from tools.owod.protocol import file_sha256
    import tools.owod.gnn_calibration as calibration
    import tools.owod.graph_adaptation as controller

    source = tmp_path / 'source.pth'
    source.write_bytes(b'completed checkpoint')
    files = {key: tmp_path / f'{key}.json' for key in ('train', 'increment_train')}
    for path in files.values():
        path.write_text('{}')
    gnn_path = tmp_path / 'gnn.pt'
    gnn = ClassInterferenceGNN(input_dim=4, hidden_dim=8).eval()
    args = SimpleNamespace(owod_stage=1, owod_manifest='fixture', owod_manifest_sha256='manifest',
                           owod_current_class_ids=[4, 3], owod_previous_class_ids=[2, 1],
                           graph_adapter_last_n=1, graph_adapter_rank=2, graph_risk_beta=2.,
                           graph_sketch_max_images=12, graph_gnn_checkpoint=str(gnn_path), num_workers=0)
    save_gnn_checkpoint(gnn, gnn_path, extra=dict(
        supervision='empirical_train_loss_increase', production_ready=True, uses_validation_labels=False,
        manifest_sha256='manifest', checkpoint_sha256=file_sha256(source), stage=0,
        last_decoder_layers=1, training_classes=[1, 2]))
    monkeypatch.setattr(controller, 'experiment_files', lambda *a: ({'image_root': str(tmp_path)}, files))
    monkeypatch.setattr(calibration, 'build_calibration_dataset', lambda root, path: path)
    sketches = {c: torch.arange(8).float() * c + c * c for c in (1, 2, 3, 4)}

    def extract(model, criterion, datasets, ids, device, **kwargs):
        assert datasets == {1: files['train'], 2: files['train'],
                            3: files['increment_train'], 4: files['increment_train']}
        assert set(ids) == {1, 2, 3, 4}  # No future-class or validation evidence.
        return sketches, {c: 1 for c in ids}

    monkeypatch.setattr(calibration, 'compute_gradient_sketches', extract)
    result = build_stage_spec(Detector(), None, args, source)
    ordered, features = compress_gradient_sketches(sketches, 4)
    scores = gnn(features)['edge_prob'].detach()
    expected = scores[[ordered.index(c) for c in (4, 3)]][:, [ordered.index(c) for c in (2, 1)]]
    assert torch.equal(torch.tensor(result['risk']), expected)
    assert result['gnn_sha256'] == file_sha256(gnn_path)
