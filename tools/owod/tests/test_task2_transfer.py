import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from models.graph_local.discovery import DiscoveryDetector, DiscoveryFFN
from models.graph_local.routed_adapter import append_stage
from models.graph_local.tests.test_discovery import args
from models.graph_local.tests.test_routed_adapter import Detector, spec
from tools.owod import task2_transfer as transfer


@pytest.mark.parametrize('epochs', [3, 50])
def test_transfer_entry_preflights_only_task1(monkeypatch, capsys, epochs):
    from tools.owod import run_sowodb_task2_routing as driver
    monkeypatch.setattr(sys, 'argv', ['transfer', '--preflight', '--epochs', str(epochs)])
    monkeypatch.setattr(driver, 'verify_parent_inputs', lambda *args: {'proof': 'fixture'})
    monkeypatch.setattr(driver, 'verify_reference', lambda *args: pytest.fail('Unexpected extra checkpoint'))
    driver.main(comparison='transfer')
    plan = json.loads(capsys.readouterr().out)
    assert plan['variants'] == ['inherit', 'scratch']
    assert plan['phases'] == transfer.schedule(epochs)
    assert plan['finetune'] == (epochs == 50)


def test_shared_calibration_and_real_transfer(tmp_path, monkeypatch):
    from tools.owod import debug_discovery_revision
    template = DiscoveryDetector(Detector(), args())
    feature = torch.tensor([0., 1., 0., 0.])
    template.memory.observe(feature[None], torch.tensor([-1]), torch.tensor([3.]))
    discovery = next(layer for layer in template.modules() if isinstance(layer, DiscoveryFFN))
    torch.nn.init.normal_(discovery.adapter.up.weight)
    discovery.snapshot([0])
    layout = spec(classes=(3, 4))
    calls = []

    def calibrate(model, *unused):
        calls.append(1)
        append_stage(model, layout)
        return layout, {3: feature}, {3: [feature]}

    monkeypatch.setattr(debug_discovery_revision, 'prepare_task2', calibrate)
    reports = {}
    opt = SimpleNamespace(output=tmp_path, rank=0, world_size=1, checkpoint='parent')
    config = SimpleNamespace(owod_known_class_ids=[1, 2, 3, 4])
    criterion = SimpleNamespace(base=SimpleNamespace())
    for arm in ('inherit', 'scratch'):
        model = copy.deepcopy(template)
        shared = transfer.shared_calibration(model, criterion, config, opt, {'epochs': 3})
        reports[arm] = transfer.promote_arm(model, *shared, arm)
        transfer.check_transfer(reports[arm], reports.get('inherit') if arm == 'scratch' else None)
    assert calls == [1]
    assert reports['inherit']['nonzero_classes'] == [3]
    assert reports['scratch']['nonzero_classes'] == []
    assert reports['inherit']['effective_mapping_norms'][4] == [0.]
    with pytest.raises(ValueError, match='outside'):
        transfer.check_transfer(dict(reports['scratch'], unchanged_state_sha256='changed'), reports['inherit'])
    with pytest.raises(ValueError, match='No nonzero'):
        transfer.check_transfer(dict(reports['inherit'], nonzero_classes=[]))
    with pytest.raises(ValueError, match='different plan'):
        transfer.shared_calibration(model, criterion, config, opt, {'epochs': 50})


def test_full_schedule_and_final_report(tmp_path):
    phases = transfer.schedule(50)
    assert transfer.schedule(3) == {'increment': [1, 3]}
    assert phases == {'increment': [1, 3, 8, 50], 'finetune': [20]}
    fields = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'unknown_precision',
              'unknown_predictions', 'wi', 'a_ose')
    for arm, score in [('inherit', .2), ('scratch', .1)]:
        for phase, epochs in phases.items():
            directory = (tmp_path if phase == 'increment' else tmp_path / phase) / arm
            directory.mkdir(parents=True)
            metrics = json.dumps({'owod_' + key: score for key in fields})
            if phase == 'increment':
                (directory / 'transfer.json').write_text('{}')
                (directory / 'initial_metrics.json').write_text(metrics)
            for epoch in epochs:
                (directory / f'metrics_epoch_{epoch}.json').write_text(metrics)
    result = transfer.report(tmp_path, phases)
    assert result['phases']['finetune'][20]['inherit_minus_scratch']['current_ap50'] == pytest.approx(.1)
    assert frozenset(result['phases']['increment']) == {0, 1, 3, 8, 50}


def test_balanced_parent_lineage(tmp_path):
    from tools.owod.train_task2_diagnostic import validate_diagnostic_resume
    plan = dict(epochs=50, world_size=2)
    rng = dict.fromkeys(('torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'loader_rng'))
    saved = dict(**rng, diagnostic_only=True, diagnostic_identity=plan, policy=7, epoch=49,
                 run_name='scratch', variant='soft', rank_states=[rng, rng], model={}, optimizer={},
                 discovery_ema={}, args=SimpleNamespace(owod_phase='increment', owod_stage=1,
                 graph_routing='soft', graph_localization='frozen', discovery_no_transfer=True,
                 graph_adapter_specs=[{'stage': 1}]))
    torch.save(saved, tmp_path / 'checkpoint.pth')
    (tmp_path / 'training_complete.json').write_text('{"epoch":49}')
    (tmp_path / 'evaluation_epoch_50_complete.json').write_text('{}')
    _, digest, specs = transfer.increment_parent(tmp_path, plan, 'scratch')
    assert len(digest) == 64 and specs == [{'stage': 1}]
    with pytest.raises(ValueError, match='run changed'):
        transfer.increment_parent(tmp_path, plan, 'inherit')
    with pytest.raises(ValueError, match='mismatch'):
        validate_diagnostic_resume(saved, dict(plan, phase='finetune'), 7)
