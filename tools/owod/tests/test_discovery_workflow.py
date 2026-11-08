import argparse
import io
import json
from pathlib import Path
import zipfile

import pytest
import torch

from tools.owod import download_data, run_discovery_study, validate_discovery


class Response(io.BytesIO):
    def __init__(self, payload, offset, total):
        super().__init__(payload)
        self.status = 206
        self.headers = {'Content-Range': f'bytes {offset}-{offset + len(payload) - 1}/{total}'}


@pytest.mark.parametrize('legacy', [False, True])
def test_checkpoint_metadata_loads_without_tensor_allocation(tmp_path, monkeypatch, legacy):
    path = tmp_path / 'checkpoint.pth'
    torch.save({'args': argparse.Namespace(epochs=50), 'model': {'weight': torch.ones(3)}}, path)
    if legacy:
        monkeypatch.delattr(torch.serialization, 'safe_globals', raising=False)
    checkpoint = validate_discovery.load_checkpoint_metadata(path)
    assert checkpoint['args'].epochs == 50
    assert checkpoint['model']['weight'].device.type == 'meta'


def test_download_resumes_and_reuses_verified_source(tmp_path, monkeypatch):
    target = tmp_path / 'source.zip'
    target.with_suffix('.zip.part').write_bytes(b'abc')
    class Opener:
        def open(self, request, timeout):
            assert request.headers['Range'].startswith('bytes=3-')
            return Response(b'def', 3, 6)
    monkeypatch.setattr(download_data.urllib.request, 'build_opener', lambda *a: Opener())
    download_data.download('https://official/source', target)
    assert target.read_bytes() == b'abcdef'
    # Reuse must verify file hash rather than trust existence.
    download_data.download('https://official/source', target)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='provenance'):
        download_data.download('https://official/source', target)


def test_completed_partial_finalizes_without_range_416(tmp_path, monkeypatch):
    target = tmp_path / 'dino.pth'
    target.with_suffix('.pth.part').write_bytes(b'complete')
    target.with_suffix('.pth.part.json').write_text(json.dumps({'url': 'https://official/weights', 'bytes': 8}))
    monkeypatch.setattr(download_data.urllib.request, 'build_opener', lambda *a: pytest.fail('Already complete'))
    download_data.download('https://official/weights', target)
    assert target.read_bytes() == b'complete'


def test_archive_cannot_escape_data_directory(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('../outside.txt', 'bad')
    with pytest.raises(ValueError, match='escapes'):
        download_data.extract(archive, tmp_path / 'raw')
    assert not (tmp_path / 'outside.txt').exists()


def completed_chain(root, delta=0):
    from main import get_args_parser
    runs = []
    for stage in range(4):
        for phase in (['increment'] if stage == 0 else ['increment', 'finetune']):
            directory = root / str(stage) / phase
            directory.mkdir(parents=True)
            command = ['python', str(validate_discovery.ROOT / 'main.py'), '--epochs', '2',
                       '--owod-stage', str(stage), '--owod-phase', phase, '--output_dir', str(directory)]
            args = get_args_parser().parse_args(command[2:])
            args.owod_manifest_sha256 = 'same'
            torch.save({'args': args, 'epoch': 1}, directory / 'checkpoint.pth')
            (directory / 'training_complete.json').write_text(json.dumps({'last_epoch': 1}))
            metrics = {f'test_owod_{key}': value + delta for key, value in
                       [('previous_ap50', 25), ('current_ap50', 30), ('known_ap50', 28),
                        ('u_recall', 40), ('wi', .1), ('a_ose', 10)]}
            (directory / 'metrics.jsonl').write_text(json.dumps({'epoch': 1, **metrics}) + '\n')
            runs.append(dict(stage=stage, phase=phase, epochs=2, command=command, output_dir=str(directory)))
    plan = dict(protocol='s-owodb', manifest_sha256='same', backbone_weights_sha256='same',
                recipe='ow-detr-paper', runs=runs)
    (root / 'benchmark_plan.json').write_text(json.dumps(plan))
    return plan


@pytest.fixture
def chains(tmp_path, monkeypatch):
    # Protocol/hashes have independent fixture tests; isolate AP final-phase selection here.
    import tools.owod.strict_setting as strict
    monkeypatch.setattr(strict, 'configure_run', lambda args: None)
    monkeypatch.setattr(strict, 'validate_checkpoint', lambda *a, **k: None)
    before, after = tmp_path / 'before', tmp_path / 'after'
    a, b = completed_chain(before), completed_chain(after, 1)
    return before, after, a, b


def test_ap_comparison_uses_all_final_phases_and_checks_actual_args(chains):
    before, after, a, b = chains
    result = validate_discovery.compare_runs(before, after)
    assert len(result['comparisons']) == 7
    assert result['mean_task_current_ap50_delta'] == 1
    assert result['ap_improved_on_this_run']
    checkpoint = Path(b['runs'][0]['output_dir']) / 'checkpoint.pth'
    with torch.serialization.safe_globals([argparse.Namespace]):
        saved = torch.load(checkpoint, weights_only=True)
    saved['args'].lr *= 2  # Plan identical, actual training changed.
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='launch arguments: lr'):
        validate_discovery.compare_runs(before, after)


def test_incomplete_or_missing_unknown_metrics_cannot_claim_ap(chains):
    before, after, a, b = chains
    directory = Path(b['runs'][0]['output_dir'])
    marker = directory / 'training_complete.json'
    marker.write_text(json.dumps({'last_epoch': 0}))
    with pytest.raises(ValueError, match='Incomplete'):
        validate_discovery.compare_runs(before, after)
    marker.write_text(json.dumps({'last_epoch': 1}))
    metrics = directory / 'metrics.jsonl'
    row = json.loads(metrics.read_text())
    del row['test_owod_wi']
    metrics.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='open-world'):
        validate_discovery.compare_runs(before, after)


def test_unpaired_pretraining_rejected(chains):
    before, after, a, b = chains
    b['backbone_weights_sha256'] = 'different'
    (after / 'benchmark_plan.json').write_text(json.dumps(b))
    with pytest.raises(ValueError, match='Unpaired'):
        validate_discovery.compare_runs(before, after)


def test_legacy_v5_metadata_without_ramp_option_can_still_be_compared(chains):
    before, after, a, b = chains
    checkpoint = Path(a['runs'][0]['output_dir']) / 'checkpoint.pth'
    saved = torch.load(checkpoint, weights_only=False)
    del saved['args'].discovery_pseudo_ramp_epochs
    torch.save(saved, checkpoint)
    result = validate_discovery.compare_runs(before, after)
    assert len(result['comparisons']) == 7


def test_pipeline_continues_s_after_m_source_failure(tmp_path, monkeypatch):
    calls = []
    def execute(command, logfile):
        calls.append(command)
        return int('--benchmark' in command and command[command.index('--benchmark') + 1] == 'm-owodb')
    monkeypatch.setattr(run_discovery_study, 'execute', execute)
    monkeypatch.setattr(run_discovery_study, 'prepare_dataset', lambda *a, **k: {'ready': True})
    monkeypatch.setattr(run_discovery_study, 'verify_dataset', lambda *a, **k: {'ready': True})
    monkeypatch.setattr(run_discovery_study, 'compare_runs', lambda *a: {'status': 'measured_single_seed'})
    result = run_discovery_study.main(['--output', str(tmp_path), '--download-attempts', '1'])
    status = json.loads((tmp_path / 'study_status.json').read_text(encoding='utf-8'))
    assert result == 1 and status['status'] == 'incomplete'
    assert status['benchmarks']['m-owodb']['status'] == 'source_unavailable'
    assert status['benchmarks']['s-owodb']['ap_measured']
    assert sum('run_benchmark.py' in ' '.join(c) for c in calls) == 3


def test_training_failure_never_generates_an_ap_report(tmp_path, monkeypatch):
    monkeypatch.setattr(run_discovery_study, 'execute', lambda command, logfile:
                        int('run_benchmark.py' in ' '.join(command)))
    monkeypatch.setattr(run_discovery_study, 'prepare_dataset', lambda *a, **k: {'ready': True})
    monkeypatch.setattr(run_discovery_study, 'verify_dataset', lambda *a, **k: {'ready': True})
    monkeypatch.setattr(run_discovery_study, 'compare_runs', lambda *a: pytest.fail('Training failed'))
    assert run_discovery_study.main(['--output', str(tmp_path), '--benchmark', 's-owodb']) == 1
    status = json.loads((tmp_path / 'study_status.json').read_text(encoding='utf-8'))
    assert not status['ap_measured']
    assert status['benchmarks']['s-owodb']['status'] == 'training_failed'
