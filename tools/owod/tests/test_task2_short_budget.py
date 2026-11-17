import json
import sys

import pytest

from tools.owod import run_sowodb_task2_routing as driver
from tools.owod.train_task2_diagnostic import validate_diagnostic_resume


@pytest.mark.parametrize('comparison,builder', [
    ('routing', driver.routing_identity), ('localization', driver.localization_identity)])
def test_three_epoch_plan_has_separate_identity_output_and_only_final_evaluation(tmp_path, comparison, builder):
    files = [tmp_path / name for name in ('manifest', 'checkpoint', 'gnn')]
    for path in files:
        path.write_text('same paired inputs')
    args = (*files, {'proof': 'exact'}, {'checkpoint_sha256': 'reference'}, 2, '0,1')
    short, extended = builder(*args, epochs=3), builder(*args)
    assert short['epochs'] == 3 and short['evaluation_epochs'] == [3]
    assert extended['epochs'] == 8 and extended['evaluation_epochs'] == [3, 8]
    assert short['suite'] != extended['suite']
    for key in ('checkpoint_sha256', 'gnn_sha256', 'manifest_sha256', 'sources', 'variants',
                'parent_provenance', 'reference', 'effective_batch_size', 'batch_size', 'world_size'):
        assert short[key] == extended[key]
    assert driver.output_directory(comparison, 3) != driver.output_directory(comparison, 8)
    for current, saved in ((short, extended), (extended, short)):
        checkpoint = dict(diagnostic_only=True, diagnostic_identity=saved, epoch=2, policy=7)
        before = dict(checkpoint)
        with pytest.raises(ValueError, match='identity'):
            validate_diagnostic_resume(checkpoint, current, 7)
        assert checkpoint == before


@pytest.mark.parametrize('comparison', ['routing', 'localization'])
@pytest.mark.parametrize('epochs', [3, 8])
def test_final_report_reads_only_scheduled_epochs_and_preserves_scope(tmp_path, comparison, epochs):
    for index, (name, _, _) in enumerate(driver.paired_variants(comparison)):
        directory = tmp_path / name
        directory.mkdir()
        for epoch in driver.evaluation_epochs(epochs):
            (directory / f'metrics_epoch_{epoch}.json').write_text(json.dumps({
                'owod_current_ap50': .01 * epoch + .01 * index,
                'owod_u_recall': .02 * epoch + .01 * index,
                'owod_unknown_predictions': 100 + 10 * index,
                'owod_wi': .04 - .001 * index, 'owod_a_ose': 1000 - 10 * index}))
    report = driver.write_reports(tmp_path, comparison, epochs)
    assert report['epochs'] == report['evaluation_epoch'] == epochs
    assert report['diagnostic_only'] and not report['interim'] and not report['finetune']
    baseline = 'hard' if comparison == 'routing' else 'frozen'
    assert 'owod_previous_ap50' not in report[baseline]
    assert json.loads((tmp_path / 'comparison.json').read_text()) == report
    if epochs == 3:
        assert 'change_from_epoch_3_to_8' not in report
        assert not (tmp_path / 'comparison_epoch_8.json').exists()
    else:
        assert report['change_from_epoch_3_to_8'][baseline]['owod_current_ap50'] == pytest.approx(.05)
        assert json.loads((tmp_path / 'comparison_epoch_3.json').read_text())['interim']


@pytest.mark.parametrize('comparison', ['routing', 'localization'])
@pytest.mark.parametrize('epochs', [3, 8])
def test_preflight_carries_cli_budget_and_default_output(monkeypatch, capsys, comparison, epochs):
    observed = []
    monkeypatch.setattr(sys, 'argv', ['paired', '--preflight', '--epochs', str(epochs)])
    monkeypatch.setattr(driver, 'verify_parent_inputs', lambda *args: {'proof': 'unit-test'})
    monkeypatch.setattr(driver, 'verify_reference', lambda *args: {'role': 'unit-test'})
    # The preflight must not allocate a process group, construct a model or train.
    monkeypatch.setattr(driver, 'run_policy', lambda *args, **kwargs: observed.append('trained'))
    driver.main(comparison=comparison)
    result = json.loads(capsys.readouterr().out)
    assert result['epochs'] == epochs
    assert result['evaluation_epochs'] == list(driver.evaluation_epochs(epochs))
    assert result['output'] == str(driver.output_directory(comparison, epochs).resolve())
    assert result['effective_batch_size'] == 2 and not result['finetune']
    assert observed == []


@pytest.mark.parametrize('epochs', [0, 1, 2, 4, 7, 9, 3., '3', True])
def test_other_budgets_are_rejected(epochs):
    with pytest.raises(ValueError, match='3 or 8'):
        driver.evaluation_epochs(epochs)


def test_three_epoch_report_rejects_eight_epoch_measurement():
    with pytest.raises(ValueError, match='predeclared'):
        driver.localization_report({}, {}, epochs=3, epoch=8)
