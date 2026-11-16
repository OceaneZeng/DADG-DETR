from types import SimpleNamespace

import pytest

from tools.owod.run_sowodb_task2_routing import (
    paired_variants, routing_identity, localization_identity, localization_report,
)
from tools.owod.train_task2_diagnostic import validate_pair_resume


def test_localization_pair_changes_only_the_coordinate_update_policy():
    control, treatment = paired_variants('localization')
    assert control == ('frozen', 'soft', 'frozen')
    assert treatment == ('bbox-last', 'soft', 'bbox-last')
    assert paired_variants('routing') == (('hard', 'hard', 'frozen'), ('soft', 'soft', 'frozen'))
    with pytest.raises(ValueError, match='comparison'):
        paired_variants('other')


def test_localization_identity_is_separate_from_completed_routing_results(tmp_path):
    inputs = [tmp_path / name for name in ('manifest', 'checkpoint', 'gnn')]
    for path in inputs:
        path.write_text('same inputs')
    args = (*inputs, {'proof': 'exact'}, {'checkpoint_sha256': 'reference'}, 2, '0,1')
    routing, localization = routing_identity(*args), localization_identity(*args)
    assert routing['suite'] != localization['suite']
    assert localization['routing'] == 'soft'
    assert localization['variants'] == ['frozen', 'bbox-last']
    assert localization['localization'] == ['frozen', 'bbox-last']
    for key in ('checkpoint_sha256', 'manifest_sha256', 'gnn_sha256', 'sources',
                'epochs', 'evaluation_epochs', 'effective_batch_size', 'parent_provenance'):
        assert localization[key] == routing[key]


def test_localization_report_preserves_scope_units_and_missing_metrics():
    report = localization_report({'owod_current_ap50': .04}, {'owod_current_ap50': .07})
    assert report['bbox_last_minus_frozen']['owod_current_ap50'] == pytest.approx(.03)
    assert 'owod_previous_ap50' not in report['frozen']
    assert 'hard' not in report and 'soft' not in report
    assert report['routing'] == 'soft' and report['diagnostic_only']
    assert report['evaluation_epoch'] == 8 and not report['interim'] and not report['finetune']
    assert localization_report({}, {}, epoch=3)['interim']
    with pytest.raises(ValueError, match='predeclared'):
        localization_report({}, {}, epoch=7)


def test_resume_rejects_other_localization_even_with_the_same_soft_routing():
    saved = dict(variant='soft', run_name='frozen',
                 args=SimpleNamespace(graph_routing='soft', graph_localization='frozen'))
    validate_pair_resume(saved, 'soft', 'frozen', 'frozen')
    with pytest.raises(ValueError, match='routing/localization/run'):
        validate_pair_resume(saved, 'soft', 'bbox-last', 'bbox-last')
    with pytest.raises(ValueError, match='routing/localization/run'):
        validate_pair_resume(saved, 'soft', 'frozen', 'another')
    with pytest.raises(ValueError, match='routing/localization/run'):
        validate_pair_resume(saved, 'hard', 'frozen')
    # Legacy routing checkpoints did not need a separate run name.
    saved.pop('run_name')
    validate_pair_resume(saved, 'soft', 'frozen')
