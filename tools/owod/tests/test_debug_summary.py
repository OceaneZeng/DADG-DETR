import copy

import pytest

from tools.owod.summarize_discovery_debug import summarize
from tools.owod.debug_discovery_revision import stable_repeated_batch_memory


def test_live_memory_compares_same_inputs_and_detects_cumulative_growth():
    rows = [dict(stream_positions=[i % 4], live_mib=600 + (i % 4) * 30)
            for i in range(20)]
    assert stable_repeated_batch_memory(rows)
    rows[-1]['live_mib'] += 17
    assert not stable_repeated_batch_memory(rows)


def test_live_memory_requires_a_repeated_batch_after_warmup():
    rows = [dict(stream_positions=[i], live_mib=600) for i in range(4)]
    assert not stable_repeated_batch_memory(rows)


def reports():
    row = dict(rank=0, world_size=2, steps=2, batch_size_per_rank=2, short_side=800,
               checkpoint_sha256='parent', training_source_sha256='train',
               status='completed', resource_guard_passed=True,
               checkpoint_unchanged=True, old_teacher_unchanged=True,
               initial_promotion={'5': [], '6': []}, policies={})
    for policy in ('5', '6'):
        row['policies'][policy] = dict(seconds=1, anonymous_support=[1, 2], measurements=[
            dict(pseudo=2, old_known_evidence=1, support_decreased_slots=[]),
            dict(pseudo=4, old_known_evidence=3, support_decreased_slots=[0, 1])])
    other = copy.deepcopy(row)
    other['rank'] = 1
    other['policies']['6']['measurements'][0]['pseudo'] = 10
    return [row, other]


def test_summary_averages_rank_evidence_but_does_not_double_count_global_resets():
    result = summarize(reports())['policies']['6']
    assert result['mean_pseudo_per_rank_step'] == 5
    assert result['reset_events_lower_bound'] == 2
    assert result['first_support_decrease_step'] == 2


def test_summary_rejects_missing_rank_and_unsynchronized_memory():
    rows = reports()
    with pytest.raises(ValueError, match='rank report'):
        summarize(rows[:1])
    rows[1]['policies']['6']['anonymous_support'] = [0, 2]
    with pytest.raises(ValueError, match='differ across ranks'):
        summarize(rows)
