"""Summarize bounded diagnostics without treating training loss as AP evidence."""
import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(rows):
    if not rows or {r['rank'] for r in rows} != set(range(rows[0]['world_size'])):
        raise ValueError('Missing or duplicate rank report')
    identities = ('checkpoint_sha256', 'training_source_sha256', 'steps',
                  'world_size', 'batch_size_per_rank', 'short_side')
    for row in rows:
        if (any(row[k] != rows[0][k] for k in identities)
                or row['status'] != 'completed'
                or not all(row[k] for k in ('resource_guard_passed',
                                           'checkpoint_unchanged', 'old_teacher_unchanged'))):
            raise ValueError('Failed guard or mismatched paired reports')
    result = {'diagnostic_only': True, 'ap_measured': False, 'policies': {}}
    for policy in ('5', '6'):
        groups = [r['policies'][policy] for r in rows]
        if any(len(g['measurements']) != rows[0]['steps'] for g in groups):
            raise ValueError('Incomplete measurement stream')
        # Memory is globally gathered and broadcast, not independent per-rank evidence.
        if any(g['anonymous_support'] != groups[0]['anonymous_support'] for g in groups):
            raise ValueError('Final anonymous supports differ across ranks')
        measurements = [m for g in groups for m in g['measurements']]
        dropped = set()
        events = 0
        first_drop = None
        for step, measurement in enumerate(groups[0]['measurements'], 1):
            slots = measurement['support_decreased_slots']
            if slots and first_drop is None:
                first_drop = step
            dropped.update(slots)
            events += len(slots)
        result['policies'][policy] = {
            'mean_pseudo_per_rank_step': mean(m['pseudo'] for m in measurements),
            'mean_old_evidence_per_rank_step': mean(m['old_known_evidence'] for m in measurements),
            'slots_with_support_decrease': sorted(dropped),
            'reset_events_lower_bound': events,
            'first_support_decrease_step': first_drop,
            'final_support_sum': sum(groups[0]['anonymous_support']),
            'max_update_seconds': max(g['seconds'] for g in groups),
            'promotion_inherited': sum(x['inherited'] for x in rows[0]['initial_promotion'][policy]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob('paired.rank*.json'))
    result = summarize([json.loads(p.read_text(encoding='utf-8')) for p in paths])
    target = args.run_dir / 'summary.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Diagnostic summary: {target}; AP not measured.')


if __name__ == '__main__':
    main()
