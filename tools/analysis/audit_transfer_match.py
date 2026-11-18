"""Explain an existing S Task 2 transfer decision on CPU, without rerunning calibration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.protocol import file_sha256
from tools.owod.task2_transfer import match_diagnostics


def audit(results, parent):
    import torch
    from models.graph_local.discovery_memory import AffinityGraph

    plan = json.loads((results / 'diagnostic_plan.json').read_text())
    if (plan.get('comparison') != 'transfer' or plan.get('phase') != 'increment'
            or plan.get('checkpoint_sha256') != file_sha256(parent)):
        raise ValueError('Expected the original transfer plan and pinned Task 1 checkpoint')
    saved = torch.load(parent, map_location='cpu', weights_only=False)['args']
    cache = torch.load(results / 'common_initialization.pth', map_location='cpu', weights_only=False)
    if cache.get('diagnostic_only') is not True or cache.get('diagnostic_identity') != plan:
        raise ValueError('Cached initialization does not match the original run plan')
    recorded = json.loads((results / 'inherit' / 'transfer.json').read_text())
    ids = cache['layout']['class_ids']
    if (recorded['arm'] != 'inherit' or set(cache['means']) != {r['class_id'] for r in recorded['promotion']}
            or {int(k) for k in recorded['effective_mapping_norms']} != set(ids)):
        raise ValueError('Promotion record does not match the cached calibration classes')
    state = cache['model']
    graph = AffinityGraph(saved.hidden_dim)
    prefix = 'affinity_graph.'
    graph.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    n = saved.discovery_capacity
    if n <= 0 or n >= len(state['memory.counts']):
        raise ValueError('Invalid anonymous memory capacity')
    details = match_diagnostics(graph.eval(), state['memory.counts'][-n:], state['memory.centers'][-n:],
                                cache['means'], ids, saved.discovery_min_support, saved.discovery_transfer_threshold,
                                recorded['promotion'])
    return dict(parent_sha256=plan['checkpoint_sha256'], recorded_matched_classes=recorded['matched_classes'],
                recorded_nonzero_classes=recorded['nonzero_classes'], matching=details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'output/sowodb-task2-v7-transfer-3epochs-2gpu/results')
    parser.add_argument('--parent', type=Path,
                        default=ROOT / 'output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth')
    opt = parser.parse_args()
    print(json.dumps(audit(opt.results, opt.parent), indent=2))
