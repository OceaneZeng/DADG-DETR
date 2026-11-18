"""Explain an existing S Task 2 transfer decision on CPU, without rerunning calibration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.protocol import file_sha256
from tools.owod.task2_transfer import match_diagnostics


def compare_matching_spaces(graph, centers, counts, means, class_ids, min_support, threshold):
    """Score identical cached centroids and eligible anonymous slots under both metrics."""
    from collections import Counter
    import torch
    import torch.nn.functional as F

    slots = torch.where(counts >= min_support)[0]
    ids = [cid for cid in class_ids if cid in means]
    result = dict(class_ids=ids, supported_slots=slots.tolist(), spaces={})
    if not ids or not len(slots):
        return result
    queries = torch.stack([torch.as_tensor(means[cid], dtype=centers.dtype) for cid in ids])
    with torch.no_grad():
        for space in ('raw', 'projected'):
            q, c = queries, centers[slots]
            if space == 'projected':
                q, c = graph.project(q), graph.project(c)
            q, c = F.normalize(q, dim=-1), F.normalize(c, dim=-1)
            scores = q @ c.T
            if not torch.isfinite(scores).all():
                raise ValueError(f'Nonfinite cosine similarity in {space} space')
            # Stable ordering resolves exact ties by the original slot index.
            order = scores.argsort(descending=True, stable=True)
            ranked = scores.gather(1, order)
            best = ranked[:, 0]
            winners = slots[order[:, 0]].tolist()
            def pairwise_median(vectors):
                a, b = torch.triu_indices(len(vectors), len(vectors), offset=1)
                return float(torch.quantile((vectors @ vectors.T)[a, b], .5)) if len(a) else None
            result['spaces'][space] = dict(
                best_slot_counts=dict(Counter(winners)), classes_at_threshold=int((best >= threshold).sum()),
                best_cosine_range=[float(best.min()), float(best.max())],
                best_cosine_median=float(torch.quantile(best, .5)),
                median_top1_top2_gap=float(torch.quantile(best - ranked[:, 1], .5)) if len(slots) > 1 else None,
                tied_top1_classes=int(((scores == best[:, None]).sum(1) > 1).sum()),
                median_slot_pair_cosine=pairwise_median(c), median_class_pair_cosine=pairwise_median(q),
                cosine_by_class_and_slot=scores.tolist())
    return result


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
    spaces = compare_matching_spaces(graph, state['memory.centers'][-n:], state['memory.counts'][-n:],
                                     cache['means'], ids, saved.discovery_min_support, saved.discovery_transfer_threshold)
    return dict(parent_sha256=plan['checkpoint_sha256'], recorded_matched_classes=recorded['matched_classes'],
                recorded_nonzero_classes=recorded['nonzero_classes'], matching=details, space_comparison=spaces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'output/sowodb-task2-v7-transfer-3epochs-2gpu/results')
    parser.add_argument('--parent', type=Path,
                        default=ROOT / 'output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth')
    parser.add_argument('--summary', action='store_true', help='Omit per-class rows and the two score matrices')
    opt = parser.parse_args()
    report = audit(opt.results, opt.parent)
    if opt.summary:
        report['matching'].pop('classes')
        for space in report['space_comparison']['spaces'].values():
            space.pop('cosine_by_class_and_slot')
    print(json.dumps(report, indent=2))
