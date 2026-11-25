"""Replay donor novelty order at the native per-image unknown score distribution."""

import subprocess
from collections import Counter
from pathlib import Path

import torch

from tools.owod.probe_s_task1_novelty_head_transfer import OUTPUT as TRANSFER, require_one_gpu, run
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256
from tools.owod.run_cross_attention_study import pin


OUTPUT = TRANSFER.with_name('task1-s-novelty-rank-remap')
TRANSFER_SHA = '801375ccea5412bf3d980a6e13d2106b40924655f43c107522bfb18da86f71f2'


def prepare():
    transfer = verified(TRANSFER / 'comparison.json', TRANSFER_SHA)
    source = verified(TRANSFER / 'plan.json', transfer['plan_sha256'])
    if (transfer['status'] != 'complete' or transfer['passed'] or source['limit'] != 50
            or set(transfer['folds']) != set(source['folds'])
            or any(fold['delta']['unknown_predictions'] >= 0 for fold in transfer['folds'].values())):
        raise ValueError('Expected the completed direct transfer with fewer unknown emissions')
    for name, field in (('control', 'checkpoint_sha256'), ('candidate', 'donor_sha256')):
        arm = source['arms'][name]
        if arm['sha256'] != transfer[field] or file_sha256(Path(arm['checkpoint'])) != arm['sha256']:
            raise ValueError(f'Changed frozen endpoint: {name}')
    plan = dict(source, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                transfer_sha256=TRANSFER_SHA,
                intervention='Reassign each image’s native unknown scores in donor-score order before competition.',
                boundary_ties='Keep native assignment if a cross-category Top-50 tie would change the unknown count.')
    pin(OUTPUT / 'plan.json', plan)
    return plan


class RankMatchedNoveltyPostprocess(CompeteWithEmittedUnknown):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.checks = Counter(images=0, boundary_tie_fallbacks=0, initial_unknown_outputs=0)

    def initial_unknown_count(self, scores):
        width = scores.shape[-1]
        selected = scores.flatten(1).topk(self.max_detections, dim=1).indices
        return ((selected % width) == len(self.known_ids)).sum(dim=1)

    def scores_before_competition(self, outputs):
        scores, categories, native = super().scores_before_competition(outputs)
        original_count = self.initial_unknown_count(scores)
        donor = outputs['pred_objectness'].sigmoid() * outputs['transferred_novelty'].sigmoid()
        donor_order = donor.argsort(dim=1, descending=True, stable=True)
        score_values = native.sort(dim=1, descending=True).values
        remapped = torch.empty_like(native).scatter_(1, donor_order, score_values)
        scores[..., -1] = remapped
        fallback = self.initial_unknown_count(scores) != original_count
        unknown = torch.where(fallback[:, None], native, remapped)
        scores[..., -1] = unknown
        if (not torch.equal(unknown.sort(dim=1, descending=True).values, score_values)
                or not torch.equal(self.initial_unknown_count(scores), original_count)):
            raise ValueError('Changed a per-image unknown score multiset or pre-competition Top-50 count')
        self.last_checks = dict(images=len(native), boundary_tie_fallbacks=int(fallback.sum()),
                                initial_unknown_outputs=int(original_count.sum()))
        return scores, categories, unknown

    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        result = super().forward(outputs, target_sizes, include_top_indices=include_top_indices)
        self.checks.update(self.last_checks)
        return result

    def take_checks(self):
        result = dict(self.checks)
        self.checks.clear()
        return result


def main():
    require_one_gpu()
    run(prepare(), OUTPUT, RankMatchedNoveltyPostprocess, variant='rank_remapped',
        score_checks=RankMatchedNoveltyPostprocess.take_checks)


if __name__ == '__main__':
    main()
