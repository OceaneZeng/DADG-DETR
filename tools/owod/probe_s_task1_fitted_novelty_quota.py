"""Measure a fitted novelty head at the native detector's per-image output quota."""

import subprocess
from collections import Counter
from pathlib import Path

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_novelty_head_transfer import (
    FrozenNoveltyViews, TransferredNoveltyPostprocess, require_one_gpu,
)
from tools.owod.probe_s_task1_rank_independent import PROTOCOL, evaluate as evaluate_released
from tools.owod.probe_s_task1_same_query_competition import METRICS
from tools.owod.probe_task1_m_s_refinement_aose import Attribution
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_frozen_novelty_fit import OUTPUT as FIT
from tools.owod.run_s_task1_known_score_order import verified_endpoint
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = FIT.with_name('task1-s-fitted-novelty-quota')
FIT_COMPARISON_SHA = 'c4c4d06210fdf437f14e5708d564e241be015bcbd871980dbba8839447aba47e'


def prepare():
    result = verified(FIT / 'comparison.json', FIT_COMPARISON_SHA)
    source = verified(FIT / 'plan.json', result['plan_sha256'])
    endpoint = verified(FIT / 'fit.json', result['fit_sha256'])
    rank = result['outcomes']['bce_rank']
    donor = endpoint['arms']['bce_rank']
    control = source['control']
    if (result['status'] != 'complete' or result['passed'] or not rank['released']['passed']
            or rank['checkpoint_sha256'] != control['sha256'] or rank['donor_sha256'] != donor['sha256']
            or source['limit'] != 50
            or any(fold['delta']['a_ose'] <= 0. for fold in rank['folds'].values())
            or any(file_sha256(Path(arm['checkpoint'])) != arm['sha256'] for arm in (control, donor))):
        raise ValueError('Expected the exact fitted head and verified excess A-OSE on both open-set folds')
    keys = ('holdout', 'holdout_sha256', 'manifest_sha256', 'batch_size', 'limit', 'min_known_ap50',
            'folds', 'fold_native', 'max_wi_increase', 'max_aose_fraction')
    plan = dict(**{key: source[key] for key in keys}, control=control, fitted=donor, expected_rank=rank,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                fit_comparison_sha256=FIT_COMPARISON_SHA, paper_comparable=False,
                intervention='Keep fitted competition scores and exactly match the final native per-image quota.',
                fallback='Use native predictions if fitted finite candidates cannot fill either quota.')
    pin(OUTPUT / 'plan.json', plan)
    return plan


def keep_best(scores, counts, limit):
    positions = scores.topk(limit, dim=1).indices
    active = torch.arange(limit, device=scores.device)[None] < counts[:, None]
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, positions, active)
    return scores.masked_fill(~keep, -torch.inf)


class MatchedFinalQuota(TransferredNoveltyPostprocess):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.native = CompeteWithEmittedUnknown(**settings)
        self.checks = Counter()

    def known_count(self, scores, columns):
        selected = scores.flatten(1).topk(self.max_detections, dim=1).indices
        return ((selected % columns) != columns - 1).sum(dim=1)

    def candidate_scores(self, outputs):
        native, _, native_unknown = self.native.candidate_scores(outputs)
        fitted, categories, fitted_unknown = super().candidate_scores(outputs)
        columns, limit = len(categories), self.max_detections
        known_count = self.known_count(native, columns)
        unknown_count = limit - known_count
        known = fitted[..., :-1].flatten(1)
        unknown = fitted[..., -1]
        fallback = ((known.isfinite().sum(dim=1) < known_count)
                    | (unknown.isfinite().sum(dim=1) < unknown_count))
        known = keep_best(known, known_count, limit).reshape_as(fitted[..., :-1])
        unknown = keep_best(unknown, unknown_count, limit)
        quota = torch.cat((known, unknown[..., None]), dim=-1)
        quota = torch.where(fallback[:, None, None], native, quota)
        if not torch.equal(self.known_count(quota, columns), known_count):
            raise ValueError('Final Top-50 known counts differ from native per-image quotas')
        self.last_checks = dict(images=len(native), fallbacks=int(fallback.sum()),
                                known_outputs=int(known_count.sum()), unknown_outputs=int(unknown_count.sum()))
        return quota, categories, torch.where(fallback[:, None], native_unknown, fitted_unknown)

    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        result = super().forward(outputs, target_sizes, include_top_indices=include_top_indices)
        self.checks.update(self.last_checks)
        return result

    def take_checks(self):
        result = dict(self.checks)
        self.checks.clear()
        return result


def evaluate_fold(model, manifest, path, plan, processors):
    known = manifest['stages'][0]['classes']
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    dataset, loader = make_loader(root, path, plan['batch_size'], 2)
    spec = dict(manifest['evaluation'],
                image_source_ids=[row['owod_source_id'] for row in dataset.coco.dataset['images']])
    traces = {view: Attribution(known, spec['iou_threshold']) for view in processors}

    def observe(outputs, targets, batch_images, score_tables):
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
        for view, (_, categories, positions) in zip(processors, score_tables, strict=True):
            for i, image in batch_images.items():
                chosen = positions[i]
                height, width = targets[i]['orig_size'].tolist()
                corners = boxes[i, chosen // len(categories)] * boxes.new_tensor([width, height, width, height])
                detections = dict(boxes=corners.cpu(), labels=categories[chosen % len(categories)].cpu())
                traces[view].add(image, detections, dataset.coco.imgToAnns[image])

    views = {name: (known, 'focal', plan['limit']) for name in processors}
    result = evaluate_views(model, loader, dataset.coco, views, spec, [], known, torch.device('cuda'),
                            include_current_stats=True, postprocessors=processors, rank_observer=observe, print_freq=128)
    result['attribution'] = {view: trace.result() for view, trace in traces.items()}
    for view, trace in result['attribution'].items():
        if trace['a_ose'] != result[view]['metrics']['a_ose']:
            raise ValueError(f'Detection attribution differs from author A-OSE in {view}')
    return result


def verify_quota(report, native_key, limit):
    checks = report['quota_checks']
    native = report[native_key]['composition']
    quota = report['quota']['composition']
    if (native['current'] != quota['current'] or native['unknown'] != quota['unknown']
            or checks['known_outputs'] != native['current'] or checks['unknown_outputs'] != native['unknown']
            or checks['images'] * limit != native['detections']
            or native['detections'] != quota['detections']):
        raise ValueError('Recorded final per-image quotas do not reconstruct total outputs')


def main():
    require_one_gpu()
    plan = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = verified(MANIFESTS[PROTOCOL], plan['manifest_sha256'])
    base, head = plan['control'], plan['fitted']
    binding = dict(plan_sha256=digest, checkpoint_sha256=base['sha256'], donor_sha256=head['sha256'])
    detector, _ = load_detector(None, Path(base['checkpoint']), torch.device('cuda'))
    model = FrozenNoveltyViews(detector, Path(head['checkpoint']))
    settings = dict(known_ids=manifest['stages'][0]['classes'], scoring='focal', max_detections=plan['limit'])
    quota = MatchedFinalQuota(**settings)
    alternatives = dict(fitted=TransferredNoveltyPostprocess(**settings), quota=quota)
    processors = dict(native=CompeteWithEmittedUnknown(**settings), **alternatives)
    released = Path(plan['holdout'])
    if file_sha256(released) != plan['holdout_sha256']:
        raise ValueError('Changed released-only evaluation inputs')
    dest = OUTPUT / 'released.json'
    if not dest.is_file():
        print('Evaluating frozen fitted novelty quotas on the released-only holdout', flush=True)
        report = evaluate_released(model, manifest, released, plan, extra_postprocessors=alternatives)
        report['quota_checks'] = quota.take_checks()
        write_json(dest, dict(**binding, report=report))
    report = verified_endpoint(dest, binding)
    verify_quota(report, 'emitted', plan['limit'])
    values = {view: report[view]['metrics']['known_ap50'] for view in ('emitted', 'fitted', 'quota')}
    small = report['released_small_learned_query_coverage']
    if (values['emitted'] != base['expected_ap'] or small != base['expected_small']
            or values['fitted'] != plan['expected_rank']['released']['transferred']
            or report['raw_current_box_recall_iou50'] != base['expected_raw']):
        raise ValueError('Frozen native and fitted released endpoints must reproduce')
    released_result = dict(known_ap50=values, small=small, quota_checks=report['quota_checks'],
                           passed=values['quota'] > plan['min_known_ap50'])

    folds = {}
    for name, entry in plan['folds'].items():
        path = Path(entry['path'])
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f'Changed S {name} fold')
        dest = OUTPUT / f'{name}.json'
        if not dest.is_file():
            print(f'Evaluating frozen fitted novelty quotas on S {name}', flush=True)
            report = evaluate_fold(model, manifest, path, plan, processors)
            report['quota_checks'] = quota.take_checks()
            write_json(dest, dict(**binding, report=report))
        report = verified_endpoint(dest, binding)
        verify_quota(report, 'native', plan['limit'])
        metrics = {view: {key: report[view]['metrics'][key] for key in METRICS} for view in processors}
        native = metrics['native']
        expected = plan['expected_rank']['folds'][name]['metrics']['transferred']
        if native != plan['fold_native'][name] or metrics['fitted'] != expected:
            raise ValueError(f'Frozen native and fitted open-set endpoints must reproduce: {name}')
        delta = {key: metrics['quota'][key] - native[key] for key in METRICS}
        gates = dict(known_ap50=delta['known_ap50'] >= 0., u_recall=delta['u_recall'] >= 0.,
                     wi=delta['wi'] <= plan['max_wi_increase'],
                     a_ose=delta['a_ose'] <= max(1., plan['max_aose_fraction'] * native['a_ose']))
        folds[name] = dict(metrics=metrics, delta=delta, gates=gates, passed=all(gates.values()),
                           quota_checks=report['quota_checks'], attribution=report['attribution'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', **binding, released=released_result,
                                               folds=folds, passed=released_result['passed'] and all(
                                                   row['passed'] for row in folds.values())))


if __name__ == '__main__':
    main()
