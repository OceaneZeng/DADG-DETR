"""Audit old-class refill and open-set confusion using already released Task 2 labels."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task2_old_dedup import OUTPUT as M_SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST as M_MANIFEST, ROOT
from tools.owod.run_s_task2_duration import MANIFEST as S_MANIFEST
from tools.owod.run_s_task2_old_dedup import DEDUP_IOU, LIMIT, OLD_POOL, OldClassRefill
from tools.owod.run_s_task2_old_dedup_holdout import OUTPUT as S_SOURCE
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/instance-task2-old-dedup-open-set'
SOURCES = {
    'm-owodb': (M_MANIFEST, M_SOURCE, ('seed42-6epochs', 'seed43-6epochs'), 'integral'),
    's-owodb': (S_MANIFEST, S_SOURCE, ('6epochs',), 'voc2007_11point'),
}


def protected_predictions(results, protected_labels):
    labels = torch.stack([row['labels'] for row in results])
    mask = torch.isin(labels, labels.new_tensor(protected_labels))
    indices = torch.stack([row['top_indices'] for row in results]).masked_fill(~mask, -1)
    indices, order = indices.sort(dim=1)
    scores = torch.stack([row['scores'] for row in results]).masked_fill(~mask, 0.)
    boxes = torch.stack([row['boxes'] for row in results]).masked_fill(~mask[..., None], 0.)
    return indices, scores.gather(1, order), boxes.gather(1, order[..., None].expand_as(boxes))


class AuditedRefill(OldClassRefill):
    def __init__(self, known, previous, *, old_only=False):
        super().__init__(known, previous, old_only=old_only)
        self.native = DiscoveryPostProcess(known_ids=known, max_detections=LIMIT, scoring='focal')
        self.checked_images = 0

    @torch.no_grad()
    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        native = self.native(outputs, target_sizes, include_top_indices=True)
        refill = super().forward(outputs, target_sizes, include_top_indices=True)
        before = protected_predictions(native, self.locked_labels)
        after = protected_predictions(refill, self.locked_labels)
        if not all(torch.equal(a, b) for a, b in zip(before, after)):
            raise ValueError('Refill changed a protected label/query, box or score')
        self.checked_images += len(native)
        if not include_top_indices:
            for row in refill:
                row.pop('top_indices')
        return refill


def inputs(protocol, manifest_path, source, arms, ap_mode):
    manifest = read_json(manifest_path)
    source_plan_path = source / 'plan.json'
    source_result_path = source / 'comparison.json'
    source_plan, source_result = read_json(source_plan_path), read_json(source_result_path)
    manifest_hash, source_plan_hash = file_sha256(manifest_path), file_sha256(source_plan_path)
    annotation = Path(source_plan['annotation'])
    annotation_hash = file_sha256(annotation)
    data = read_json(annotation)
    order = [row['owod_source_id'] for row in data['images']]
    order_hash = sha256('\n'.join(order).encode()).hexdigest()
    previous, current = (manifest['stages'][index]['classes'] for index in (0, 1))
    evaluation = manifest['evaluation']
    if (source_result['status'] != 'complete' or not source_result['follow_up'] or
            source_result['plan_sha256'] != source_plan_hash or source_plan['protocol'] != protocol or
            manifest_hash != source_plan['manifest_sha256'] or evaluation['ap_mode'] != ap_mode or
            annotation_hash != source_plan['annotation_sha256'] or order_hash != source_plan['eval_order_sha256'] or
            len(order) != 1024 or len(set(order)) != len(order) or set(order).intersection(evaluation['image_source_ids']) or
            {a['category_id'] for a in data['annotations']} != set(previous + current) or
            (source_plan['max_detections'], source_plan['old_pool'], source_plan['nms_iou']) != (LIMIT, OLD_POOL, DEDUP_IOU)):
        raise ValueError(f'Changed {protocol} holdout or rule sources')
    checkpoints = {arm: source_plan['checkpoints'][arm] for arm in arms}
    for arm, entry in checkpoints.items():
        if file_sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Changed {protocol} checkpoint: {arm}')
    subsets = {'released': current, 'proxy_odd': current[1::2], 'proxy_even': current[::2]}
    return dict(annotation=str(annotation), annotation_sha256=annotation_hash, eval_order_sha256=order_hash,
                manifest_sha256=manifest_hash, source_plan_sha256=source_plan_hash,
                source_result_sha256=file_sha256(source_result_path), ap_mode=ap_mode, checkpoints=checkpoints,
                visible_current_ids=subsets, images=len(order))


def evaluate_protocol(protocol, manifest_path, source, plan, plan_hash, *, output=OUTPUT, old_only=False):
    manifest = read_json(manifest_path)
    source_arms = read_json(source / 'comparison.json')['arms']
    annotation = Path(plan['annotation'])
    order = [row['owod_source_id'] for row in read_json(annotation)['images']]
    dataset, loader = make_loader(resolve_image_root(manifest_path, manifest), annotation, 2, 2)
    spec = dict(manifest['evaluation'], image_source_ids=order)
    previous, current = (manifest['stages'][index]['classes'] for index in (0, 1))
    variants, processors, auditors = {}, {}, {}
    for fold, visible in plan['visible_current_ids'].items():
        known = previous + visible
        auditor = AuditedRefill(known, previous, old_only=old_only)
        for view, processor in (('native', auditor.native), ('dedup', auditor)):
            key = f'{fold}_{view}'
            variants[key] = (known, 'focal', LIMIT)
            processors[key] = processor
        auditors[fold] = auditor

    rows = {}
    for arm, entry in plan['checkpoints'].items():
        path = output / f'{protocol}-{arm}.json'
        check_key = 'current_and_unknown_identity_checks' if old_only else 'unknown_identity_checks'
        if not path.is_file():
            print(f'Evaluating {protocol} {arm}: released labels and two proxy folds', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            for auditor in auditors.values():
                auditor.checked_images = 0
            views = evaluate_views(model, loader, dataset.coco, variants, spec, previous, current, 'cuda',
                                   postprocessors=processors, print_freq=64, include_wi_sources=old_only)
            write_json(path, dict(plan_sha256=plan_hash, checkpoint_sha256=entry['sha256'], views=views,
                                  **{check_key: {k: v.checked_images for k, v in auditors.items()}}))
            del model
        report = read_json(path)
        if (report['plan_sha256'] != plan_hash or report['checkpoint_sha256'] != entry['sha256'] or
                report[check_key] != dict.fromkeys(auditors, len(order))):
            raise ValueError(f'Incomplete protected prediction check: {path}')
        views = report['views']
        if old_only:
            parent = read_json(OUTPUT / path.name)
            if (parent['checkpoint_sha256'] != entry['sha256'] or
                    parent['plan_sha256'] != file_sha256(OUTPUT / 'plan.json')):
                raise ValueError(f'Changed source endpoint: {path.name}')
            for fold in auditors:
                name = f'{fold}_native'
                if any(views[name][k] != parent['views'][name][k] for k in ('metrics', 'composition')):
                    raise ValueError(f'Native replay differs: {path.name}, {name}')
        for view in (('native',) if old_only else ('native', 'dedup')):
            metrics = views[f'released_{view}']['metrics']
            if (metrics['unknown_gt'] or any(metrics[k] != v for k, v in source_arms[arm][view]['metrics'].items()) or
                    views[f'released_{view}']['composition'] != source_arms[arm][view]['composition']):
                raise ValueError(f'Prior released-category result differs from {path}: {view}')
        folds = {}
        for fold in ('proxy_odd', 'proxy_even'):
            modes = {view: views[f'{fold}_{view}']['metrics'] for view in ('native', 'dedup')}
            native, dedup = modes['native'], modes['dedup']
            if (native['unknown_gt'] <= 0 or any(native[key] != dedup[key] for key in ('unknown_gt', 'u_recall',
                                                                                    'unknown_predictions'))):
                raise ValueError(f'Invalid proxy metrics: {path}, {fold}')
            folds[fold] = dict(**modes, dedup_minus_native={key: dedup[key] - value for key, value in native.items()})
        delta = {key: mean(f['dedup_minus_native'][key] for f in folds.values())
                 for key in ('previous_ap50', 'current_ap50', 'known_ap50', 'wi')}
        delta['a_ose'] = sum(f['dedup_minus_native']['a_ose'] for f in folds.values())
        follow_up = (delta['previous_ap50'] >= .0075 and delta['current_ap50'] >= -.0025 and delta['known_ap50'] > 0
                     and delta['a_ose'] <= 0 and delta['wi'] <= 0)
        rows[arm] = dict(folds=folds, combined_delta=delta, follow_up=follow_up)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-only', action='store_true', help='Reserve native Current and unknown predictions')
    old_only = parser.parse_args().old_only
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name():
        raise ValueError('Select one RTX 3090 for the frozen open-set audit')
    print('Verifying completed M/S holdouts and all three frozen checkpoints', flush=True)
    protocols = {protocol: inputs(protocol, *entry) for protocol, entry in SOURCES.items()}
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                protocols=protocols, paper_comparable=False, max_detections=LIMIT, old_pool=OLD_POOL, nms_iou=DEDUP_IOU,
                device=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(),
                batch_size=2, torch_version=str(torch.__version__),
                proxy_policy='Alternate manifest-ordered, already released current classes between two masked folds. '
                             'Remove those logits from scoring and use their existing boxes as proxy unknown ground truth.',
                criterion='Exact per-image unknown query/box/score invariance in every view. For every checkpoint, '
                          'across the two proxy folds: mean Previous delta >= +0.75 pp, Current >= -0.25 pp, '
                          'Known > 0 pp, summed A-OSE and mean WI do not increase.')
    output = OUTPUT
    protocol_order = list(SOURCES)
    if old_only:
        attribution = ROOT / 'output/s-instance-task2-dedup-wi-sources/comparison.json'
        parent = read_json(OUTPUT / 'plan.json')
        if (file_sha256(attribution) != 'fe635a708a38c890dc5e6e6c70621c030d7d617214e0f2d729bbe8ce0d6d6100' or
                file_sha256(OUTPUT / 'comparison.json') != '8e98021a8b491bb4b0c8113953c8dd1c87b08d4cde237b2a6ac44ece26d36398' or
                protocols != parent['protocols'] or file_sha256(OUTPUT / 'plan.json') != read_json(OUTPUT / 'comparison.json')['plan_sha256']):
            raise ValueError('Expected archived WI attribution and parent inputs')
        output = OUTPUT.with_name('instance-task2-old-only-refill')
        protocol_order = ['s-owodb', 'm-owodb']
        plan.update(wi_attribution_sha256=file_sha256(attribution), parent_comparison_sha256=file_sha256(OUTPUT / 'comparison.json'),
                    policy='Fill vacated slots only from Previous. Preserve each native Current and unknown '
                           'query/class, box and score. Use the existing two-fold gate. Stop after S if it fails, '
                           'otherwise evaluate both M seeds before considering independent validation.')
    plan_path = output / 'plan.json'
    pin(plan_path, plan)
    plan_hash = file_sha256(plan_path)
    rows = {}
    for protocol in protocol_order:
        manifest_path, source, _, _ = SOURCES[protocol]
        rows[protocol] = evaluate_protocol(protocol, manifest_path, source, protocols[protocol], plan_hash,
                                           output=output, old_only=old_only)
        if old_only and not all(row['follow_up'] for row in rows[protocol].values()):
            break
    result = output / 'comparison.json'
    write_json(result, dict(status='complete', plan_sha256=plan_hash, paper_comparable=False, protocols=rows,
                            follow_up=all(row['follow_up'] for arms in rows.values() for row in arms.values())))
    print('Comparison saved:', result, flush=True)


if __name__ == '__main__':
    main()
