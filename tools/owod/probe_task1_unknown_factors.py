"""Decompose the native unknown score at fixed per-image deployment quotas."""
from collections import Counter
from pathlib import Path
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import probe_task1_unknown_budget as budget
from tools.owod import probe_task1_view_boxes as paired
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = ROOT / 'output/m-instance-task1-unknown-factors'
SLICES = {'reference1024': holdout.OUTPUT / 'heldout.json', 'fresh1024': budget.OUTPUT / 'fresh1024.json'}
ARMS = ('native', 'objectness', 'novelty')


def prepare():
    parent = read_json(budget.OUTPUT / 'plan.json')
    digest = file_sha256(budget.OUTPUT / 'plan.json')
    result = read_json(budget.OUTPUT / 'comparison.json')
    if (result['status'] != 'complete' or result['plan_sha256'] != digest
            or file_sha256(MANIFEST) != parent['manifest_sha256']
            or file_sha256(SLICES['reference1024']) != parent['holdout_sha256']
            or file_sha256(SLICES['fresh1024']) != parent['fresh_sha256']
            or torch.cuda.get_device_name() != parent['gpu'] or str(torch.__version__) != parent['torch_version']):
        raise ValueError('Changed input or environment for score-factor diagnosis')
    references = {}
    for seed, job in parent['jobs'].items():
        if file_sha256(Path(job['parent'])) != job['parent_sha256']:
            raise ValueError(f'Changed Task 1 checkpoint for seed {seed}')
        for label in SLICES:
            path = budget.OUTPUT / f'seed{seed}-{label}.json'
            ref = read_json(path)
            if ref['plan_sha256'] != digest or ref['counts'] != result['rows'][seed][label]:
                raise ValueError(f'Unexpected frozen reference {path}')
            references[path.name] = file_sha256(path)
    plan = {key: parent[key] for key in ('jobs', 'batch_size', 'gpu', 'torch_version', 'manifest_sha256',
                                          'holdout_sha256', 'fresh_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                parent_run=budget.OUTPUT.name, parent_plan_sha256=digest, reference_sha256=references,
                parent_result_sha256=file_sha256(budget.OUTPUT / 'comparison.json'),
                rule='Retain native known detections and per-image unknown quota. Compare native focal product, '
                'objectness alone and novelty alone without suppression. Stable query-index tie breaking for '
                'single factors. Labels are read only after each list is fixed.',
                criterion='Descriptive factor ablation on frozen models. Flag a factor for a released-only '
                'follow-up if both seeds gain >=5 pp in fresh proxy recall with no loss on reference. Report '
                'known-only overlap per unknown output and paired gains/losses. No tuning or deployment.')
    pin(OUTPUT / 'plan.json', plan)
    return plan


def best_target(boxes, truth):
    if not len(truth):
        return torch.full((len(boxes),), -1, dtype=torch.long, device=boxes.device)
    iou, best = paired.overlap(boxes, truth).max(1)
    return best.masked_fill(iou < .5, -1)


@torch.inference_mode()
def evaluate(model, post, loader, known, proxy, reference):
    rows, totals = {}, Counter()
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        detections = post(outputs, sizes, include_top_indices=True)
        obj = outputs['pred_objectness'].sigmoid()
        nov = outputs['pred_novelty'].sigmoid()
        for i, target in enumerate(targets):
            key = str(target['image_id'].item())
            ref = reference[key]
            result = detections[i]
            unknown = result['labels'] == 91
            ids = result['top_indices']
            native = ids[unknown] // (len(known) + 1)
            quota = len(native)
            orders = dict(native=native, objectness=torch.argsort(obj[i], descending=True, stable=True)[:quota],
                          novelty=torch.argsort(nov[i], descending=True, stable=True)[:quota])
            if (key in rows or ids[~unknown].tolist() != ref['known_positions'] or native.tolist() != ref['native']
                    or result['scores'][~unknown].tolist() != ref['known_scores']
                    or (obj[i] * nov[i]).tolist() != ref['unknown_scores']):
                raise ValueError(f'Changed native predictions for image {key}')
            labels, truth = target['labels'].cuda(), target['boxes'].cuda()
            boxes = outputs['pred_boxes'][i]
            proxy_gt = truth[torch.isin(labels, proxy)]
            proxy_id = best_target(boxes, proxy_gt)
            known_id = best_target(boxes, truth[torch.isin(labels, known)])
            if proxy_id.tolist() != ref['best_gt'] or len(proxy_gt) != ref['gt']:
                raise ValueError(f'Changed paired proxy targets for image {key}')
            counts = dict(images=1, quota=quota, known=len(ids)-quota, gt=len(proxy_gt))
            for arm, picks in orders.items():
                p, k = proxy_id[picks], known_id[picks]
                counts[arm + '_hits'] = len(p[p >= 0].unique())
                counts[arm + '_known_only'] = int(((k >= 0) & (p < 0)).sum())
                counts[arm + '_both'] = int(((k >= 0) & (p >= 0)).sum())
                counts[arm + '_proxy_only'] = int(((k < 0) & (p >= 0)).sum())
                counts[arm + '_neither'] = int(((k < 0) & (p < 0)).sum())
            if counts['native_hits'] != ref['native_hits']:
                raise ValueError(f'Changed native proxy hits for image {key}')
            rows[key] = dict(counts=counts, picks={arm: v.tolist() for arm, v in orders.items()},
                             objectness=obj[i].tolist(), novelty=nov[i].tolist(), known_gt=known_id.tolist())
            totals.update(counts)
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluated {step}/{len(loader)} batches', flush=True)
    if set(rows) != set(reference) or len(rows) != len(loader.dataset):
        raise ValueError('Incomplete paired evaluation')
    return dict(images=rows, counts=totals)


def main():
    plan = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    root = resolve_image_root(MANIFEST, manifest)
    known = torch.tensor(manifest['stages'][0]['active_classes'], device='cuda')
    proxy = torch.tensor(manifest['stages'][1]['classes'], device='cuda')
    totals = {}
    for seed, job in plan['jobs'].items():
        checkpoint = Path(job['parent'])
        args = load_checkpoint_metadata(checkpoint)['args']
        if (args.discovery_scoring != 'focal' or args.owod_max_detections != 50 or args.owod_stage != 0
                or list(args.owod_known_class_ids) != known.tolist()):
            raise ValueError('Expected native Task 1 focal Top-50 detector')
        post = DiscoveryPostProcess(known_ids=args.owod_known_class_ids, max_detections=args.owod_max_detections,
                                    unknown_threshold=args.unknown_threshold, scoring=args.discovery_scoring)
        model, _ = load_detector(args, checkpoint, torch.device('cuda'))
        model.eval()
        totals[seed] = {}
        for label, annotation in SLICES.items():
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                ref = read_json(budget.OUTPUT / path.name)['images']
                _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                print(f'Seed {seed} {label}: unknown score factors', flush=True)
                result = evaluate(model, post, loader, known, proxy, ref)
                write_json(path, dict(plan_sha256=digest, **result))
            saved = read_json(path)
            if saved['plan_sha256'] != digest or len(saved['images']) != 1024:
                raise ValueError(f'Incomplete endpoint {path}')
            totals[seed][label] = saved['counts']
        del model
    flags = {}
    for arm in ARMS[1:]:
        hits = arm + '_hits'
        flags[arm] = all(100 * (row['fresh1024'][hits] - row['fresh1024']['native_hits'])
                         >= 5 * row['fresh1024']['gt'] and row['reference1024'][hits] >= row['reference1024']['native_hits']
                         for row in totals.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals,
                                                released_only_followup=flags))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
