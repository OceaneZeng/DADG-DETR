"""Fit paired proposal quality heads on released balanced memory with a frozen detector."""
from collections import Counter
from copy import deepcopy
from functools import partial
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from models.graph_local.instance_proposals import nearby_anchor_ids
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_shortlist_depth import OUTPUT as NATIVE, STAGES, compare, record_geometry
from tools.owod.probe_s_task2_tail_cell_ranks import OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task2_duration import MANIFEST, SCREEN
from tools.owod.run_s_task2_increment_giou import evaluate_heldout
from tools.owod.screening_data import screening_files, source_ids, write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = SOURCE.with_name('s-instance-task2-cell-pair-quality')
ARMS = ('iou', 'rank')
CRITERION = ('On both slices the added ranking loss must beat matched IoU-only fitting in Current AP50 and '
             'recover more baseline-missed tail GT into the top 300. Against the unmodified detector require '
             'recovery of at least half of the misses, higher final tail coverage, Current +1 pp or better, '
             'Previous no worse than -1 pp, higher Known, no lower mean proxy U-Recall and no higher mean WI or '
             'summed A-OSE. Inspect each arm even if the mechanism fails.')


def prepare():
    src = read_json(SOURCE/'plan.json')
    result = read_json(SOURCE/'comparison.json')
    if (file_sha256(SOURCE/'comparison.json') != 'c1d2e0395853f86dff3202c4dd89d3ca43cd203888e0a6ed74cc8c29935ea052'
            or result['decision'] != 'quality_ranking' or result['status'] != 'complete'
            or result['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or file_sha256(Path(src['checkpoint']['path'])) != src['checkpoint']['sha256']
            or file_sha256(MANIFEST) != src['manifest_sha256'] or file_sha256(SCREEN) !=
            '34f48520fefc31528c087111bdadc1371a1a6eebf5eedcdfec6a29e3240c70e0'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != src['gpu'] or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed rank evidence, checkpoint, data or evaluation environment')
    manifest, files = screening_files(MANIFEST, SCREEN, 1)
    fit = files['train']
    blocked = set(manifest['evaluation']['image_source_ids'])
    for name, entry in src['slices'].items():
        ann = Path(entry['annotation'])
        if (file_sha256(ann) != entry['annotation_sha256']
                or file_sha256(NATIVE/f'{name}-native.json') != src['native_sha256'][name]
                or file_sha256(Path(entry['native_targets'])) != entry['native_targets_sha256']
                or file_sha256(Path(entry['heldout']['path'])) != entry['heldout']['sha256']):
            raise ValueError(f'Changed frozen reference for {name}')
        blocked.update(source_ids(read_json(ann)))
    sources = source_ids(read_json(fit))
    if len(sources) != 1679 or sources & blocked:
        raise ValueError('Expected disjoint paired screening memory')
    plan = {k: src[k] for k in ('seed', 'protocol', 'paper_comparable', 'manifest_sha256', 'checkpoint',
                                 'slices', 'batch_size', 'device', 'gpu', 'torch_version', 'tail_class_ids',
                                 'native_sha256')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_comparison_sha256=file_sha256(SOURCE/'comparison.json'), fit_annotation=str(fit),
                fit_sha256=file_sha256(fit), fit_images=len(sources), fit_passes=1, lr=1e-4, weight_decay=1e-4,
                rank_coefficient=1., min_positive_iou=.5, max_competitor_iou=.3, criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest


def extract_pairs(geo, i, target, annotations, plan):
    """Pair a sampled accurate cell with a high-scoring poorly localized cell inside the same object."""
    if target['labels'].tolist() != [ann['category_id'] for ann in annotations]:
        raise ValueError('Changed released training annotation order')
    ids = torch.where(geo['valid'][i])[0]
    gt = target['boxes'].to(ids.device)
    if not len(gt):
        return []
    boxes = geo['boxes'][i, ids]
    truth = box_cxcywh_to_xyxy(gt)
    ious = box_iou(truth, box_cxcywh_to_xyxy(boxes))[0]
    max_iou = ious.amax(0)
    near = torch.searchsorted(ids, nearby_anchor_ids(gt, geo['anchors'][i], ids))
    logits = geo['logits'][i, ids]
    pairs = []
    for j, ann in enumerate(annotations):
        if ann.get('difficult'):
            continue
        pos = near[j, ious[j, near[j]].argmax()]
        if ious[j, pos] <= plan['min_positive_iou']:
            continue
        inside = ((boxes[:, :2] >= truth[j, :2]) & (boxes[:, :2] <= truth[j, 2:])).all(-1)
        candidates = torch.where(inside & (max_iou < plan['max_competitor_iou']))[0]
        if not len(candidates):
            continue
        neg = candidates[logits[candidates].argmax()]
        cells = ids[torch.stack((pos, neg))]
        features = torch.cat((geo['memory'][i, cells], geo['boxes'][i, cells]), -1)
        pairs.append((features, torch.stack((ious[j, pos], max_iou[neg])), ann['category_id']))
    return pairs


def fit_heads(model, image_root, plan, plan_hash):
    out = OUTPUT/'fit.json'
    if not out.exists():
        ds, loader = make_loader(image_root, Path(plan['fit_annotation']), plan['batch_size'], 2)
        native_head = model.detr.transformer.instance_proposals.quality
        heads = {arm: deepcopy(native_head).requires_grad_(True) for arm in ARMS}
        opts = {arm: torch.optim.AdamW(head.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
                for arm, head in heads.items()}
        step_count, classes, losses = 0, Counter(), Counter()
        for batch, (samples, targets) in enumerate(loader, 1):
            with torch.no_grad():
                geo = model(samples.to('cuda'))['geometry']
                pairs = []
                for i, target in enumerate(targets):
                    annotations = valid_annotations(ds, int(target['image_id']))
                    pairs.extend(extract_pairs(geo, i, target, annotations, plan))
            if pairs:
                features = torch.stack([x for x, _, _ in pairs])
                expected = torch.stack([y for _, y, _ in pairs])
                classes.update(label for _, _, label in pairs)
                for arm, head in heads.items():
                    logits = head(features).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, expected)
                    if arm == 'rank':
                        loss = loss + plan['rank_coefficient'] * F.softplus(logits[:, 1]-logits[:, 0]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite quality fitting loss')
                    opts[arm].zero_grad()
                    loss.backward()
                    opts[arm].step()
                    losses[arm] += loss.item()
                step_count += 1
            if batch == 1 or batch % 64 == 0 or batch == len(loader):
                print(f'Fit {batch}/{len(loader)} batches, {step_count} updates, {sum(classes.values())} pairs', flush=True)
        if not step_count or all(torch.equal(a, b) for a, b in zip(heads['rank'].parameters(), heads['iou'].parameters())):
            raise ValueError('No paired quality-head learning occurred')
        weights = {arm: {k: v.detach().cpu().tolist() for k, v in head.state_dict().items()} for arm, head in heads.items()}
        write_json(out, dict(plan_sha256=plan_hash, updates=step_count, pairs_by_class=classes,
                             mean_loss={arm: losses[arm]/step_count for arm in ARMS}, heads=weights))
    evidence = read_json(out)
    if evidence['plan_sha256'] != plan_hash or set(evidence['heads']) != set(ARMS):
        raise ValueError('Changed fitted head identity')
    return evidence


def main():
    print('Validating released balanced memory, rank evidence and native controls', flush=True)
    plan, manifest = prepare()
    plan_hash = file_sha256(OUTPUT/'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    model.requires_grad_(False).eval()
    proposal = model.detr.transformer.instance_proposals
    if proposal.shortlist_multiplier != 12 or proposal.rank_within_level or proposal.match_encoder_counts:
        raise ValueError('Expected native global top-300 proposals')
    fit = fit_heads(model, root, plan, plan_hash)
    fit_hash = file_sha256(OUTPUT/'fit.json')
    summaries = {}
    for name, entry in plan['slices'].items():
        ann = Path(entry['annotation'])
        ds, loader = make_loader(root, ann, plan['batch_size'], 2)
        order = [im['owod_source_id'] for im in read_json(ann)['images']]
        native = read_json(NATIVE/f'{name}-native.json')
        arms = {}
        for arm in ARMS:
            path = OUTPUT/f'{name}-{arm}.json'
            if not path.exists():
                proposal.quality.load_state_dict({k: torch.tensor(v) for k, v in fit['heads'][arm].items()})
                print(f'Evaluating {name}-{arm} on {len(ds)} images', flush=True)
                hits, choices = {stage: set() for stage in STAGES}, {}
                observer = partial(record_geometry, hits, choices, ds.coco, set(plan['tail_class_ids']), 12,
                                   manifest['evaluation']['iou_threshold'])
                views = evaluate_heldout(model, ds, loader, manifest, order, proxies=True, rank_observer=observer)
                write_json(path, dict(plan_sha256=plan_hash, fit_sha256=fit_hash, arm=arm, views=views,
                                      hits={k: sorted(v) for k, v in hits.items()}, choices=choices))
            endpoint = read_json(path)
            if endpoint['plan_sha256'] != plan_hash or endpoint['fit_sha256'] != fit_hash or endpoint['arm'] != arm:
                raise ValueError(f'Changed endpoint {path}')
            arms[arm] = compare(native, endpoint, entry)
        extra = (arms['rank']['delta']['current_ap50'] > arms['iou']['delta']['current_ap50']
                 and len(arms['rank']['recovered_in_shortlist']) > len(arms['iou']['recovered_in_shortlist']))
        summaries[name] = dict(arms=arms, beats_paired_iou=extra, follow_up=extra and arms['rank']['follow_up'])
    write_json(OUTPUT/'comparison.json', dict(status='complete', plan_sha256=plan_hash, fit_sha256=fit_hash,
                                               paper_comparable=False, slices=summaries,
                                               follow_up=all(s['follow_up'] for s in summaries.values())))
    print('Paired quality head comparison saved', flush=True)


if __name__ == '__main__':
    main()
