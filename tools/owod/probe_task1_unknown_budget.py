"""Audit deployed unknown predictions under the checkpoint's fixed joint Top-50 budget."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torchvision.ops import nms

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod import prepare_benchmarks
from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_encoder_counts as counts
from tools.owod import probe_task1_encoder_neighborhood as local
from tools.owod import probe_task1_view_boxes as paired
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import screening_files, source_ids, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = ROOT / 'output/m-instance-task1-unknown-budget'


def prepare():
    plan, manifest, _, reference = paired.paired_plan(
        counts.OUTPUT, 'Use checkpoint focal Top-50. Preserve every known entry and the number of unknown '
        'entries per image. Rank all final queries by the original unknown score. Suppress unknown-to-unknown '
        'IoU > .7 and refill missing slots by original order if needed. No labels enter selection.',
        'Both seeds: improve score-ordered Task 2 proxy hits by >=5 pp on fresh1024, no loss on '
        'reference1024, identical known detections and unknown counts. Report same-quota matching ceiling.',
        arms=('global',))
    previous = read_json(local.OUTPUT / 'comparison.json')
    last = read_json(local.OUTPUT / 'plan.json')
    if (previous['status'] != 'complete' or previous['plan_sha256'] != file_sha256(local.OUTPUT / 'plan.json')
            or last['jobs'] != plan['jobs'] or file_sha256(local.PARENT / 'val-data.json') != last['val_sha256']):
        raise ValueError('Changed previous diagnostic or parent inputs')
    blocked = set(manifest['evaluation']['image_source_ids']) | source_ids(read_json(reference))
    blocked.update(source_ids(read_json(local.PARENT / 'val-data.json')))
    screens = set()
    for job in plan['jobs'].values():
        screen = Path(job['screen'])
        if file_sha256(screen) != job['screen_sha256']:
            raise ValueError('Changed parent training screen')
        if screen not in screens:
            for stage in (0, 1):
                _, files = screening_files(MANIFEST, screen, stage)
                for path in files.values():
                    blocked.update(source_ids(read_json(path)))
            screens.add(screen)
    root = resolve_image_root(MANIFEST, manifest)
    selected = []
    for task in ('t1_train', 't2_train'):
        # Membership and image identity alone determine the new slice.
        names = sorted(set(manifest['splits'][task]) - blocked,
                       key=lambda name: hashlib.sha256(f'45:{name}'.encode()).digest())[:512]
        if len(names) != 512:
            raise ValueError(f'Insufficient fresh images in {task}')
        selected.extend(names)
        blocked.update(names)
    xml = {name: prepare_benchmarks.read_voc(root.parent, name) for name in selected}
    fresh = prepare_benchmarks.subset(selected, manifest['stages'][1]['active_classes'], xml)
    encoded = (json.dumps(fresh, indent=2, sort_keys=True) + '\n').encode()
    plan.update(previous_study=local.OUTPUT.name, previous_result_sha256=file_sha256(local.OUTPUT / 'comparison.json'),
                selection_seed=45, fresh_sha256=hashlib.sha256(encoded).hexdigest(), fresh_per_source=512,
                annotation_policy='Author training sources, first 40 released categories only. Task 2 is a proxy '
                'for unknown. Score-ordered hits use normalized IoU >= .5 and one credit per best GT.')
    pin(OUTPUT / 'plan.json', plan)
    path = OUTPUT / 'fresh1024.json'
    if not path.exists():
        write_json(path, fresh)
    if file_sha256(path) != plan['fresh_sha256']:
        raise ValueError('Fresh annotation mismatch')
    return plan, manifest, reference, path


def distinct_unknown(boxes, scores, original):
    if not len(original):
        return original, 0
    remaining = torch.ones_like(scores, dtype=torch.bool)
    remaining[original] = False
    ids = remaining.nonzero().flatten()
    order = torch.cat((original, ids[torch.argsort(scores[ids], descending=True, stable=True)]))
    if (scores[order][1:] > scores[order][:-1]).any():
        raise ValueError('Native unknown entries do not lead the unknown-score order')
    priority = torch.arange(len(order), 0, -1, device=boxes.device, dtype=torch.float32)
    kept = order[nms(box_cxcywh_to_xyxy(boxes[order]).float(), priority, .7)[:len(original)]]
    missing = len(original) - len(kept)
    if missing:
        kept = torch.cat((kept, order[~torch.isin(order, kept)][:missing]))
    return order[torch.isin(order, kept)], missing


@torch.inference_mode()
def evaluate(model, post, loader, current, prior):
    classes = torch.tensor(current, device='cuda')
    rows, totals = {}, Counter()
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        native = post(out, sizes, include_top_indices=True)
        _, categories, unknown_scores = post.candidate_scores(out)
        for i, target in enumerate(targets):
            result = native[i]
            positions = result['top_indices']
            queries = positions // len(categories)
            unknown = result['labels'] == 91
            chosen = queries[unknown]
            boxes = out['pred_boxes'][i]
            alt, fill = distinct_unknown(boxes, unknown_scores[i], chosen)
            if (len(positions) != 50 or len(alt.unique()) != len(chosen)
                    or not torch.equal(result['scores'][unknown], unknown_scores[i, chosen])):
                raise ValueError('Changed native scores or output budget')
            # Both arms use the identical native list of known query/class/score entries.
            known = positions[~unknown].tolist()
            im = int(target['image_id'].item())
            truth = target['boxes'].cuda()[torch.isin(target['labels'].cuda(), classes)]
            overlap = paired.overlap(boxes, truth).cpu()
            edges = overlap >= .5
            counts = dict(images=1, gt=len(truth), quota=len(chosen), known=len(known), refill=fill,
                          raw_covered=int(edges.any(0).sum()), replaced=int((~torch.isin(alt, chosen)).sum()))
            if prior is not None and (counts['gt'] != prior[str(im)]['gt']
                                      or counts['raw_covered'] != prior[str(im)]['raw_final']):
                raise ValueError(f'Unpaired reference decoder coverage at {im}')
            full = gates.measure_boxes(edges, torch.ones(len(edges), dtype=torch.bool), len(chosen))
            counts.update(raw_matching=full['one_to_one'], quota_upper=full['same_quota'])
            best = overlap.argmax(1) if len(truth) else torch.full((len(boxes),), -1)
            best = torch.where(edges.any(1), best, -1)
            row = dict(known_positions=known, known_scores=result['scores'][~unknown].tolist(),
                       unknown_scores=unknown_scores[i].tolist(), best_gt=best.tolist(),
                       query_gt=[e.nonzero().flatten().tolist() for e in edges])
            for arm, picks in [('native', chosen), ('distinct', alt)]:
                ids = picks.cpu()
                mask = torch.zeros(len(boxes), dtype=torch.bool)
                mask[ids] = True
                # Processing high-score detections first, each best-overlap GT can earn one hit.
                assigned = best[ids]
                counts[arm + '_hits'] = len(assigned[assigned >= 0].unique())
                counts[arm + '_matching'] = gates.measure_boxes(edges, mask, len(ids))['one_to_one']
                row[arm] = ids.tolist()
            rows[im] = dict(**counts, **row)
            totals.update(counts)
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluated {step}/{len(loader)} batches', flush=True)
    if set(rows) != set(loader.dataset.ids):
        raise ValueError('Incomplete inference slice')
    return dict(images=rows, counts=totals)


def main():
    plan, manifest, reference, fresh = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    rows, passed = {}, {}
    for seed, job in plan['jobs'].items():
        checkpoint = Path(job['parent'])
        args = load_checkpoint_metadata(checkpoint)['args']
        if (args.owod_stage != 0 or args.owod_max_detections != 50 or args.discovery_scoring != 'focal'
                or list(args.owod_known_class_ids) != list(manifest['stages'][0]['active_classes'])):
            raise ValueError('Expected original focal Top-50 Task 1 checkpoint')
        post = DiscoveryPostProcess(known_ids=args.owod_known_class_ids, max_detections=args.owod_max_detections,
                                    unknown_threshold=args.unknown_threshold, scoring=args.discovery_scoring)
        model, _ = load_detector(args, checkpoint, torch.device('cuda'))
        model.eval()
        rows[seed] = {}
        for label, annotation in [('reference1024', reference), ('fresh1024', fresh)]:
            path = OUTPUT / f'seed{seed}-{label}.json'
            if not path.exists():
                prior = None
                if label == 'reference1024':
                    prior = read_json(counts.OUTPUT / f'seed{seed}-holdout1024-global.json')['images']
                _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                print(f'Seed {seed} {label}: native and fixed-quota distinct unknowns', flush=True)
                result = evaluate(model, post, loader, manifest['stages'][1]['classes'], prior)
                write_json(path, dict(plan_sha256=digest, **result))
            saved = read_json(path)
            if saved['plan_sha256'] != digest or len(saved['images']) != 1024:
                raise ValueError('Unexpected inference endpoint')
            rows[seed][label] = saved['counts']
        a, b = rows[seed]['reference1024'], rows[seed]['fresh1024']
        passed[seed] = (a['distinct_hits'] >= a['native_hits'] and b['gt'] > 0
                        and 100 * (b['distinct_hits'] - b['native_hits']) >= 5 * b['gt'])
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
