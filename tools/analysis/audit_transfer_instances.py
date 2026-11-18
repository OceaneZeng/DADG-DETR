"""Replay the original Task 2 training calibration and inspect instance-level slot matches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

import torch
import torch.nn.functional as F
from tools.owod.protocol import file_sha256, resolve_image_root, stage_files
from util.box_ops import box_cxcywh_to_xyxy, box_iou


def summarize(rows, threshold):
    located = [r for r in rows if r['assigned_iou'] >= .5]
    result = dict(assigned=len(rows), assigned_iou50=len(located))
    for space in ('raw', 'projected'):
        def counts(group):
            passed = [r for r in group if r[space]['score'] >= threshold]
            return dict(top1_slots=dict(Counter(r[space]['slot'] for r in group)),
                        passed=len(passed), passing_classes=len({r['class_id'] for r in passed}),
                        passing_images=len({r['image_id'] for r in passed}),
                        passing_slots=dict(Counter(r[space]['slot'] for r in passed)))
        result[space] = dict(all=counts(rows), localized=counts(located))
    return result


@torch.inference_mode()
def audit(opt):
    from models.graph_local.routed_adapter import append_stage
    from tools.owod.gnn_calibration import (build_calibration_dataset, class_loader, load_detector,
                                           move_targets, target_for_class)
    plan = json.loads((opt.results / 'diagnostic_plan.json').read_text())
    if (plan.get('comparison') != 'transfer' or plan.get('phase') != 'increment'
            or plan['checkpoint_sha256'] != file_sha256(opt.parent)
            or plan['manifest_sha256'] != file_sha256(opt.manifest)):
        raise ValueError('Expected the original transfer increment plan, parent and manifest')
    manifest, files = stage_files(opt.manifest, 1)
    cache = torch.load(opt.results / 'common_initialization.pth', map_location='cpu', weights_only=False)
    spec = cache['layout']
    if (not cache.get('diagnostic_only') or cache['diagnostic_identity'] != plan
            or spec['class_ids'] != manifest['stages'][1]['classes']
            or spec['training_annotation_sha256']['increment_train'] != file_sha256(files['increment_train'])):
        raise ValueError('Cached calibration layout or training data do not match')
    device = torch.device(opt.device)
    model, criterion = load_detector(None, opt.parent, device)
    append_stage(model, spec)
    model.load_state_dict(cache.pop('model'), strict=True)
    model.known_ids = criterion.known_ids = criterion.base.active_class_ids = list(manifest['stages'][1]['active_classes'])
    model.eval()
    criterion.eval()
    dataset = build_calibration_dataset(resolve_image_root(opt.manifest, manifest), files['increment_train'])
    memory = model.memory
    slots = torch.where(memory.counts[memory.known_slots:] >= model.min_support)[0]
    if not slots.numel():
        raise ValueError('No supported anonymous slots to compare')
    centers = memory.centers[memory.known_slots:][slots]
    rows, by_class = [], {}
    for cid in spec['class_ids']:
        print(f'Checking released class {cid}', file=sys.stderr, flush=True)
        group, features = [], []
        n_gt = n_covered = 0
        for images, targets in class_loader(dataset, cid, 1, 0, spec['sketch_max_images']):
            image_id = int(targets[0]['image_id'].item())
            target = target_for_class(move_targets(targets, device)[0], cid)
            outputs = model(images.to(device))
            queries, matches = criterion.matcher(outputs, [target])[0]
            n_gt += len(target['boxes'])
            if not len(target['boxes']):
                continue
            overlap = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][0]),
                              box_cxcywh_to_xyxy(target['boxes']))[0]
            n_covered += int((overlap.amax(0) >= .5).sum())
            if not len(queries):
                continue
            query_features = outputs['query_features'][0, queries]
            features.append(query_features)
            scores = dict(raw=F.normalize(query_features) @ F.normalize(centers).T,
                          projected=model.affinity_graph.similarities(query_features, centers))
            if not torch.isfinite(overlap).all() or any(not torch.isfinite(s).all() for s in scores.values()):
                raise ValueError(f'Nonfinite scores or boxes for image {image_id}')
            best = {key: value.max(1) for key, value in scores.items()}
            ious = overlap[queries, matches].tolist()
            for i, iou in enumerate(ious):
                group.append(dict(class_id=cid, image_id=image_id, assigned_iou=iou, **{
                    k: dict(slot=int(slots[v.indices[i]]), score=float(v.values[i])) for k, v in best.items()}))
        stats = summarize(group, model.transfer_threshold)
        stats.update(gt=n_gt, any_query_iou50=n_covered)
        if features and cid in cache['means']:
            mean = torch.cat(features).mean(0)
            original = mean.new_tensor(cache['means'][cid])
            stats['replayed_mean_cosine'] = float(F.cosine_similarity(mean, original, dim=0))
            stats['replayed_mean_max_abs_diff'] = float((mean - original).abs().max())
        by_class[cid] = stats
        rows.extend(group)
    total = summarize(rows, model.transfer_threshold)
    total.update(gt=sum(s['gt'] for s in by_class.values()),
                 any_query_iou50=sum(s['any_query_iou50'] for s in by_class.values()))
    cosines = [s['replayed_mean_cosine'] for s in by_class.values() if 'replayed_mean_cosine' in s]
    total['replayed_classes'] = len(cosines)
    total['min_replayed_mean_cosine'] = min(cosines, default=None)
    return dict(parent_sha256=plan['checkpoint_sha256'], manifest_sha256=plan['manifest_sha256'],
                source='released Task 2 increment training calibration instances', threshold=model.transfer_threshold,
                summary=total, by_class=by_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'output/sowodb-task2-v7-transfer-3epochs-2gpu/results')
    parser.add_argument('--parent', type=Path,
                        default=ROOT / 'output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path)
    opt = parser.parse_args()
    report = audit(opt)
    destination = opt.output or opt.results.parent / 'match-instances.json'
    from tools.owod.train_task2_diagnostic import dump
    dump(destination, report)
    print(json.dumps(report['summary'], indent=2))
    print(f'Per-class results: {destination}', file=sys.stderr)
