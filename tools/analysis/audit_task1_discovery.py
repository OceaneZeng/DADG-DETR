"""Trace released Task 2 objects in S-OWODB Task 1 training images, offline."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.benchmark_spec import load_manifest, manifest_path
from tools.owod.prepare_benchmarks import image_id
from tools.owod.protocol import file_sha256, resolve_image_root, resolve_manifest_path

MANIFEST = manifest_path('s-owodb')
RUN = ROOT / 'output/sowodb-task2-v7-transfer-3epochs-2gpu'
PARENT = ROOT / 'output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/checkpoint0049.pth'
PARENT_HASH = '0f010e57faf3c0ecac4edb00ce251ce9e71ea5bf2dfa2be9a9d779acfbafa04a'


def exposure(manifest, sample):
    """Count the complete split. Retain scoring boxes only for preselected images."""
    stems = set(manifest['splits']['t1_train'])
    new_ids = set(manifest['stages'][1]['classes'])
    known_ids = set(manifest['stages'][0]['classes'])
    scored_ids = known_ids | new_ids
    seen, positive_images = set(), set()
    counts, class_images = Counter(), defaultdict(set)
    scoring = {stem: [] for stem in sample}
    for filename, expected in manifest['source_hashes'].items():
        path = resolve_manifest_path(MANIFEST, filename)
        if file_sha256(path) != expected:
            raise ValueError(f'Raw annotation hash mismatch: {path}')
        with path.open(encoding='utf-8') as f:
            payload = json.load(f)
        images = {}
        for im in payload['images']:
            stem = Path(im['file_name']).stem
            if stem in stems:
                if stem in seen:
                    raise ValueError(f'Duplicate source image: {stem}')
                seen.add(stem)
                images[im['id']] = (stem, im['width'], im['height'])
        for ann in payload['annotations']:
            info = images.get(ann['image_id'])
            if info is None:
                continue
            stem, width, height = info
            cid = ann['category_id']
            if cid in new_ids:
                counts[cid] += 1
                class_images[cid].add(stem)
                positive_images.add(stem)
            if stem not in scoring or cid not in scored_ids:
                continue
            # prepare_benchmarks.load_coco uses the author's integer VOC conversion,
            # including crowd boxes. The dataset then clips to image bounds.
            x, y, w, h = ann['bbox']
            x1, y1 = max(0, int(x + 1) - 1), max(0, int(y + 1) - 1)
            x2, y2 = min(width, int(x + w + 1)), min(height, int(y + h + 1))
            if x2 > x1 and y2 > y1:
                scoring[stem].append((cid, [x1/width, y1/height, x2/width, y2/height]))
        del payload
    if seen != stems:
        raise ValueError(f'{len(stems - seen)} Task 1 images missing from raw COCO annotations')
    return dict(images=len(seen), images_with_task2=len(positive_images), task2_annotations=sum(counts.values()),
                by_class={c: dict(images=len(class_images[c]), annotations=counts[c]) for c in sorted(new_ids)}), scoring


def frozen_slots(model, features):
    """v5 existing-slot assignment against the frozen bank, before any update."""
    import torch
    import torch.nn.functional as F

    memory = model.memory
    counts = memory.counts[memory.known_slots:]
    centers = memory.centers[memory.known_slots:]
    scores = F.cosine_similarity(features[:, None], centers[None], dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite anonymous-slot similarities')
    best, slots = scores.masked_fill(counts == 0, -2).max(1)
    # PrototypeMemory.observe defaults to raw cosine >= .8 for updates.
    return torch.where(best >= .8, slots, -1), counts


def score_image(boxes, pseudo, slots, counts, min_support, annotations, new_ids, by_class, by_slot):
    """Any-query coverage counts each GT once per step. Slot composition counts proposals."""
    import torch
    from util.box_ops import box_iou

    gt = boxes.new_tensor([b for _, b in annotations]).reshape(-1, 4)
    ious = box_iou(boxes, gt)[0]
    if not torch.isfinite(ious).all():
        raise ValueError('Nonfinite box IoUs')
    mature = (slots >= 0) & (counts[slots.clamp_min(0)] >= min_support)
    stages = {'any_query': ious, 'pseudo': ious[pseudo], 'mature_slot': ious[pseudo][mature]}
    covered = {k: (v >= .5).any(0).tolist() for k, v in stages.items()}
    for j, (cid, _) in enumerate(annotations):
        if cid in new_ids:
            by_class[cid]['gt'] += 1
            for k in covered:
                by_class[cid][k] += int(covered[k][j])
    for row, slot in zip(ious[pseudo], slots.tolist()):
        group = 'unresolved'
        if row.numel() and row.max() >= .5:
            cid = annotations[int(row.argmax())][0]
            group = f'task2_class_{cid}' if cid in new_ids else 'task1_known'
        by_slot[str(slot) if slot >= 0 else 'no_existing_slot'][group] += 1


def replay(manifest, sample, scoring):
    import torch
    from main import load_local_checkpoint
    from models.graph_local.discovery import make_ema
    from models.graph_local.discovery_candidates import flip_samples, mine_candidates
    from tools.owod.gnn_calibration import build_calibration_dataset, load_detector, move_targets
    from util.box_ops import box_cxcywh_to_xyxy
    from util.misc import nested_tensor_from_tensor_list

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for checkpoint replay')
    primary = torch.device('cuda:0')
    secondary = torch.device('cuda:1') if torch.cuda.device_count() >= 2 else primary
    stage = manifest['stages'][0]
    annotation = resolve_manifest_path(MANIFEST, stage['files']['increment_train'])
    if file_sha256(annotation) != stage['sha256']['increment_train']:
        raise ValueError('Task 1 training annotation hash mismatch')
    model, criterion = load_detector(None, PARENT, primary)
    if model.known_ids != stage['classes'] or criterion.revision != 5:
        raise ValueError('Expected Task 1 v5 student')
    model.eval()
    teacher = make_ema(model).to(secondary)
    teacher.load_state_dict(load_local_checkpoint(PARENT)['discovery_ema'], strict=True)
    dataset = build_calibration_dataset(resolve_image_root(MANIFEST, manifest), annotation)
    index = {n: i for i, n in enumerate(dataset.ids)}
    new_ids = set(manifest['stages'][1]['classes'])
    by_class = {c: Counter(gt=0, any_query=0, pseudo=0, mature_slot=0) for c in sorted(new_ids)}
    by_slot = defaultdict(Counter)
    config = dict(student=str(primary), ema=str(secondary), top_k=criterion.top_k,
                  saliency=criterion.saliency_threshold, consistency=criterion.consistency_threshold,
                  min_support=model.min_support, slot_cosine=.8, coverage_iou=.5,
                  slot_support=model.memory.counts[model.memory.known_slots:].tolist())
    with torch.inference_mode():
        for i, stem in enumerate(sample, 1):
            im, target = dataset[index[image_id(stem)]]
            if set(target['labels'].tolist()) - set(model.known_ids):
                raise ValueError('Unexpected unreleased class in miner input')
            samples = nested_tensor_from_tensor_list([im])
            # The compiled deformable-attention op launches on the current device.
            with torch.cuda.device(secondary):
                other = teacher(flip_samples(samples.to(secondary)))
            with torch.cuda.device(primary):
                out = model(samples.to(primary))
            for k in ('pred_boxes', 'query_features', 'attention_feature'):
                other[k] = other[k].to(primary)
            targets = move_targets([target], primary)
            matches = criterion.matcher(out, targets)
            cand = mine_candidates(out, targets, matches, model.known_ids, other, criterion.top_k,
                                   criterion.saliency_threshold, criterion.consistency_threshold)
            pseudo = cand['pseudo'][0]
            slots, counts = frozen_slots(model, out['query_features'][0, pseudo])
            score_image(box_cxcywh_to_xyxy(out['pred_boxes'][0]), pseudo, slots, counts, model.min_support,
                        scoring[stem], new_ids, by_class, by_slot)
            del out, other, cand
            if i % 50 == 0:
                print(f'Replayed {i}/{len(sample)} images', file=sys.stderr, flush=True)
    total = {k: sum(row[k] for row in by_class.values()) for k in ('gt', 'any_query', 'pseudo', 'mature_slot')}
    total['pseudo_proposals'] = sum(sum(row.values()) for row in by_slot.values())
    total['pseudo_without_existing_slot'] = sum(by_slot.get('no_existing_slot', {}).values())
    total['slot3_proposals_by_overlap'] = dict(by_slot.get('3', {}))
    return dict(config=config, summary=total, by_class=by_class, proposals_by_slot=by_slot, source_ids=sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=int, default=512, help='Uniform sample size for frozen replay')
    parser.add_argument('--exposure-only', action='store_true', help='Full annotation census on CPU')
    opt = parser.parse_args()
    manifest = load_manifest('s-owodb')
    population = sorted(set(manifest['splits']['t1_train']))
    if not 1 <= opt.images <= len(population):
        parser.error(f'--images must be between 1 and {len(population)}')
    sample = [] if opt.exposure_only else random.Random(42).sample(population, opt.images)
    if sample and file_sha256(PARENT) != PARENT_HASH:
        raise ValueError('Expected the original seed 42 epoch 49 parent')
    census, scoring = exposure(manifest, sample)
    report = dict(manifest_sha256=file_sha256(MANIFEST), exposure=census)
    print(json.dumps({k: v for k, v in census.items() if k != 'by_class'}), flush=True)
    filename = 'task1-exposure.json'
    if sample:
        report.update(parent_sha256=PARENT_HASH, seed=42, replay=replay(manifest, sample, scoring))
        print(json.dumps(report['replay']['summary']), flush=True)
        filename = f'task1-discovery-{len(sample)}.json'
    RUN.mkdir(exist_ok=True, parents=True)
    from tools.owod.train_task2_diagnostic import dump
    dump(RUN / filename, report)
    print(RUN / filename)


if __name__ == '__main__':
    main()
