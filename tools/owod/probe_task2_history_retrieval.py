"""Measure how frozen Task 1 instance slots identify released Task 2 objects on disjoint images."""
from collections import Counter
from pathlib import Path
import subprocess

from scipy.optimize import linear_sum_assignment
import torch

from models.graph_local.instance_transfer import Representatives
from tools.owod import probe_task2_duration_holdout as holdout
from tools.owod import run_task2_history_alignment as alignment
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = ROOT / 'output/m-instance-task2-history-retrieval'


def restore_bank(entry, heldout_ids, current):
    cache = read_json(Path(entry['cache_path']))
    data = cache['data']
    bank = Representatives(cache['identity']['settings'])
    bank.slots = [[(r['image'], torch.tensor(r['feature']), r['weight'], r['quality']) for r in slot]
                  for slot in data['representatives']]
    labels = {}
    source = {image for slot in bank.slots for image, *_ in slot}
    for r in data['records']:
        if labels.setdefault(r['slot'], r['label']) != r['label'] or r['label'] not in current:
            raise ValueError('The registered slot must have one released Task 2 class')
        source.add(r['image'])
    if source & heldout_ids or cache['identity']['parent'] != entry['parent_sha256']:
        raise ValueError('Holdout overlaps the frozen bank or its registered parent changed')
    return bank, labels


@torch.inference_mode()
def measure(model, loader, current, bank, labels, image_ids, *, slot_hits=None):
    seen = set()
    current_ids = torch.tensor(current)
    by_class = {c: Counter(gt=0, localized=0, retrieved=0, named=0, correct=0) for c in current}
    # Rows are true released classes; columns are the class registered for the winning slot.
    confusion = {c: Counter() for c in current}
    for step, (images, targets) in enumerate(loader, 1):
        outputs = model(images.to('cuda'))
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes']).cpu()
        features = outputs['instance_features'].cpu()
        for row, target in enumerate(targets):
            image = int(target['image_id'].item())
            if image in seen:
                raise ValueError(f'Repeated holdout image {image}')
            seen.add(image)
            keep = torch.isin(target['labels'], current_ids)
            truth = target['labels'][keep].tolist()
            for c in truth:
                by_class[c]['gt'] += 1
            if not truth:
                continue
            overlap = box_iou(boxes[row], box_cxcywh_to_xyxy(target['boxes'][keep]))[0]
            queries, objects = linear_sum_assignment(-overlap.numpy())
            for q, g in zip(queries, objects):
                if overlap[q, g] < bank.settings['localization_iou']:
                    continue
                c = truth[g]
                counts = by_class[c]
                counts['localized'] += 1
                hit = bank.query(features[row, q], image)
                if hit is None:
                    continue
                counts['retrieved'] += 1
                if slot_hits is not None:
                    slot_hits.append((image, hit[1], c))
                predicted = labels.get(hit[1])
                if predicted is not None:
                    counts['named'] += 1
                    counts['correct'] += int(predicted == c)
                    confusion[c][predicted] += 1
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(seen)} images', flush=True)
    if seen != image_ids:
        raise ValueError('Incomplete holdout pass')
    totals = {k: sum(counts[k] for counts in by_class.values()) for k in next(iter(by_class.values()))}
    named = totals['named']
    predicted_counts = sum(confusion.values(), Counter())
    expected = sum(by_class[c]['named'] * predicted_counts[c] for c in current)
    return dict(images=len(seen), totals=totals, by_class=by_class, confusion=confusion,
                named_accuracy=totals['correct'] / named if named else None,
                independent_label_accuracy=expected / named**2 if named else None,
                correct_gt_coverage=totals['correct'] / totals['gt'])


def main():
    source = read_json(alignment.OUTPUT / 'plan.json')
    result = read_json(alignment.OUTPUT / 'comparison.json')
    annotation = holdout.OUTPUT / 'heldout.json'
    manifest = read_json(MANIFEST)
    if (result['status'] != 'complete' or result['plan_sha256'] != file_sha256(alignment.OUTPUT / 'plan.json')
            or file_sha256(annotation) != source['annotations']['holdout']
            or file_sha256(MANIFEST) != source['manifest_sha256'] or str(torch.__version__) != source['torch_version']):
        raise ValueError('The completed alignment comparison and its original holdout are required')
    entries = {}
    for seed, job in source['jobs'].items():
        command = job['commands']['matched']['increment']
        parent = command[command.index('--pretrained') + 1]
        if file_sha256(Path(parent)) != job['parent_sha256'] or file_sha256(Path(job['cache_path'])) != job['cache_sha256']:
            raise ValueError(f'Changed frozen input for seed {seed}')
        entries[seed] = dict(parent=parent, parent_sha256=job['parent_sha256'], cache_path=job['cache_path'],
                             cache_sha256=job['cache_sha256'])
    current = sorted(manifest['stages'][1]['classes'])
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_plan_sha256=result['plan_sha256'], source_result_sha256=file_sha256(alignment.OUTPUT / 'comparison.json'),
                manifest_sha256=source['manifest_sha256'], annotation_sha256=source['annotations']['holdout'],
                entries=entries, current_classes=current, torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                batch_size=2, paper_comparable=False, localization='IoU assignment against Task 2 classes only',
                criterion='Both seeds: named accuracy >= 2/3 and correct coverage >= 5% of all released Task 2 GT')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, plan['batch_size'], 2)
    ids = set(dataset.ids)
    if len(ids) != 1024 or len(dataset) != len(ids):
        raise ValueError('Expected the 1024 unique training-source holdout images')
    rows = {}
    for seed, entry in entries.items():
        path = OUTPUT / f'seed{seed}.json'
        if not path.is_file():
            bank, labels = restore_bank(entry, ids, current)
            model, _ = load_detector(None, Path(entry['parent']), torch.device('cuda'))
            print(f'Seed {seed}: frozen parent, {len(labels)} named slots, {len(ids)} images', flush=True)
            row = measure(model.eval(), loader, current, bank, labels, ids)
            write_json(path, dict(plan_sha256=digest, **row))
            del model
        row = read_json(path)
        if row['plan_sha256'] != digest or row['images'] != len(ids):
            raise ValueError(f'Incomplete or mismatched seed result: {path}')
        rows[seed] = row
    passed = all((r['named_accuracy'] or 0) >= 2 / 3 and r['correct_gt_coverage'] >= .05 for r in rows.values())
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passes_retrieval_screen=passed))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
