"""Test whether frozen S Task 2 features support better class-conditional query ranking."""

from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_ffn_score_components import OUTPUT as SCORE
from tools.owod.probe_s_task2_ffn_target_transfer import MANIFEST, OUTPUT as REPLAY, PARENT
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = REPLAY.with_name('s-instance-task2-frozen-query-readout')
SOURCE_SHA256 = '168f03ccd0734a94773dddc70f05db76d8882c82d89bc427ddf816d57c9a4740'
MIN_PAIRS = 16
RIDGE = .1
SEED = 42
CRITERION = (
    'On each held-out slice, among new-class targets with an original box and native within-class '
    'rank below Top-5, the linear pairwise readout must beat the best original or shuffled '
    'within-class Top-5 coverage by at least 10 pp. Require two common classes with a net gain '
    'over each of those controls. Unfitted classes stay in the denominator at original rank.'
)


def prepare():
    score = read_json(SCORE / 'comparison.json')
    replay = read_json(REPLAY / 'plan.json')
    source = read_json(PARENT / 'plan.json')
    manifest = read_json(MANIFEST)
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (file_sha256(SCORE / 'comparison.json') != SOURCE_SHA256 or score['supported']
            or score['status'] != 'complete' or score['plan_sha256'] != file_sha256(SCORE / 'plan.json')
            or file_sha256(MANIFEST) != replay['manifest_sha256']
            or not device.isdecimal() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != replay['gpu'] or str(torch.__version__) != replay['torch_version']):
        raise ValueError('Changed score evidence, benchmark or environment')
    control = replay['checkpoints']['control']
    checkpoint = Path(control['path'])
    if (control != source['control']['finetune'] or file_sha256(checkpoint) != control['sha256']
            or read_json(checkpoint.parent / 'training_complete.json')['last_epoch'] != 0):
        raise ValueError('Changed original frozen finetune endpoint')
    classes = manifest['stages'][1]['classes']
    occupied = set(manifest['evaluation']['image_source_ids'])
    specs = {'train': dict(annotation=source['train_a_annotation'], annotation_sha256=source['train_a_annotation_sha256'])}
    specs.update(replay['slices'])
    cohorts = {}
    for name, item in specs.items():
        path = Path(item['annotation'])
        payload = read_json(path)
        order = [im['owod_source_id'] for im in payload['images']]
        count = 1536 if name == 'train' else 1024
        if (file_sha256(path) != item['annotation_sha256'] or len(order) != count or len(set(order)) != count
                or occupied.intersection(order) or not set(classes) <= {a['category_id'] for a in payload['annotations']}):
            raise ValueError(f'Changed or overlapping diagnostic images: {name}')
        occupied.update(order)
        order_sha = sha256('\n'.join(order).encode()).hexdigest()
        cohort = dict(annotation=str(path), annotation_sha256=item['annotation_sha256'], order_sha256=order_sha)
        if name != 'train':
            reference = REPLAY / f'{name}-control.json'
            ref = read_json(reference)
            if (order_sha != item['order_sha256'] or ref['plan_sha256'] != file_sha256(REPLAY / 'plan.json')
                    or ref['checkpoint_sha256'] != control['sha256']):
                raise ValueError(f'Changed frozen reference: {name}')
            cohort.update(reference=str(reference), reference_sha256=file_sha256(reference))
        cohorts[name] = cohort
    plan = {k: replay[k] for k in ('protocol', 'gpu', 'batch_size', 'torch_version', 'manifest_sha256')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                device=device, seed=SEED, paper_comparable=False, checkpoint=control, cohorts=cohorts,
                source_sha256=SOURCE_SHA256, classes=classes, ridge=RIDGE, min_pairs=MIN_PAIRS,
                negative_iou=.3, criterion=CRITERION)
    pin(OUTPUT / 'plan.json', plan)
    return plan, manifest


@contextmanager
def capture_classifier(model):
    detector = model.detr
    classifier = detector.class_embed[detector.transformer.decoder.num_layers - 1]
    result = []

    def capture(_module, inputs, output):
        result[:] = [inputs[0], output]

    def features(outputs):
        if len(result) != 2 or not torch.equal(result[1], outputs['pred_logits']):
            raise ValueError('The recorded classifier input does not reproduce the final head')
        return result[0]

    handle = classifier.register_forward_hook(capture)
    try:
        yield features
    finally:
        handle.remove()


@torch.inference_mode()
def query_images(model, loader, processor, classes, name):
    class_ids = set(classes)
    with capture_classifier(model) as classifier_features:
        for step, (samples, targets) in enumerate(loader, 1):
            outputs = model(samples.to('cuda'))
            features = classifier_features(outputs).cpu().numpy()
            boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
            focal, columns, _ = processor.candidate_scores(outputs)
            classifier_scores = outputs['pred_logits'][..., classes].sigmoid().cpu().numpy()
            index = {cls: i for i, cls in enumerate(columns.tolist())}
            positions = [index[cls] for cls in classes]
            focal = focal[..., positions].cpu().numpy()
            for i, target in enumerate(targets):
                image_id = int(target['image_id'])
                coco = loader.dataset.coco
                h, w = target['orig_size'].tolist()
                scaled = boxes[i] * boxes.new_tensor([w, h, w, h])
                query_boxes = np.asarray([voc_prediction_box(box) for box in scaled.cpu().tolist()])
                anns = [a for a in coco.imgToAnns.get(image_id, []) if a['category_id'] in class_ids]
                matches = [(ann, overlaps(np.asarray(ann['voc_bbox']), query_boxes)) for ann in anns]
                yield image_id, features[i], focal[i], classifier_scores[i], matches
            if step % 64 == 0 or step == len(loader):
                print(f'Inference {step}/{len(loader)} batches: {name}', flush=True)


def linear_pair_weights(differences, rng, ridge):
    x = np.asarray(differences, dtype=np.float64)
    scale = x.std(axis=0).clip(min=.01)
    design = x / scale
    normal = design.T @ design + len(x) * ridge * np.eye(x.shape[1])
    signs = np.ones(len(x))
    signs[rng.permutation(len(x))[:len(x) // 2]] = -1
    native = np.linalg.solve(normal, design.sum(axis=0)) / scale
    shuffled = np.linalg.solve(normal, design.T @ signs) / scale
    return native, shuffled


def fit(model, loader, processor, plan, threshold, plan_hash):
    path = OUTPUT / 'readout.json'
    if not path.exists():
        rng = np.random.default_rng(plan['seed'])
        pairs = defaultdict(list)
        for _, features, _, _, matches in query_images(model, loader, processor, plan['classes'], 'fit'):
            groups = defaultdict(list)
            for ann, ious in matches:
                groups[ann['category_id']].append((ann, ious))
            for cls, cohort in groups.items():
                negatives = np.flatnonzero(np.maximum.reduce([ious for _, ious in cohort]) < plan['negative_iou'])
                if not len(negatives):
                    continue
                for ann, ious in cohort:
                    positive = int(ious.argmax())
                    if not ann.get('difficult') and ious[positive] > threshold:
                        negative = rng.choice(negatives)
                        pairs[cls].append(features[positive] - features[negative])
        heads = {}
        for cls in plan['classes']:
            examples = pairs[cls]
            row = dict(pairs=len(examples))
            if len(examples) >= plan['min_pairs']:
                weights, control = linear_pair_weights(examples, rng, plan['ridge'])
                row.update(weights=weights.tolist(), control=control.tolist())
            heads[str(cls)] = row
        write_json(path, dict(plan_sha256=plan_hash, heads=heads))
        print('Training-pool readout fitted', flush=True)
    report = read_json(path)
    if report['plan_sha256'] != plan_hash:
        raise ValueError('Fit belongs to a different plan')
    return report['heads']


def rank(scores, covered):
    order = np.argsort(-scores, kind='stable')
    return 1 + int(np.flatnonzero(covered[order])[0])


def evaluate(model, loader, processor, plan, cohort, name, heads, threshold, plan_hash):
    path = OUTPUT / f'{name}.json'
    if not path.exists():
        ref = read_json(Path(cohort['reference']))
        classes = plan['classes']
        columns = {cls: i for i, cls in enumerate(classes)}
        targets = {r['annotation_id']: r for r in ref['rows'] if (r['category'] in columns
                   and r['within_class_rank'] is not None and r['within_class_rank'] > 5)}
        rows = []
        weights = {int(cls): np.stack([row['weights'], row['control']], axis=1)
                   for cls, row in heads.items() if 'weights' in row}
        for image_id, features, focal, original, matches in query_images(model, loader, processor, classes, name):
            local_scores = {}
            for ann, ious in matches:
                ref_row = targets.get(ann['id'])
                if ref_row is None:
                    continue
                cls = ann['category_id']
                covered = ious > threshold
                if not covered.any():
                    raise ValueError(f'Previously covered target has no matching box: {ann["id"]}')
                column = columns[cls]
                native = rank(focal[:, column], covered)
                if native != ref_row['within_class_rank'] or float(ious.max()) != ref_row['best_iou'] or cls != ref_row['category']:
                    raise ValueError(f'Original box, category or focal rank changed: {ann["id"]}')
                class_only = rank(original[:, column], covered)
                linear = shuffled = native
                if cls in weights:
                    if cls not in local_scores:
                        local_scores[cls] = features @ weights[cls]
                    scores = local_scores[cls]
                    linear, shuffled = (rank(scores[:, index], covered) for index in range(2))
                rows.append(dict(annotation_id=ann['id'], image_id=image_id, category=cls, native=native,
                                 class_only=class_only, linear=linear, shuffled=shuffled))
        if len(rows) != len(targets) or {row['annotation_id'] for row in rows} != targets.keys():
            raise ValueError('The complete fixed holdout target set did not replay')
        write_json(path, dict(plan_sha256=plan_hash, rows=rows))
    result = read_json(path)
    if result['plan_sha256'] != plan_hash:
        raise ValueError(f'Changed readout report: {name}')
    return result['rows']


def summarize(rows):
    controls = ('native', 'class_only', 'shuffled')
    names = (*controls, 'linear')
    totals = {k: sum(row[k] <= 5 for row in rows) for k in names}
    per_class = defaultdict(lambda: dict.fromkeys(names, 0))
    for row in rows:
        summary = per_class[row['category']]
        for k in names:
            summary[k] += int(row[k] <= 5)
    gain = totals['linear'] - max(totals[k] for k in controls)
    improved = [cls for cls, count in per_class.items() if count['linear'] > max(count[k] for k in controls)]
    return dict(targets=len(rows), totals=totals, per_class=per_class, improved=sorted(improved),
                gain=gain, margin_pass=len(rows) > 0 and 10 * gain >= len(rows))


def main():
    plan, manifest = prepare()
    threshold = manifest['evaluation']['iou_threshold']
    plan_hash = file_sha256(OUTPUT / 'plan.json')
    model, _ = load_detector(None, Path(plan['checkpoint']['path']), torch.device('cuda'))
    model.eval()
    processor = DiscoveryPostProcess(known_ids=manifest['stages'][1]['active_classes'], max_detections=50, scoring='focal')
    image_root = resolve_image_root(MANIFEST, manifest)
    loaders = {name: make_loader(image_root, Path(cohort['annotation']), plan['batch_size'], 2)[1]
               for name, cohort in plan['cohorts'].items()}
    heads = fit(model, loaders['train'], processor, plan, threshold, plan_hash)
    results = {name: summarize(evaluate(model, loaders[name], processor, plan, plan['cohorts'][name],
                                          name, heads, threshold, plan_hash)) for name in ('original', 'independent')}
    common = set.intersection(*(set(row['improved']) for row in results.values()))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=plan_hash, paper_comparable=False,
               fit_pairs={cls: row['pairs'] for cls, row in heads.items()}, slices=results,
               common_improved=sorted(common),
               supported=len(common) >= 2 and all(row['margin_pass'] for row in results.values())))
    print('Frozen query readout comparison saved', flush=True)


if __name__ == '__main__':
    main()
