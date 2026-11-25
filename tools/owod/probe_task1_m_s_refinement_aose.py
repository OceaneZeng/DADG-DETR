"""Attribute changes in author A-OSE after Task 1 decoder refinement."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import evaluate_author_benchmark, overlaps, voc_prediction_box
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import ARMS, MANIFESTS, OUTPUT as SOURCE
from tools.owod.run_task1_m_s_released_small_extent import measure
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('task1-m-s-refinement-aose')
SOURCE_SHA = '949ef0e69fe467a537a4020d1ae02a77420671ada61670b819399c086e14c315'
ENDPOINTS_SHA = '789b130917c50abfc84faa03d11b44bb42e447fdd20752fe1902cf4fb8dbd4cc'
METRICS = ('known_ap50', 'u_recall', 'a_ose', 'wi', 'unknown_predictions')
UNKNOWN = 91


def prepare():
    result = verified(SOURCE / 'comparison.json', SOURCE_SHA)
    if result['status'] != 'complete' or result['passed']:
        raise ValueError('Expected the completed iterative refinement screen')
    source = verified(SOURCE / 'plan.json', result['plan_sha256'])
    endpoints_file = SOURCE / 'endpoint-sha256.txt'
    if file_sha256(endpoints_file) != ENDPOINTS_SHA:
        raise ValueError('Changed archived refinement checkpoint digests')
    endpoints = {name: digest for digest, name in
                 (line.split() for line in endpoints_file.read_text(encoding='utf-8').splitlines())}
    jobs = {}
    for protocol, job in source['jobs'].items():
        folder = Path(job['command'][job['command'].index('--output_dir') + 1])
        candidate = folder / 'checkpoint.pth'
        checkpoints = dict(control=(Path(job['control']), job['control_sha256']),
                           refinement=(candidate, endpoints[candidate.resolve().relative_to(SOURCE.resolve()).as_posix()]))
        arms = {}
        for arm, (checkpoint, digest) in checkpoints.items():
            if file_sha256(checkpoint) != digest:
                raise ValueError(f'Changed frozen checkpoint: {protocol}/{arm}')
            path = SOURCE / f'{protocol}-{arm}.json'
            reference = read_json(path)
            if reference['plan_sha256'] != result['plan_sha256'] or reference['checkpoint_sha256'] != digest:
                raise ValueError(f'Changed reference metrics: {protocol}/{arm}')
            arms[arm] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, report=str(path),
                             report_sha256=file_sha256(path))
        jobs[protocol] = dict(arms=arms, folds=job['folds'], manifest_sha256=job['manifest_sha256'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_sha256=SOURCE_SHA, source_plan_sha256=result['plan_sha256'], endpoints_sha256=ENDPOINTS_SHA,
                jobs=jobs, batch_size=source['eval_batch'], limit=source['max_detections'], paper_comparable=False,
                quota='Refinement scores select exactly the control number of known and unknown items in each image.')


class Attribution:
    def __init__(self, known_ids, iou_threshold, *, trace_detections=False):
        self.known = set(known_ids)
        self.threshold = iou_threshold
        self.counts = Counter()
        self.targets = Counter()
        self.categories = Counter()
        self.source_categories = Counter()
        self.detections = [] if trace_detections else None

    def add(self, image, predictions, annotations):
        unknown = [ann for ann in annotations if ann['category_id'] not in self.known]
        unknown_boxes = [ann['voc_bbox'] for ann in unknown]
        released = {}
        for ann in annotations:
            if ann['category_id'] in self.known:
                released.setdefault(ann['category_id'], []).append(ann['voc_bbox'])
        scores = predictions['scores'].tolist() if self.detections is not None else None
        detections = zip(predictions['labels'].tolist(), predictions['boxes'].tolist(), strict=True)
        for rank, (label, box) in enumerate(detections):
            if label == UNKNOWN:
                continue
            self.counts['known_outputs'] += 1
            voc_box = np.asarray(voc_prediction_box(box))
            iou = overlaps(voc_box, unknown_boxes)
            if not len(iou):
                continue
            best = int(iou.argmax())
            if iou[best] <= self.threshold:
                continue
            ann = unknown[best]
            self.targets[image, ann['id']] += 1
            self.categories[label] += 1
            self.source_categories[ann['category_id']] += 1
            self.counts['a_ose'] += 1
            same_class = overlaps(voc_box, released.get(label, []))
            self.counts['also_overlaps_same_known_class'] += bool(len(same_class) and same_class.max() > self.threshold)
            self.counts['unknown_difficult'] += bool(ann.get('difficult'))
            if self.detections is not None:
                self.detections.append(dict(image_id=image, unknown_gt=ann['id'], unknown_class=ann['category_id'],
                                            known_prediction=label, score=scores[rank], overlap=float(iou[best])))

    def result(self):
        hits = self.counts['a_ose']
        result = dict(known_outputs=self.counts['known_outputs'], a_ose=hits, unique_unknown_targets=len(self.targets),
                      duplicate_hits=hits - len(self.targets),
                      also_overlaps_same_known_class=self.counts['also_overlaps_same_known_class'],
                      unknown_difficult=self.counts['unknown_difficult'],
                      by_known_prediction=dict(self.categories), by_unknown_source_category=dict(self.source_categories),
                      rate=hits / max(self.counts['known_outputs'], 1))
        if self.detections is not None:
            result['detections'] = self.detections
        return result


def with_quota(scores, categories, boxes, known_count, limit):
    known_columns = len(categories) - 1
    known_score, known_pos = scores[:, :-1].flatten().topk(known_count)
    unknown_score, unknown_query = scores[:, -1].topk(limit - known_count)
    query = torch.cat((known_pos // known_columns, unknown_query))
    labels = torch.cat((categories[known_pos % known_columns], categories[-1].expand(len(unknown_query))))
    values = torch.cat((known_score, unknown_score))
    order = values.argsort(descending=True, stable=True)
    return dict(scores=values[order].cpu(), labels=labels[order].cpu(), boxes=boxes[query[order]].cpu())


def replay(model, manifest, root, annotations, plan, reference, control_counts=None):
    known_ids = manifest['stages'][0]['classes']
    threshold = manifest['evaluation']['iou_threshold']
    native = Attribution(known_ids, threshold)
    balanced = Attribution(known_ids, threshold)
    quotas, predictions = {}, {}
    spec, coco = None, None

    def observe(dataset, _samples, targets, outputs, detections, sizes, post):
        nonlocal spec, coco
        if spec is None:
            coco = dataset.coco
            spec = dict(manifest['evaluation'], image_source_ids=[im['owod_source_id'] for im in coco.dataset['images']])
        if control_counts is not None:
            scores, categories, _ = post.candidate_scores(outputs)
            h, w = sizes.unbind(1)
            boxes = box_cxcywh_to_xyxy(outputs['pred_boxes']) * torch.stack((w, h, w, h), dim=1)[:, None]
        for i, (target, detection) in enumerate(zip(targets, detections, strict=True)):
            image = int(target['image_id'])
            anns = coco.imgToAnns[image]
            native.add(image, detection, anns)
            quotas[image] = int((detection['labels'] != UNKNOWN).sum())
            if control_counts is not None:
                pred = with_quota(scores[i], categories, boxes[i], control_counts[image], plan['limit'])
                predictions[image] = pred
                balanced.add(image, pred, anns)

    report = measure(model, manifest, root, annotations, plan['batch_size'], plan['limit'], observer=observe)
    metrics = report['metrics']
    if metrics != reference['metrics'] or report['matched'] != reference['matched']:
        raise ValueError('Frozen endpoint no longer reproduces the original VOC metrics and matches')
    outcome = native.result()
    if outcome['a_ose'] != metrics['a_ose']:
        raise ValueError('Per-detection trace does not reconstruct author A-OSE')
    result = dict(native=dict(metrics={key: metrics[key] for key in METRICS}, trace=outcome))
    if control_counts is not None:
        adjusted = evaluate_author_benchmark(predictions, coco, known_ids, [], known_ids, spec)
        trace = balanced.result()
        if trace['a_ose'] != adjusted['a_ose'] or trace['known_outputs'] != sum(control_counts.values()):
            raise ValueError('Per-image quota replay does not reconstruct the author counts')
        result['control_quota'] = dict(metrics={key: adjusted[key] for key in METRICS}, trace=trace)
    return result, quotas


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    results = {}
    for protocol, job in plan['jobs'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        root = resolve_image_root(path, manifest)
        arms, reports = {}, {}
        for arm, entry in job['arms'].items():
            arms[arm], _ = load_detector(None, Path(entry['checkpoint']), torch.device('cuda'))
            arms[arm].eval()
            reports[arm] = verified(Path(entry['report']), entry['report_sha256'])['folds']
        results[protocol] = {}
        for fold, entry in job['folds'].items():
            dest = OUTPUT / f'{protocol}-{fold}.json'
            if not dest.exists():
                annotations = Path(entry['path'])
                if file_sha256(annotations) != entry['sha256']:
                    raise ValueError(f'Changed frozen validation slice: {protocol}/{fold}')
                print(f'Evaluating {protocol}/{fold} frozen control', flush=True)
                control, counts = replay(arms['control'], manifest, root, annotations, plan, reports['control'][fold])
                print(f'Evaluating {protocol}/{fold} frozen refinement and control quota', flush=True)
                refine, _ = replay(arms['refinement'], manifest, root, annotations, plan,
                                   reports['refinement'][fold], counts)
                write_json(dest, dict(plan_sha256=digest, control=control['native'],
                                      refinement=refine['native'], refinement_control_quota=refine['control_quota']))
            result = read_json(dest)
            if result.pop('plan_sha256') != digest:
                raise ValueError(f'Changed frozen diagnostic plan: {dest}')
            results[protocol][fold] = result
        del arms
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=results))


if __name__ == '__main__':
    main()
