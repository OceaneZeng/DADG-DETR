"""Decompose native focal score margins for shared Current matches lost in finetuning."""

from collections import Counter, defaultdict
from math import log
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('task2-m-s-score-margin')
PARENT_PLAN = 'de4bb126945143a69b5a60e60b83b7761c6270e6e71734fe127129948fd35712'
PARENT_RESULT = '0b412be1003930809eaa19270c59a2e604ea2e62f36e75838769e151f11b9a1a'
FACTORS = ('classification', 'objectness', 'cutoff')
MIN_SHARE = .60


@torch.inference_mode()
def replay(model, loader, wanted, known, phase, limit, iou_threshold):
    post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=limit)
    category_positions = {category: index for index, category in enumerate(known)}
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        names = [loader.dataset.coco.imgs[int(t['image_id'])]['owod_source_id'] for t in targets]
        if any(name in wanted for name in names):
            outputs = model(samples.to('cuda'))
            scores, categories, _ = post.candidate_scores(outputs)
            top, indices = scores.flatten(1).topk(limit, dim=1)
            boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
            for i, (name, target) in enumerate(zip(names, targets, strict=True)):
                if name not in wanted:
                    continue
                height, width = target['orig_size'].tolist()
                scaled = boxes[i] * boxes.new_tensor([width, height, width, height])
                raw = np.asarray([voc_prediction_box(box) for box in scaled.tolist()])
                selected = indices[i].tolist()
                for pair in wanted[name]:
                    category = pair['category']
                    index = category_positions[category]
                    eligible = np.flatnonzero(overlaps(np.asarray(pair['voc_bbox']), raw) > iou_threshold)
                    if not len(eligible):
                        raise ValueError(f'Changed geometry: {phase}/{name}/{category}')
                    candidates = eligible.tolist()
                    winner = max(candidates, key=lambda q: float(scores[i, q, index]))
                    emitted = any(q * len(categories) + index in selected for q in candidates)
                    if emitted != (phase == 'increment'):
                        raise ValueError(f'Changed native Top-50 coverage: {phase}/{name}/{category}')
                    rows.append(dict(target=pair['target'], query=winner, score=float(scores[i, winner, index]),
                                     classification=float(outputs['pred_logits'][i, winner, category].sigmoid()),
                                     objectness=float(outputs['pred_objectness'][i, winner].sigmoid()),
                                     cutoff=float(top[i, -1])))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} tracked targets', flush=True)
    expected = {p['target'] for values in wanted.values() for p in values}
    if len(rows) != len(expected) or {r['target'] for r in rows} != expected:
        raise ValueError('Incomplete matched-target replay')
    return rows


def summarize(before, after):
    right = {r['target']: r for r in after}
    counts = Counter(dict.fromkeys(FACTORS, 0))
    rows = []
    for inc in before:
        ft = right[inc['target']]
        drops = {k: log(inc[k]) - log(ft[k]) for k in FACTORS}
        drops['cutoff'] = -drops['cutoff']
        dominant = max(FACTORS, key=drops.__getitem__)
        margin_drop = log(inc['score'] / inc['cutoff']) - log(ft['score'] / ft['cutoff'])
        if abs(margin_drop - sum(drops.values())) > 1e-5:
            raise ValueError(f'Unexpected native focal factors on target {inc["target"]}')
        counts[dominant] += 1
        rows.append(dict(target=inc['target'], drops=drops, margin_drop=margin_drop, dominant=dominant))
    return dict(targets=len(rows), dominant_counts=counts, rows=rows)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    parent = verified(SOURCE / 'plan.json', PARENT_PLAN)
    result = verified(SOURCE / 'comparison.json', PARENT_RESULT)
    if result['status'] != 'complete' or result['plan_sha256'] != PARENT_PLAN:
        raise ValueError('Changed shared-class diagnosis')
    protocols = parent['protocols']
    cohorts = parent['cohorts']
    reports = {}
    for protocol, info in protocols.items():
        for phase in info['endpoints']:
            for fold in cohorts:
                path = SOURCE / f'{protocol}-{phase}-{fold}.json'
                reports[path.name] = file_sha256(path)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                parent_plan=PARENT_PLAN, parent_result=PARENT_RESULT, reports=reports,
                batch_size=parent['batch_size'], max_detections=parent['max_detections'],
                minimum_shared_dominance=MIN_SHARE, paper_comparable=False)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    summarized = {}
    for protocol, info in protocols.items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, info['manifest_sha256'])
        root = resolve_image_root(manifest_path, manifest)
        known = info['previous'] + info['current']
        wanted_by_fold, phase_rows = {}, defaultdict(dict)
        for fold, cohort in cohorts.items():
            pairs = verified(SOURCE / cohort['paired_file'], cohort['paired_sha256'])
            views = []
            for phase in ('increment', 'finetune'):
                name = f'{protocol}-{phase}-{fold}.json'
                payload = verified(SOURCE / name, reports[name])
                if payload['plan_sha256'] != PARENT_PLAN:
                    raise ValueError(f'Changed source endpoint: {name}')
                views.append(payload['outcomes'])
            wanted = defaultdict(list)
            for n, (pair, before, after) in enumerate(zip(pairs, *views, strict=True)):
                if before == 'matched' and after == 'no_correct_class_output':
                    wanted[pair['image']].append(dict(pair, target=n))
            wanted_by_fold[fold] = wanted
        for phase, endpoint in info['endpoints'].items():
            checkpoint = Path(endpoint['path'])
            if file_sha256(checkpoint) != endpoint['sha256']:
                raise ValueError(f'Changed frozen weight: {checkpoint}')
            model, _ = load_detector(None, checkpoint, torch.device('cuda'))
            model.eval()
            if set(model.known_ids) != set(known):
                raise ValueError(f'Changed active classes: {protocol}/{phase}')
            for fold, cohort in cohorts.items():
                entry = cohort['by_protocol'][protocol]
                annotation = SOURCE / entry['file']
                verified(annotation, entry['sha256'])
                path = OUTPUT / f'{protocol}-{phase}-{fold}.json'
                if not path.exists():
                    print(f'Evaluating {protocol}/{phase}/{fold} native score margins', flush=True)
                    _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                    rows = replay(model, loader, wanted_by_fold[fold], known, phase, plan['max_detections'],
                                  manifest['evaluation']['iou_threshold'])
                    write_json(path, dict(plan_sha256=digest, rows=rows))
                report = read_json(path)
                if report['plan_sha256'] != digest:
                    raise ValueError(f'Changed score-margin plan: {path}')
                phase_rows[fold][phase] = report['rows']
            del model
        summarized[protocol] = {fold: summarize(rows['increment'], rows['finetune'])
                                for fold, rows in phase_rows.items()}
    folds = [row for values in summarized.values() for row in values.values()]
    if sum(r['targets'] for r in folds) != 410:
        raise ValueError('Changed target cohort')
    priority = next((factor for factor in FACTORS if all(
        fold['dominant_counts'][factor] >= MIN_SHARE * fold['targets'] for fold in folds)), None)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=summarized, shared_priority=priority, paper_comparable=False))


if __name__ == '__main__':
    main()
