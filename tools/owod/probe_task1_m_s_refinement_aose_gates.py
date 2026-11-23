"""Join frozen Top-50 A-OSE detections to native EMA mining gates."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery_candidates import flip_samples, mine_candidates
from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.probe_task1_discovery_gates import frozen_miner
from tools.owod.probe_task1_m_s_refinement_aose import UNKNOWN
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import TOP_KNOWN_SHA
from tools.owod.probe_task1_m_s_refinement_top_known import OUTPUT as SOURCE, TopKnownPerQuery
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_released_small_extent import measure, size_group
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('task1-m-s-refinement-aose-gates')
GATES = ('released_gt', 'invalid_size', 'student_saliency', 'no_teacher_match',
         'consistency', 'teacher_saliency')


class IndexedTopKnown(TopKnownPerQuery):
    def forward(self, outputs, target_sizes):
        return super().forward(outputs, target_sizes, include_top_indices=True)


def prepare():
    reference = verified(SOURCE / 'comparison.json', TOP_KNOWN_SHA)
    if reference['status'] != 'complete' or reference['passed']:
        raise ValueError('Expected the completed single-known-label comparison')
    parent = verified(SOURCE / 'plan.json', reference['plan_sha256'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_comparison_sha256=TOP_KNOWN_SHA, source_plan_sha256=reference['plan_sha256'],
                jobs=parent['jobs'], limit=parent['limit'], batch_size=parent['batch_size'],
                expected=reference['protocols'], paper_comparable=False,
                first_hit='Highest native Top-50 known output per target and image, in emitted order.',
                mining='Student eval and flipped checkpoint EMA. Only released boxes and labels enter mining.')


def voc_hits(detection, annotations, known_ids, threshold):
    unknown = [ann for ann in annotations if ann['category_id'] not in known_ids]
    if not unknown:
        return []
    boxes = [ann['voc_bbox'] for ann in unknown]
    seen, hits = set(), []
    labels = detection['labels'].tolist()
    queries = (detection['top_indices'] // (len(known_ids) + 1)).tolist()
    for rank, (label, query, box) in enumerate(zip(labels, queries, detection['boxes'].tolist(), strict=True), 1):
        if label == UNKNOWN:
            continue
        ious = overlaps(np.asarray(voc_prediction_box(box)), boxes)
        best = int(ious.argmax())
        if ious[best] <= threshold:
            continue
        ann = unknown[best]
        hits.append(dict(query=query, known_label=label, rank=rank, annotation=ann['id'], iou=float(ious[best]),
                         difficult=bool(ann.get('difficult')), size=size_group(ann), first=ann['id'] not in seen))
        seen.add(ann['id'])
    return hits


def mining_rows(trace, mined, outputs, pairs, batch, hits, known_ids, budget):
    queries = [hit['query'] for hit in hits]
    ids = torch.tensor(queries, device=outputs['pred_boxes'].device)
    gate = trace[batch]
    selected = mined['pseudo'][batch]
    if (selected & ~gate['confident']).any():
        raise ValueError('Selected candidate outside the original gates')
    over_budget = gate['confident'] & ~selected & ~gate['deduplicated']
    if over_budget.any() and int(selected.sum()) != budget:
        raise ValueError('Skipped eligible query before budget exhaustion')
    matched_queries = set(pairs[batch][0].tolist())
    matches = mined['matches'][batch]
    failures = torch.stack((gate['excluded'], ~gate['valid'], ~gate['salient'],
                            matches < 0, (matches >= 0) & ~gate['consistent'],
                            (matches >= 0) & ~gate['other_salient']), dim=1)[ids].tolist()
    states = torch.stack((gate['deduplicated'], over_budget, selected, mined['background'][batch]), dim=1)[ids].tolist()
    known = outputs['pred_logits'][batch, ids][:, known_ids].sigmoid().amax(-1)
    novelty = outputs['pred_novelty'][batch, ids].sigmoid()
    margins = (novelty - known).tolist()
    for hit, flags, state, margin in zip(hits, failures, states, margins, strict=True):
        failed = [name for name, flag in zip(GATES, flags, strict=True) if flag]
        outcomes = [name for name, flag in zip(('deduplication', 'budget', 'selected'), state[:3]) if flag]
        if len(outcomes) + bool(failed) != 1:
            raise ValueError('A-OSE query does not have one native mining outcome')
        outcome = failed[0] if failed else outcomes[0]
        if outcome == 'released_gt':
            outcome = 'released_match' if hit['query'] in matched_queries else 'released_overlap'
        hit.update(gate=outcome, failed_gates=failed, background=state[3], novelty_margin=margin)


def summarize(rows):
    return dict(hits=len(rows), by_gate=dict(Counter(row['gate'] for row in rows)),
                by_failed_gates=dict(Counter('+'.join(row['failed_gates']) or 'none' for row in rows)),
                background=sum(row['background'] for row in rows),
                selected_score_conflicts=sum(row['gate'] == 'selected' and row['novelty_margin'] <= 0 for row in rows))


def evaluate(model, criterion, ema, manifest, image_root, path, plan, reference):
    known_ids = manifest['stages'][0]['classes']
    known = set(known_ids)
    threshold = manifest['evaluation']['iou_threshold']
    post = IndexedTopKnown(known_ids=known_ids, scoring='focal', max_detections=plan['limit'])
    rows = []

    def observe(dataset, samples, targets, outputs, detections, _sizes, _post):
        image_ids = [int(target['image_id']) for target in targets]
        hits = [voc_hits(detection, dataset.coco.imgToAnns[image], known, threshold)
                for image, detection in zip(image_ids, detections, strict=True)]
        if not any(hits):
            return
        # Keep the full batch for the miner's batch-level released semantic anchors.
        released = []
        for target in targets:
            keep = torch.isin(target['labels'], target['labels'].new_tensor(known_ids))
            released.append({key: target[key][keep].cuda() for key in ('labels', 'boxes')})
        pairs = criterion.matcher(outputs, released)
        teacher = ema.forward_teacher(flip_samples(samples.to('cuda')))
        trace = []
        mined = mine_candidates(outputs, released, pairs, model.known_ids, teacher, criterion.top_k,
                                criterion.saliency_threshold, criterion.consistency_threshold, gate_trace=trace)
        for batch, image_hits in enumerate(hits):
            if image_hits:
                mining_rows(trace, mined, outputs, pairs, batch, image_hits, known_ids, criterion.top_k)
                source = dataset.coco.imgs[image_ids[batch]]['owod_source_id']
                rows.extend(dict(image=source, **hit) for hit in image_hits)

    result = measure(model, manifest, image_root, path, plan['batch_size'], plan['limit'],
                     observer=observe, postprocessor=post)
    first = [row for row in rows if row['first']]
    duplicates = [row for row in rows if not row['first']]
    metrics = {key: result['metrics'][key] for key in reference['metrics']}
    if (metrics != reference['metrics'] or len(rows) != metrics['a_ose']
            or len(first) != reference['trace']['unique_unknown_targets']
            or len(duplicates) != reference['trace']['duplicate_hits']):
        raise ValueError('Query gate join failed to reconstruct the frozen author VOC result')
    small = [row for row in rows if row['size'] != '32_sq_plus']
    return dict(metrics=metrics, groups=dict(all=summarize(rows), first=summarize(first),
                                             duplicate=summarize(duplicates), small=summarize(small)), events=rows)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        root = resolve_image_root(path, manifest)
        frozen = job['arms']['refinement']
        checkpoint = Path(frozen['checkpoint'])
        if file_sha256(checkpoint) != frozen['checkpoint_sha256']:
            raise ValueError(f'Changed frozen endpoint: {protocol}')
        model, criterion, ema = frozen_miner(checkpoint, manifest)
        protocols[protocol] = {}
        for fold, entry in job['folds'].items():
            dest = OUTPUT / f'{protocol}-{fold}.json'
            if not dest.exists():
                annotations = Path(entry['path'])
                if file_sha256(annotations) != entry['sha256']:
                    raise ValueError(f'Changed frozen validation annotations: {protocol}/{fold}')
                print(f'Evaluating {protocol}/{fold} native A-OSE query gates', flush=True)
                data = evaluate(model, criterion, ema, manifest, root, annotations, plan, plan['expected'][protocol][fold])
                write_json(dest, dict(plan_sha256=digest, **data))
            data = read_json(dest)
            if data['plan_sha256'] != digest:
                raise ValueError(f'Changed frozen diagnostic plan: {dest}')
            protocols[protocol][fold] = data['groups']
        del model, criterion, ema
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
