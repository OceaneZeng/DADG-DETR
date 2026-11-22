"""Locate the Previous candidates that compete with recovered Current matches."""

from collections import Counter, defaultdict
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_head_restore import (
    DISCOVERY_OUTPUT, DISCOVERY_RESULT, FFN_OUTPUT, FFN_RESULT, ROLE_OUTPUT, restored_detector,
)
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


OUTPUT = SOURCE.with_name('task2-m-s-previous-competition')
ROLE_RESULT = '7b137773982dacc07babaec3d5fe9954c9fed04fae3883b5a1985850b2ea88d2'


def selected_queries(posts, current_position, class_count, limit):
    indices = posts.flatten().topk(limit).indices
    return set((indices[indices % class_count == current_position] // class_count).tolist())


def attribution(full, hybrid, correct_queries, current_position, previous_count, limit):
    class_count = full.shape[-1]
    full_queries = selected_queries(full, current_position, class_count, limit)
    if not full_queries.intersection(correct_queries):
        raise ValueError('A registered full-restoration match has no eligible detection')
    hybrid_queries = selected_queries(hybrid, current_position, class_count, limit)
    if hybrid_queries.intersection(correct_queries):
        return 'already_emitted'
    within = hybrid.clone()
    within[correct_queries, :previous_count] = full[correct_queries, :previous_count]
    outside = full.clone()
    outside[correct_queries, :previous_count] = hybrid[correct_queries, :previous_count]
    active = [bool(selected_queries(values, current_position, class_count, limit).intersection(correct_queries))
              for values in (within, outside)]
    return {
        (True, True): 'either',
        (True, False): 'overlapping_queries_only',
        (False, True): 'other_queries_only',
        (False, False): 'both_required',
    }[tuple(active)]


@torch.inference_mode()
def replay(model, loader, wanted, previous, current, limit, threshold):
    post = DiscoveryPostProcess(known_ids=previous + current, scoring='focal', max_detections=limit)
    positions = {category: index for index, category in enumerate(previous + current)}
    rows = []
    for step, (samples, targets) in enumerate(loader, 1):
        names = [loader.dataset.coco.imgs[int(target['image_id'])]['owod_source_id'] for target in targets]
        if any(name in wanted for name in names):
            output = model(samples.to('cuda'))
            full, _, _ = post.candidate_scores(output)
            logits = output['pred_logits'].clone()
            logits[..., previous] = output['native_outputs']['pred_logits'][..., previous]
            hybrid, _, _ = post.candidate_scores(dict(output, pred_logits=logits))
            boxes = box_cxcywh_to_xyxy(output['pred_boxes'])
            for image, (name, target) in enumerate(zip(names, targets, strict=True)):
                if name not in wanted:
                    continue
                height, width = target['orig_size'].tolist()
                scaled = boxes[image] * boxes.new_tensor([width, height, width, height])
                voc_boxes = np.asarray([voc_prediction_box(box) for box in scaled.tolist()])
                for pair in wanted[name]:
                    queries = np.flatnonzero(overlaps(np.asarray(pair['voc_bbox']), voc_boxes) > threshold).tolist()
                    if not queries:
                        raise ValueError(f'No compatible query for {pair["target"]}: {name}')
                    result = attribution(full[image], hybrid[image], queries, positions[pair['category']],
                                         len(previous), limit)
                    rows.append(dict(target=pair['target'], overlap_queries=len(queries), attribution=result))
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(rows)} tracked targets', flush=True)
    expected = {pair['target'] for pairs in wanted.values() for pair in pairs}
    if len(rows) != len(expected) or {row['target'] for row in rows} != expected:
        raise ValueError('Incomplete Previous competition replay')
    return rows


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    references = {}
    for key, path, digest in (('full', FFN_OUTPUT, FFN_RESULT), ('hybrid', DISCOVERY_OUTPUT, DISCOVERY_RESULT),
                              ('fixed_counts', ROLE_OUTPUT, ROLE_RESULT)):
        result = verified(path / 'comparison.json', digest)
        if result['status'] != 'complete':
            raise ValueError(f'Missing comparison: {path}')
        references[key] = (result, verified(path / 'plan.json', result['plan_sha256']))
    source_sha = references['full'][1]['source_plan']
    if any(plan['source_plan'] != source_sha for _, plan in references.values()):
        raise ValueError('Changed paired cohort identity')
    source = verified(SOURCE / 'plan.json', source_sha)
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_plan=source_sha, full_result=FFN_RESULT, hybrid_result=DISCOVERY_RESULT,
                fixed_counts_result=ROLE_RESULT, paper_comparable=False)
    plan_path = OUTPUT / 'plan.json'
    pin(plan_path, plan)
    digest = file_sha256(plan_path)
    summaries = {}
    for protocol, info in source['protocols'].items():
        model = restored_detector(info, restore_ffn=True, current_logits=True, fixed_role_counts=True)
        manifest = verified(MANIFESTS[protocol], info['manifest_sha256'])
        root = resolve_image_root(MANIFESTS[protocol], manifest)
        summaries[protocol] = {}
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            full, hybrid = (set(references[key][0]['protocols'][protocol][fold]['recovered'])
                            for key in ('full', 'hybrid'))
            targets = full - hybrid
            pairs = verified(SOURCE / cohort['paired_file'], cohort['paired_sha256'])
            wanted = defaultdict(list)
            for pair in pairs:
                target = pair['annotation_ids'][protocol]
                if target in targets:
                    wanted[pair['image']].append(dict(pair, target=target))
            if sum(map(len, wanted.values())) != len(targets):
                raise ValueError(f'Changed registered target identities: {label}')
            path = OUTPUT / f'{label}.json'
            if not path.exists():
                annotation = cohort['by_protocol'][protocol]
                annotation_path = SOURCE / annotation['file']
                verified(annotation_path, annotation['sha256'])
                _, loader = make_loader(root, annotation_path, source['batch_size'], 2)
                print(f'Evaluating Previous competition on {protocol}/{fold}', flush=True)
                rows = replay(model, loader, wanted, info['previous'], info['current'],
                              source['max_detections'], manifest['evaluation']['iou_threshold'])
                write_json(path, dict(plan_sha256=digest, rows=rows))
            report = read_json(path)
            if report['plan_sha256'] != digest or {row['target'] for row in report['rows']} != targets:
                raise ValueError(f'Changed matched target replay: {label}')
            summaries[protocol][fold] = dict(targets=len(targets),
                                             counts=dict(Counter(row['attribution'] for row in report['rows'])))
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest,
               protocols=summaries, paper_comparable=False))


if __name__ == '__main__':
    main()
