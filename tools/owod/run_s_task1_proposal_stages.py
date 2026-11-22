"""Trace S Task 1 target coverage at each step of the unmodified detector."""
from collections import Counter, defaultdict
from pathlib import Path
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_unknown_factors import best_target
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_input_scale import OUTPUT as SCALE, valid_annotations
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT
from tools.owod.screening_data import write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = SCALE.with_name('s-instance-task1-proposal-stages')


def stage_boxes(out, i, shortlist_multiplier=12):
    geo = out['geometry']
    ids = torch.where(geo['valid'][i])[0]
    scores = geo['logits'][i, ids]
    quota = geo['chosen'].shape[1]
    pool = ids[scores.topk(min(len(ids), quota * shortlist_multiplier)).indices]
    selected = geo['chosen'][i, geo['active'][i]]
    if not torch.isin(selected, pool).all():
        raise ValueError('Native selected cells do not belong to the reconstructed shortlist')
    boxes = geo['boxes'][i]
    stages = dict(anchor=geo['anchors'][i, ids], dense=boxes[ids], shortlist=boxes[pool], selected=boxes[selected])
    for depth, layer in enumerate(out['aux_outputs'], 1):
        stages[f'decoder{depth}'] = layer['pred_boxes'][i]
    stages['final'] = out['pred_boxes'][i]
    if len(out['aux_outputs']) != 5:
        raise ValueError('Expected the six-layer frozen decoder')
    return stages


def stage_coverage(out, i, truth):
    # Coverage is any-box IoU >= .5, including overlapping GT. Report the
    # baseline best-target convention separately to reconcile the parent run.
    return {name: (overlap(b, truth) >= .5).any(0).cpu().tolist() for name, b in stage_boxes(out, i).items()}


@torch.inference_mode()
def evaluate(model, post, loader, known, proxy, reference=None):
    images, totals = {}, defaultdict(Counter)
    for step, (samples, targets) in enumerate(loader, 1):
        out = model(samples.to('cuda'))
        native = post(out, torch.stack([t['orig_size'] for t in targets]).cuda(), include_top_indices=True)
        for i, target in enumerate(targets):
            key = str(target['image_id'].item())
            ref = reference[key] if reference is not None else None
            truth = target['boxes'].cuda()
            labels = target['labels'].tolist()
            anns = valid_annotations(loader.dataset, int(key))
            if labels != [a['category_id'] for a in anns]:
                raise ValueError(f'Changed annotation order on {key}')
            detections = native[i]
            unknown = detections['labels'] == 91
            positions = detections['top_indices']
            queries = positions // (len(known) + 1)
            if ref is not None and (queries[unknown].tolist() != ref['picks']['native']
                                    or positions[~unknown].tolist() != ref['known_positions']):
                raise ValueError(f'Changed native Top-50 on {key}')
            hits = stage_coverage(out, i, truth)
            final_best, emitted_best = [], []
            for group, ids, expected in [('known', known, 'known_gt'), ('proxy', proxy, 'hidden_gt')]:
                indices = [j for j, label in enumerate(labels) if label in ids]
                assigned = best_target(out['pred_boxes'][i], truth[indices]).tolist()
                if ref is not None and assigned != ref[expected]:
                    raise ValueError(f'Changed final query assignments on {key}')
                chosen = queries[unknown if group == 'proxy' else ~unknown].tolist()
                emitted_labels = detections['labels'][unknown if group == 'proxy' else ~unknown].tolist()
                emitted = {assigned[q] for q, label in zip(chosen, emitted_labels) if assigned[q] >= 0
                           and (group == 'proxy' or label == labels[indices[assigned[q]]])}
                best = set(assigned)
                final_best.extend(anns[indices[j]]['id'] for j in sorted(best - {-1}))
                emitted_best.extend(anns[indices[j]]['id'] for j in sorted(emitted))
                for local, j in enumerate(indices):
                    area = anns[j]['area']
                    size = 'small' if area < 32**2 else 'medium' if area < 96**2 else 'large'
                    counts = totals[group + '_' + size]
                    counts.update(gt=1, final_best=int(local in best), emitted_best=int(local in emitted))
                    for stage, covered in hits.items():
                        counts[stage] += int(covered[j])
                    if not hits['final'][j]:
                        if not hits['dense'][j]:
                            cause = 'no_dense_cover'
                        elif not hits['shortlist'][j]:
                            cause = 'outside_shortlist'
                        elif not hits['selected'][j]:
                            cause = 'outside_selected'
                        else:
                            cause = 'selected_then_missing'
                        counts[cause] += 1
            geo = out['geometry']
            images[key] = dict(annotation_ids=[a['id'] for a in anns], stages=hits, final_best=final_best,
                               emitted_best=emitted_best, unknown_quota=int(unknown.sum()),
                               valid_cells=int(geo['valid'][i].sum()), selected_cells=int(geo['active'][i].sum()))
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluated {step}/{len(loader)} batches', flush=True)
    if set(images) != {str(im) for im in loader.dataset.ids} or (reference is not None and images.keys() != reference.keys()):
        raise ValueError('Stage trace did not cover the full paired slice')
    return images, totals


def main():
    base = read_json(SCALE / 'plan.json')
    result = read_json(SCALE / 'comparison.json')
    data = PARENT / 'evaluation/fresh1024.json'
    checkpoint = Path(base['job']['parent'])
    reference = read_json(SCALE / 'standard.json')
    if (not file_sha256(SCALE / 'plan.json') == result['plan_sha256'] == reference['plan_sha256']
            or result['status'] != 'complete' or result['follow_up'] or file_sha256(data) != base['fresh_sha256']
            or file_sha256(checkpoint) != base['job']['parent_sha256'] or file_sha256(MANIFEST) != base['manifest_sha256']
            or torch.cuda.get_device_name() != base['gpu'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed parent scale diagnosis or inference environment')
    plan = {k: base[k] for k in ('seed', 'protocol', 'job', 'batch_size', 'gpu', 'torch_version',
                                 'fresh_sha256', 'manifest_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], parent_plan_sha256=result['plan_sha256'],
                parent_result_sha256=file_sha256(SCALE / 'comparison.json'), reference_sha256=file_sha256(SCALE / 'standard.json'),
                criterion='Descriptive on the original 800/1333 input. Attribute final-uncovered GT to disjoint '
                          'dense, shortlist, NMS plus truncation, and selected-then-missing groups. For small proxy GT, '
                          'use the largest group with at least 5 pp of all small proxy GT to motivate the next '
                          'paired control. Report same-size known targets alongside. No label-guided predictions.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    manifest = read_json(MANIFEST)
    known, proxy = [s['classes'] for s in manifest['stages'][:2]]
    args = load_checkpoint_metadata(checkpoint)['args']
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    model.eval()
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    _, loader = make_loader(resolve_image_root(MANIFEST, manifest), data, plan['batch_size'], 2)
    images, totals = evaluate(model, post, loader, known, proxy, reference['images'])
    base_counts = result['rows']['standard']['counts']
    for size in ('small', 'medium', 'large'):
        counts = totals['proxy_' + size]
        if (counts['gt'], counts['final_best'], counts['emitted_best']) != (
                base_counts[size + '_gt'], base_counts[size + '_covered'], base_counts[size + '_native']):
            raise ValueError(f'Unpaired {size} baseline')
    write_json(OUTPUT / 'stage-trace.json', dict(plan_sha256=digest, images=images))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=totals,
                                                paper_comparable=False, images=len(images)))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
