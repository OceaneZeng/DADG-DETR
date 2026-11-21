"""Cross accurate and matched-miss encoder geometry/content on released known objects."""
from collections import Counter
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess

import torch

from models.graph_local.discovery import DiscoveryPostProcess
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_encoder_transfer import identities
from tools.owod.probe_task1_view_boxes import overlap
from tools.owod.protocol import file_sha256, read_json, resolve_image_root, stage_files
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_anchor_choice import OUTPUT as CHOICE
from tools.owod.run_s_task1_input_scale import valid_annotations
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT
from tools.owod.screening_data import screening_files, source_ids, subset, write_json
from tools.owod.validate_discovery import load_checkpoint_metadata


OUTPUT = PARENT.with_name('s-instance-task1-released-seed')
ARMS = {'sham': ('miss', 'miss'), 'geometry': ('hit', 'miss'),
        'content': ('miss', 'hit'), 'both': ('hit', 'hit')}


def prepare():
    previous = read_json(CHOICE / 'comparison.json')
    base = read_json(CHOICE / 'plan.json')
    manifest, official = stage_files(MANIFEST, 0)
    screen = PARENT / 'data/screening.json'
    _, files = screening_files(MANIFEST, screen, 0)
    checkpoint = Path(base['job']['parent'])
    if (previous['status'] != 'complete' or previous['follow_up'] or file_sha256(CHOICE / 'plan.json') != previous['plan_sha256']
            or file_sha256(MANIFEST) != base['manifest_sha256'] or file_sha256(screen) != base['screen_sha256']
            or file_sha256(files['increment_train']) != base['train_sha256'] or file_sha256(checkpoint) != base['job']['parent_sha256']
            or torch.cuda.get_device_name() != base['gpu'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed frozen parent, data or inference environment')
    known = manifest['stages'][0]['classes']
    all_train = read_json(official['increment_train'])
    excluded = source_ids(read_json(files['increment_train'])) | set(read_json(screen)['heldout_source_ids'])
    excluded.update(manifest['evaluation']['image_source_ids'])
    excluded.update(read_json(PARENT / 'selection/inputs.json')['origins'])
    pool = set(manifest['splits']['t1_train']) - excluded
    selected = sorted(pool, key=lambda s: sha256(f'50:{s}'.encode()).digest())[:2048]
    data = subset(all_train, set(selected), known)
    if len(selected) != 2048 or set(selected) != source_ids(data) or {a['category_id'] for a in data['annotations']} - set(known):
        raise ValueError('Incomplete released-only diagnostic slice')
    data_hash = sha256((json.dumps(data, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    plan = {k: base[k] for k in ('seed', 'protocol', 'job', 'gpu', 'torch_version', 'batch_size',
                                 'manifest_sha256', 'screen_sha256', 'paper_comparable')}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], parent_plan_sha256=previous['plan_sha256'],
                parent_result_sha256=file_sha256(CHOICE / 'comparison.json'), selection_seed=50, images=len(selected),
                selection_sha256=identities(selected), exclusions_sha256=identities(excluded), data_sha256=data_hash,
                rule='Released Task 1 targets only. One small object per eligible fresh image: lowest annotation ID '
                     'missed by all native final and selected encoder boxes at IoU .5, with an accurate dense cell. '
                     'Insert its best-IoU cell at the last live encoder slot. Pair a nonselected valid cell on the '
                     'same level with IoU <.1, closest raw quality logit. Cross both position embedding/center and '
                     'memory content. Keep native and all four paired outcomes; GT-assisted diagnostic only.',
                criterion='Pursue deployable proposal ranking only with at least 50 paired images, full insertion '
                          'recovering >=10 pp more class-correct target hits than sham, and no net loss of total '
                          'correct known hits on eligible images vs native. Report slot and whole-set geometry.')
    pin(OUTPUT / 'plan.json', plan)
    annotation = OUTPUT / 'released2048.json'
    if not annotation.exists():
        write_json(annotation, data)
    if file_sha256(annotation) != data_hash:
        raise ValueError('Changed released annotations')
    return plan, manifest, checkpoint, annotation


def choose(out, i, truth, annotations):
    geo = out['geometry']
    live = torch.where(geo['active'][i])[0]
    small = sorted((j for j, a in enumerate(annotations) if a['area'] < 32**2), key=lambda j: annotations[j]['id'])
    if not len(live) or not small:
        return None
    chosen = geo['chosen'][i, live]
    boxes, quality = geo['boxes'][i], geo['logits'][i]
    targets = truth[small]
    missed = (overlap(out['pred_boxes'][i], targets).amax(0) < .5) & (overlap(boxes[chosen], targets).amax(0) < .5)
    valid = torch.where(geo['valid'][i])[0]
    edges = overlap(boxes[valid], targets)
    maximum, best = edges.max(0)
    lengths = [mask[i].numel() for mask in geo['masks']]
    for col in torch.where(missed & (maximum >= .5))[0].tolist():
        hit = int(valid[best[col]])
        start = 0
        for length in lengths:
            if hit < start + length:
                break
            start += length
        matches = (valid >= start) & (valid < start + length) & (edges[:, col] < .1) & ~torch.isin(valid, chosen)
        ids = valid[matches]
        if len(ids):
            miss = int(ids[(quality[ids] - quality[hit]).abs().argmin()])
            j = small[col]
            return dict(j=j, annotation_id=annotations[j]['id'], slot=int(live[-1]), donor=int(chosen[-1]),
                        hit=hit, miss=miss, hit_iou=float(maximum[col]), miss_iou=float(overlap(boxes[miss:miss+1], targets[col:col+1])[0, 0]),
                        hit_quality=float(quality[hit]), miss_quality=float(quality[miss]))
    return None


def replace(module, inputs, output, *, choices, geometry, content):
    query, memory, center, geo = output
    for i, item in enumerate(choices):
        if item is None:
            continue
        slot = item['slot']
        if int(geo['chosen'][i, slot]) != item['donor'] or not geo['active'][i, slot]:
            raise ValueError('Changed native recipient slot')
        box = geo['boxes'][i, item[geometry]]
        query[i, slot] = module.position(box)
        center[i, slot] = box[:2]
        memory[i, slot] = geo['memory'][i, item[content]]
    return query, memory, center, geo


def outcome(out, i, detections, truth, labels, item, known):
    j, slot = item['j'], item['slot']
    overlaps = overlap(out['pred_boxes'][i], truth)
    value, assigned = overlaps.max(1)
    positions = detections['top_indices']
    queries = positions // (len(known) + 1)
    correct = (value[queries] >= .5) & (detections['labels'] == labels[assigned[queries]])
    hit_ids = assigned[queries[correct]].unique().tolist()
    layer_iou = [float(overlap(layer['pred_boxes'][i,slot:slot+1], truth[j:j+1])[0, 0]) for layer in out['aux_outputs']]
    return dict(slot_iou=layer_iou + [float(overlaps[slot, j])], any_final_iou=float(overlaps[:, j].max()),
                correct_ids=hit_ids, target_correct=j in hit_ids,
                slot_correct=bool(((queries == slot) & (assigned[queries] == j) & correct).any()))


@torch.inference_mode()
def evaluate(model, post, loader, known):
    rows, eligible = {}, 0
    totals = {arm: Counter() for arm in ('native', *ARMS)}
    module = model.detr.transformer.instance_proposals
    for step, (samples, targets) in enumerate(loader, 1):
        samples = samples.to('cuda')
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        native = model(samples)
        boxes = [t['boxes'].cuda() for t in targets]
        labels = [t['labels'].cuda() for t in targets]
        choices, batch_rows = [], []
        for i, t in enumerate(targets):
            key = int(t['image_id'].item())
            anns = valid_annotations(loader.dataset, key)
            if [a['category_id'] for a in anns] != t['labels'].tolist():
                raise ValueError('Changed released annotation order')
            choice = choose(native, i, boxes[i], anns)
            choices.append(choice)
            rows[key] = dict(choice=choice, annotation_ids=[a['id'] for a in anns], outcomes={})
            batch_rows.append(rows[key])
        if any(choices):
            eligible += sum(c is not None for c in choices)
            for arm in totals:
                if arm == 'native':
                    out = native
                else:
                    geom, content = ARMS[arm]
                    hook = partial(replace, choices=choices, geometry=geom, content=content)
                    with module.register_forward_hook(hook):
                        out = model(samples)
                dets = post(out, sizes, include_top_indices=True)
                for i, item in enumerate(choices):
                    if item is None:
                        continue
                    record = outcome(out, i, dets[i], boxes[i], labels[i], item, known)
                    batch_rows[i]['outcomes'][arm] = record
                    totals[arm].update(images=1, final_covered=int(record['any_final_iou'] >= .5),
                                       slot_covered=int(record['slot_iou'][-1] >= .5), slot_correct=int(record['slot_correct']),
                                       target_correct=int(record['target_correct']), all_correct=len(record['correct_ids']))
        if step == 1 or step % 64 == 0 or step == len(loader):
            print(f'Evaluated {step}/{len(loader)} batches, {eligible} paired images', flush=True)
    if set(rows) != set(loader.dataset.ids) or totals['native']['final_covered']:
        raise ValueError('Changed paired image population or native eligibility')
    return rows, totals, eligible


def main():
    plan, manifest, checkpoint, data = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    args = load_checkpoint_metadata(checkpoint)['args']
    model, _ = load_detector(args, checkpoint, torch.device('cuda'))
    model.eval()
    known = manifest['stages'][0]['classes']
    post = DiscoveryPostProcess(known_ids=known, max_detections=50, scoring='focal')
    _, loader = make_loader(resolve_image_root(MANIFEST, manifest), data, plan['batch_size'], 2)
    images, totals, n = evaluate(model, post, loader, known)
    write_json(OUTPUT / 'paired.json', dict(plan_sha256=digest, images=images))
    passes = (n >= 50 and 10 * (totals['both']['target_correct'] - totals['sham']['target_correct']) >= n
              and totals['both']['all_correct'] >= totals['native']['all_correct'])
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, eligible=n,
                                                rows=totals, follow_up=passes, paper_comparable=False))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
