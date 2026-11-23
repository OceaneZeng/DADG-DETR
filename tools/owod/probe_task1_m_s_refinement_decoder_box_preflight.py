"""Measure last-two-layer box averaging on released Task 1 training images."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_known_dedup import COMPETITION_OUTPUT, CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_local_saliency import SOURCE_SHA as TRAIN_REPLAY_SHA
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import OUTPUT as TRAIN_REPLAY, training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = COMPETITION_OUTPUT.with_name('task1-m-s-refinement-decoder-box-preflight')
COMPETITION_SHA = 'b3801feb4573a0b6d9586d70f5479d8be541acea665678baaf5f76f974570f67'


def prepare():
    comparison = verified(COMPETITION_OUTPUT / 'comparison.json', COMPETITION_SHA)
    plan = verified(COMPETITION_OUTPUT / 'plan.json', comparison['plan_sha256'])
    if comparison['status'] != 'complete' or comparison['passed']:
        raise ValueError('Expected the completed known/unknown competition screen')
    source = verified(TRAIN_REPLAY / 'comparison.json', TRAIN_REPLAY_SHA)
    train = verified(TRAIN_REPLAY / 'plan.json', source['plan_sha256'])
    jobs = {}
    for protocol, job in plan['jobs'].items():
        reference = read_json(TRAIN_REPLAY / f'{protocol}.json')
        if reference['plan_sha256'] != source['plan_sha256'] or job['manifest_sha256'] != train['jobs'][protocol]['manifest_sha256']:
            raise ValueError(f'Changed released image replay: {protocol}')
        data = train['jobs'][protocol]
        jobs[protocol] = dict(checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'],
                              manifest_sha256=job['manifest_sha256'], classes=data['classes'],
                              training=data['training'], training_sha256=data['training_sha256'],
                              expected_order=reference['sample_order_sha256'])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                competition_sha256=COMPETITION_SHA, training_replay_sha256=TRAIN_REPLAY_SHA,
                jobs=jobs, seed=train['seed'], images=train['images'], batch_size=train['batch_size'],
                limit=plan['limit'], paper_comparable=False, small_area=32**2,
                factor='Average the normalized boxes at decoder layers 5 and 6 by native query identity.',
                advance='Require positive released-small dense coverage in S and no decline in M. '
                        'Report class-correct Top-50 matches and the any-last-two-layer geometry upper bound.')


@torch.inference_mode()
def measure(model, known, loader, plan):
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    stats = {key: Counter() for key in ('all', 'small')}
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        layers = outputs['aux_outputs']
        if len(layers) != 5:
            raise ValueError('Expected the locked six-layer decoder')
        final = outputs['pred_boxes']
        previous = layers[-1]['pred_boxes']
        blend = final.lerp(previous, .5)
        revised = dict(outputs, pred_boxes=blend)
        unit = final.new_ones((len(targets), 2))
        native, blended = post(outputs, unit), post(revised, unit)
        for batch, target in enumerate(targets):
            labels = target['labels'].cuda()
            truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
            small = target['source_box_area'].cuda() < plan['small_area']
            dense = [(box_iou(box_cxcywh_to_xyxy(boxes[batch]), truth)[0] > .5).any(dim=0)
                     for boxes in (final, previous, blend)]
            matches = [((box_iou(row['boxes'], truth)[0] > .5) & (row['labels'][:, None] == labels)).any(dim=0)
                       for row in (native[batch], blended[batch])]
            for name, mask in (('all', torch.ones_like(small)), ('small', small)):
                original, penultimate, averaged = (x[mask] for x in dense)
                native_tp, blend_tp = (x[mask] for x in matches)
                stats[name].update(gt=int(mask.sum()), final=int(original.sum()),
                                   penultimate=int(penultimate.sum()), blend=int(averaged.sum()),
                                   either=int((original | penultimate).sum()), gained=int((averaged & ~original).sum()),
                                   lost=int((original & ~averaged).sum()), native_known_tp=int(native_tp.sum()),
                                   blend_known_tp=int(blend_tp.sum()))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images'] or stats['small']['gt'] <= 0:
        raise ValueError('Incomplete released replay or missing small targets')
    return stats


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            path = MANIFESTS[protocol]
            manifest = verified(path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed matched-count endpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(path, manifest),
                                            job, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            model, _ = load_detector(None, checkpoint, torch.device('cuda'))
            model.eval()
            print(f'Evaluating {protocol} released small-box geometry', flush=True)
            groups = measure(model, job['classes'], loader, plan)
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, groups=groups))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen plan: {dest}')
        protocols[protocol] = report['groups']
    m, s = (protocols[key]['small'] for key in ('m-owodb', 's-owodb'))
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=(m['blend'] >= m['final'] and s['blend'] > s['final'])))


if __name__ == '__main__':
    main()
