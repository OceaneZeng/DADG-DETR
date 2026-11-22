"""Screen one Task 2 focal-loss change against the pinned three-epoch M/S controls."""

import argparse
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from datasets import build_dataset
from main import get_args_parser
from models.deformable_detr import PREVIOUS_HARD_NEGATIVE_MAX_IOU, PREVIOUS_HARD_NEGATIVE_TOPK
from tools.owod import run_instance_pilot as pilot
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector, move_targets
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.probe_task2_m_s_current_fp_open_set import OUTPUT as FULL
from tools.owod.probe_task2_m_s_current_score import measure
from tools.owod.probe_task2_m_s_shared_current import MANIFESTS, OUTPUT as SOURCE, endpoints
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task2_finetune_half_lr import optimizer_groups
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.screening_data import screening_files, source_ids, write_json
from tools.owod.strict_setting import configure_run
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import collate_fn


OUTPUT = SOURCE.with_name('task2-m-s-previous-hard-negatives')
BOX_OUTPUT = SOURCE.with_name('task2-m-s-current-low-iou-boxes')
BLEND_SHA = '9c7fa2e4b524568932e32c0de5a305fb4c91436fbe6e062c63472b67ac12fa98'
SOURCE_SHA = 'de4bb126945143a69b5a60e60b83b7761c6270e6e71734fe127129948fd35712'
FULL_SHA = '3d280049a0969d3634dc5ef861e6205975fb54c54d4345b3d82990d70bc8dafd'
FLAG = '--screen-double-previous-hard-negatives'
METRICS = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'wi', 'a_ose')
DEVICE = torch.device('cuda')
GATES = dict(previous_ap50_at_least=-.0005, current_ap50_at_least=.0005,
             known_ap50_at_least=0., net_current_voc_matches_at_least=1)


def prepare(output=OUTPUT, low_iou=False):
    source = verified(SOURCE / 'plan.json', SOURCE_SHA)
    full = verified(FULL / 'plan.json', FULL_SHA)
    if full['parent_plan'] != SOURCE_SHA:
        raise ValueError('Full annotations use a different cohort plan')
    if low_iou:
        blend = verified(OUTPUT.with_name('task2-m-s-native-box-blend') / 'comparison.json', BLEND_SHA)
        if blend['status'] != 'complete' or blend['follow_up']:
            raise ValueError('Expected the completed fixed-box comparison')
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if devices != ['NVIDIA GeForce RTX 3090'] * 2:
        raise ValueError('Use two RTX 3090s for the matched finetune')

    jobs = {}
    for protocol, entry in source['protocols'].items():
        manifest_path = MANIFESTS[protocol]
        manifest = verified(manifest_path, entry['manifest_sha256'])
        locked = endpoints(protocol)
        if locked != entry['endpoints']:
            raise ValueError(f'Changed control endpoints: {protocol}')
        folder = Path(locked['finetune']['path']).parent
        inputs = read_json(folder / 'input.json')
        command = inputs['command'].copy()
        command[command.index('--output_dir') + 1] = str((output / protocol / 'finetune').resolve())
        command.append('--screen-double-current-low-iou-boxes' if low_iou else FLAG)
        argv = next(i for i, token in enumerate(command) if Path(token).name == 'main.py') + 1
        args = get_args_parser().parse_args(command[argv:])
        configure_run(args)
        screen = Path(args.owod_screening)
        trained = set(manifest['evaluation']['image_source_ids'])
        trained.update(read_json(screen)['heldout_source_ids'])
        for stage, keys in ((0, ('increment_train', 'full_val')), (1, ('increment_train', 'train', 'full_val'))):
            _, paths = screening_files(manifest_path, screen, stage)
            for key in keys:
                trained.update(source_ids(read_json(paths[key])))
        folds = {}
        seen = set()
        for fold, cohort in source['cohorts'].items():
            label = f'{protocol}-{fold}'
            released = cohort['by_protocol'][protocol]
            scored = full['reports'][label]
            order = cohort['order']
            names = set(order)
            released_data = verified(SOURCE / released['file'], released['sha256'])
            full_data = verified(FULL / scored['file'], scored['sha256'])
            if (len(names) != len(order) or names & (seen | trained)
                    or source_ids(released_data) != names or source_ids(full_data) != names
                    or not names <= set(manifest['splits']['t2_train'])):
                raise ValueError(f'Unexpected overlap or source for {label}')
            seen.update(names)
            folds[fold] = dict(order=order, released=released, scored=scored)
        if (args.owod_screening_sha256 != entry['screen_sha256']
                or inputs['screen_sha256'] != entry['screen_sha256']
                or file_sha256(Path(args.graph_gnn_checkpoint)) != inputs['--graph-gnn-checkpoint']
                or file_sha256(Path(args.pretrained)) != locked['increment']['sha256']):
            raise ValueError(f'Changed predecessor or screening inputs: {protocol}')
        jobs[protocol] = dict(command=command, endpoints=locked, folds=folds,
                              train_sha256=file_sha256(Path(args.train_ann)),
                              previous=entry['previous'], current=entry['current'],
                              manifest_sha256=entry['manifest_sha256'], screen_sha256=entry['screen_sha256'])

    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                sources=dict(cohorts=SOURCE_SHA, full_annotations=FULL_SHA), paper_comparable=False,
                devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpus=devices, torch_version=str(torch.__version__),
                train_batch_per_gpu=2, eval_batch=2, seed=42, epochs=1, max_detections=50,
                selection=dict(topk=PREVIOUS_HARD_NEGATIVE_TOPK, max_iou=PREVIOUS_HARD_NEGATIVE_MAX_IOU,
                               focal_term_multiplier=2, score='previous-class logit'),
                gates=GATES, future_annotations_for_reporting_only=True, jobs=jobs)
    if low_iou:
        plan.update(source_box_blend_sha256=BLEND_SHA,
                    selection=dict(max_iou=.5, box_term_multiplier=2, category_group='current',
                                   terms=['loss_bbox', 'loss_giou'], normalization='original matched-GT denominator'))
    pin(output / 'plan.json', plan)
    return plan


def low_iou_gradients(base, model, loader, job, args):
    current = torch.tensor(job['current'], device=DEVICE)
    known = job['previous'] + job['current']
    totals = [dict(weighted_terms=0, protected_matches=0) for _ in range(args.dec_layers)]
    image_ids = []
    for samples, targets in islice(loader, 8):
        targets = move_targets(targets, DEVICE)
        if list(base.active_class_ids) != known or any(not set(t['labels'].tolist()) <= set(known) for t in targets):
            raise ValueError('Unexpected released class supervision')
        image_ids.extend(int(t['image_id']) for t in targets)
        with torch.no_grad():
            outputs = model(samples.to(DEVICE))
        layers = [*outputs['aux_outputs'], outputs]
        num_boxes = max(sum(len(t['labels']) for t in targets), 1)
        for layer, row in zip(layers, totals, strict=True):
            indices = base.matcher(layer, targets)
            pred_ids = base._get_src_permutation_idx(indices)
            labels = torch.cat([t['labels'][idx] for t, (_, idx) in zip(targets, indices, strict=True)])
            targets_box = torch.cat([t['boxes'][idx] for t, (_, idx) in zip(targets, indices, strict=True)])
            boxes = layer['pred_boxes'].detach().requires_grad_()
            iou = box_iou(box_cxcywh_to_xyxy(boxes[pred_ids].detach()), box_cxcywh_to_xyxy(targets_box))[0].diag()
            expected = torch.isin(labels, current) & (iou <= .5)
            mask = torch.zeros(boxes.shape[:2], dtype=torch.bool, device=DEVICE)
            mask[pred_ids] = expected
            base.current_low_iou_box_ids = None
            before = base.loss_boxes(dict(pred_boxes=boxes), targets, indices, num_boxes)
            base.current_low_iou_box_ids = current
            after = base.loss_boxes(dict(pred_boxes=boxes), targets, indices, num_boxes)
            for term in ('loss_bbox', 'loss_giou'):
                before_grad, = torch.autograd.grad(before[term], boxes, retain_graph=True)
                after_grad, = torch.autograd.grad(after[term], boxes, retain_graph=True)
                torch.testing.assert_close(after_grad, before_grad * (1 + mask[..., None]), rtol=1e-6, atol=1e-7)
            weighted = int(after['current_low_iou_box_matches'])
            if weighted != int(expected.sum()):
                raise ValueError('Unexpected low-IoU box mask count')
            row['weighted_terms'] += weighted
            row['protected_matches'] += len(expected) - weighted
        if all(row['weighted_terms'] and row['protected_matches'] for row in totals):
            return dict(train_annotation_sha256=job['train_sha256'], image_ids=image_ids, layers=totals)
    raise ValueError('Eight batches did not exercise weighted and protected boxes in each decoder layer')


def probe(protocol, job, low_iou=False):
    command = job['command']
    argv = next(i for i, token in enumerate(command) if Path(token).name == 'main.py') + 1
    args = get_args_parser().parse_args(command[argv:])
    configure_run(args)
    dataset = build_dataset('train', args)
    torch.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_fn)
    model, criterion = load_detector(None, Path(job['endpoints']['increment']['path']), DEVICE)
    model.eval()
    base = criterion.base.train()
    if low_iou:
        return low_iou_gradients(base, model, loader, job, args)
    samples, targets = next(iter(loader))
    targets = move_targets(targets, DEVICE)
    known = job['previous'] + job['current']
    if (list(base.active_class_ids) != known
            or any(not set(t['labels'].tolist()) <= set(known) for t in targets)):
        raise ValueError(f'Unexpected target classes or classification columns: {protocol}')
    with torch.no_grad():
        outputs = model(samples.to(DEVICE))
    layers = [*outputs['aux_outputs'], outputs]
    matches = [base.matcher(layer, targets) for layer in layers]
    num_boxes = max(sum(len(t['labels']) for t in targets), 1)
    layer_inputs = [dict(pred_logits=layer['pred_logits'].detach().requires_grad_(),
                         pred_boxes=layer['pred_boxes']) for layer in layers]

    def losses_and_gradients():
        results = []
        for layer, indices in zip(layer_inputs, matches, strict=True):
            losses = base.loss_labels(layer, targets, indices, num_boxes, log=False)
            gradient, = torch.autograd.grad(losses['loss_ce'], layer['pred_logits'])
            results.append((losses, gradient))
        return results

    control = losses_and_gradients()
    base.enable_previous_hard_negatives(job['previous'])
    base.to(DEVICE)
    weighted = losses_and_gradients()
    records = []
    for layer, indices, (_, before), (losses, after) in zip(layer_inputs, matches, control, weighted, strict=True):
        mask = base.previous_hard_negative_mask(
            layer['pred_logits'][..., known], layer['pred_boxes'], targets, indices, None)
        expanded = torch.zeros_like(before, dtype=torch.bool)
        expanded[..., known] = mask
        protected = sum(len(pred) for pred, _ in indices)
        terms = int(losses['previous_hard_negative_terms'])
        if (terms <= 0 or protected <= 0 or terms != int(expanded.sum())
                or any(bool(expanded[i, pred].any()) for i, (pred, _) in enumerate(indices))
                or not bool((before[expanded] > 0).any())
                or not torch.equal(after[~expanded], before[~expanded])
                or not torch.equal(after[expanded], 2 * before[expanded])):
            raise ValueError(f'Unexpected focal gradient or matched-query weighting: {protocol}/{len(records)}')
        records.append(dict(weighted_terms=terms, protected_matches=protected))
    if len(records) != args.dec_layers:
        raise ValueError(f'Unexpected decoder depth: {protocol}')
    return dict(train_annotation_sha256=job['train_sha256'], image_ids=[int(t['image_id']) for t in targets],
                layers=records)


@torch.inference_mode()
def evaluate(checkpoint, root, manifest, job, plan, previous_source=None):
    model, _ = load_detector(None, checkpoint, DEVICE)
    model.eval()
    reports = {}
    previous, current = job['previous'], job['current']
    if previous_source is not None:
        donor, _ = load_detector(None, previous_source, DEVICE)
        source, target = donor.detr.class_embed[-1], model.detr.class_embed[-1]
        if not isinstance(source, torch.nn.Linear) or not isinstance(target, torch.nn.Linear):
            raise ValueError('Expected linear final classifiers for the old-class row swap')
        rows = torch.tensor(previous, device=target.weight.device)
        target.weight.index_copy_(0, rows, source.weight.index_select(0, rows))
        target.bias.index_copy_(0, rows, source.bias.index_select(0, rows))
        del donor, source
    for fold, cohort in job['folds'].items():
        print(f'Evaluating {checkpoint.parent.parent.name}/{fold}', flush=True)
        released = SOURCE / cohort['released']['file']
        scored = FULL / cohort['scored']['file']
        if file_sha256(released) != cohort['released']['sha256'] or file_sha256(scored) != cohort['scored']['sha256']:
            raise ValueError(f'Changed frozen evaluation annotations: {fold}')
        dataset, loader = make_loader(root, released, plan['eval_batch'], 2)
        coco = COCO(str(scored))
        spec = dict(manifest['evaluation'], image_source_ids=cohort['order'])
        reports[fold] = measure(model, dataset, loader, previous + current, previous, current,
                                spec, plan['max_detections'], evaluation_coco=coco,
                                include_perturbed=False, audit_previous_matches=True)
    return reports


def compare(control, candidate):
    before, after = (r['views']['native'] for r in (control, candidate))
    delta = {k: after['metrics'][k] - before['metrics'][k] for k in METRICS}
    old_matches = [{ann for ids in r['previous_matches']['native'].values() for ann in ids}
                   for r in (control, candidate)]
    new_matches = [set(r['matched']) for r in (before, after)]
    changes = {label: dict(gained=len(b - a), lost=len(a - b))
               for label, (a, b) in (('previous', old_matches), ('current', new_matches))}
    passed = (delta['previous_ap50'] >= GATES['previous_ap50_at_least']
              and delta['current_ap50'] >= GATES['current_ap50_at_least']
              and delta['known_ap50'] >= GATES['known_ap50_at_least']
              and len(new_matches[1]) - len(new_matches[0]) >= GATES['net_current_voc_matches_at_least'])
    return dict(delta=delta, top50_voc_matches=changes, passed=passed)


def main(low_iou=False):
    output = BOX_OUTPUT if low_iou else OUTPUT
    name = 'current low-IoU boxes' if low_iou else 'previous-class hard negatives'
    count_name = 'current_low_iou_box_matches' if low_iou else 'previous_hard_negative_terms'
    plan = prepare(output, low_iou)
    digest = file_sha256(output / 'plan.json')
    mechanics = output / 'mechanics.json'
    if not mechanics.exists():
        probes = {}
        for protocol, job in plan['jobs'].items():
            path = MANIFESTS[protocol]
            pilot.run([sys.executable, str(pilot.ROOT / 'tools/owod/manage_data.py'), 'verify',
                       '--benchmark', protocol, '--output-root', str(path.parent.parent)])
            print(f'Checking {name} gradients on {protocol} released finetune targets', flush=True)
            probes[protocol] = probe(protocol, job, low_iou)
        write_json(mechanics, dict(plan_sha256=digest, protocols=probes))
    if read_json(mechanics)['plan_sha256'] != digest:
        raise ValueError('Changed mechanism checks')

    summaries = {}
    for protocol, job in plan['jobs'].items():
        marker = output / f'{protocol}.json'
        if not marker.exists():
            print(f'Training {protocol} {name}', flush=True)
            candidate, args = run_job(job['command'], job['screen_sha256'])
            control = Path(job['endpoints']['finetune']['path'])
            row = json.loads((candidate.parent / 'metrics.jsonl').read_text().splitlines()[-1])
            counts = {k: v for k, v in row.items() if k.startswith('train_' + count_name)
                      and k.endswith('_unscaled')}
            if (row['epoch'] != 0 or len(counts) != args.dec_layers or min(counts.values()) <= 0
                    or optimizer_groups(candidate.parent / 'train.log') != optimizer_groups(control.parent / 'train.log')):
                raise ValueError(f'Unexpected term counts, optimization endpoint or optimizer groups: {protocol}')
            manifest_path = MANIFESTS[protocol]
            manifest = read_json(manifest_path)
            root = resolve_image_root(manifest_path, manifest)
            before = evaluate(control, root, manifest, job, plan)
            after = evaluate(candidate, root, manifest, job, plan)
            comparisons = {fold: compare(report, after[fold]) for fold, report in before.items()}
            write_json(marker, dict(plan_sha256=digest, candidate_sha256=file_sha256(candidate),
                                    control_sha256=job['endpoints']['finetune']['sha256'],
                                    counts=counts, control=before, candidate=after, comparisons=comparisons))
        report = read_json(marker)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed protocol result: {protocol}')
        summaries[protocol] = report['comparisons']
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=summaries,
               follow_up=all(r['passed'] for folds in summaries.values() for r in folds.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-iou-boxes', action='store_true')
    main(parser.parse_args().low_iou_boxes)
