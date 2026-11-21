"""Track the same released GT identities through Task 1 unknown and Task 2 Current VOC matches."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.graph_local.discovery import DiscoveryPostProcess
from models.owod_benchmark_eval import evaluate_author_benchmark, overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import CHECKPOINT, MANIFEST, OUTPUT as DURATION
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


SOURCE = DURATION.with_name('s-instance-task2-increment-classifier-lr')
OUTPUT = DURATION.with_name('s-instance-task2-unknown-transition')


def prepare():
    source = read_json(SOURCE/'comparison.json')
    training = read_json(SOURCE/'plan.json')
    controls = read_json(DURATION/'comparison.json')
    manifest = read_json(MANIFEST)
    annotation = Path(training['annotation'])
    data = read_json(annotation)
    order = [im['owod_source_id'] for im in data['images']]
    if (file_sha256(SOURCE/'comparison.json') != '681a62664dc4fdf70b840c613e94d7ddc498d05c1d831334a7cafcc2865ed753'
            or source['follow_up'] or source['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or file_sha256(DURATION/'comparison.json') != training['source_duration_result_sha256']
            or file_sha256(MANIFEST) != training['identity']['manifest_sha256']
            or file_sha256(annotation) != training['annotation_sha256'] or len(set(order)) != len(order) or len(order) != 1024
            or sha256('\n'.join(order).encode()).hexdigest() != training['eval_order_sha256']
            or set(order) & set(manifest['evaluation']['image_source_ids'])
            or {a['category_id'] for a in data['annotations']} != set(manifest['stages'][1]['active_classes'])
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != training['gpus'][0]
            or str(torch.__version__) != training['torch_version']):
        raise ValueError('Changed source evidence, release-only holdout, or environment')
    endpoints = {'parent':dict(path=str(CHECKPOINT), sha256=training['identity']['checkpoint_sha256'])}
    for phase in ('increment', 'finetune'):
        base = training['control'][phase]
        if base['sha256'] != controls['arms']['3epochs'][phase]['checkpoint_sha256']:
            raise ValueError('Three-epoch control differs')
        endpoints['control-'+phase] = base
        endpoints['candidate-'+phase] = dict(path=str(SOURCE/phase/'checkpoint.pth'),
                                            sha256=source['phases'][phase]['checkpoint_sha256'])
    for name, entry in endpoints.items():
        path = Path(entry['path'])
        epoch = 11 if name == 'parent' else (2 if name.endswith('increment') else 0)
        if file_sha256(path) != entry['sha256'] or read_json(path.parent/'training_complete.json')['last_epoch'] != epoch:
            raise ValueError(f'Changed checkpoint or incomplete endpoint: {name}')
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                endpoints=endpoints, protocol='s-owodb', seed=42, paper_comparable=False,
                annotation=str(annotation), annotation_sha256=training['annotation_sha256'], manifest_sha256=file_sha256(MANIFEST),
                eval_order_sha256=training['eval_order_sha256'], source_sha256=file_sha256(SOURCE/'comparison.json'),
                batch_size=2, gpu=torch.cuda.get_device_name(), device=os.environ['CUDA_VISIBLE_DEVICES'],
                torch_version=str(torch.__version__), scoring='focal', max_detections=50, ap_mode=manifest['evaluation']['ap_mode'],
                criterion='Prioritize classification of formerly discovered objects if the parent has >=50 unique '
                          'unknown VOC TPs, >=60% lack a correct Current VOC TP at both control endpoints, '
                          'and >=60% of those failures at each endpoint have a raw box but no correct Top-50 output. '
                          'Partition the finetune candidate-minus-control TP gains/losses by parent status.')
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest, order


@torch.inference_mode()
def measure(model, dataset, loader, known, previous, current, specification, parent):
    post = DiscoveryPostProcess(known_ids=known, scoring='focal', max_detections=50)
    predictions, raw_hits, output_hits = {}, set(), set()
    model.eval()
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
        native = post(outputs, sizes)
        h,w = sizes.unbind(1)
        boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])*torch.stack([w,h,w,h], -1)[:,None]
        for index,target in enumerate(targets):
            image_id = int(target['image_id'].item())
            if image_id in predictions:
                raise ValueError('Repeated image')
            det = predictions[image_id] = {k:native[index][k].cpu() for k in ('scores','boxes','labels')}
            raw = np.asarray([voc_prediction_box(b) for b in boxes[index].tolist()])
            emitted = np.asarray([voc_prediction_box(b) for b in det['boxes'].tolist()])
            labels = det['labels'].numpy()
            for ann in dataset.coco.imgToAnns.get(image_id, []):
                if ann.get('difficult',0) or ann['category_id'] not in current:
                    continue
                truth = np.asarray(ann['voc_bbox'])
                if np.any(overlaps(truth,raw) > specification['iou_threshold']):
                    raw_hits.add(ann['id'])
                eligible = emitted[labels == (91 if parent else ann['category_id'])]
                if np.any(overlaps(truth,eligible) > specification['iou_threshold']):
                    output_hits.add(ann['id'])
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches, {len(predictions)} images', flush=True)
    metrics = evaluate_author_benchmark(predictions, dataset.coco, known, [] if parent else previous,
                                        previous if parent else current, specification, include_current_stats=True,
                                        include_matched_ids=True)
    matched = metrics.pop('matched_annotation_ids')
    hits = set(matched[-1]) if parent else {i for c in current for i in matched[c]}
    expected = metrics['unknown_tp'] if parent else sum(c['tp'] for c in metrics['current_class_stats'].values())
    if len(hits) != expected or not hits <= output_hits <= raw_hits:
        raise ValueError('Raw geometry, Top-50 or VOC match identities disagree')
    return dict(metrics=metrics, raw=sorted(raw_hits), emitted=sorted(output_hits), matched=sorted(hits))


def summarize(views, universe):
    parent = set(views['parent']['matched'])
    raw = set(views['parent']['raw'])
    groups = dict(parent_unknown=parent, parent_box_only=raw-parent, parent_no_box=universe-raw)
    rows = {}
    for arm, v in views.items():
        if arm == 'parent':
            continue
        covered, emitted, matched = (set(v[k]) for k in ('raw','emitted','matched'))
        rows[arm] = {g:dict(gt=len(ids), correct=len(ids & matched), emitted_unmatched=len(ids & (emitted-matched)),
                            box_without_correct_output=len(ids & (covered-emitted)), no_box=len(ids-covered))
                     for g,ids in groups.items()}
    candidate, baseline = (set(views[a+'-finetune']['matched']) for a in ('candidate','control'))
    changes = {g:dict(gained=len(ids & (candidate-baseline)), lost=len(ids & (baseline-candidate)))
               for g,ids in groups.items()}
    criteria = []
    for phase in ('increment','finetune'):
        row = rows['control-'+phase]['parent_unknown']
        failed = row['gt']-row['correct']
        criteria.append(failed > 0 and failed >= .6*row['gt'] and row['box_without_correct_output'] >= .6*failed)
    return dict(parent_unknown_tp=len(parent), groups=rows, finetune_changes=changes,
                prioritize_classification=len(parent)>=50 and all(criteria))


def main():
    plan, manifest, order = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), 2, 2)
    old, current = (s['classes'] for s in manifest['stages'][:2])
    universe = {ann['id'] for ann in ds.coco.anns.values() if ann['category_id'] in current and not ann.get('difficult',0)}
    spec = dict(manifest['evaluation'], image_source_ids=order)
    baseline = read_json(DURATION.with_name('s-instance-task2-current-phase')/'comparison.json')['arms']['3epochs']
    trained = read_json(SOURCE/'comparison.json')['phases']
    views = {}
    for arm, entry in plan['endpoints'].items():
        path = OUTPUT/f'{arm}.json'
        parent = arm == 'parent'
        if not path.exists():
            print(f'Evaluating {arm} on 1,024 release-only images', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            view = measure(model, ds, loader, old if parent else manifest['stages'][1]['active_classes'],
                           old, current, spec, parent)
            write_json(path, dict(plan_sha256=digest, checkpoint_sha256=entry['sha256'], view=view))
            del model
        result = read_json(path)
        if result['plan_sha256'] != digest or result['checkpoint_sha256'] != entry['sha256']:
            raise ValueError(f'Changed replay: {arm}')
        view = views[arm] = result['view']
        if not set(view['raw']) <= universe:
            raise ValueError('Unexpected ground-truth identity')
        if parent:
            if view['metrics']['unknown_gt'] != len(universe):
                raise ValueError('Parent unknown evaluation must be limited to the newly released classes')
        else:
            group,phase = arm.split('-')
            ref = baseline[phase] if group=='control' else trained[phase]['heldout']
            if any(view['metrics'][k] != ref[k] for k in ('previous_ap50','current_ap50','known_ap50')):
                raise ValueError(f'Native AP did not replay: {arm}')
            if group == 'control':
                archived = read_json(DURATION.with_name('s-instance-task2-current-phase')/f'3epochs-{phase}.json')['view']['native']
            else:
                archived = read_json(SOURCE/f'{arm}-heldout.json')['views']['released']
            for label, row in view['metrics']['current_class_stats'].items():
                if any(row[k] != archived['current_class_stats'][label][k] for k in ('gt','tp','fp','ap50')):
                    raise ValueError(f'VOC replay differs for {arm}, class {label}')
    summary = summarize(views, universe)
    if sum(v['gained']-v['lost'] for v in summary['finetune_changes'].values()) != 32:
        raise ValueError('Changes must reproduce the archived 325 minus 293 Current TPs')
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                              gt=len(universe), **summary))
    print('Unknown-to-Current transitions saved:', OUTPUT/'comparison.json', flush=True)


if __name__ == '__main__':
    main()
