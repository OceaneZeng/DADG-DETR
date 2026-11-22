"""Compare small-object novelty across the locked S Task 1 checkpoints."""

from functools import partial
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_prescreen import EXTENT_REPORTS
from tools.owod.probe_task1_m_s_prescreen_replacement import choose, coverage, replace_cells
from tools.owod.probe_task1_m_s_query_trace import OUTPUT as TRACE, observe
from tools.owod.probe_task1_paired_discovery import LOCKED, verified, voc_coords
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_released_small_extent import MANIFESTS, OUTPUT as PARENT, measure, size_group
from tools.owod.screening_data import source_ids, write_json


OUTPUT = PARENT.with_name('s-task1-novelty-budget')
PLAN_HASH = 'fd056b5878413cdb35917a3b5c3e3c9fefbcde55cad22ab89c0cff08a6cf3691'
TRACE_HASHES = {
    'first': 'd4f422a157017726b23148de867f81938de2038a7ac6712cc94a56ea6e12a653',
    'second': '06d025913d7c37f8d51380bcae679b12d304596911a59a9bb0416b3a6110dbaf',
}
PROTOCOL = 's-owodb'
HEADS = ('objectness', 'novelty')


def capture(released, injected, model, known, cohort, dataset, samples, targets, out, detections, sizes, post):
    objects, assignments, cells = {}, {}, {}
    for i, target in enumerate(targets):
        image_id = int(target['image_id'])
        anns = dataset.coco.imgToAnns[image_id]
        h, w = target['orig_size'].tolist()
        boxes = voc_coords(out['pred_boxes'][i], w, h)
        scores = [out['pred_'+key][i].sigmoid().tolist() for key in HEADS]
        for ann in anns:
            if ann['category_id'] in known and not ann.get('difficult') and size_group(ann) != '32_sq_plus':
                iou = overlaps(np.asarray(ann['voc_bbox']), boxes)
                best = int(iou.argmax())
                released.append(dict(annotation_id=ann['id'], iou=float(iou[best]),
                                     **{key: values[best] for key, values in zip(HEADS, scores, strict=True)}))
        objects[i] = [ann for ann in anns if ann['id'] in cohort]
        if not objects[i]:
            continue
        before = coverage(out, detections, i, target, objects[i])
        assignments[i] = []
        candidates = choose(out, i, target, objects[i], before, assignments=assignments[i])
        if len(candidates):
            cells[i] = candidates
    if cells:
        proposal = model.detr.transformer.instance_proposals
        with proposal.register_forward_hook(partial(replace_cells, replacements=cells)):
            changed = model(samples.to('cuda'))
        if not torch.equal(out['geometry']['active'], changed['geometry']['active']):
            raise ValueError('Changed native query counts')
        detect = post(changed, sizes, include_top_indices=True)
        observe(injected, known, changed, detect, targets, objects, cells, assignments)


def summarize_pairs(reports, cohort, targets):
    models = {label: {r['annotation_id']: r for r in report['injected']}
              for label, report in reports.items()}
    dense_ids = {model: set(report['native']['coverage']['dense']) for model, report in reports.items()}
    common_dense = targets.intersection(*dense_ids.values())
    common_queries = targets.intersection(*(set(r) for r in models.values()))
    summary = dict(targets=len(targets), cohort=len(cohort), common_dense=len(common_dense),
                   common_injected=len(common_queries), common_target_ids=sorted(common_queries), models={})
    for label, report in reports.items():
        common = [models[label][i] for i in sorted(common_queries)]
        known = [r for r in report['released'] if r['iou'] > .5]
        summary['models'][label] = dict(dense=len(dense_ids[label] & targets),
            injected=len(models[label].keys() & targets), score_admitted=sum(r['unknown_top50'] for r in common),
            native_small_tp=report['native']['groups']['small']['unknown_tp'],
            native_metrics=report['native']['metrics'], released_small_localized=len(known),
            median_released_novelty=median(r['novelty'] for r in known) if known else None,
            median_common_novelty=median(r['novelty'] for r in common) if common else None,
            median_common_objectness=median(r['objectness'] for r in common) if common else None,
            median_score_to_cutoff=median(r['unknown_score']/r['top50_joint_cutoff'] for r in common) if common else None)
    return summary


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090')
    parent = verified(PARENT/'plan.json', PLAN_HASH)
    mpath = MANIFESTS[PROTOCOL]
    manifest = verified(mpath, LOCKED[PROTOCOL]['manifest_sha256'])
    root = resolve_image_root(mpath, manifest)
    known = manifest['stages'][0]['classes']
    prior = {arm: verified(PARENT/f'{PROTOCOL}-{arm}.json', digest) for arm,digest in EXTENT_REPORTS.items()}
    paths = {arm: dict(path=str(PARENT/f'{PROTOCOL}/{arm}/task1/checkpoint.pth'),
                       sha256=report['checkpoint_sha256']) for arm,report in prior.items()}
    paths['parent12'] = LOCKED[PROTOCOL]['checkpoint']
    configs = {}
    config_fields = ('train_ann', 'owod_manifest_sha256', 'owod_screening_sha256', 'backbone_weights_sha256',
                     'lr', 'batch_size', 'world_size', 'seed', 'discovery_warmup', 'discovery_pseudo_ramp_epochs',
                     'discovery_scoring')
    for model, row in paths.items():
        path = Path(row['path'])
        if file_sha256(path) != row['sha256']:
            raise ValueError(f'Changed {model} checkpoint')
        configs[model] = read_json(path.with_name('run_config.json'))
        if prior.get(model, {}).get('plan_sha256', PLAN_HASH) != PLAN_HASH:
            raise ValueError(f'Changed {model} native provenance')
    for key in config_fields:
        if len({config[key] for config in configs.values()}) != 1:
            raise ValueError(f'Training provenance differs for {key}')
    if [config['epochs'] for config in configs.values()] != [3, 3, 12]:
        raise ValueError('Changed checkpoint training lengths')
    train = source_ids(read_json(Path(configs['control']['train_ann'])))
    specs = parent['protocols'][PROTOCOL]['slices']
    expected, cohorts, seen = {}, {}, set()
    for label, spec in specs.items():
        data = verified(PARENT/spec['file'], spec['sha256'])
        image_ids = source_ids(data)
        if len(image_ids) != 1024 or image_ids & (train | seen):
            raise ValueError(f'Overlapping validation images in {label}')
        seen |= image_ids
        small = {a['id'] for a in data['annotations'] if size_group(a) != '32_sq_plus'}
        before, after = (set(prior[a]['slices'][label]['coverage']['dense']) for a in ('control','extent'))
        cohorts[label] = small & (after-before)
        expected[label] = verified(TRACE/f's-extent-{label}.json', TRACE_HASHES[label])['rows']
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                original_plan=PLAN_HASH, checkpoint_inputs=paths, slices=specs, trace_hashes=TRACE_HASHES,
                epochs={name:cfg['epochs'] for name,cfg in configs.items()}, paper_comparable=False)
    pin(OUTPUT/'plan.json', plan)
    digest = file_sha256(OUTPUT/'plan.json')
    reports = {label: {} for label in specs}
    for name, row in paths.items():
        model, _ = load_detector(None, Path(row['path']), torch.device('cuda'))
        model.eval()
        for label,spec in specs.items():
            path = OUTPUT/f'{name}-{label}.json'
            if not path.exists():
                released, injected = [], []
                print(f'Evaluating {name}-{label} with native and matched proposal scoring', flush=True)
                metrics = measure(model, manifest, root, PARENT/spec['file'], parent['validation_batch_size'],
                                  parent['max_detections'],
                                  observer=partial(capture,released,injected,model,known,cohorts[label]))
                write_json(path, dict(plan_sha256=digest, native=metrics, released=released, injected=injected))
            report = read_json(path)
            assigned = {r['annotation_id'] for r in report['injected']}
            if (report['plan_sha256'] != digest or not assigned <= cohorts[label]
                    or len(assigned) != len(report['injected'])):
                raise ValueError(f'Changed frozen report {name}-{label}')
            if name in prior and report['native'] != prior[name]['slices'][label]:
                raise ValueError(f'Changed native evaluation {name}-{label}')
            if name == 'extent' and report['injected'] != expected[label]:
                raise ValueError(f'Changed short extent query trace {label}')
            reports[label][name] = report
        del model
    summaries = {label:summarize_pairs(rows, cohorts[label], {r['annotation_id'] for r in expected[label]})
                 for label, rows in reports.items()}
    write_json(OUTPUT/'comparison.json', dict(status='complete', plan_sha256=digest,
                                             paper_comparable=False, slices=summaries))


if __name__ == '__main__':
    main()
