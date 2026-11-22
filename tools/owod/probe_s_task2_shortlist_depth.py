"""Test the pre-NMS encoder shortlist size at the frozen S Task 2 control endpoint."""
from functools import partial
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.owod_benchmark_eval import overlaps, voc_prediction_box
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_proposal_stages import stage_boxes
from tools.owod.run_s_task2_duration import MANIFEST, OUTPUT as DURATION
from tools.owod.run_s_task2_increment_giou import evaluate_heldout, paired_gates
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy


PARENT = DURATION.with_name('s-instance-task2-tail-affinity')
OUTPUT = DURATION.with_name('s-instance-task2-shortlist-depth')
ORIGINAL_TRACE = DURATION.with_name('s-instance-task2-proposal-stages')/'finetune.json'
STAGES = ('dense', 'shortlist', 'selected', 'final')
CRITERION = ('On each slice recover at least half of the baseline final-missed low-frequency GT that had dense '
             'coverage but were outside the native shortlist into the larger shortlist. Require a net increase '
             'in final low-frequency coverage on both slices. On each slice require Current >= +1 pp, Previous '
             '>= -1 pp, Known > 0 change, mean proxy U-Recall not lower, mean WI and summed A-OSE not higher. '
             'Report the actual per-image selected IDs and counts under the same maximum of 25 encoder queries.')


def prepare():
    src = read_json(PARENT/'plan.json')
    parent = read_json(PARENT/'comparison.json')
    cp = src['checkpoints']['control']
    if (file_sha256(PARENT/'comparison.json') != '20f259ed5b7bab280bfc7732184e4724a2c482a010ece2bc00b927f58612e56a'
            or parent['supported'] or parent['plan_sha256'] != file_sha256(PARENT/'plan.json')
            or file_sha256(Path(cp['path'])) != cp['sha256']
            or read_json(Path(cp['path']).parent/'training_complete.json')['last_epoch'] != 0
            or file_sha256(ORIGINAL_TRACE) != '3972bed3b7b71babdea3a83767e0fe30e227b8facc2681d680ebbbb657aaecca'
            or file_sha256(MANIFEST) != src['manifest_sha256'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != src['gpu']
            or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed paired geometry evidence, data or evaluation environment')
    manifest = read_json(MANIFEST)
    excluded = set(manifest['evaluation']['image_source_ids'])
    slices = {}
    for name, entry in src['slices'].items():
        annotation = Path(entry['annotation'])
        order = [im['owod_source_id'] for im in read_json(annotation)['images']]
        ref = entry['references']['control']
        native_file = PARENT/(name+'-control.json')
        native = read_json(native_file)
        if (file_sha256(annotation) != entry['annotation_sha256'] or len(set(order)) != 1024 or len(order) != 1024
                or excluded.intersection(order) or sha256('\n'.join(order).encode()).hexdigest() != entry['order_sha256']
                or file_sha256(Path(ref['path'])) != ref['sha256'] or read_json(Path(ref['path']))['checkpoint_sha256'] != cp['sha256']
                or native['plan_sha256'] != parent['plan_sha256'] or native['checkpoint_sha256'] != cp['sha256']):
            raise ValueError(f'Changed baseline reference or validation image IDs: {name}')
        excluded.update(order)
        slices[name] = dict(annotation=str(annotation), annotation_sha256=entry['annotation_sha256'],
                            order_sha256=entry['order_sha256'], heldout=ref, native_targets=str(native_file),
                            native_targets_sha256=file_sha256(native_file))
    plan = dict(protocol=src['protocol'], seed=src['seed'], paper_comparable=False,
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                device='0', gpu=src['gpu'], torch_version=src['torch_version'], batch_size=2, slices=slices,
                checkpoint=cp, manifest_sha256=src['manifest_sha256'], parent_comparison_sha256=file_sha256(PARENT/'comparison.json'),
                tail_class_ids=src['tail_class_ids'], multipliers={'native':12, 'expanded':24}, max_encoder=25,
                decoder_queries=100, max_detections=50, original_trace_sha256=file_sha256(ORIGINAL_TRACE), criterion=CRITERION)
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest


def record_geometry(hits, choices, coco, tail, multiplier, threshold, outputs, targets, batch_images, score_tables):
    geo = outputs['geometry']
    if outputs['pred_boxes'].shape[1] != 100 or geo['chosen'].shape[1] != 25:
        raise ValueError('Changed query budget')
    for i, image_id in batch_images.items():
        key = coco.imgs[image_id]['owod_source_id']
        choices[key] = geo['chosen'][i, geo['active'][i]].cpu().tolist()
        anns = [a for a in coco.imgToAnns.get(image_id, []) if a['category_id'] in tail and not a.get('difficult')]
        if not anns:
            continue
        h, w = targets[i]['orig_size'].tolist()
        boxes = stage_boxes(outputs, i, shortlist_multiplier=multiplier)
        for stage in STAGES:
            scaled = box_cxcywh_to_xyxy(boxes[stage]) * boxes[stage].new_tensor([w,h,w,h])
            if stage == 'dense':
                # Use the same vectorized pixel conversion as the archived dense proposal trace.
                voc = scaled.cpu().numpy().astype(np.float64)
                voc[:, :2] += 1
                np.round(voc, decimals=1, out=voc)
            else:
                voc = np.asarray([voc_prediction_box(b) for b in scaled.cpu().tolist()])
            for ann in anns:
                if np.any(overlaps(np.asarray(ann['voc_bbox']),voc) > threshold):
                    hits[stage].add(ann['id'])


def compare(native, expanded, source, historical_stages=None):
    target_rows = read_json(Path(source['native_targets']))['rows']
    by_id = {r['annotation_id']: r for r in target_rows}
    n = {k:set(v) for k,v in native['hits'].items()}
    e = {k:set(v) for k,v in expanded['hits'].items()}
    if n['final'] != {i for i,r in by_id.items() if r['best_iou'] > .5} or n['dense'] != e['dense']:
        raise ValueError('Frozen final identities or dense encoder geometry changed')
    for row in (n,e):
        if not row['selected'] <= row['shortlist'] <= row['dense'] <= by_id.keys():
            raise ValueError('Invalid encoder coverage sets')
    if historical_stages is not None and any(n[k] != set(historical_stages[k]) & by_id.keys() for k in STAGES):
        raise ValueError('Native stage replay differs from the earlier S geometry trace')
    previous = read_json(Path(source['heldout']['path']))['views']
    if native['views'] != previous:
        raise ValueError('Native full-class or masked-fold benchmark replay changed')
    a, b = native['choices'], expanded['choices']
    if a.keys()!=b.keys() or len(a)!=1024:
        raise ValueError('Incomplete per-image encoder choices')
    missing = n['dense'] - n['shortlist'] - n['final']
    recovered = missing & e['shortlist']
    delta = {k:expanded['views']['released']['metrics'][k]-native['views']['released']['metrics'][k]
             for k in ('previous_ap50','current_ap50','known_ap50')}
    folds, known_pass, open_pass = paired_gates(expanded['views'],native['views'],delta)
    geom_pass = bool(missing and len(recovered)*2>=len(missing) and len(e['final'])>len(n['final']))
    return dict(native_covered={k:len(v) for k,v in n.items()}, expanded_covered={k:len(v) for k,v in e.items()},
                missing_before_shortlist=sorted(missing), recovered_in_shortlist=sorted(recovered),
                gained_final=sorted(e['final']-n['final']), lost_final=sorted(n['final']-e['final']),
                images_with_changed_encoder_queries=sum(a[k]!=b[k] for k in a),
                images_with_changed_encoder_count=sum(len(a[k])!=len(b[k]) for k in a),
                native_at_capacity=sum(len(ids)==25 for ids in a.values()),
                changed_at_capacity=sum(len(a[k])==25 and a[k]!=b[k] for k in a),
                total_encoder_queries=[sum(map(len,choices.values())) for choices in (a,b)], delta=delta, proxy_deltas=folds,
                geometry_pass=geom_pass, known_pass=known_pass, open_pass=open_pass, follow_up=geom_pass and known_pass and open_pass)


def main():
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    model, _ = load_detector(None,Path(plan['checkpoint']['path']),torch.device('cuda'))
    proposal = model.detr.transformer.instance_proposals
    if proposal.shortlist_multiplier!=12 or proposal.rank_within_level or proposal.match_encoder_counts:
        raise ValueError('Expected the original global proposal ranking')
    trace = read_json(ORIGINAL_TRACE)
    if trace['checkpoint_sha256'] != plan['checkpoint']['sha256']:
        raise ValueError('Expected the same native geometry checkpoint')
    summaries = {}
    for name, entry in plan['slices'].items():
        ann = Path(entry['annotation'])
        order = [im['owod_source_id'] for im in read_json(ann)['images']]
        ds, loader = make_loader(resolve_image_root(MANIFEST,manifest),ann,2,2)
        arms = {}
        for arm, multiplier in plan['multipliers'].items():
            path = OUTPUT/(name+'-'+arm+'.json')
            if not path.exists():
                print(f'Evaluating {name}-{arm} on {len(ds)} images',flush=True)
                proposal.shortlist_multiplier = multiplier
                hits, choices = {k:set() for k in STAGES}, {}
                observer = partial(record_geometry,hits,choices,ds.coco,set(plan['tail_class_ids']),multiplier,
                                   manifest['evaluation']['iou_threshold'])
                views = evaluate_heldout(model,ds,loader,manifest,order,proxies=True,rank_observer=observer)
                write_json(path,dict(views=views,hits={k:sorted(v) for k,v in hits.items()},choices=choices,
                                     plan_sha256=digest,checkpoint_sha256=plan['checkpoint']['sha256'],multiplier=multiplier))
            arms[arm] = read_json(path)
            if (arms[arm]['plan_sha256']!=digest or arms[arm]['checkpoint_sha256']!=plan['checkpoint']['sha256']
                    or arms[arm]['multiplier']!=multiplier):
                raise ValueError(f'Changed output: {path}')
        summaries[name] = compare(arms['native'],arms['expanded'],entry,trace['hits'] if name=='original' else None)
    write_json(OUTPUT/'comparison.json',dict(status='complete',paper_comparable=False,plan_sha256=digest,
               slices=summaries,follow_up=all(v['follow_up'] for v in summaries.values())))
    print('Encoder shortlist comparison saved',flush=True)


if __name__ == '__main__':
    main()
