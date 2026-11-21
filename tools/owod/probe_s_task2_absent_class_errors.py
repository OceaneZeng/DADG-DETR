"""Audit high-ranked unresolved Current errors on images without that released label."""
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_finetune_half_lr import OUTPUT as HALF
from tools.owod.screening_data import write_json


SOURCE = HALF.with_name('s-instance-task2-finetune-current-positive')
OUTPUT = HALF.with_name('s-instance-task2-absent-class-errors')


def analyze(view, reference):
    if view['metrics'] != reference['metrics'] or view['composition'] != reference['composition']:
        raise ValueError('Native AP or prediction composition differs from the paired study')
    breakdown = []
    for label, row in view['current_class_stats'].items():
        families = row['fp_sources']
        audit = families['absent_unresolved']
        base = reference['current_class_stats'][label]
        if (any(row[k] != base[k] for k in ('ap50', 'gt', 'tp', 'fp'))
                or any(families[k] != base['fp_sources'][k] for k in ('families', 'confused_with'))
                or not 0 <= audit['images'] <= audit['eligible_fp'] <= families['families']['unresolved']['fp']):
            raise ValueError(f'Changed VOC matches or FP attribution for class {label}')
        breakdown.append(audit)
    highest = mean(x['highest_ap50'] for x in breakdown)
    random = mean(x['random_ap50_mean'] for x in breakdown)
    chair = view['current_class_stats']['62']
    return dict(eligible_fp=sum(x['eligible_fp'] for x in breakdown),
                removed=sum(x['images'] for x in breakdown), highest_ap50=highest, random_ap50_mean=random,
                chair=dict(native_ap50=chair['ap50'], fp=chair['fp'], **chair['fp_sources']['absent_unresolved']),
                highest_gain=highest-view['metrics']['current_ap50'], highest_minus_random=highest-random)


def main():
    print('Checking matched positive-loss study and frozen checkpoints', flush=True)
    source = read_json(SOURCE/'plan.json')
    result = read_json(SOURCE/'comparison.json')
    manifest = read_json(MANIFEST)
    annotation = Path(source['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    parent_hash = file_sha256(SOURCE/'comparison.json')
    if (parent_hash != '8a9eb86b158f03c038c19df4da4a63069ea4700b9411cbe7018772cf2c17ccf5'
            or result['status'] != 'complete' or result['follow_up'] or result['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or result['fp_deltas_vs_half']['unresolved'] != 2406 or file_sha256(annotation) != source['annotation_sha256']
            or file_sha256(MANIFEST) != source['identity']['manifest_sha256'] or len(order) != 1024 or len(set(order)) != 1024
            or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != source['eval_order_sha256']
            or os.environ['CUDA_VISIBLE_DEVICES'] != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source['gpus'][0] or str(torch.__version__) != source['torch_version']):
        raise ValueError('Changed parent evidence, held-out data or evaluation environment')
    refs = {'half': HALF.with_name('s-instance-task2-step-classifier')/'half-half.json',
            'positive': SOURCE/'candidate-finetune-heldout.json'}
    endpoints = {}
    for arm, folder in [('half', HALF), ('positive', SOURCE)]:
        comparison = read_json(folder/'comparison.json')
        checkpoint = folder/'finetune/checkpoint.pth'
        cp_hash = file_sha256(checkpoint)
        ref_hash = file_sha256(refs[arm])
        expected = source['half_reference_sha256'] if arm == 'half' else '71bc081ad85b5ec24a90d4953ca85dfae6b119d1421e9cd51c8dfab83ff3e179'
        if (ref_hash != expected or cp_hash != comparison['checkpoint_sha256']
                or comparison['plan_sha256'] != file_sha256(folder/'plan.json')
                or read_json(checkpoint.parent/'input.json')['--pretrained'] != source['increment_sha256']
                or read_json(checkpoint.parent/'training_complete.json')['last_epoch'] != 0):
            raise ValueError(f'Changed trained endpoint: {arm}')
        endpoints[arm] = dict(path=str(checkpoint), sha256=cp_hash, reference=str(refs[arm]), reference_sha256=ref_hash)
    plan = {k: source[k] for k in ('annotation','annotation_sha256','eval_order_sha256','protocol','paper_comparable','seed','increment_sha256')}
    plan.update(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_comparison_sha256=parent_hash, manifest_sha256=file_sha256(MANIFEST), endpoints=endpoints,
                torch_version=str(torch.__version__), device='0', gpu=torch.cuda.get_device_name(), batch_size=2,
                scoring='focal', max_detections=50, random_draws=32, random_seed=42,
                criterion='For each label-image with no released annotation of that label and at least one unresolved FP, '
                          'remove the highest-ranked such FP or one uniform such FP (32 fixed draws). Keep native VOC ranks '
                          'and TP assignments. At both endpoints require highest removal to gain >=1 pp Current AP50 over native '
                          'and >=0.5 pp over equal-count random removal. Require >=half of the net 2406 extra unresolved FPs '
                          'to be on class-absent images. Report chair separately. No deployment of annotation filters.')
    pin(OUTPUT/'plan.json',plan)
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST,manifest), annotation, 2, 2)
    prev, cur = manifest['stages'][:2]
    spec = dict(manifest['evaluation'], image_source_ids=order, diagnostic_current_fp_sources=True,
                diagnostic_absent_unresolved=True, absent_random_draws=plan['random_draws'], absent_random_seed=plan['random_seed'])
    arms = {}
    for arm, entry in endpoints.items():
        out = OUTPUT/(arm+'.json')
        if not out.exists():
            print(f'Evaluating {arm} on {len(ds)} images', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            views = evaluate_views(model,loader,ds.coco,{'native':(cur['active_classes'],'focal',50)},spec,
                                   prev['classes'],cur['classes'],'cuda',include_current_stats=True,print_freq=64)
            write_json(out,dict(plan_sha256=digest,checkpoint_sha256=entry['sha256'],view=views['native']))
            del model
        row = read_json(out)
        if row['plan_sha256'] != digest or row['checkpoint_sha256'] != entry['sha256']:
            raise ValueError(f'Changed frozen replay: {arm}')
        arms[arm] = analyze(row['view'],read_json(refs[arm])['views']['released'])
    extra = arms['positive']['eligible_fp'] - arms['half']['eligible_fp']
    rank_pass = all(x['highest_gain'] >= .01 and x['highest_minus_random'] >= .005 for x in arms.values())
    share_pass = extra * 2 >= result['fp_deltas_vs_half']['unresolved']
    write_json(OUTPUT/'comparison.json', dict(status='complete',paper_comparable=False,plan_sha256=digest,
        arms=arms,extra_absent_unresolved=extra,rank_pass=rank_pass,share_pass=share_pass,supported=rank_pass and share_pass))
    print('Absent-class error comparison saved',flush=True)


if __name__ == '__main__':
    main()
