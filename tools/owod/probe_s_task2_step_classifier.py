"""Cross a frozen output head between matched S finetune step-size endpoints."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_current_fp_sources import summarize as summarize_fp
from tools.owod.probe_s_task2_quality_swap import OUTPUT as SWAP, SOURCE
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_finetune_half_lr import OUTPUT as HALF
from tools.owod.run_s_task2_increment_giou import evaluate_heldout
from tools.owod.screening_data import write_json


OUTPUT = HALF.with_name('s-instance-task2-step-classifier')
ARMS = ('full', 'half')
FOLDS = ('released', 'proxy_even', 'proxy_odd')


def prepare():
    result = read_json(HALF/'comparison.json')
    previous = read_json(HALF/'plan.json')
    source = read_json(SOURCE/'comparison.json')
    result_sha = file_sha256(HALF/'comparison.json')
    manifest = read_json(MANIFEST)
    annotation = Path(previous['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    if (result_sha != 'e967ab1361d6797b83fc7843f0c24890bf7732dc822470f9db4c810c7ddcaa75'
            or result['follow_up'] or result['plan_sha256'] != file_sha256(HALF/'plan.json')
            or file_sha256(SOURCE/'comparison.json') != previous['source_training_sha256']
            or source['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or file_sha256(MANIFEST) != previous['identity']['manifest_sha256']
            or file_sha256(annotation) != previous['annotation_sha256'] or len(order) != 1024 or len(set(order)) != len(order)
            or sha256('\n'.join(order).encode()).hexdigest() != previous['eval_order_sha256']
            or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != previous['gpus'][0]
            or os.environ['CUDA_VISIBLE_DEVICES'] != '0' or str(torch.__version__) != previous['torch_version']):
        raise ValueError('Changed finetune study, holdout or evaluation environment')

    endpoints, references = {}, {}
    for arm, folder, expected, reference, ref_hash in (
            ('full', SOURCE, source['phases']['finetune']['checkpoint_sha256'], SWAP/'candidate-candidate.json',
             'd421feb40ac152e3477f1a60cfc57978b29a5252a0b5f2cb95b6a5054a8ec7f2'),
            ('half', HALF, result['checkpoint_sha256'], HALF/'candidate-finetune-heldout.json',
             '4212a7809cd7f2f03e61be49aad4c5e158095f1e8625df365691142d5a64e4de')):
        ckpt = folder/'finetune/checkpoint.pth'
        row = read_json(reference)
        if (file_sha256(ckpt) != expected or file_sha256(reference) != ref_hash
                or row.get('body_sha256', row.get('checkpoint_sha256')) != expected
                or row['plan_sha256'] != file_sha256(reference.parent/'plan.json')
                or read_json(ckpt.parent/'training_complete.json')['last_epoch'] != 0
                or read_json(ckpt.parent/'run_config.json')['two_stage']
                or read_json(ckpt.parent/'input.json')['--pretrained'] != previous['increment_sha256']):
            raise ValueError(f'Changed checkpoint or native reference: {arm}')
        references[arm] = row['views']
        endpoints[arm] = dict(checkpoint=str(ckpt), sha256=expected, reference=str(reference), reference_sha256=ref_hash)

    gain = result['comparisons']['candidate']['delta']['current_ap50']
    if gain <= 0 or gain != references['half']['released']['metrics']['current_ap50'] - references['full']['released']['metrics']['current_ap50']:
        raise ValueError('Expected a positive matched Current AP50 gain')
    plan = {k: previous[k] for k in ('annotation', 'annotation_sha256', 'eval_order_sha256', 'seed', 'protocol',
                                      'paper_comparable', 'torch_version', 'increment_sha256')}
    plan.update(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256=result_sha, endpoints=endpoints, device='0', gpu=torch.cuda.get_device_name(), batch_size=2,
                manifest_sha256=file_sha256(MANIFEST), threshold=gain/2,
                criterion='Flag a classifier contribution if exchanging full for half classifier gives at least half '
                          'the native Current gain at both fixed bodies. Use the same rule for the remaining weights '
                          'at both fixed classifiers. Report both or neither, interaction, Previous, Known, both proxy '
                          'folds and released FP families. The frozen crosses are diagnostics only.')
    return plan, manifest, order, references


def register_box_probe(plan, refs):
    offset = HALF.with_name('s-instance-task2-finetune-offset-rate')
    result = read_json(offset/'comparison.json')
    offset_plan = read_json(offset/'plan.json')
    half_record = OUTPUT/'half-half.json'
    half = read_json(half_record)
    endpoint = plan['endpoints']['half']
    if (file_sha256(offset/'comparison.json') != 'ffe0ed26c5819ac3a56be5a325db7ef287cd3e936b787420a2b024d0a8c113da'
            or result['plan_sha256'] != file_sha256(offset/'plan.json') or result['follow_up'] or result['old_coverage_pass']
            or offset_plan['increment_sha256'] != plan['increment_sha256']
            or offset_plan['source_classifier_sha256'] != file_sha256(OUTPUT/'comparison.json')
            or offset_plan['half_reference_sha256'] != file_sha256(half_record)
            or half['plan_sha256'] != file_sha256(OUTPUT/'plan.json')
            or half['body_checkpoint'] != endpoint['sha256'] or half['classifier_checkpoint'] != endpoint['sha256']):
        raise ValueError('Changed offset study or half-rate native endpoint')
    for fold in FOLDS:
        if any(half['views'][fold][k] != refs['half'][fold][k] for k in ('composition', 'metrics')):
            raise ValueError('Different native half-rate predictions')
    refs['half'] = half['views']
    endpoint.update(reference=str(half_record), reference_sha256=file_sha256(half_record))
    counts = [refs[arm]['raw_old_box_recall_iou50'] for arm in ARMS]
    if counts[0]['gt'] != counts[1]['gt'] or counts[0]['hits'] - counts[1]['hits'] != 162:
        raise ValueError('Changed native old geometry deficit')
    plan.update(offset_result_sha256=file_sha256(offset/'comparison.json'), threshold=offset_plan['old_hits_to_recover'],
                criterion='Use full minus half for both factors. Attribute old coverage to a factor if both '
                          'conditional gains reach 81 old targets with all-query IoU > 0.5. Report both factors, '
                          'interaction, released AP50, proxy folds and FP sources. Do not advance a hybrid checkpoint.')


def conditional_effects(values, positive, negative, factor, threshold):
    head = {b: values[b+'-'+positive] - values[b+'-'+negative] for b in ARMS}
    rest = {h: values[positive+'-'+h] - values[negative+'-'+h] for h in ARMS}
    return {factor: head, 'rest': rest, 'interaction': head[positive]-head[negative], 'threshold': threshold,
            'supported': [k for k, v in ((factor, head), ('rest', rest)) if min(v.values()) >= threshold]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--box-head', action='store_true', help='Attribute old geometry using the frozen box readout')
    box = parser.parse_args().box_head
    output = OUTPUT.with_name('s-instance-task2-step-box') if box else OUTPUT
    print('Checking matched finetune endpoints and frozen native references', flush=True)
    plan, manifest, order, refs = prepare()
    if box:
        register_box_probe(plan, refs)
    pin(output/'plan.json', plan)
    digest = file_sha256(output/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), 2, 2)
    models, heads, modules = {}, {}, {}
    for arm, record in plan['endpoints'].items():
        models[arm], _ = load_detector(None, Path(record['checkpoint']), torch.device('cuda'))
        detr = models[arm].detr
        if box and (detr.transformer.decoder.bbox_embed is not None or len({id(m) for m in detr.bbox_embed}) != 1):
            raise ValueError('Box readout probe requires the shared non-iterative box head')
        modules[arm] = detr.bbox_embed if box else detr.class_embed
        heads[arm] = {k: v.detach().cpu().clone() for k, v in modules[arm].state_dict().items()}
    if all(torch.equal(v, heads['full'][k]) for k, v in heads['half'].items()):
        raise ValueError('Matched head weights are identical')

    cells = {}
    for body in ARMS:
        for head in ARMS:
            name = f'{body}-{head}'
            if name == 'full-full' or (box and body == head):
                views = refs[body]
            else:
                path = output/f'{name}.json'
                identity = dict(plan_sha256=digest, body_checkpoint=plan['endpoints'][body]['sha256'])
                identity['bbox_checkpoint' if box else 'classifier_checkpoint'] = plan['endpoints'][head]['sha256']
                if not path.exists():
                    print(f'Evaluating {name} on {len(ds)} images', flush=True)
                    model = models[body]
                    modules[body].load_state_dict(heads[head], strict=True)
                    views = evaluate_heldout(model, ds, loader, manifest, order, proxies=True, fp_sources=True)
                    write_json(path, dict(**identity, views=views))
                row = read_json(path)
                if any(row[k] != v for k, v in identity.items()):
                    raise ValueError(f'Changed classifier cross: {name}')
                views = row['views']
            if box:
                if any(views[f]['composition'] != refs[body][f]['composition'] for f in FOLDS):
                    raise ValueError(f'Box intervention changed native score composition: {name}')
            else:
                for key in ('raw_old_box_recall_iou50', 'raw_current_box_recall_iou50'):
                    if views[key] != refs[body][key]:
                        raise ValueError(f'Changed all-query box coverage at fixed body: {name}, {key}')
            if body == head and any(views[f][k] != refs[body][f][k] for f in FOLDS for k in ('composition', 'metrics')):
                raise ValueError(f'Native predictions did not replay: {name}')
            rel = views['released']
            families = summarize_fp(rel, refs[body]['released'] if body == head else rel)
            cells[name] = dict(metrics={f: views[f]['metrics'] for f in FOLDS}, composition=rel['composition'],
                               fp_families=families)

            if box:
                for group in ('old', 'current'):
                    key = f'raw_{group}_box_recall_iou50'
                    if views[key]['gt'] != refs[body][key]['gt']:
                        raise ValueError(f'Different target population: {name}, {group}')
                    cells[name][key] = views[key]

    result = dict(status='complete', plan_sha256=digest, paper_comparable=False, cells=cells)
    if box:
        hits = {name: c['raw_old_box_recall_iou50']['hits'] for name, c in cells.items()}
        result['old_hit_effects'] = conditional_effects(hits, 'full', 'half', 'box_head', plan['threshold'])
    else:
        current = {name: c['metrics']['released']['current_ap50'] for name, c in cells.items()}
        result['current_effects'] = conditional_effects(current, 'half', 'full', 'classifier', plan['threshold'])
    write_json(output/'comparison.json', result)
    print('Finetune head comparison saved', flush=True)


if __name__ == '__main__':
    main()
