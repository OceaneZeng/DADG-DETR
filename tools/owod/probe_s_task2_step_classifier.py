"""Separate classification-head and remaining-weight effects in the S finetune step-size comparison."""
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
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest, order, references


def main():
    print('Checking matched finetune endpoints and frozen native references', flush=True)
    plan, manifest, order, refs = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), 2, 2)
    models, heads = {}, {}
    for arm, record in plan['endpoints'].items():
        models[arm], _ = load_detector(None, Path(record['checkpoint']), torch.device('cuda'))
        heads[arm] = {k: v.detach().cpu().clone() for k, v in models[arm].detr.class_embed.state_dict().items()}
    if all(torch.equal(v, heads['full'][k]) for k, v in heads['half'].items()):
        raise ValueError('Classifier weights are identical')

    cells = {}
    for body in ARMS:
        for head in ARMS:
            name = f'{body}-{head}'
            if name == 'full-full':
                views = refs['full']
            else:
                path = OUTPUT/f'{name}.json'
                identity = dict(plan_sha256=digest, body_checkpoint=plan['endpoints'][body]['sha256'],
                                classifier_checkpoint=plan['endpoints'][head]['sha256'])
                if not path.exists():
                    print(f'Evaluating {name} on {len(ds)} images', flush=True)
                    model = models[body]
                    model.detr.class_embed.load_state_dict(heads[head], strict=True)
                    views = evaluate_heldout(model, ds, loader, manifest, order, proxies=True, fp_sources=True)
                    write_json(path, dict(**identity, views=views))
                row = read_json(path)
                if any(row[k] != v for k, v in identity.items()):
                    raise ValueError(f'Changed classifier cross: {name}')
                views = row['views']
            for key in ('raw_old_box_recall_iou50', 'raw_current_box_recall_iou50'):
                if views[key] != refs[body][key]:
                    raise ValueError(f'Changed all-query box coverage at fixed body: {name}, {key}')
            if body == head and any(views[f][k] != refs[body][f][k] for f in FOLDS for k in ('composition', 'metrics')):
                raise ValueError(f'Native predictions did not replay: {name}')
            rel = views['released']
            families = summarize_fp(rel, refs[body]['released'] if body == head else rel)
            cells[name] = dict(metrics={f: views[f]['metrics'] for f in FOLDS}, composition=rel['composition'],
                               fp_families=families)

    current = {name: c['metrics']['released']['current_ap50'] for name, c in cells.items()}
    classifier = {b: current[b+'-half'] - current[b+'-full'] for b in ARMS}
    rest = {h: current['half-'+h] - current['full-'+h] for h in ARMS}
    effects = dict(classifier=classifier, rest=rest, interaction=classifier['half']-classifier['full'],
                   threshold=plan['threshold'], supported=[k for k, v in (('classifier', classifier), ('rest', rest))
                                                           if min(v.values()) >= plan['threshold']])
    write_json(OUTPUT/'comparison.json', dict(status='complete', plan_sha256=digest, paper_comparable=False,
                                              current_effects=effects, cells=cells))
    print('Finetune classifier comparison saved', flush=True)


if __name__ == '__main__':
    main()
