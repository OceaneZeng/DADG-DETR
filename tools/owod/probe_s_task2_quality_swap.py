"""Cross two frozen S finetune detectors with the two trained proposal quality heads."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_current_fp_sources import OUTPUT as FP_SOURCE, summarize as summarize_fp
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import OUTPUT as GIOU, evaluate_heldout
from tools.owod.s_task2_proposal_quality import quality_digest
from tools.owod.screening_data import write_json


SOURCE = GIOU.with_name('s-instance-task2-increment-proposal-quality')
OUTPUT = SOURCE.with_name('s-instance-task2-finetune-quality-swap')
ARMS = ('control', 'candidate')


def prepare():
    src, result = (read_json(SOURCE / f) for f in ('plan.json', 'comparison.json'))
    manifest = read_json(MANIFEST)
    annotation = Path(src['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    result_sha = file_sha256(SOURCE / 'comparison.json')
    if (result_sha != 'e6b8660f31feffdba7f338621ed18317b409238b495bf0b334f44698ae2a3c17'
            or result['plan_sha256'] != file_sha256(SOURCE / 'plan.json') or result['follow_up']
            or file_sha256(annotation) != src['annotation_sha256'] or file_sha256(MANIFEST) != src['identity']['manifest_sha256']
            or sha256('\n'.join(order).encode()).hexdigest() != src['eval_order_sha256'] or len(set(order)) != 1024
            or len(order) != 1024 or set(order) & set(manifest['evaluation']['image_source_ids'])
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != src['gpus'][0]
            or str(torch.__version__) != src['torch_version']):
        raise ValueError('Changed quality adaptation evidence, holdout or replay environment')
    endpoints = {}
    for arm in ARMS:
        path = Path(src['control']['finetune']['path']) if arm == 'control' else SOURCE / 'finetune/checkpoint.pth'
        row = result['phases']['finetune']
        expected = src['control']['finetune']['sha256'] if arm == 'control' else row['checkpoint_sha256']
        head = src['parent_quality_sha256'] if arm == 'control' else row['quality_sha256']
        ref = SOURCE / f'{arm}-finetune-heldout.json'
        record = read_json(ref)
        if (file_sha256(path) != expected or quality_digest(path) != head or record['checkpoint_sha256'] != expected
                or record['plan_sha256'] != result['plan_sha256'] or read_json(path.parent/'training_complete.json')['last_epoch'] != 0):
            raise ValueError(f'Changed finetune endpoint: {arm}')
        endpoints[arm] = dict(path=str(path), checkpoint_sha256=expected, quality_sha256=head,
                              reference=str(ref), reference_sha256=file_sha256(ref))
    fp = FP_SOURCE / 'control-finetune.json'
    fp_hash = file_sha256(fp)
    if (fp_hash != 'a963b1d113cf904107da881bfd2e8bff317c7d6255058ae65ac028c18514a9b6'
            or read_json(fp)['checkpoint_sha256'] != endpoints['control']['checkpoint_sha256']
            or read_json(fp)['plan_sha256'] != file_sha256(FP_SOURCE/'plan.json')
            or read_json(FP_SOURCE/'plan.json')['annotation_sha256'] != src['annotation_sha256']):
        raise ValueError('Changed native control FP attribution')
    plan = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                protocol='s-owodb', paper_comparable=False, seed=42, endpoints=endpoints, source_sha256=result_sha,
                fp_reference_sha256=fp_hash, annotation=str(annotation), annotation_sha256=src['annotation_sha256'],
                eval_order_sha256=src['eval_order_sha256'], manifest_sha256=file_sha256(MANIFEST),
                device=os.environ['CUDA_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__), batch_size=2, scoring='focal', max_detections=50,
                criterion='Report the two conditional Current AP50 effects of each factor, candidate minus control, '
                          'and their interaction. Flag a factor if both conditional effects reduce Current by at least '
                          'half the magnitude of the observed native finetune deficit. Report FP families, Previous, '
                          'Known and both proxy folds. No hybrid selection or advancement.')
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest, order


def effects(values):
    control, candidate = values['control'], values['candidate']
    head = {b: values[b]['candidate']-values[b]['control'] for b in ARMS}
    body = {h: candidate[h]-control[h] for h in ARMS}
    deficit = control['control']-candidate['candidate']
    if deficit <= 0:
        raise ValueError('Expected to replay the paired Current AP50 deficit')
    return dict(head=head, rest=body, interaction=head['candidate']-head['control'], threshold=deficit/2,
                supported=[name for name, conditional in [('quality',head),('rest',body)]
                           if all(v <= -deficit/2 for v in conditional.values())])


def main():
    plan, manifest, order = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), plan['batch_size'], 2)
    models, heads, references = {}, {}, {}
    for arm, entry in plan['endpoints'].items():
        models[arm], _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
        module = models[arm].detr.transformer.instance_proposals.quality
        heads[arm] = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        references[arm] = read_json(Path(entry['reference']))['views']
    fp_control = read_json(FP_SOURCE/'control-finetune.json')['view']
    summarize_fp(fp_control, references['control']['released'])
    summary, current = {}, {arm: {} for arm in ARMS}
    for body in ARMS:
        for head in ARMS:
            name = f'{body}-{head}'
            path = OUTPUT / f'{name}.json'
            identity = dict(plan_sha256=digest, body_sha256=plan['endpoints'][body]['checkpoint_sha256'],
                            head_sha256=plan['endpoints'][head]['quality_sha256'])
            if not path.exists():
                if name == 'control-control':
                    views = dict(references['control'])
                    views['released'] = dict(views['released'], current_class_stats=fp_control['current_class_stats'])
                else:
                    print(f'Evaluating {name} on {len(ds)} images', flush=True)
                    model = models[body]
                    model.detr.transformer.instance_proposals.quality.load_state_dict(heads[head], strict=True)
                    views = evaluate_heldout(model, ds, loader, manifest, order, proxies=True, fp_sources=True)
                write_json(path, dict(**identity, reused=name=='control-control', views=views))
            row = read_json(path)
            if any(row[k] != v for k,v in identity.items()):
                raise ValueError(f'Changed frozen combination: {name}')
            views = row['views']
            if body == head:
                for fold in ('released', 'proxy_even', 'proxy_odd'):
                    view, ref = views[fold], references[body][fold]
                    if (view['metrics'] != ref['metrics'] or view['composition'] != ref['composition']
                            or any(any(view['current_class_stats'][c][k] != n for k,n in counts.items())
                                   for c,counts in ref['current_class_stats'].items())):
                        raise ValueError(f'Native endpoint failed to replay: {name} {fold}')
            families = summarize_fp(views['released'], references[body]['released'] if body == head else views['released'])
            summary[name] = dict(metrics={f: views[f]['metrics'] for f in ('released','proxy_even','proxy_odd')},
                                 fp_families=families)
            current[body][head] = views['released']['metrics']['current_ap50']
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                              arms=summary, current_effects=effects(current)))
    print('Frozen quality head comparison saved', flush=True)


if __name__ == '__main__':
    main()
