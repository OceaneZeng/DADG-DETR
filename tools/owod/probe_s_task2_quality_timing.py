"""Locate the fixed-quality-head Current AP50 gap before and after balanced finetuning."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_s_task2_quality_swap import ARMS, OUTPUT as FINETUNE, SOURCE, load_frozen_pair
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.run_s_task2_increment_giou import PHASE_SOURCE, evaluate_heldout
from tools.owod.s_task2_proposal_quality import quality_digest
from tools.owod.screening_data import write_json


OUTPUT = SOURCE.with_name('s-instance-task2-quality-timing')
AP_KEYS = ('previous_ap50', 'current_ap50', 'known_ap50')


def prepare():
    ft = read_json(FINETUNE/'comparison.json')
    base = read_json(FINETUNE/'plan.json')
    fit = read_json(SOURCE/'comparison.json')
    training = read_json(SOURCE/'plan.json')
    manifest = read_json(MANIFEST)
    annotation = Path(base['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    ft_hash = file_sha256(FINETUNE/'comparison.json')
    if (ft_hash != '85beed2821d2d54ac7f53e40f6f99aa3b3a933bce02f7d4d7c9defb9cac5f335'
            or ft['current_effects']['supported'] != ['rest'] or ft['plan_sha256'] != file_sha256(FINETUNE/'plan.json')
            or base['source_sha256'] != file_sha256(SOURCE/'comparison.json') or fit['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or file_sha256(annotation) != base['annotation_sha256'] or file_sha256(MANIFEST) != base['manifest_sha256']
            or len(order) != 1024 or len(set(order)) != 1024 or set(order) & set(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != base['eval_order_sha256']
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != base['gpu']
            or os.environ['CUDA_VISIBLE_DEVICES'] != base['device'] or str(torch.__version__) != base['torch_version']):
        raise ValueError('Changed finetune swap, training evidence, heldout identities or environment')
    endpoints, natives = {}, {}
    for arm in ARMS:
        control = arm == 'control'
        entry = training['control']['increment'] if control else fit['phases']['increment']
        path = Path(entry['path']) if control else SOURCE/'increment/checkpoint.pth'
        digest = entry['sha256'] if control else entry['checkpoint_sha256']
        ref = PHASE_SOURCE/'3epochs-increment.json' if control else SOURCE/'candidate-increment-heldout.json'
        record = read_json(ref)
        successor_input = read_json(Path(base['endpoints'][arm]['path']).parent/'input.json')
        if (file_sha256(path) != digest or record['checkpoint_sha256'] != digest
                or record['plan_sha256'] != file_sha256(ref.parent/'plan.json')
                or file_sha256(PHASE_SOURCE/'comparison.json') != training['source_phase_result_sha256']
                or read_json(path.parent/'training_complete.json')['last_epoch'] != 2
                or successor_input['--pretrained'] != digest or quality_digest(path) != base['endpoints'][arm]['quality_sha256']):
            raise ValueError(f'Unpaired increment predecessor or quality head: {arm}')
        view = record['view']['native'] if control else record['views']['released']
        expected = (read_json(PHASE_SOURCE/'comparison.json')['arms']['3epochs']['increment'] if control
                    else fit['phases']['increment']['heldout'])
        if any(view['metrics'][k] != expected[k] for k in AP_KEYS):
            raise ValueError(f'Native increment AP differs from the archived comparison: {arm}')
        natives[arm] = view
        endpoints[arm] = dict(path=str(path), checkpoint_sha256=digest, quality_sha256=base['endpoints'][arm]['quality_sha256'],
                              reference=str(ref), reference_sha256=file_sha256(ref))
    plan = {k:base[k] for k in ('annotation','annotation_sha256','eval_order_sha256','manifest_sha256','device','gpu',
                                 'torch_version','batch_size','seed','protocol','paper_comparable','scoring','max_detections')}
    plan.update(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                endpoints=endpoints, source_finetune_sha256=ft_hash, source_training_sha256=base['source_sha256'],
                threshold=ft['current_effects']['threshold'],
                criterion='At each fixed quality head compare candidate minus control remaining weights. Flag increment '
                          'if both increment Current losses reach the inherited half-deficit threshold; flag finetuning '
                          'if both fixed-head gaps worsen by that threshold from increment to finetune. Report both or '
                          'neither when appropriate, including the classwise contributions. Use released views only.')
    pin(OUTPUT/'plan.json', plan)
    return plan, manifest, order, natives, ft


def compare_phases(increment, finetune, threshold):
    rest = {h: increment['candidate-'+h]['current_ap50']-increment['control-'+h]['current_ap50'] for h in ARMS}
    change = {h: finetune[h]-rest[h] for h in ARMS}
    supported = [phase for phase, diffs in [('increment', rest), ('finetune', change)]
                 if all(v <= -threshold for v in diffs.values())]
    return dict(increment_rest=rest, finetune_rest=finetune, change=change, threshold=threshold, supported=supported)


def main():
    plan, manifest, order, natives, ft = prepare()
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), Path(plan['annotation']), plan['batch_size'], 2)
    models, heads = load_frozen_pair(plan['endpoints'])
    metrics, classes = {}, {}
    for body in ARMS:
        for head in ARMS:
            name = f'{body}-{head}'
            if body == head:
                view = natives[body]
            else:
                path = OUTPUT/f'{name}.json'
                identity = dict(plan_sha256=digest, body_sha256=plan['endpoints'][body]['checkpoint_sha256'],
                                head_sha256=plan['endpoints'][head]['quality_sha256'])
                if not path.exists():
                    print(f'Evaluating increment {name} on {len(ds)} images', flush=True)
                    model = models[body]
                    model.detr.transformer.instance_proposals.quality.load_state_dict(heads[head], strict=True)
                    views = evaluate_heldout(model, ds, loader, manifest, order, proxies=False)
                    write_json(path, dict(**identity, view=views['released']))
                row = read_json(path)
                if any(row[k] != v for k,v in identity.items()):
                    raise ValueError(f'Changed paired condition: {name}')
                view = row['view']
            metrics[name] = {k:view['metrics'][k] for k in AP_KEYS}
            classes[name] = {c:r['ap50'] for c,r in view['current_class_stats'].items()}
    timing = compare_phases(metrics, ft['current_effects']['rest'], plan['threshold'])
    contributions = {}
    for head in ARMS:
        cells = [body+'-'+head for body in ARMS]
        previous = [read_json(FINETUNE/(n+'.json')) for n in cells]
        if any(row['plan_sha256'] != ft['plan_sha256'] or row['views']['released']['metrics'] != ft['arms'][n]['metrics']['released']
               for row,n in zip(previous,cells)):
            raise ValueError('Changed finetune cell')
        old, new = [v['views']['released']['current_class_stats'] for v in previous]
        contributions[head] = {c: (new[c]['ap50']-old[c]['ap50'])-(classes[cells[1]][c]-classes[cells[0]][c]) for c in old}
        if abs(sum(contributions[head].values())/len(old)-timing['change'][head]) > 1e-10:
            raise ValueError('Per-class AP changes do not reconcile with macro Current AP50')
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                              increment_metrics=metrics, timing=timing, per_class_change=contributions))
    print('Quality timing comparison saved', flush=True)


if __name__ == '__main__':
    main()
