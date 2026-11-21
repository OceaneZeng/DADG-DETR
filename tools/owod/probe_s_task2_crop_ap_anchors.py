"""Decompose the paired crop study's S Current AP50 with the original VOC ranking."""
from hashlib import sha256
from math import isclose
import os
from pathlib import Path
import subprocess

import torch

from tools.owod.diagnose_learning import evaluate_views, make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import ROOT
from tools.owod.run_s_task2_duration import MANIFEST
from tools.owod.screening_data import write_json


SOURCE = ROOT / 'output/s-instance-task2-increment-no-random-crop'
CONTROL = SOURCE.with_name('s-instance-task2-current-phase')
OUTPUT = SOURCE.with_name('s-instance-task2-crop-ap-anchors')
PHASES = ('increment', 'finetune')
CRITERION = ('The chair recall-zero precision contribution must explain at least half of both the '
             'final Current deficit and the change in the paired Current gap from increment to finetune. '
             'Use all 21 classes, native VOC2007 11-point AP, no reranking and no author test data.')
SOURCE_RESULT = '668e0b6d1ead81e92cab94f44fc4173967730ca0d95bafc7b58b0f5c692330c8'
REFERENCE_HASHES = {
    'control-increment': '538191b3759898d96ff015ad2b73552b0012e070c9a1626a0556bcf3e8402cdb',
    'control-finetune': '705ff887b5e159117ffce31e672009309ef87fbaccda912a1c3d0b804d1cee5b',
    'candidate-increment': '93310607741db17225b2d11ac0ad009f0f1f1510f4b37b4bd3c806064b95e82c',
    'candidate-finetune': 'ccd7b66420ae67d70fede8100c13610b258950f7741b80c544cc46dfb59b4ca2',
}


def endpoints(source, result):
    entries = {}
    for phase in PHASES:
        for arm in ('control', 'candidate'):
            key = f'{arm}-{phase}'
            if arm == 'control':
                checkpoint = Path(source['control'][phase]['path'])
                expected = source['control'][phase]['sha256']
                ref = CONTROL / f'3epochs-{phase}.json'
                field, view = 'view', 'native'
            else:
                checkpoint = SOURCE / phase / 'checkpoint.pth'
                expected = result['phases'][phase]['checkpoint_sha256']
                ref = SOURCE / f'candidate-{phase}-heldout.json'
                field, view = 'views', 'released'
            if (file_sha256(checkpoint) != expected or file_sha256(ref) != REFERENCE_HASHES[key]
                    or read_json(ref)['checkpoint_sha256'] != expected
                    or read_json(checkpoint.parent/'training_complete.json')['last_epoch'] != (2 if phase == 'increment' else 0)):
                raise ValueError(f'Frozen endpoint or original metrics changed: {key}')
            entries[key] = dict(path=str(checkpoint), sha256=expected, reference=str(ref),
                                reference_sha256=REFERENCE_HASHES[key], field=field, view=view)
    return entries


def verify_replay(view, reference, classes):
    if (view['scoring'] != reference['scoring'] or view['max_detections'] != reference['max_detections']
            or view['composition'] != reference['composition']
            or any(view['metrics'][k] != reference['metrics'][k] for k in ('previous_ap50', 'current_ap50', 'known_ap50'))):
        raise ValueError('Native metrics or prediction composition changed')
    rows = view['current_class_stats']
    if set(rows) != {str(c) for c in classes} or len(rows) != 21:
        raise ValueError('Unexpected S Task 2 classes')
    for label, row in rows.items():
        orig = reference['current_class_stats'][label]
        if (any(row[k] != orig[k] for k in ('gt', 'tp', 'fp', 'ap50'))
                or len(row['anchors']) != 11
                or not isclose(sum(a['precision'] for a in row['anchors'])/11, row['ap50'], abs_tol=1e-12)):
            raise ValueError(f'VOC ranks fail to reproduce class {label}')


def decompose(views, classes, result):
    # Units in this report are fractions of macro AP50. Positive loss favors the original crop.
    losses = {}
    for label in classes:
        key = str(label)
        rows = {k: v['current_class_stats'][key]['anchors'] for k, v in views.items()}
        contributions = {}
        for phase in PHASES:
            control, candidate = rows[f'control-{phase}'], rows[f'candidate-{phase}']
            contributions[phase] = [(a['precision'] - b['precision'])/(11 * len(classes))
                                    for a, b in zip(control, candidate)]
        contributions['stage_change'] = [b-a for a,b in zip(contributions['increment'], contributions['finetune'])]
        losses[key] = contributions
    gaps = {ph: -result['phases'][ph]['delta']['current_ap50'] for ph in PHASES}
    gaps['stage_change'] = gaps['finetune'] - gaps['increment']
    for k, gap in gaps.items():
        if not isclose(sum(sum(row[k]) for row in losses.values()), gap, abs_tol=1e-12):
            raise ValueError(f'AP decomposition does not close for {k}')
    shares = {k: losses['62'][k][0]/gaps[k] for k in ('finetune', 'stage_change')}
    leading = {k: sorted((dict(category_id=int(label), recall_index=i, contribution=v)
                          for label, row in losses.items() for i,v in enumerate(row[k])),
                         key=lambda a: abs(a['contribution']), reverse=True)[:12] for k in shares}
    return dict(control_minus_candidate=gaps, contributions=losses, chair_zero_shares=shares,
                largest_absolute_contributions=leading, supported=all(x >= .5 for x in shares.values()))


def main():
    source = read_json(SOURCE/'plan.json')
    result = read_json(SOURCE/'comparison.json')
    manifest = read_json(MANIFEST)
    annotation = Path(source['annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    if (file_sha256(SOURCE/'comparison.json') != SOURCE_RESULT or result['status'] != 'complete'
            or result['follow_up'] or result['plan_sha256'] != file_sha256(SOURCE/'plan.json')
            or file_sha256(MANIFEST) != source['identity']['manifest_sha256']
            or file_sha256(annotation) != source['annotation_sha256'] or len(order) != 1024 or len(set(order)) != 1024
            or set(order).intersection(manifest['evaluation']['image_source_ids'])
            or sha256('\n'.join(order).encode()).hexdigest() != source['eval_order_sha256']
            or manifest['evaluation']['ap_mode'] != 'voc2007_11point'
            or os.environ['CUDA_VISIBLE_DEVICES'] != '0' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != source['gpus'][0] or str(torch.__version__) != source['torch_version']):
        raise ValueError('Changed crop experiment, holdout, protocol or device')
    entries = endpoints(source, result)
    plan = {k: source[k] for k in ('annotation', 'annotation_sha256', 'eval_order_sha256', 'protocol',
                                    'seed', 'paper_comparable', 'torch_version')}
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    plan_file = OUTPUT/'plan.json'
    plan_commit = code_commit
    # Keep the existing inference result attached to its original immutable plan.
    if plan_file.exists() and file_sha256(plan_file) == 'acbfba229cc304ade5c25624eac5fda9817323e05cca3b401ea08dad6e88878a':
        plan_commit = read_json(plan_file)['commit']
        if code_commit != plan_commit:
            parent = subprocess.check_output(['git', 'rev-parse', 'HEAD^'], cwd=ROOT, text=True).strip()
            changed = subprocess.check_output(['git', 'diff', '--name-only', plan_commit, code_commit],
                                              cwd=ROOT, text=True).splitlines()
            if parent != plan_commit or changed != [Path(__file__).resolve().relative_to(ROOT).as_posix()]:
                raise ValueError('Only the immediate runner repair can resume this pinned plan')
    plan.update(commit=plan_commit,
                manifest_sha256=file_sha256(MANIFEST), source_comparison_sha256=SOURCE_RESULT,
                endpoints=entries, device='0', gpu=torch.cuda.get_device_name(), batch_size=2,
                scoring='focal', max_detections=50, criterion=CRITERION, prefix_class=62, prefix_length=12)
    pin(OUTPUT/'plan.json', plan)
    digest = file_sha256(OUTPUT/'plan.json')
    ds, loader = make_loader(resolve_image_root(MANIFEST,manifest), annotation, 2, 2)
    prev, cur = manifest['stages'][:2]
    spec = dict(manifest['evaluation'], image_source_ids=order, diagnostic_current_ap_anchors=True,
                diagnostic_rank_prefix_class=plan['prefix_class'])
    views = {}
    for key, entry in entries.items():
        out = OUTPUT/(key+'.json')
        if not out.exists():
            print(f'Evaluating {key} on {len(ds)} images', flush=True)
            model, _ = load_detector(None, Path(entry['path']), torch.device('cuda'))
            report = evaluate_views(model, loader, ds.coco, {'native':(cur['active_classes'],'focal',50)},
                                    spec, prev['classes'], cur['classes'], 'cuda', include_current_stats=True, print_freq=64)
            write_json(out, dict(view=report['native'], plan_sha256=digest, code_commit=code_commit,
                                 checkpoint_sha256=entry['sha256']))
            del model
        row = read_json(out)
        if row['plan_sha256'] != digest or row['checkpoint_sha256'] != entry['sha256']:
            raise ValueError(f'Changed replay file: {key}')
        views[key] = row['view']
        reference = read_json(Path(entry['reference']))
        verify_replay(row['view'], reference[entry['field']][entry['view']], cur['classes'])
    comparison = decompose(views, cur['classes'], result)
    write_json(OUTPUT/'comparison.json', dict(status='complete', paper_comparable=False, plan_sha256=digest,
                                              code_commit=code_commit, **comparison))
    print('VOC2007 Current AP decomposition saved', flush=True)


if __name__ == '__main__':
    main()
