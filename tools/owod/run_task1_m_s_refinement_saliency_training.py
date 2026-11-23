"""Train the single local-saliency change on the locked Task 1 screen."""

from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_local_saliency import OUTPUT as PREFLIGHT
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import TOP_KNOWN_SHA
from tools.owod.probe_task1_m_s_refinement_top_known import OUTPUT as REFERENCE, TopKnownPerQuery
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS, OUTPUT as TRAINING, compare, prepare as prepare_refinement
from tools.owod.run_task1_m_s_released_small_extent import measure
from tools.owod.screening_data import write_json


OUTPUT = TRAINING.with_name('task1-m-s-refinement-saliency-training')
PREFLIGHT_SHA = '3be9cf68082bbc19224ce2123d7fea0a0ec808a1226d884fcf9a9c25be5c1056'
GATE_FLAG = '--discovery-saliency-top-points'


def prepare():
    evidence = verified(PREFLIGHT / 'comparison.json', PREFLIGHT_SHA)
    if evidence['status'] != 'complete' or not evidence['passed']:
        raise ValueError('Both released training-source saliency checks must pass')
    preflight = verified(PREFLIGHT / 'plan.json', evidence['plan_sha256'])
    top = verified(REFERENCE / 'comparison.json', TOP_KNOWN_SHA)
    reference = verified(REFERENCE / 'plan.json', top['plan_sha256'])
    prior = verified(TRAINING / 'plan.json', reference['training_plan_sha256'])
    checked = prepare_refinement()
    if prior['jobs'] != checked['jobs'] or prior['gates'] != reference['gates']:
        raise ValueError('Prepared inputs differ from the locked Task 1 screen')
    jobs = {}
    for protocol, source in checked['jobs'].items():
        command = list(source['command'])
        command[command.index('--output_dir') + 1] = str((OUTPUT / protocol / 'increment').resolve())
        command.extend((GATE_FLAG, str(preflight['strongest_points'])))
        control = reference['jobs'][protocol]['arms']['control']
        jobs[protocol] = dict(source, command=command, control_report=control['report'],
                              control_report_sha256=control['report_sha256'],
                              refinement_reference=top['protocols'][protocol])
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                preflight_sha256=PREFLIGHT_SHA, training_plan_sha256=reference['training_plan_sha256'],
                top_known_sha256=TOP_KNOWN_SHA, jobs=jobs, limit=reference['limit'], batch_size=reference['batch_size'],
                device_ids=checked['device_ids'], devices=checked['devices'], epochs=checked['epochs'],
                seed=checked['seed'], strongest_points=preflight['strongest_points'],
                gates=reference['gates'], paper_comparable=False,
                start='Fresh shared DINO initialization with the original refinement architecture and schedule.',
                decision='Apply all original author-VOC screening gates on all four locked folds. Compare the score '
                         'and coverage differences to the refinement endpoint under the same TopKnown postprocessor.')


def evaluate(checkpoint, protocol, job, plan, digest):
    checkpoint_sha = file_sha256(checkpoint)
    path = MANIFESTS[protocol]
    manifest = verified(path, job['manifest_sha256'])
    root = resolve_image_root(path, manifest)
    control = verified(Path(job['control_report']), job['control_report_sha256'])['folds']
    model, _ = load_detector(None, checkpoint, torch.device('cuda'))
    model.eval()
    post = TopKnownPerQuery(known_ids=manifest['stages'][0]['classes'], scoring='focal', max_detections=plan['limit'])
    folds = {}
    for fold, entry in job['folds'].items():
        dest = OUTPUT / f'{protocol}-{fold}.json'
        if not dest.exists():
            annotations = verified_annotation(entry)
            print(f'Evaluating {protocol}/{fold} saliency training', flush=True)
            report = measure(model, manifest, root, annotations, plan['batch_size'], plan['limit'], postprocessor=post)
            outcome = compare(control[fold], report, plan['gates'])
            reference = job['refinement_reference'][fold]
            outcome.update(metrics=report['metrics'],
                           deltas_vs_refinement={key: report['metrics'][key] - before
                                                 for key, before in reference['metrics'].items()},
                           gate_deltas_vs_refinement={key: value - reference['deltas'][key]
                                                      for key, value in outcome['deltas'].items()})
            write_json(dest, dict(plan_sha256=digest, checkpoint_sha256=checkpoint_sha, outcome=outcome, report=report))
        data = read_json(dest)
        if data['plan_sha256'] != digest or data['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError(f'Changed training plan or checkpoint: {dest}')
        folds[fold] = data['outcome']
    return folds


def verified_annotation(entry):
    path = Path(entry['path'])
    if file_sha256(path) != entry['sha256']:
        raise ValueError(f'Changed frozen validation annotations: {path}')
    return path


def main():
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        print(f'Fitting {protocol} Task 1 local saliency', flush=True)
        checkpoint, saved = run_job(job['command'], job['screen_sha256'])
        if (not saved.with_box_refine or saved.backbone_weights_sha256 != job['backbone_sha256']
                or saved.discovery_saliency_top_points != plan['strongest_points']):
            raise ValueError(f'Unexpected training endpoint: {protocol}')
        protocols[protocol] = evaluate(checkpoint, protocol, job, plan, digest)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for folds in protocols.values() for row in folds.values())))


if __name__ == '__main__':
    main()
