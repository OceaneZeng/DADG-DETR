"""Repeat both fixed-budget fine seed reservations on disjoint released images."""

from hashlib import sha256
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_fine_rank import (
    DECODER_OUTPUT, MINED_OUTPUT, MINED_SHA, mined_selector,
)
from tools.owod.probe_task1_m_s_refinement_center_selector import source_ids
from tools.owod.probe_task1_m_s_refinement_fixed_injection import measure_injection
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = DECODER_OUTPUT.with_name('task1-m-s-refinement-mined-seed-holdout')
DECODER_SHA = '63680e2a22eff04678e8d37645b672b11d4a200dcc37c0d0e34f38adb13ffef2'


def holdout_loader(protocol, job, plan):
    path = MANIFESTS[protocol]
    manifest = verified(path, job['manifest_sha256'])
    checkpoint = Path(job['checkpoint'])
    if file_sha256(checkpoint) != job['checkpoint_sha256']:
        raise ValueError(f'Changed released endpoint: {protocol}')
    offset = plan['replay_offset']
    loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(path, manifest),
                                   job, plan, source_box_area=True, offset=offset)
    base = loader.dataset.dataset
    prefix = torch.randperm(len(base), generator=torch.Generator().manual_seed(plan['seed']))[:offset]
    previous = [base.coco.imgs[base.ids[i]]['owod_source_id'] for i in prefix.tolist()]
    boundary = plan['images']
    first = sha256('\n'.join(previous[:boundary]).encode()).hexdigest()
    fit = sha256('\n'.join(previous[boundary:]).encode()).hexdigest()
    current = source_ids(loader)
    if (first != job['expected_order'] or fit != plan['fit_order_sha256'][protocol]
            or len(current) != boundary or len(set(current)) != boundary or set(current).intersection(previous)):
        raise ValueError(f'Changed or overlapping released holdout: {protocol}')
    return loader, order


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the independent released replay')
    original = verified(DECODER_OUTPUT / 'comparison.json', DECODER_SHA)
    plan = verified(DECODER_OUTPUT / 'plan.json', original['plan_sha256'])
    mined = verified(MINED_OUTPUT / 'comparison.json', MINED_SHA)
    if (original['status'] != 'complete' or not all(row['groups']['passed'] for row in original['protocols'].values())
            or plan['mined_sha256'] != MINED_SHA):
        raise ValueError('Expected the passed first-slice decoder replay and pinned rankers')
    plan.pop('expected_reservations')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                decoder_sha256=DECODER_SHA, replay_offset=plan['images'] + plan['fit_images'],
                factor='Use the next disjoint released slice. Freeze rankers and compare raw-quality and mined '
                       'reservations with identical thresholds, slot cap and unmodified detector control.',
                advance='Apply the original decoder gates to both protocols. Independently require more accurate '
                        'seed replacements than raw quality on each protocol to replicate the ranking result.')
    loaders = {}
    plan['holdout_order_sha256'] = {}
    for protocol, job in plan['jobs'].items():
        loaders[protocol], plan['holdout_order_sha256'][protocol] = holdout_loader(protocol, job, plan)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        order = plan['holdout_order_sha256'][protocol]
        if not dest.exists():
            model, _ = load_detector(None, Path(job['checkpoint']), torch.device('cuda'))
            model.eval()
            loader = loaders[protocol]
            state = loader.generator.get_state()
            print(f'Evaluating {protocol} independent raw quality', flush=True)
            raw = measure_injection(model, loader, job['classes'], plan, protocol, native_selection=True)
            loader.generator.set_state(state)
            print(f'Evaluating {protocol} independent mined readout', flush=True)
            select = mined_selector(mined['protocols'][protocol]['ranker'])
            revised = measure_injection(model, loader, job['classes'], plan, protocol, True, select)
            for group in ('small', 'all'):
                if any(raw['groups'][group][key] != revised['groups'][group][key]
                       for key in ('gt', 'final', 'native_known_tp')):
                    raise ValueError(f'Changed native baseline between independent arms: {protocol}')
            raw, revised = raw['groups'], revised['groups']
            rank_improved = revised['transfer']['assigned'] > raw['transfer']['assigned']
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, raw=raw, mined=revised,
                                  rank_improved=rank_improved, passed=revised['passed'] and rank_improved))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest or report['order_sha256'] != order:
            raise ValueError(f'Changed independent replay record: {protocol}')
        protocols[protocol] = {key: report[key] for key in ('raw', 'mined', 'rank_improved', 'passed')}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for row in protocols.values())))


if __name__ == '__main__':
    main()
