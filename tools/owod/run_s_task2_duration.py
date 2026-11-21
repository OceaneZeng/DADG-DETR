"""Test the Task 2 increment-duration tradeoff on an independent S training-source holdout."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
import subprocess
import sys

import torch

from tools.owod import run_instance_pilot as pilot
from tools.owod.diagnose_learning import make_loader
from tools.owod.probe_task1_known_queries import fresh_origins
from tools.owod.probe_task2_duration_holdout import evaluate_finetune_holdout
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_s_task1_crop_proxy import OUTPUT as PREVIOUS_PROXY
from tools.owod.run_s_task1_known_fraction import MANIFEST, OUTPUT as PARENT, proxy_data
from tools.owod.run_s_task1_region_ranking import OUTPUT as RANKING
from tools.owod.run_schedule_pair import endpoint
from tools.owod.run_stronger_parent_study import run_job
from tools.owod.screening_data import screening_files, source_ids, write_json
from tools.owod.shared_task1 import validate_inputs


OUTPUT = PARENT.with_name('s-instance-task2-duration')
SCREEN = PARENT / 'data/screening.json'
CHECKPOINT = PARENT / 'off/task1/checkpoint.pth'
PHASES = ('increment', 'finetune')
CACHE_RECOVERY_FILES = {
    'models/graph_local/instance_transfer.py',
    'tools/owod/run_instance_pilot.sh',
    'tools/owod/run_s_task2_duration.py',
}


def commands():
    gnn = OUTPUT / 'off/gnn/gnn_stage0.pt'
    jobs = {}
    for duration in (3, 6):
        arm = f'{duration}epochs'
        jobs[arm] = {}
        for phase in PHASES:
            previous = CHECKPOINT if phase == 'increment' else OUTPUT / arm / 'increment/checkpoint.pth'
            command = pilot.command('off', phase, previous, gnn, output=OUTPUT, screen=SCREEN,
                                    epochs=duration if phase == 'increment' else 1, seed=42, manifest=MANIFEST)
            command[command.index('--output_dir') + 1] = str(OUTPUT / arm / phase)
            command[command.index('--graph-localization') + 1] = 'decoder-bbox'
            command.extend(['--decoder-bbox-lr-mult', '0.75', '--screen-branch-checkpoint', str(CHECKPOINT)])
            if phase == 'finetune':
                command.extend(['--discovery-pseudo-epoch-offset', str(duration)])
            jobs[arm][phase] = command
    return jobs


def prepare(recover_cache_timeout=False):
    parent_plan = read_json(PARENT / 'plan.json')
    parent = read_json(PARENT / 'comparison.json')
    ranking = read_json(RANKING / 'comparison.json')
    ranking_plan = read_json(RANKING / 'plan.json')
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if (parent['plan_sha256'] != file_sha256(PARENT / 'plan.json') or parent['status'] != 'complete' or
            file_sha256(CHECKPOINT) != parent['parent_sha256'] or
            read_json(CHECKPOINT.parent / 'training_complete.json')['last_epoch'] != 11 or
            file_sha256(MANIFEST) != parent_plan['manifest_sha256'] or
            file_sha256(pilot.WEIGHTS) != parent_plan['backbone_sha256'] or
            devices != parent_plan['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != parent_plan['devices'] or
            str(torch.__version__) != parent_plan['torch_version'] or ranking['status'] != 'complete' or
            ranking['follow_up'] or ranking['plan_sha256'] != file_sha256(RANKING / 'plan.json')):
        raise ValueError('Changed S parent checkpoint, prior decisions, manifest, backbone or devices')
    manifest, first = screening_files(MANIFEST, SCREEN, 0)
    _, second = screening_files(MANIFEST, SCREEN, 1)
    sources = [source_ids(read_json(path)) for path in (first['increment_train'], second['increment_train'], second['train'])]
    screen_hash = file_sha256(SCREEN)
    selection = read_json(PARENT / 'selection/inputs.json')
    if (screen_hash != selection['screen_sha256'] or
            [len(group) for group in sources] != [12288, 1536, 1679] or
            manifest['evaluation']['ap_mode'] != 'voc2007_11point'):
        raise ValueError('Incompatible S training data or comparator')
    blocked = set(manifest['evaluation']['image_source_ids']) | set(read_json(SCREEN)['heldout_source_ids'])
    blocked.update(set.union(*sources))
    blocked.update(selection['origins'])
    blocked.update(read_json(PREVIOUS_PROXY / 'selection/origins.json'))
    blocked.update(ranking_plan['split'])
    origins = fresh_origins(manifest, blocked, 53)
    if set(origins) & blocked or Counter(origins.values()) != {'t1_train': 512, 't2_train': 512}:
        raise ValueError('Independent holdout is not source balanced')
    # Only classes already released in Task 2 are retained from the verified source annotations.
    data = proxy_data(manifest, origins)
    if (len(data['images']) != 1024 or source_ids(data) != set(origins) or
            {a['category_id'] for a in data['annotations']} != set(manifest['stages'][1]['active_classes'])):
        raise ValueError('Incomplete released-classes holdout')
    annotation_hash = sha256((json.dumps(data, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    plan = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                protocol=manifest['protocol'], paper_comparable=False, seed=42, selection_seed=53,
                parent_plan_sha256=parent['plan_sha256'], parent_result_sha256=file_sha256(PARENT/'comparison.json'),
                ranking_result_sha256=file_sha256(RANKING/'comparison.json'), checkpoint_sha256=parent['parent_sha256'],
                manifest_sha256=parent_plan['manifest_sha256'], weights_sha256=parent_plan['backbone_sha256'],
                screen_sha256=screen_hash, devices=parent_plan['devices'], gpus=devices, torch_version=str(torch.__version__),
                jobs=commands(), origins=origins, annotation_sha256=annotation_hash, batch_size_per_gpu=2,
                train_images=1536, finetune_images=1679, global_batch_size=4, increments=[3, 6],
                primary='Final balanced-finetune Current AP50 on the fresh 1024-image, 40-released-class S holdout. '
                        'Report Previous and Known AP50, VOC2007 11-point, focal Top-50. '
                        'Author-list scores for each stage are descriptive only.',
                criterion='Recheck a second seed only if 6 minus 3 epochs yields Current >=1 pp, Known >0 pp '
                          'and Previous >=-1 pp at the heldout finetune endpoint.')
    if recover_cache_timeout:
        original = read_json(OUTPUT / 'plan.json')
        runtime_commit = plan['commit']
        plan['commit'] = original['commit']

        def git(*args):
            return subprocess.check_output(['git', *args], cwd=pilot.ROOT, text=True).strip()

        if (git('rev-parse', f'{runtime_commit}^') != original['commit'] or
                set(git('diff', '--name-only', original['commit'], runtime_commit).splitlines()) != CACHE_RECOVERY_FILES or
                git('diff', '--name-only', runtime_commit)):
            raise ValueError('Cache recovery requires the transport-only change on top of the pinned commit')
    pin(OUTPUT / 'plan.json', plan)
    annotation = OUTPUT / 'heldout.json'
    if not annotation.exists():
        write_json(annotation, data)
    if file_sha256(annotation) != annotation_hash:
        raise ValueError('Changed heldout annotations')
    if recover_cache_timeout:
        recovery_path = OUTPUT / 'cache_recovery.json'
        recovery = dict(planned_commit=plan['commit'], runtime_commit=runtime_commit,
                        plan_sha256=file_sha256(OUTPUT / 'plan.json'),
                        change='CPU Gloo transport for frozen instance evidence; training settings unchanged')
        if recovery_path.exists() and read_json(recovery_path) != recovery:
            raise ValueError(f'Changed cache recovery identity: {recovery_path}')
        write_json(recovery_path, recovery)
    return plan, manifest, annotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recover-cache-timeout', action='store_true',
                        help='resume the pinned duration study after the frozen instance-cache timeout')
    launcher_args = parser.parse_args()
    print('Checking S Task 1 endpoint and Task 2 train/holdout identities', flush=True)
    plan, manifest, annotation = prepare(launcher_args.recover_cache_timeout)
    digest = file_sha256(OUTPUT / 'plan.json')
    pilot.run([sys.executable, str(pilot.ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 's-owodb', '--output-root', str(MANIFEST.parent.parent)])
    gnn = pilot.calibrate('off', CHECKPOINT, output=OUTPUT, screen=SCREEN, seed=42, manifest=MANIFEST)
    identity = validate_inputs(MANIFEST, SCREEN, CHECKPOINT, gnn, pilot.WEIGHTS, 42)
    if identity['screening_sha256'] != plan['screen_sha256'] or identity['checkpoint_sha256'] != plan['checkpoint_sha256']:
        raise ValueError('Shared branch identity differs from the plan')
    dataset, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    spec = dict(manifest['evaluation'], image_source_ids=list(plan['origins']), known_only_screening=True)
    report = dict(status='partial', paper_comparable=False, plan_sha256=digest, identity=identity, arms={})
    for arm, phases in plan['jobs'].items():
        outcome = report['arms'][arm] = {}
        for phase in PHASES:
            path, args = run_job(phases[phase], plan['screen_sha256'])
            adapter = args.graph_adapter_specs[-1]
            cache = OUTPUT / 'retrieval' / f'instances-stage-1-{identity["checkpoint_sha256"]}.json'
            cache_hash = file_sha256(cache)
            if (args.owod_protocol != 's-owodb' or args.discovery_version != 10 or args.paper_comparable_protocol or
                    adapter['source_checkpoint_sha256'] != identity['checkpoint_sha256'] or
                    adapter['gnn_sha256'] != identity['gnn_sha256'] or adapter['instances']['cache_sha256'] != cache_hash or
                    cache_hash != report.setdefault('common_cache_sha256', cache_hash)):
                raise ValueError(f'Changed shared S adapter inputs for {arm}/{phase}')
            outcome[phase] = dict(checkpoint_sha256=file_sha256(path),
                                  author=endpoint(path.parent, phase, epoch=args.epochs - 1))
            if phase == 'finetune':
                heldout = evaluate_finetune_holdout(path, args, dataset, loader, spec, digest)
                outcome['heldout'] = heldout['view']['top50']['metrics']
            write_json(OUTPUT / 'progress.json', report)
    short, long = (report['arms'][arm]['heldout'] for arm in ('3epochs', '6epochs'))
    deltas = {k: 100*(long[k]-short[k]) for k in ('previous_ap50', 'current_ap50', 'known_ap50')}
    report.update(status='complete', six_minus_three_pp=deltas,
                  follow_up=deltas['current_ap50'] >= 1 and deltas['known_ap50'] > 0 and deltas['previous_ap50'] >= -1)
    write_json(OUTPUT / 'comparison.json', report)
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
