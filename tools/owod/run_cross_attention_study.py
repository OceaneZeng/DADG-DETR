"""Fixed M Task 2 study of decoder adaptation, instance transfer and pseudo supervision."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod import run_instance_pilot as pilot
from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_localization_pair import endpoint_args, verified_pilot_inputs
from tools.owod.run_schedule_pair import coverage, endpoint
from tools.owod.screening_data import write_json

OUTPUT = ROOT / 'output/m-instance-cross-attention'
ARMS = ('off', 'matched', 'shuffled')
ENDPOINTS = ('increment', 'finetune', 'finetune-continued')


def study_commands(parent, gnn):
    commands = {}
    for arm in ARMS:
        commands[arm] = {}
        for name in ENDPOINTS:
            phase = 'increment' if name == 'increment' else 'finetune'
            previous = parent if phase == 'increment' else OUTPUT / arm / 'increment/checkpoint.pth'
            cmd = pilot.command(arm, phase, previous, gnn)
            cmd[cmd.index('--output_dir') + 1] = str(OUTPUT / arm / name)
            cmd[cmd.index('--graph-localization') + 1] = 'cross-attn-last'
            cmd.extend(['--screen-branch-checkpoint', str(parent)])
            if name == 'finetune-continued':
                cmd.extend(['--discovery-pseudo-epoch-offset', str(pilot.EPOCHS['increment'])])
            commands[arm][name] = cmd
    return commands


def pin(path, identity):
    if path.is_file():
        if read_json(path) != identity:
            raise ValueError(f'Experiment inputs changed: {path}')
    else:
        if path.parent.is_dir() and any(p.name not in ('run.lock', 'console.log') for p in path.parent.iterdir()):
            raise ValueError(f'Existing experiment has no matching plan: {path}')
        write_json(path, identity)


def endpoint_coverage(checkpoint, loader, path):
    digest = file_sha256(checkpoint)
    if path.is_file():
        cached = read_json(path)
        if cached['checkpoint_sha256'] != digest:
            raise ValueError(f'Coverage checkpoint changed: {checkpoint}')
        return cached['coverage']
    observed = coverage(checkpoint, loader, include_classification=True, include_ranks=True)
    write_json(path, dict(checkpoint_sha256=digest, coverage=observed))
    return observed


def run_endpoint(command, reference, loader):
    folder = Path(command[command.index('--output_dir') + 1])
    previous = Path(command[command.index('--pretrained') + 1])
    phase = command[command.index('--owod-phase') + 1]
    pin(folder / 'input.json', dict(command=command, predecessor_sha256=file_sha256(previous)))
    checkpoint = folder / 'checkpoint.pth'
    marker = folder / 'training_complete.json'
    if not marker.exists():
        pilot.run(command + (['--resume', str(checkpoint)] if checkpoint.exists() else []))
    saved = endpoint_args(checkpoint, phase, command, reference)
    metrics = endpoint(folder, phase)
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    row = next(r for r in rows if r['epoch'] == pilot.EPOCHS[phase] - 1 and 'test_owod_known_ap50' in r)
    classes = row['test_owod_current_class_stats']
    spec = saved.graph_adapter_specs[-1]
    accepted = Counter(r['label'] for r in spec['instances']['records'])
    ap_by_support = {}
    for name, has_match in [('with_match', True), ('without_match', False)]:
        ids = [c for c in saved.owod_current_class_ids if (accepted[c] > 0) == has_match]
        ap_by_support[name] = dict(ids=ids, ap50=sum(classes[str(c)]['ap50'] for c in ids) / len(ids) if ids else None)
    return dict(metrics=metrics, current_classes=classes, current_ap_by_retrieval=ap_by_support,
                retrieval=spec['alignment_summary'], accepted_by_class=dict(accepted),
                heldout=endpoint_coverage(checkpoint, loader, folder / 'coverage.json'),
                trainable_parameters=row['n_parameters'], peak_allocated_mib_rank0=row['train_peak_allocated_mib'],
                last_invocation_seconds=read_json(marker)['training_time_seconds'])


def differences(left, right):
    return {key: left[key] - right[key] for key in (*pilot.METRICS, 'unknown_tp')}


def summarize(rows, baseline):
    capacity = {name: differences(rows['off'][name]['metrics'], baseline[name])
                for name in ('increment', 'finetune')}
    transfer = {name: {arm: differences(rows['matched'][name]['metrics'], rows[arm][name]['metrics'])
                       for arm in ('off', 'shuffled')} for name in ENDPOINTS}
    discovery = {arm: differences(rows[arm]['finetune-continued']['metrics'], rows[arm]['finetune']['metrics'])
                 for arm in ARMS}
    for name in ENDPOINTS:
        matched = rows['matched'][name]['current_ap_by_retrieval']
        for arm, delta in transfer[name].items():
            control = rows[arm][name]['current_ap_by_retrieval']
            for group, values in matched.items():
                if values['ids'] != control[group]['ids']:
                    raise ValueError(f'Retrieval class groups changed in {arm}/{name}')
                delta['current_ap50_' + group] = (
                    values['ap50'] - control[group]['ap50'] if values['ids'] else None)
    evidence = (rows['shuffled']['increment']['retrieval']['shuffled_across_slots'] > 0 and
                all(rows[arm]['increment']['metrics']['mean_alignment_count_per_rank_batch'] > 0
                    for arm in ('matched', 'shuffled')))
    return dict(cross_attention_minus_frozen_off=capacity, matched_minus_controls=transfer,
                continued_minus_reset=discovery, distinct_transfer_exercised=evidence,
                capacity_screen_passed=(capacity['finetune']['current_ap50'] >= .01 and
                                        capacity['finetune']['previous_ap50'] >= -.01),
                matched_current_above_both={name: evidence and all(
                    delta['current_ap50'] > 0 for delta in transfer[name].values()) for name in ENDPOINTS})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print the nine jobs without starting training')
    args = parser.parse_args(argv)
    parent_dir = pilot.OUTPUT / 'matched'
    commands = study_commands(parent_dir / 'task1/checkpoint.pth', parent_dir / 'gnn/gnn_stage0.pt')
    if args.plan:
        print(json.dumps(commands, indent=2))
        return

    from torch.utils.data import DataLoader
    from tools.owod.gnn_calibration import build_calibration_dataset
    from util.misc import collate_fn
    reference, files, parent, gnn = verified_pilot_inputs()
    source = pilot.OUTPUT
    frozen = {phase: source / 'off' / phase / 'checkpoint.pth' for phase in ('increment', 'finetune')}
    plan = dict(commands=commands, commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_plan=file_sha256(source / 'plan.json'),
        source_results=file_sha256(source / 'comparison.json'),
        frozen_checkpoints={phase: file_sha256(path) for phase, path in frozen.items()})
    pin(OUTPUT / 'plan.json', plan)
    baseline = {}
    for phase, path in frozen.items():
        previous = parent if phase == 'increment' else frozen['increment']
        saved = endpoint_args(path, phase, pilot.command('off', phase, previous, gnn), reference)
        baseline[phase] = endpoint(path.parent, phase)
        if any(baseline[phase][k] != v for k, v in reference['rows']['off'][phase].items()):
            raise ValueError(f'Original off endpoint differs from the completed pilot: {path}')
    dataset = build_calibration_dataset(Path(saved.owod_image_root), files['full_val'])
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn)
    frozen_coverage = {phase: endpoint_coverage(path, loader, OUTPUT / f'frozen-{phase}-coverage.json')
                       for phase, path in frozen.items()}
    rows = {}
    report = dict(status='partial', task1=reference['task1'], bank=reference['bank'],
                  frozen_off=baseline, frozen_off_heldout=frozen_coverage, rows=rows,
                  units='AP, precision, recall and their deltas are fractions; A-OSE and predictions are counts',
                  primary_endpoint='finetune', seed=42, paper_comparable=False)
    for arm, jobs in commands.items():
        rows[arm] = {}
        for name, command in jobs.items():
            rows[arm][name] = run_endpoint(command, reference, loader)
            write_json(OUTPUT / 'progress.json', report)
    summary = summarize(rows, baseline)
    report.update(status='complete', summary=summary)
    write_json(OUTPUT / 'comparison.json', report)
    write_json(OUTPUT / 'progress.json', report)
    print(json.dumps(summary, indent=2), flush=True)
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
