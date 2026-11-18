"""Paired Task 2 finetuning from the completed expanded M-OWODB pilot."""
import json
import os
from collections import Counter
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod import run_instance_pilot as pilot
from tools.owod.protocol import file_sha256, read_json
from tools.owod.screening_data import screening_files, write_json

OUTPUT = ROOT / 'output/m-instance-schedule-pair'


def endpoint(folder, phase):
    epoch = pilot.EPOCHS[phase] - 1
    if read_json(folder / 'training_complete.json')['last_epoch'] != epoch:
        raise ValueError(f'Incomplete {phase}: {folder}')
    values = pilot.metrics(folder, epoch)
    rows = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines() if s]
    row = next(r for r in rows if r['epoch'] == epoch and 'test_owod_known_ap50' in r)
    for key in ('discovery_pseudo_scale', 'discovery_pseudo_per_image'):
        values[key] = row['train_' + key]
    values['unknown_gt'] = row['test_owod_unknown_gt']
    values['unknown_tp'] = round(values['u_recall'] * values['unknown_gt'])
    return values


def coverage(checkpoint, loader):
    import torch
    from tools.owod.gnn_calibration import load_detector, move_targets
    from tools.owod.supervision_probe import current_endpoint_hits, summarize_endpoint
    from models.graph_local.discovery import DiscoveryPostProcess

    args = torch.load(checkpoint, map_location='meta', weights_only=False)['args']
    device = torch.device('cuda')
    detector, _ = load_detector(args, checkpoint, device)
    detector.eval()
    known, current = args.owod_known_class_ids, args.owod_current_class_ids
    process = DiscoveryPostProcess(known_ids=known, max_detections=args.owod_max_detections,
                                   scoring=args.discovery_scoring, unknown_threshold=args.unknown_threshold)
    hits = []
    with torch.no_grad():
        for samples, targets in loader:
            targets = move_targets(targets, device)
            outputs = detector(samples.to(device))
            results = process(outputs, torch.stack([t['orig_size'] for t in targets]))
            for i, (target, result) in enumerate(zip(targets, results)):
                hits.append(current_endpoint_hits(outputs, i, target, result, known, current).cpu())
    return summarize_endpoint(hits, current)


def main():
    import torch
    from torch.utils.data import DataLoader
    from tools.owod.gnn_calibration import build_calibration_dataset
    from tools.owod.validate_discovery import validate_plan_arguments
    from util.misc import collate_fn

    source = pilot.OUTPUT
    old = read_json(source / 'plan.json')
    comparison = read_json(source / 'comparison.json')
    if comparison.get('status') != 'complete' or old['epochs'] != pilot.EPOCHS:
        raise ValueError('Expected the completed expanded pilot')
    if (torch.cuda.device_count() != 2 or [torch.cuda.get_device_name(i) for i in range(2)] != old['gpus']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != old['devices'] or str(torch.__version__) != old['torch_version']):
        raise ValueError('Use the same two GPUs and tree-detr environment as the completed pilot')
    _, files = screening_files(pilot.MANIFEST, pilot.SCREEN, 1)
    increment = source / 'off/increment/checkpoint.pth'
    baseline = source / 'off/finetune/checkpoint.pth'
    parent = source / 'matched/task1/checkpoint.pth'
    gnn = source / 'matched/gnn/gnn_stage0.pt'
    for path, expected in ((pilot.MANIFEST, old['manifest']), (pilot.SCREEN, old['screen']),
                           (pilot.WEIGHTS, old['weights']), (parent, comparison['common_parent_sha256']),
                           (gnn, comparison['common_gnn_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError(f'Original input changed: {path}')
    for phase, path, previous in [('increment', increment, parent), ('finetune', baseline, increment)]:
        saved = torch.load(path, map_location='meta', weights_only=False)
        if saved['epoch'] != pilot.EPOCHS[phase] - 1:
            raise ValueError(f'Unexpected endpoint: {path}')
        validate_plan_arguments({'command': pilot.command('off', phase, previous, gnn)}, saved['args'])
    control = endpoint(baseline.parent, 'finetune')
    previous = comparison['rows']['off']['finetune']
    if any(control[k] != v for k, v in previous.items()):
        raise ValueError('Control metrics differ from the completed pilot')

    folder = OUTPUT / 'continued/finetune'
    cmd = pilot.command('off', 'finetune', increment, gnn)
    cmd[cmd.index('--output_dir') + 1] = str(folder)
    cmd += ['--discovery-pseudo-epoch-offset', str(pilot.EPOCHS['increment'])]
    identity = dict(command=cmd, source_plan=file_sha256(source / 'plan.json'),
                    source_results=file_sha256(source / 'comparison.json'),
                    increment_sha256=file_sha256(increment), control_sha256=file_sha256(baseline),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    plan_path = OUTPUT / 'plan.json'
    if plan_path.exists() and read_json(plan_path) != identity:
        raise ValueError(f'Changed schedule comparison inputs: {plan_path}')
    if not plan_path.exists() and folder.exists():
        raise ValueError('Missing plan for existing treatment')
    write_json(plan_path, identity)
    treatment = folder / 'checkpoint.pth'
    if not (folder / 'training_complete.json').is_file():
        resume = ['--resume', str(treatment)] if treatment.exists() else []
        pilot.run(cmd + resume)
    continued = endpoint(folder, 'finetune')

    dataset = build_calibration_dataset(Path(saved['args'].owod_image_root), files['full_val'])
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_fn)
    report = dict(control=control, continued=continued, units='AP/recall/precision are fractions',
                  continued_minus_control={k: continued[k] - control[k] for k in (*pilot.METRICS, 'unknown_tp')},
                  treatment_sha256=file_sha256(treatment), heldout_images=len(dataset), coverage={})
    for name, path in [('increment', increment), ('control', baseline), ('continued', treatment)]:
        report['coverage'][name] = coverage(path, loader)
    records = saved['args'].graph_adapter_specs[-1]['instances']['records']
    accepted = {(r['image'], r['obj'], r['label']) for r in records}
    counts = {}
    for phase in ('increment_train', 'train'):
        gt, present = Counter(), Counter()
        for ann in read_json(files[phase])['annotations']:
            label = ann['category_id']
            gt[label] += 1
            obj = int(str(ann['owod_source_annotation_id']).rsplit(':', 1)[-1])
            if (ann['image_id'], obj, label) in accepted:
                present[label] += 1
        counts[phase] = {str(c): dict(gt=gt[c], accepted_in_split=present[c], applied_off=0)
                         for c in saved['args'].owod_current_class_ids}
    report.update(retrieval_by_class=counts, status='complete')
    write_json(OUTPUT / 'comparison.json', report)
    print(json.dumps(report['continued_minus_control'], indent=2), flush=True)
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
