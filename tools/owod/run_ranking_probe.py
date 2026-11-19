"""Compare output budgets or score factors on the frozen M-OWODB Task 2 checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.owod import run_instance_pilot as pilot
from tools.owod.protocol import file_sha256, read_json, stage_files
from tools.owod.run_schedule_pair import endpoint
from tools.owod.screening_data import write_json

OUTPUT = ROOT / 'output/m-instance-ranking-probe'
OBJECTNESS_OUTPUT = ROOT / 'output/m-instance-objectness-probe'


def current_counts(view):
    rows = view['current_class_stats'].values()
    counts = {key: sum(row[key] for row in rows) for key in ('gt', 'tp', 'fp')}
    return {**counts, 'recall': counts['tp'] / max(1, counts['gt']),
            'precision': counts['tp'] / max(1, counts['tp'] + counts['fp'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objectness', action='store_true',
                        help='At fixed Top-50, omit the objectness factor from known-class scores in the treatment')
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import BatchSampler, DataLoader
    from datasets import build_dataset
    from datasets.samplers import DistributedSampler
    from tools.owod.diagnose_learning import evaluate_views
    from tools.owod.gnn_calibration import load_detector
    from tools.owod.validate_discovery import load_checkpoint_metadata
    from util.misc import collate_fn
    from util.training_runtime import loader_worker_options

    folder = pilot.OUTPUT / 'off/finetune'
    checkpoint = folder / 'checkpoint.pth'
    original_metrics = endpoint(folder, 'finetune')
    metadata = load_checkpoint_metadata(checkpoint)
    saved = metadata['args']
    manifest, files = stage_files(pilot.MANIFEST, 1)
    known = manifest['stages'][1]['active_classes']
    current = manifest['stages'][1]['classes']
    previous = manifest['stages'][0]['active_classes']
    manifest_hash = file_sha256(pilot.MANIFEST)
    expected = dict(owod_protocol='m-owodb', owod_stage=1, owod_phase='finetune',
                    owod_discovery=True, discovery_scoring='focal', instance_mode='off',
                    graph_localization='frozen', owod_max_detections=50, cache_mode=False,
                    epochs=pilot.EPOCHS['finetune'], owod_manifest_sha256=manifest_hash,
                    owod_known_class_ids=known, owod_current_class_ids=current,
                    owod_previous_class_ids=previous, owod_evaluation=manifest['evaluation'])
    if (metadata['epoch'] != expected['epochs'] - 1
            or any(getattr(saved, key, None) != value for key, value in expected.items())
            or Path(saved.val_ann).resolve() != files['full_val']):
        raise ValueError('Expected the completed frozen Task 2 checkpoint with the full author test split')
    if file_sha256(Path(saved.backbone_weights)) != saved.backbone_weights_sha256:
        raise ValueError('Checkpoint backbone input changed')

    diagnostic = 'class_only_top50' if args.objectness else 'top300'
    diagnostic_scoring, diagnostic_budget = ('focal_class_only', 50) if args.objectness else (saved.discovery_scoring, 300)
    variants = {'top50': (known, saved.discovery_scoring, 50),
                diagnostic: (known, diagnostic_scoring, diagnostic_budget)}
    identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                    manifest_sha256=manifest_hash, test_annotation=str(files['full_val']),
                    test_annotation_sha256=manifest['stages'][1]['sha256']['full_val'],
                    scoring=saved.discovery_scoring, budgets=[50, diagnostic_budget],
                    original_eval_world_size=saved.world_size, batch_size=saved.batch_size)
    if args.objectness:
        identity['diagnostic_scoring'] = diagnostic_scoring
    report_path = (OBJECTNESS_OUTPUT if args.objectness else OUTPUT) / 'comparison.json'
    if report_path.is_file():
        result = read_json(report_path)
        if result['inputs'] != identity:
            raise ValueError(f'Existing report has different inputs: {report_path}')
        print(json.dumps(result['summary'], indent=2), flush=True)
        print('Comparison already complete:', report_path, flush=True)
        return

    dataset = build_dataset('val', saved)
    # Reproduce the original rank-local batch grouping on one GPU. The shared
    # evaluator keeps the first prediction for sampler padding, as in training.
    batches = [
        batch
        for rank in range(saved.world_size)
        for batch in BatchSampler(
            DistributedSampler(dataset, num_replicas=saved.world_size, rank=rank, shuffle=False),
            batch_size=saved.batch_size, drop_last=False)
    ]
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate_fn, pin_memory=True,
                        num_workers=saved.num_workers, **loader_worker_options(saved.num_workers))
    device = torch.device('cuda:0')
    model, _ = load_detector(saved, checkpoint, device)
    print(f'Evaluating {len(dataset)} unique M-OWODB test images with a shared forward pass', flush=True)
    views = evaluate_views(model, loader, dataset.coco, variants, manifest['evaluation'],
                           previous, current, device, include_current_stats=True,
                           compare_current_ranks=args.objectness, print_freq=250)
    reference_metrics, diagnostic_metrics = (views[name]['metrics'] for name in variants)
    delta_name = 'class_only_minus_top50' if args.objectness else 'top300_minus_top50'
    summary = {
        'top50': reference_metrics,
        diagnostic: diagnostic_metrics,
        delta_name: {key: diagnostic_metrics[key] - reference_metrics[key] for key in reference_metrics},
        'top50_minus_saved': {key: reference_metrics[key] - original_metrics[key] for key in reference_metrics},
        'current_counts': {name: current_counts(views[name]) for name in variants},
    }
    if args.objectness:
        summary['current_candidate_ranks'] = views['current_candidate_ranks']['overall']
    write_json(report_path, dict(inputs=identity, unique_test_images=len(dataset),
                                 author_test_entries=len(manifest['evaluation']['image_source_ids']),
                                 reference_budget=50, diagnostic_budget=diagnostic_budget,
                                 units='AP50, recall, precision and H-Score are fractions',
                                 summary=summary, views=views, status='complete'))
    print(json.dumps(summary, indent=2), flush=True)
    print('Comparison saved:', report_path, flush=True)


if __name__ == '__main__':
    main()
