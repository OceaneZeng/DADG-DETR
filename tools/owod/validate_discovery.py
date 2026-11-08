"""Mechanism diagnostics or strictly paired author-benchmark AP comparison."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_checkpoint_metadata(path):
    """Read trusted local experiment metadata without allocating model tensors."""
    import torch
    safe_globals = getattr(torch.serialization, 'safe_globals', None)
    if safe_globals is None:
        # Match main.load_local_checkpoint's PyTorch 2.4 compatibility path.
        return torch.load(path, map_location='meta', weights_only=False)
    with safe_globals([argparse.Namespace]):
        return torch.load(path, map_location='meta', weights_only=True)


def validate_plan_arguments(run, saved):
    """Check actual saved training arguments, including defaults, against launch."""
    from main import get_args_parser
    from tools.owod.strict_setting import configure_run
    command = run['command']
    index = next(i for i, item in enumerate(command) if Path(item).name == 'main.py')
    planned = get_args_parser().parse_args(command[index + 1:])
    configure_run(planned)
    for field, value in vars(planned).items():
        if field in ('start_epoch', 'resume', 'dist_url'):
            continue  # resume advances this counter without changing the recipe
        default = 0 if field == 'discovery_pseudo_ramp_epochs' else None
        if getattr(saved, field, default) != value:
            raise ValueError(f'Saved checkpoint differs from launch arguments: {field}')
    world_size = int(command[command.index('--nproc_per_node') + 1]) if '--nproc_per_node' in command else 1
    if getattr(saved, 'world_size', 1) != world_size:
        raise ValueError('Saved checkpoint world size differs from launch arguments')


def compare_runs(control, treatment):
    import math
    import torch
    from tools.owod.strict_setting import configure_run, validate_checkpoint
    def read(root):
        plan = json.loads((root / 'benchmark_plan.json').read_text())
        runs = []
        for run in plan['runs']:
            if run.get('kind') == 'calibration':
                continue
            directory = Path(run['output_dir'])
            marker = json.loads((directory / 'training_complete.json').read_text())
            if marker['last_epoch'] != run['epochs'] - 1 or not (directory / 'checkpoint.pth').is_file():
                raise ValueError(f'Incomplete training phase: {directory}')
            checkpoint = load_checkpoint_metadata(directory / 'checkpoint.pth')
            saved = copy.deepcopy(checkpoint['args'])
            validate_plan_arguments(run, saved)
            configure_run(saved)
            validate_checkpoint(saved, checkpoint, resume=True)
            if (saved.owod_manifest_sha256 != plan['manifest_sha256']
                    or saved.owod_stage != run['stage'] or saved.owod_phase != run['phase']
                    or saved.epochs != run['epochs'] or checkpoint['epoch'] != saved.epochs - 1):
                raise ValueError('Checkpoint provenance differs from the benchmark plan')
            rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            final = [row for row in rows if row['epoch'] == run['epochs'] - 1]
            if len(final) != 1:
                raise ValueError('Require exactly one final evaluation, not a best-epoch selection')
            runs.append((run, final[0]))
        return plan, runs
    baseline, before = read(control)
    experiment, after = read(treatment)
    for field in ('protocol', 'manifest_sha256', 'backbone_weights_sha256', 'recipe'):
        if baseline[field] != experiment[field]:
            raise ValueError(f'Unpaired AP comparison: {field}')
    if len(before) != 7 or len(after) != 7:
        raise ValueError('Require completed four-task chains, with separate finetuning')
    # Permit only method arguments, source/output paths to differ.
    def common(command):
        ignored = {'--output_dir', '--pretrained', '--resume', '--graph-gnn-checkpoint'}
        boolean = {'--graph-routed-adapter', '--owod-discovery', '--discovery-no-transfer', '--discovery-no-message'}
        result, i = [], 0
        while i < len(command):
            item = command[i]
            if item in ignored:
                i += 2
            elif item in boolean:
                i += 1
            elif item.startswith('--graph-') or item.startswith('--discovery-'):
                i += 2
            else:
                result.append(item)
                i += 1
        return result
    comparisons = []
    for (a, x), (b, y) in zip(before, after):
        if ((a['stage'], a['phase'], a['epochs']) != (b['stage'], b['phase'], b['epochs'])
                or common(a['command']) != common(b['command'])):
            raise ValueError('Training data, schedule, seed or optimizer changed')
        metrics = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'h_score',
                   'a_ose', 'wi', 'unknown_precision')
        differences = {}
        for metric in metrics:
            key = 'test_owod_' + metric
            if key in x and key in y:
                if not math.isfinite(x[key]) or not math.isfinite(y[key]):
                    raise ValueError('Nonfinite benchmark metric')
                differences[metric] = {'control': x[key], 'treatment': y[key], 'delta': y[key] - x[key]}
        if 'known_ap50' not in differences or 'current_ap50' not in differences:
            raise ValueError('Missing author AP50; COCO AP cannot replace it')
        if a['stage'] > 0 and 'previous_ap50' not in differences:
            raise ValueError('Missing previous-class AP50')
        if a['stage'] < 3 and not {'u_recall', 'wi', 'a_ose'}.issubset(differences):
            raise ValueError('Missing open-world evaluation')
        comparisons.append({'task': a['stage'] + 1, 'phase': a['phase'], 'metrics': differences})
    expected = [(1, 'increment')] + [(t, p) for t in range(2, 5) for p in ('increment', 'finetune')]
    if [(row['task'], row['phase']) for row in comparisons] != expected:
        raise ValueError('Task/phase sequence differs from the official four-task chain')
    final = [row for row in comparisons if row['phase'] == 'finetune' or row['task'] == 1]
    known_delta = sum(row['metrics']['known_ap50']['delta'] for row in final) / 4
    current_delta = sum(row['metrics']['current_ap50']['delta'] for row in final) / 4
    return {'status': 'measured_single_seed', 'protocol': baseline['protocol'],
            'comparisons': comparisons, 'mean_task_known_ap50_delta': known_delta,
            'mean_task_current_ap50_delta': current_delta,
            'metric_scale': 'AP50 and recall are fractions; delta_points fields multiply AP deltas by 100.',
            'mean_task_known_ap50_delta_points': 100 * known_delta,
            'mean_task_current_ap50_delta_points': 100 * current_delta,
            'ap_improved_on_this_run': known_delta > 0 and current_delta > 0,
            'limitation': 'Paired single-seed evidence; repeat seeds before claiming robustness.'}


def mechanisms(steps=100, device='cpu'):
    import torch
    from torch import nn
    from models.deformable_transformer import DeformableTransformerDecoderLayer
    from models.deformable_detr import SetCriterion
    from models.matcher import HungarianMatcher
    from models.graph_local.discovery import DiscoveryDetector, DiscoveryCriterion, make_ema, update_ema
    from models.graph_local.discovery_candidates import flip_samples
    from models.graph_local.routed_adapter import append_stage, configure_trainable, mask_inactive_adapter_gradients
    from util.misc import nested_tensor_from_tensor_list
    from types import SimpleNamespace

    torch.manual_seed(42)
    torch.set_num_threads(2)
    class ControlledProposals(nn.Module):
        # Fixed proposals isolate learning/state mechanics. This is NOT an AP experiment.
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.decoder = nn.Module()
            self.transformer.decoder.layers = nn.ModuleList([
                DeformableTransformerDecoderLayer(d_model=4, d_ffn=8, dropout=0, n_heads=1)])
            self.class_embed = nn.ModuleList([nn.Linear(4, 7)])
            self.bbox_embed = nn.Linear(4, 4)
            nn.init.constant_(self.class_embed[0].bias, -4.)

        def forward(self, samples):
            features = torch.eye(4, device=samples.tensors.device)[:3][None]
            features = self.transformer.decoder.layers[0].forward_ffn(features)
            boxes = features.new_tensor([[[.2, .2, .2, .2], [.7, .7, .2, .2], [.85, .15, .1, .1]]])
            if samples.tensors[0, 0, 0, 0] > .5:
                boxes[..., 0] = 1 - boxes[..., 0]
            heat = features.new_zeros(1, 1, 10, 10)
            heat[:, :, 5:9, 5:9] = 1
            if samples.tensors[0, 0, 0, 0] > .5:
                heat = heat.flip(-1)
            from models.graph_local.routed_adapter import routing_outputs
            return dict(pred_logits=self.class_embed[0](features), pred_boxes=boxes,
                        decoder_features=features[None], attention_feature=heat,
                        padded_size=(10, 10), adapter_routes=routing_outputs(self))

    config = SimpleNamespace(owod_known_class_ids=[1], discovery_no_transfer=False,
        discovery_transfer_threshold=.8, discovery_min_support=3., hidden_dim=4,
        discovery_capacity=4, graph_adapter_last_n=1, graph_adapter_rank=2,
        discovery_warmup=0, discovery_top_k=5, discovery_saliency_threshold=.65,
        discovery_consistency_threshold=.5)
    model = DiscoveryDetector(ControlledProposals(), config).to(device)
    base = SetCriterion(7, HungarianMatcher(), {'loss_ce': 2., 'loss_bbox': 5., 'loss_giou': 2.}, ['labels', 'boxes'])
    base.active_class_ids = [1]
    criterion = DiscoveryCriterion(base, model, config).to(device)
    teacher = make_ema(model)
    samples = nested_tensor_from_tensor_list([torch.linspace(0, 1, 10, device=device)[None, None].expand(3, 10, 10)])
    targets = [{'labels': torch.tensor([1], device=device), 'boxes': torch.tensor([[.2, .2, .2, .2]], device=device)}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
    history = []
    for step in range(steps):
        outputs = model(samples)
        with torch.no_grad():
            outputs['discovery_teacher'] = teacher(flip_samples(samples))
        losses = criterion(outputs, targets)
        total = sum(value * criterion.weight_dict[key] for key, value in losses.items() if key in criterion.weight_dict)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        model.observe_training(criterion.evidence)
        update_ema(teacher, model, decay=.9)
        history.append({'step': step, 'loss': total.item(), 'pseudo': losses['discovery_pseudo_count'].item(),
                        'unknown_objectness': outputs['pred_objectness'].sigmoid()[0, 1].item(),
                        'background_objectness': outputs['pred_objectness'].sigmoid()[0, 2].item(),
                        'unknown_novelty': outputs['pred_novelty'].sigmoid()[0, 1].item()})
    with torch.no_grad():
        observations = model(samples)
        mean = observations['query_features'][0, 1].clone()
        route_mean = observations['discovery_inputs'][0][0, 1].clone()
    layout = dict(stage=1, class_ids=[3], old_ids=[1], rank=2, last_n=1, risk_beta=2., risk=[[0.]])
    scratch = copy.deepcopy(model)
    append_stage(model, layout)
    promoted = model.promote({3: mean}, layout, {3: [route_mean]})
    scratch.transfer_enabled = False
    append_stage(scratch, layout)
    scratch.promote({3: mean}, layout, {3: [route_mean]})
    transition = []
    for candidate in (model, scratch):
        candidate.known_ids = [1, 3]
        configure_trainable(candidate, [3], [1, 3])
        # Isolate adapter inheritance, holding the discovery extractor fixed.
        opt = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=.01, weight_decay=0)
        target = [{'labels': torch.tensor([3], device=device), 'boxes': torch.tensor([[.7, .7, .2, .2]], device=device)}]
        new_base = SetCriterion(7, HungarianMatcher(), {'loss_ce': 2., 'loss_bbox': 5., 'loss_giou': 2., 'loss_adapter_route': 1.}, ['labels', 'boxes'])
        new_base.active_class_ids = [1, 3]
        curve = []
        for _ in range(30):
            output = candidate(samples)
            losses = new_base(output, target)
            loss = sum(value * new_base.weight_dict[key] for key, value in losses.items() if key in new_base.weight_dict)
            opt.zero_grad()
            loss.backward()
            mask_inactive_adapter_gradients(candidate)
            opt.step()
            curve.append(loss.item())
        transition.append(curve)
    success = (history[-1]['unknown_objectness'] > history[-1]['background_objectness']
               and history[-1]['unknown_novelty'] > .5 and any(r['inherited'] for r in promoted)
               and transition[0][-1] < transition[0][0])
    return {'status': 'mechanism_passed' if success else 'mechanism_failed',
            'evidence_scope': 'Controlled fixed query proposals; neither real-image discovery nor benchmark AP.',
            'history': history, 'promotion': promoted,
            'promotion_new_class_loss_curve': transition[0], 'scratch_new_class_loss_curve': transition[1],
            'ap_improvement': None,
            'next_if_failed': ['inspect candidate quality and null routing',
                               'reduce transfer strength or reject low-confidence prototypes',
                               'measure a detection baseline before adding more graph complexity']}


def real_batches(manifest, weights, batches=3):
    """Exercise actual engine/prefetching on legal portrait/landscape train images.

    This is a resource check on a fresh disposable model, not a benchmark run.
    The official study still starts fresh and uses every author training image.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from main import get_args_parser
    from engine import train_one_epoch
    from models.graph_local.discovery import build, make_ema
    from tools.owod.strict_setting import configure_run
    from tools.owod.gnn_calibration import build_calibration_dataset
    from util.misc import collate_fn
    args = get_args_parser().parse_args([
        '--owod-manifest', str(manifest.resolve()), '--owod-stage', '0',
        '--owod-recipe', 'ow-detr-paper', '--backbone-weights', str(weights.resolve()),
        '--owod-discovery', '--activation-checkpointing', '--num_classes', '91',
        '--num_queries', '100', '--dim_feedforward', '1024'])
    configure_run(args)
    payload = json.loads(Path(args.train_ann).read_text(encoding='utf-8'))
    landscape = [i['id'] for i in payload['images'] if i['width'] / i['height'] >= 1.6][:batches]
    portrait = [i['id'] for i in payload['images'] if i['height'] / i['width'] >= 1.6][:batches]
    if len(landscape) != batches or len(portrait) != batches:
        raise ValueError('Insufficient legal train images for mixed-orientation resource check')
    dataset = build_calibration_dataset(Path(args.owod_image_root), Path(args.train_ann))
    positions = {image_id: index for index, image_id in enumerate(dataset.ids)}
    selected = [i for pair in zip(landscape, portrait) for i in pair]
    loader = DataLoader(Subset(dataset, [positions[i] for i in selected]), batch_size=2,
                        num_workers=0, collate_fn=collate_fn, pin_memory=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    model, criterion, _ = build(args)
    model.cuda()
    criterion.cuda()
    teacher = make_ema(model)
    groups = {2e-4: [], 2e-5: []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rate = 2e-5 if ('backbone.0' in name or any(k in name for k in
                            ('sampling_offsets', 'reference_points'))) else 2e-4
            groups[rate].append(parameter)
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': rate}
                                  for rate, parameters in groups.items()], lr=2e-4, weight_decay=1e-4)
    stats = train_one_epoch(model, criterion, loader, optimizer, torch.device('cuda'),
                            epoch=0, max_norm=.1, print_freq=1, discovery_teacher=teacher)
    return {'status': 'real_batch_execution_passed',
            'evidence_scope': 'Resource/engine check on fresh model and legal train images; no AP evidence.',
            'image_ids': selected, 'batches': batches, 'batch_size': 2,
            'activation_checkpointing': True, 'manifest_sha256': args.owod_manifest_sha256,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
            'train_stats': stats, 'ap_improvement': None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('mechanisms')
    smoke.add_argument('--steps', type=int, default=100)
    smoke.add_argument('--device', default='cpu')
    report = sub.add_parser('compare')
    report.add_argument('--control', type=Path, required=True)
    report.add_argument('--treatment', type=Path, required=True)
    hardware = sub.add_parser('real-batches')
    hardware.add_argument('--manifest', type=Path, required=True)
    hardware.add_argument('--weights', type=Path, required=True)
    hardware.add_argument('--batches', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'mechanisms':
            result = mechanisms(args.steps, args.device)
        elif args.command == 'real-batches':
            result = real_batches(args.manifest, args.weights, args.batches)
        else:
            result = compare_runs(args.control, args.treatment)
    except (FileNotFoundError, ValueError) as error:
        result = {'status': 'not_measured', 'ap_improvement': None, 'reason': str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'history' not in k and 'curve' not in k}, indent=2))
    return 0 if result['status'] in ('mechanism_passed', 'measured_single_seed', 'real_batch_execution_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
