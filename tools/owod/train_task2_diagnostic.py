"""Three-epoch full-author-data Task 2 diagnostic; never a formal-chain parent."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.protocol import file_sha256, stage_files


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def diagnostic_identity(manifest, checkpoint, gnn):
    # Include implementation changes so --resume cannot silently change methods.
    sources = [ROOT / 'main.py', ROOT / 'engine.py']
    for folder in ('models', 'datasets', 'util', 'tools/owod'):
        sources.extend(p for p in sorted((ROOT / folder).rglob('*.py')) if 'tests' not in p.parts)
    return dict(manifest_sha256=file_sha256(manifest), checkpoint_sha256=file_sha256(checkpoint),
                gnn_sha256=file_sha256(gnn), epochs=3, batch_size=2, world_size=1,
                policies=[5, 7], diagnostic_only=True, finetune=False,
                sources={p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sources})


def validate_diagnostic_resume(checkpoint, identity, policy):
    if (checkpoint.get('diagnostic_only') is not True
            or checkpoint.get('diagnostic_identity') != identity
            or checkpoint.get('policy') != policy
            or not 0 <= checkpoint.get('epoch', -1) < 3):
        raise ValueError('Diagnostic resume identity/policy/epoch mismatch')
    required = {'model', 'optimizer', 'discovery_ema', 'torch_rng', 'cuda_rng',
                'numpy_rng', 'python_rng', 'loader_rng', 'args'}
    if not required <= checkpoint.keys():
        raise ValueError('Incomplete diagnostic resume state')


def make_optimizer(model, args):
    import torch
    groups = [[], [], [], []]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = (1 if any(k in name for k in args.lr_backbone_names) else
                 2 if any(k in name for k in args.lr_linear_proj_names) else
                 3 if 'class_embed' in name else 0)
        groups[group].append(parameter)
    return torch.optim.AdamW([
        {'params': groups[0], 'lr': args.lr},
        {'params': groups[1], 'lr': args.lr_backbone},
        {'params': groups[2], 'lr': args.lr * args.lr_linear_proj_mult},
        {'params': groups[3], 'lr': args.lr * args.class_embed_lr_mult, 'weight_decay': 0.},
    ], lr=args.lr, weight_decay=args.weight_decay)


def run_policy(opt, identity, policy):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    from datasets import build_dataset, get_coco_api_from_dataset
    from engine import train_one_epoch, evaluate
    from main import get_args_parser
    from models.graph_local.discovery import DiscoveryPostProcess, make_ema
    from models.graph_local.routed_adapter import configure_trainable
    from tools.owod.debug_discovery_revision import task2_arguments, prepare_task2
    from tools.owod.gnn_calibration import load_detector, atomic_torch_save
    from tools.owod.strict_setting import configure_run, validate_checkpoint
    from util.misc import collate_fn

    directory = opt.output / f'v{policy}'
    directory.mkdir(exist_ok=True)
    parent = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    source = parent['args']
    if (parent.get('diagnostic_only') or source.owod_stage != 0
            or source.owod_phase != 'increment' or getattr(source, 'owod_screening', None)
            or getattr(source, 'discovery_version', None) != 5
            or parent.get('epoch') != source.epochs - 1
            or not getattr(source, 'owod_discovery', False)
            or not getattr(source, 'graph_routed_adapter', False)):
        raise ValueError('Requires completed full-data v5 discovery Task 1 parent')
    manifest, files = stage_files(opt.manifest, 1)
    # Fill newly introduced ordinary parser defaults, preserving actual parent settings.
    args = get_args_parser().parse_args([])
    vars(args).update(vars(copy.deepcopy(source)))
    args = task2_arguments(args, manifest, files, opt.manifest, opt.gnn)
    args.pretrained = str(opt.checkpoint)
    args.resume = ''
    args.output_dir = str(directory)
    args.coco_path = manifest['image_root']
    args.epochs = 3
    args.start_epoch = 0
    args.batch_size = 2
    args.num_workers = 2
    args.lr = 2e-4
    args.lr_backbone = 2e-5
    args.lr_drop = 50  # Keep increment LR constant during this short budget.
    args.weight_decay = 1e-4
    args.sgd = False
    args.eval = False
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.gpu = 0
    args.device = 'cuda'
    args.discovery_policy = 'v5'
    configure_run(args)
    # Strictly validate the unmodified v5 parent before the explicit diagnostic fork.
    validate_checkpoint(args, parent)
    if args.discovery_warmup >= 3:
        raise ValueError('Three epochs would contain no unknown-learning updates')
    if args.seed != 42 or args.lightweight or args.no_augmentation:
        raise ValueError('Diagnostic requires seed42 and ordinary full-resolution augmentation')
    del parent
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    path = directory / 'checkpoint.pth'
    restored = torch.load(path, map_location='cpu', weights_only=False) if opt.resume and path.exists() else None
    if restored:
        validate_diagnostic_resume(restored, identity, policy)
        if (restored['epoch'] == 2 and (directory / 'training_complete.json').is_file()
                and (directory / 'final_metrics.json').is_file()):
            print(f'v{policy} already completed; preserving its results.', flush=True)
            return
        # Reconstruct the diagnostic model from its own complete saved bank.
        model, criterion = load_detector(args, path, device)
        args = restored['args']
    else:
        model, criterion = load_detector(source, opt.checkpoint, device)
        layout, means, routes = prepare_task2(model, criterion, args, opt.checkpoint)
        model.promote({c: torch.tensor(v, device=device) for c, v in means.items()}, layout,
                      {c: [torch.tensor(v, device=device) for v in values] for c, values in routes.items()})
        args.graph_adapter_specs = [layout]
        if policy == 7:
            model.memory.enable_bounded(args.discovery_min_support)
        criterion.known_ids = criterion.base.active_class_ids = list(args.owod_known_class_ids)
        criterion.old_ids = list(args.owod_previous_class_ids)
    args.discovery_version = policy
    args.discovery_policy = f'v{policy}'
    args.diagnostic_only = True
    args.paper_comparable_protocol = False
    criterion.revision = policy
    hooks = configure_trainable(model, args.owod_current_class_ids, args.owod_known_class_ids,
                                localization=args.graph_localization)
    model.train_discovery()
    criterion.weight_dict['loss_adapter_route'] = args.graph_route_loss_coef
    frozen, _ = load_detector(source, opt.checkpoint, device)
    frozen.eval().requires_grad_(False)
    teacher = make_ema(model)
    optimizer = make_optimizer(model, args)
    generator = torch.Generator().manual_seed(args.seed)
    train = build_dataset('train', args)
    validation = build_dataset('val', args)
    # Reset shuffle and worker RNG independently of policy-specific model construction.
    train_loader = DataLoader(train, batch_size=2, sampler=RandomSampler(train, generator=generator),
                              drop_last=True, num_workers=2, collate_fn=collate_fn,
                              pin_memory=True, generator=generator, multiprocessing_context='spawn')
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    val_loader = DataLoader(validation, batch_size=2, sampler=SequentialSampler(validation),
                            num_workers=2, collate_fn=collate_fn, pin_memory=True,
                            generator=val_generator, multiprocessing_context='spawn')
    base_ds = get_coco_api_from_dataset(validation)
    postprocessors = {'bbox': DiscoveryPostProcess(known_ids=args.owod_known_class_ids,
                        max_detections=args.owod_max_detections,
                        unknown_threshold=args.unknown_threshold, scoring=args.discovery_scoring)}
    dump(directory / 'run_config.json', vars(args))

    def evaluate_now():
        return evaluate(model, criterion, postprocessors, val_loader, base_ds, device, str(directory),
                        owod_known_class_ids=args.owod_known_class_ids,
                        owod_unknown_threshold=args.unknown_threshold,
                        owod_previous_class_ids=args.owod_previous_class_ids,
                        owod_current_class_ids=args.owod_current_class_ids,
                        owod_specification=args.owod_evaluation, print_freq=100)[0]

    start = 0
    if restored:
        optimizer.load_state_dict(restored['optimizer'])
        teacher.load_state_dict(restored['discovery_ema'], strict=True)
        start = restored['epoch'] + 1
        torch.set_rng_state(restored['torch_rng'])
        torch.cuda.set_rng_state_all(restored['cuda_rng'])
        np.random.set_state(restored['numpy_rng'])
        random.setstate(restored['python_rng'])
        generator.set_state(restored['loader_rng'])
        del restored
    else:
        criterion.epoch = 0
        dump(directory / 'initial_metrics.json', evaluate_now())
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    for epoch in range(start, 3):
        criterion.epoch = epoch
        started = time.monotonic()
        stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch,
                                args.clip_max_norm, print_freq=100, teacher=frozen,
                                distill_old_class_ids=args.owod_previous_class_ids,
                                distill_class_coef=args.distill_class_coef,
                                distill_bbox_coef=args.distill_bbox_coef,
                                distill_score_threshold=args.distill_score_threshold,
                                distill_max_queries=args.distill_max_queries,
                                discovery_teacher=teacher)
        payload = dict(diagnostic_only=True, diagnostic_identity=identity, policy=policy,
                       epoch=epoch, args=args, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       discovery_ema=teacher.state_dict(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                       python_rng=random.getstate(), loader_rng=generator.get_state())
        atomic_torch_save(payload, path)
        dump(directory / f'epoch_{epoch}.json', dict(epoch=epoch, train=stats,
             seconds=time.monotonic()-started, anonymous_support=model.memory.counts[91:].cpu().tolist(),
             recycle_count=int(model.memory.recycle_count) if policy == 7 else None))
        del payload
    criterion.epoch = 2
    dump(directory / 'final_metrics.json', evaluate_now())
    dump(directory / 'training_complete.json', dict(diagnostic_only=True, epoch=2, policy=policy))
    for hook in hooks:
        hook.remove()
    del model, criterion, teacher, frozen, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gnn', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    opt = parser.parse_args()
    import torch
    if int(os.environ.get('WORLD_SIZE', 1)) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('Requires exactly one visible CUDA GPU; do not use torchrun')
    for field in ('manifest', 'checkpoint', 'gnn', 'output'):
        setattr(opt, field, getattr(opt, field).resolve())
    identity = diagnostic_identity(opt.manifest, opt.checkpoint, opt.gnn)
    opt.output.mkdir(parents=True, exist_ok=True)
    plan = opt.output / 'diagnostic_plan.json'
    if plan.exists():
        if not opt.resume or json.loads(plan.read_text()) != identity:
            raise ValueError('Use --resume with unchanged code and inputs, or a new output directory')
    elif any(opt.output.iterdir()):
        raise ValueError('Use an empty diagnostic output directory')
    dump(plan, identity)
    for policy in (5, 7):
        run_policy(opt, identity, policy)
    if (file_sha256(opt.checkpoint) != identity['checkpoint_sha256']
            or file_sha256(opt.gnn) != identity['gnn_sha256']):
        raise ValueError('Parent artifacts changed during diagnostic')
    summary = {f'v{p}': {phase: json.loads((opt.output / f'v{p}' / f'{phase}_metrics.json').read_text())
                        for phase in ('initial', 'final')} for p in (5, 7)}
    dump(opt.output / 'comparison.json', dict(diagnostic_only=True, results=summary))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
