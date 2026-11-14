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


def diagnostic_identity(manifest, checkpoint, gnn, world_size=1):
    # Include implementation changes so --resume cannot silently change methods.
    sources = [ROOT / 'main.py', ROOT / 'engine.py']
    for folder in ('models', 'datasets', 'util', 'tools/owod'):
        sources.extend(p for p in sorted((ROOT / folder).rglob('*.py')) if 'tests' not in p.parts)
    return dict(manifest_sha256=file_sha256(manifest), checkpoint_sha256=file_sha256(checkpoint),
                gnn_sha256=file_sha256(gnn), epochs=3, batch_size=2 // world_size,
                world_size=world_size, effective_batch_size=2,
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
    if identity.get('world_size', 1) > 1:
        states = checkpoint.get('rank_states', [])
        fields = {'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'loader_rng'}
        if len(states) != identity['world_size'] or any(not fields <= s.keys() for s in states):
            raise ValueError('Incomplete per-rank diagnostic resume state')


def capture_rank_state(generator):
    import numpy as np
    import torch
    return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                loader_rng=generator.get_state())


def restore_rank_state(state, generator):
    import numpy as np
    import torch
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    generator.set_state(state['loader_rng'])


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


def run_policy(opt, identity, policy, *, variant=None, provenance=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
    from datasets import build_dataset, get_coco_api_from_dataset
    from engine import train_one_epoch, evaluate
    from main import get_args_parser
    from models.graph_local.discovery import DiscoveryPostProcess, make_ema
    from models.graph_local.routed_adapter import configure_trainable
    from tools.owod.debug_discovery_revision import task2_arguments, prepare_task2
    from tools.owod.gnn_calibration import load_detector, atomic_torch_save
    from tools.owod.strict_setting import configure_run, validate_checkpoint
    from util.misc import collate_fn

    rank, world, gpu = opt.rank, opt.world_size, opt.gpu
    distributed = world > 1

    def write(path, value):
        if rank == 0:
            dump(path, value)

    if variant is not None and (variant not in ('hard', 'soft') or policy != 7 or provenance is None):
        raise ValueError('Routing comparison requires an explicit verified v7 diagnostic fork')
    directory = opt.output / (variant or f'v{policy}')
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
    args.epochs = 3
    args.start_epoch = 0
    args.batch_size = 2 // world
    args.num_workers = 2
    args.lr = 2e-4
    args.lr_backbone = 2e-5
    args.lr_drop = 50  # Keep increment LR constant during this short budget.
    args.weight_decay = 1e-4
    args.sgd = False
    args.eval = False
    args.distributed = distributed
    args.world_size = world
    args.rank = rank
    args.gpu = gpu
    args.device = 'cuda'
    args.discovery_policy = 'v5'
    configure_run(args)
    args.coco_path = args.owod_image_root
    # Strictly validate the unmodified v5 parent before the explicit diagnostic fork.
    if provenance is None:
        validate_checkpoint(args, parent)
        initialization_args = args
    else:
        from tools.owod.task2_parent_provenance import legacy_validation_arguments
        initialization_args = legacy_validation_arguments(args, parent, opt.manifest, provenance)
        args.diagnostic_parent_provenance = provenance
    if variant is not None:
        # Only after strict validation: Task 1 has no class bank, so both new
        # banks can be initialized identically with different routing rules.
        args.graph_routing = initialization_args.graph_routing = variant
        args.graph_localization = initialization_args.graph_localization = 'frozen'
        # Calibration uses deterministic transforms. Do not fork workers after
        # CUDA initialization in its class-by-class loaders on Linux.
        initialization_args.num_workers = 0
    if args.discovery_warmup >= 3:
        raise ValueError('Three epochs would contain no unknown-learning updates')
    if args.seed != 42 or args.lightweight or args.no_augmentation:
        raise ValueError('Diagnostic requires seed42 and ordinary full-resolution augmentation')
    del parent
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda', gpu)
    path = directory / 'checkpoint.pth'
    restored = torch.load(path, map_location='cpu', weights_only=False) if opt.resume and path.exists() else None
    resuming = restored is not None
    if restored:
        validate_diagnostic_resume(restored, identity, policy)
        if variant is not None and restored.get('variant') != variant:
            raise ValueError('Routing diagnostic variant changed on resume')
        if (restored['epoch'] == 2 and (directory / 'training_complete.json').is_file()
                and (directory / 'final_metrics.json').is_file()):
            print(f'v{policy} already completed; preserving its results.', flush=True)
            return
        # Reconstruct the diagnostic model from its own complete saved bank.
        model, criterion = load_detector(args, path, device)
        args = restored['args']
        args.rank, args.gpu = rank, gpu
    else:
        model, criterion = load_detector(source, opt.checkpoint, device)
        layout, means, routes = prepare_task2(model, criterion, initialization_args, opt.checkpoint)
        if distributed:
            # Every rank enters calibration, whose criterion uses collectives.
            packet = [(layout, means, routes)]
            torch.distributed.broadcast_object_list(packet, src=0)
            layout, means, routes = packet[0]
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
    optimizer = make_optimizer(model, args)
    network = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
               if distributed else model)
    teacher = make_ema(model)
    generator = torch.Generator().manual_seed(args.seed + rank)
    train = build_dataset('train', args)
    validation = build_dataset('val', args)
    # Reset shuffle and worker RNG independently of policy-specific model construction.
    train_sampler = (DistributedSampler(train, num_replicas=world, rank=rank,
                                       seed=args.seed, drop_last=True) if distributed
                     else RandomSampler(train, generator=generator))
    val_sampler = (DistributedSampler(validation, num_replicas=world, rank=rank, shuffle=False)
                   if distributed else SequentialSampler(validation))
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=train_sampler,
                              drop_last=True, num_workers=2, collate_fn=collate_fn,
                              pin_memory=True, generator=generator, multiprocessing_context='spawn')
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    val_loader = DataLoader(validation, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=2, collate_fn=collate_fn, pin_memory=True,
                            generator=val_generator, multiprocessing_context='spawn')
    base_ds = get_coco_api_from_dataset(validation)
    postprocessors = {'bbox': DiscoveryPostProcess(known_ids=args.owod_known_class_ids,
                        max_detections=args.owod_max_detections,
                        unknown_threshold=args.unknown_threshold, scoring=args.discovery_scoring)}
    write(directory / 'run_config.json', vars(args))

    def evaluate_now():
        return evaluate(model, criterion, postprocessors, val_loader, base_ds, device, str(directory),
                        owod_known_class_ids=args.owod_known_class_ids,
                        owod_unknown_threshold=args.unknown_threshold,
                        owod_previous_class_ids=args.owod_previous_class_ids,
                        owod_current_class_ids=args.owod_current_class_ids,
                        owod_specification=args.owod_evaluation, print_freq=100)[0]

    def learning_probe(phase):
        from tools.owod.task2_learning_probe import probe
        write(directory / f'{phase}_learning_probe.json',
              probe(model, criterion, args, opt.manifest, rank, world))

    start = 0
    if restored:
        optimizer.load_state_dict(restored['optimizer'])
        teacher.load_state_dict(restored['discovery_ema'], strict=True)
        start = restored['epoch'] + 1
        restore_rank_state(restored['rank_states'][rank] if distributed else restored, generator)
        del restored
    else:
        if variant is not None:
            from tools.owod.task2_learning_probe import state_sha256
            digest = state_sha256(model)
            if distributed:
                digests = [None] * world
                torch.distributed.all_gather_object(digests, digest)
                if len(set(digests)) != 1:
                    raise ValueError('Task 2 initialization differs across ranks')
            if variant == 'soft':
                initial = json.loads((opt.output / 'hard' / 'initialization.json').read_text())
                if initial['state_sha256'] != digest:
                    raise ValueError('hard/soft must start with identical weights and memory')
            write(directory / 'initialization.json', dict(state_sha256=digest, routing=variant,
                  diagnostic_only=True, parent_checkpoint_sha256=identity['checkpoint_sha256']))
        criterion.epoch = 0
        write(directory / 'initial_metrics.json', evaluate_now())
        if variant is not None:
            learning_probe('initial')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if not resuming and distributed:
        # Identical model initialization, distinct but policy-paired augmentation streams.
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
    for epoch in range(start, 3):
        if distributed:
            train_sampler.set_epoch(epoch)
        criterion.epoch = epoch
        started = time.monotonic()
        stats = train_one_epoch(network, criterion, train_loader, optimizer, device, epoch,
                                args.clip_max_norm, print_freq=100, teacher=frozen,
                                distill_old_class_ids=args.owod_previous_class_ids,
                                distill_class_coef=args.distill_class_coef,
                                distill_bbox_coef=args.distill_bbox_coef,
                                distill_score_threshold=args.distill_score_threshold,
                                distill_max_queries=args.distill_max_queries,
                                discovery_teacher=teacher)
        payload = dict(diagnostic_only=True, diagnostic_identity=identity, policy=policy,
                       epoch=epoch, args=args, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       discovery_ema=teacher.state_dict(), **capture_rank_state(generator))
        if variant is not None:
            payload['variant'] = variant
        if distributed:
            states = [None] * world
            torch.distributed.all_gather_object(states, capture_rank_state(generator))
            payload['rank_states'] = states
        if rank == 0:
            atomic_torch_save(payload, path)
        write(directory / f'epoch_{epoch}.json', dict(epoch=epoch, train=stats,
             seconds=time.monotonic()-started, anonymous_support=model.memory.counts[91:].cpu().tolist(),
             recycle_count=int(model.memory.recycle_count) if policy == 7 else None))
        del payload
        if distributed:
            torch.distributed.barrier()
    criterion.epoch = 2
    write(directory / 'final_metrics.json', evaluate_now())
    if variant is not None:
        from tools.owod.task2_learning_probe import parameter_audit
        learning_probe('final')
        write(directory / 'parameter_audit.json', parameter_audit(
            model, frozen.state_dict(), args.owod_current_class_ids, args.owod_previous_class_ids))
    write(directory / 'training_complete.json', dict(diagnostic_only=True, epoch=2, policy=policy))
    if distributed:
        torch.distributed.barrier()
    for hook in hooks:
        hook.remove()
    del network, model, criterion, teacher, frozen, optimizer
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
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.rank = int(os.environ.get('RANK', 0))
    opt.gpu = int(os.environ.get('LOCAL_RANK', 0))
    if (opt.world_size not in (1, 2) or not torch.cuda.is_available()
            or torch.cuda.device_count() != opt.world_size):
        parser.error('Requires one GPU or two GPUs via torchrun; effective batch is fixed at 2')
    torch.cuda.set_device(opt.gpu)
    if opt.world_size > 1:
        from datetime import timedelta
        torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30))
    for field in ('manifest', 'checkpoint', 'gnn', 'output'):
        setattr(opt, field, getattr(opt, field).resolve())
    identity = diagnostic_identity(opt.manifest, opt.checkpoint, opt.gnn, opt.world_size)
    opt.output.mkdir(parents=True, exist_ok=True)
    plan = opt.output / 'diagnostic_plan.json'
    if opt.rank == 0:
        if plan.exists():
            if not opt.resume or json.loads(plan.read_text()) != identity:
                raise ValueError('Use --resume with unchanged code and inputs, or a new output directory')
        elif any(opt.output.iterdir()):
            raise ValueError('Use an empty diagnostic output directory')
        dump(plan, identity)
    if opt.world_size > 1:
        torch.distributed.barrier()
    for policy in (5, 7):
        run_policy(opt, identity, policy)
    if (file_sha256(opt.checkpoint) != identity['checkpoint_sha256']
            or file_sha256(opt.gnn) != identity['gnn_sha256']):
        raise ValueError('Parent artifacts changed during diagnostic')
    summary = {f'v{p}': {phase: json.loads((opt.output / f'v{p}' / f'{phase}_metrics.json').read_text())
                        for phase in ('initial', 'final')} for p in (5, 7)}
    if opt.rank == 0:
        dump(opt.output / 'comparison.json', dict(diagnostic_only=True, results=summary))
        print(json.dumps(summary, indent=2))
    if opt.world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
