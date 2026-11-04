"""Bounded paired CUDA debug from an existing checkpoint; never save weights.

Input fixture contains only released training labels. This consumes a repeated
small training stream, not a benchmark split, AP result or formal continuation.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def resident_mib():
    """Process RAM, without adding a profiling package dependency."""
    import os
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_ws', 'ws', 'peak_paged', 'paged',
                                                    'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.c_void_p(get_process()), ctypes.byref(counters), counters.cb):
            raise OSError('Cannot measure process RAM')
        return counters.ws / 2**20
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**20


def main():
    import torch
    from PIL import Image
    from datasets.coco import make_coco_transforms
    from models.graph_local.discovery import make_ema, update_ema
    from models.graph_local.discovery_candidates import flip_samples
    from tools.owod.gnn_calibration import load_detector
    from tools.owod.protocol import file_sha256
    from util.misc import collate_fn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fixture', type=Path)
    source.add_argument('--manifest', type=Path,
                        help='Read the real author train annotation on this host; no fixture needed')
    parser.add_argument('--images', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--full-resolution', action='store_true', help='800 short side instead of debug 384')
    parser.add_argument('--old-teacher-checkpoint', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', default=40, type=int)
    parser.add_argument('--memory-limit-mib', default=6000, type=int)
    parser.add_argument('--ram-limit-mib', default=8192, type=int)
    parser.add_argument('--seconds-per-policy', default=180, type=int)
    opt = parser.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        opt.output = opt.output.with_name(opt.output.stem + f'.rank{rank}' + opt.output.suffix)
    if (not 3 <= opt.steps <= 100 or opt.output.exists() or
            min(opt.memory_limit_mib, opt.ram_limit_mib, opt.seconds_per_policy) <= 0 or
            not 1 <= opt.batch_size <= 2 or not world * opt.batch_size <= opt.images <= 32):
        parser.error('Use 3..100 steps, positive resource budgets and a fresh output path')
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; no unbounded CPU training fallback')
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        from datetime import timedelta
        torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=120))
    identity = file_sha256(opt.checkpoint)
    payload = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    saved = payload['args']
    if not getattr(saved, 'owod_discovery', False) or saved.owod_stage not in (0, 1):
        parser.error('This paired debug accepts discovery Task 1/2 checkpoints only')
    if saved.owod_stage == 1 and not opt.old_teacher_checkpoint:
        parser.error('Task 2 debug requires the actual Task 1 frozen teacher')
    released = set(saved.owod_known_class_ids)
    transforms = make_coco_transforms('val', lightweight=not opt.full_resolution)
    stream = []
    if opt.manifest:
        from datasets.coco import CocoDetection
        from tools.owod.protocol import stage_files
        if file_sha256(opt.manifest) != saved.owod_manifest_sha256:
            parser.error('Manifest/checkpoint mismatch')
        manifest, files = stage_files(opt.manifest, saved.owod_stage)
        dataset = CocoDetection(manifest['image_root'], files['increment_train'], transforms,
                                return_masks=False, cache_mode=False)
        # Only retain dataset indices; decode/resize batches on demand.
        stream = list(range(min(opt.images, len(dataset))))
        if len(stream) < world * opt.batch_size:
            parser.error('Insufficient author train images for the requested batch/world size')
        source_identity = file_sha256(files['increment_train'])
        fixture = {'images': []}
    else:
        fixture = json.loads(opt.fixture.read_text(encoding='utf-8'))
        if fixture['manifest_sha256'] != saved.owod_manifest_sha256:
            parser.error('Fixture/checkpoint manifest mismatch')
        source_identity = file_sha256(opt.fixture)
    for entry in fixture['images']:
        if not set(entry['labels']) <= released:
            parser.error('Fixture has unreleased labels')
        path = opt.fixture.parent / entry['file']
        if file_sha256(path) != entry['sha256']:
            parser.error('Image hash mismatch')
        target = {'labels': torch.tensor(entry['labels']),
                  'boxes': torch.tensor(entry['boxes'], dtype=torch.float32).reshape(-1, 4)}
        image, target = transforms(Image.open(path).convert('RGB'), target)
        stream.append((image, target))
    if not stream or len(stream) > 32:
        parser.error('Fixture must contain 1..32 training images')
    # Parent optimizer/EMA state is not used to initialize diagnostic updates.
    del payload
    result = {'diagnostic_only': True, 'ap_measured': False,
              'checkpoint_sha256': identity, 'training_source_sha256': source_identity,
              'scope': f'Task {saved.owod_stage + 1} repeated released-label train stream; fresh AdamW',
              'gpu': torch.cuda.get_device_name(), 'steps': opt.steps, 'policies': {},
              'world_size': world, 'rank': rank, 'batch_size_per_rank': opt.batch_size,
              'short_side': 800 if opt.full_resolution else 384}
    opt.output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        opt.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    try:
        frozen = None
        if saved.owod_stage == 1:
            teacher_identity = file_sha256(opt.old_teacher_checkpoint)
            source = getattr(saved, 'graph_adapter_specs', [])[-1]['source_checkpoint_sha256']
            if teacher_identity != source:
                raise ValueError('Frozen teacher is not the recorded Task 2 parent')
            frozen, _ = load_detector(saved, opt.old_teacher_checkpoint, device)
            frozen.eval().requires_grad_(False)
            result['old_teacher_sha256'] = teacher_identity
        for policy in (5, 6):
            torch.manual_seed(42)
            model, criterion = load_detector(saved, opt.checkpoint, device)
            if policy == 6:
                model.memory.enable_adaptive()
            criterion.revision = policy
            criterion.epoch = max(criterion.warmup, 1)
            model.train()
            criterion.train()
            hooks = []
            if saved.owod_stage == 1:
                from models.graph_local.routed_adapter import configure_trainable
                hooks = configure_trainable(model, saved.owod_current_class_ids,
                                            saved.owod_known_class_ids,
                                            localization=saved.graph_localization)
                model.train_discovery()
                criterion.weight_dict['loss_adapter_route'] = saved.graph_route_loss_coef
            teacher = make_ema(model)
            network = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
                       if world > 1 else model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
            initial_memory = model.memory.centers[91:].detach().cpu().clone()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            measurements = []
            for step in range(opt.steps):
                if time.monotonic() - started > opt.seconds_per_policy:
                    raise RuntimeError('Diagnostic wall-time budget reached')
                positions = [(step * world * opt.batch_size + rank * opt.batch_size + j) % len(stream)
                             for j in range(opt.batch_size)]
                samples, targets = collate_fn([dataset[stream[j]] if opt.manifest else stream[j]
                                               for j in positions])
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                outputs = network(samples)
                with torch.no_grad():
                    outputs['discovery_teacher'] = teacher(flip_samples(samples))
                    if frozen is not None:
                        outputs['discovery_old_teacher'] = frozen(samples)
                losses = criterion(outputs, targets)
                if frozen is not None:
                    from models.graph_local.distillation import old_class_distillation_losses
                    extra, _ = old_class_distillation_losses(
                        outputs, outputs['discovery_old_teacher'], saved.owod_previous_class_ids,
                        saved.distill_score_threshold, saved.distill_max_queries)
                    losses.update(extra)
                    criterion.weight_dict.update(loss_distill_cls=saved.distill_class_coef,
                                                 loss_distill_bbox=saved.distill_bbox_coef)
                total = sum(v * criterion.weight_dict[k] for k, v in losses.items()
                            if k in criterion.weight_dict)
                if not torch.isfinite(total):
                    raise RuntimeError('Nonfinite loss')
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                optimizer.step()
                model.observe_training(criterion.evidence)
                update_ema(teacher, model)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2**20
                ram = resident_mib()
                if peak > opt.memory_limit_mib or ram > opt.ram_limit_mib:
                    raise RuntimeError('Diagnostic CUDA/RAM memory budget reached')
                measurements.append({'loss': float(total), 'grad_norm': float(norm),
                                     'pseudo': float(losses['discovery_pseudo_count']),
                                     'old_known_evidence': int(torch.isin(
                                         criterion.evidence[1], criterion.evidence[1].new_tensor(
                                             saved.owod_previous_class_ids)).sum()),
                                     'rss_mib': ram,
                                     'live_mib': torch.cuda.memory_allocated() / 2**20})
                del outputs, losses, total
            result['policies'][str(policy)] = {
                'seconds': time.monotonic() - started, 'peak_allocated_mib': peak,
                'peak_sampled_rss_mib': max(m['rss_mib'] for m in measurements),
                'measurements': measurements,
                'anonymous_centers_changed': not torch.equal(initial_memory, model.memory.centers[91:].cpu()),
                'anonymous_support': model.memory.counts[91:].cpu().tolist(),
                'parameter_count': sum(p.numel() for p in model.parameters())}
            print(json.dumps({'policy': policy, **{k: v for k, v in result['policies'][str(policy)].items()
                                                 if k != 'measurements'}}), flush=True)
            write()
            for hook in hooks:
                hook.remove()
            del network, model, criterion, teacher, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        baseline, treatment = (result['policies'][str(v)] for v in (5, 6))
        result['resource_guard_passed'] = (
            treatment['peak_allocated_mib'] <= baseline['peak_allocated_mib'] * 1.2 + 64 and
            treatment['peak_sampled_rss_mib'] <= baseline['peak_sampled_rss_mib'] * 1.2 + 128 and
            treatment['measurements'][-1]['live_mib'] <= treatment['measurements'][2]['live_mib'] + 16
            if opt.steps >= 3 else False)
        result['status'] = 'completed'
        if frozen is not None:
            result['old_teacher_unchanged'] = file_sha256(opt.old_teacher_checkpoint) == teacher_identity
    except Exception as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        result['checkpoint_unchanged'] = file_sha256(opt.checkpoint) == identity
        write()
        if world > 1:
            torch.distributed.destroy_process_group()
    if not result['resource_guard_passed'] or not result['checkpoint_unchanged']:
        raise RuntimeError('Diagnostic resource/identity guard failed')


if __name__ == '__main__':
    main()
