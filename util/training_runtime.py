"""Runtime optimizations that do not change the training recipe or state format."""
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist


def atomic_torch_save(value, destination, **kwargs):
    """Keep the previous checkpoint intact until the new file is complete."""
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp',
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(value, handle, **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@torch.no_grad()
def gather_evidence(evidence):
    """Gather variable-length evidence in rank order, preserving local row order.

    NCCL receives CUDA tensors directly. No pickle, CPU staging, or semantic
    label conversion is needed. Empty ranks still participate in collectives.
    """
    evidence = tuple(tensor.detach() for tensor in evidence)
    if not dist.is_available() or not dist.is_initialized():
        return evidence
    count = evidence[0].new_tensor([len(evidence[0])], dtype=torch.long)
    counts = [torch.empty_like(count) for _ in range(dist.get_world_size())]
    dist.all_gather(counts, count)
    lengths = torch.cat(counts).cpu().tolist()
    maximum = max(lengths)
    if maximum == 0:
        return evidence
    result = []
    for tensor in evidence:
        padded = tensor.new_zeros((maximum, *tensor.shape[1:]))
        padded[:len(tensor)].copy_(tensor)
        gathered = [torch.empty_like(padded) for _ in lengths]
        dist.all_gather(gathered, padded)
        result.append(torch.cat([part[:length] for part, length in zip(gathered, lengths)]))
    return tuple(result)


@torch.no_grad()
def foreach_ema_parameters(teacher, model, decay):
    """Use separate multiply/add passes, retaining the original EMA arithmetic."""
    groups = defaultdict(lambda: ([], []))
    for target, source in zip(teacher.parameters(), model.parameters()):
        targets, sources = groups[(target.device, target.dtype)]
        targets.append(target)
        sources.append(source)
    for targets, sources in groups.values():
        torch._foreach_mul_(targets, decay)
        torch._foreach_add_(targets, sources, alpha=1 - decay)


@torch.no_grad()
def copy_model_buffers(teacher, model):
    target_buffers = dict(teacher.named_buffers())
    groups = defaultdict(lambda: ([], []))
    for name, source in model.named_buffers():
        target = target_buffers[name]
        targets, sources = groups[(target.device, target.dtype)]
        targets.append(target)
        sources.append(source)
    for targets, sources in groups.values():
        torch._foreach_copy_(targets, sources)


def loader_worker_options(num_workers):
    # Recreated workers must not fork a rank after CUDA has been initialized.
    # Keep non-persistent workers so epoch-level worker seeding is unchanged.
    return {'multiprocessing_context': 'spawn'} if num_workers > 0 else {}


def training_epoch_limit(args):
    """An execution pause never shortens the optimizer's registered schedule."""
    budget = getattr(args, 'pause_after_epochs', 0)
    if budget < 0 or budget > args.epochs:
        raise ValueError('pause-after-epochs must be between 0 and the registered epoch count')
    control = not any(getattr(args, name, False) for name in
                      ('owod_discovery', 'graph_routed_adapter', 'paper_baseline', 'with_tree'))
    if budget and (not getattr(args, 'owod_manifest', '')
                   or not (control or getattr(args, 'owod_discovery', False))
                   or getattr(args, 'owod_stage', None) != 0
                   or getattr(args, 'owod_phase', None) != 'increment'
                   or getattr(args, 'owod_screening', '')):
        raise ValueError('pause-after-epochs is restricted to full-data control/discovery Task 1')
    return budget or args.epochs


def write_training_status(directory, epochs, last_epoch, seconds, merged_checkpoint=None):
    """A paused stage must never acquire a completion marker."""
    directory = Path(directory)
    complete = last_epoch == epochs - 1
    opposite = directory / ('training_paused.json' if complete else 'training_complete.json')
    if opposite.exists():
        opposite.unlink()
    path = directory / ('training_complete.json' if complete else 'training_paused.json')
    status = dict(epochs=epochs, last_epoch=last_epoch, training_time_seconds=int(seconds),
                  merged_checkpoint=merged_checkpoint)
    if not complete:
        status.update(completed_epochs=last_epoch + 1, stage_complete=False,
                      reason='execution_pause_budget', resumable=True)
    path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
