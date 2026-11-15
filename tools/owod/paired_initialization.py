"""Evidence for paired diagnostics; no RNG draws or training-state changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime_sha256():
    """Bind source contents, including uncommitted new source files, on resume."""
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
        '--exclude-standard', '--', 'main.py', 'engine.py', 'models', 'datasets',
        'util', 'tools/owod', 'environment.yml'], cwd=ROOT).decode('utf-8').split('\0')
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        path = ROOT / name
        if not path.is_file() or '/tests/' in name or path.suffix not in ('.py', '.sh', '.yml'):
            continue
        digest.update(name.encode('utf-8') + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def state_sha256(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{list(tensor.shape)}\0'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_initialization(model, args):
    """Run after DDP's parameter/buffer broadcast; all ranks receive failures."""
    import torch.distributed as dist
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    packet = [None]
    if rank == 0:
        try:
            shared = Path(args.initialization_audit)
            local = Path(args.output_dir) / 'initialization.json'
            identity = dict(source_sha256=runtime_sha256(), seed=args.seed,
                world_size=args.world_size, manifest_sha256=args.owod_manifest_sha256,
                backbone_sha256=args.backbone_weights_sha256)
            if args.resume:
                record = json.loads(local.read_text(encoding='utf-8'))
                if record != json.loads(shared.read_text(encoding='utf-8')):
                    raise ValueError('Local and shared initialization proof differ')
            else:
                record = dict(identity, model_sha256=state_sha256(model))
                if shared.exists():
                    if record != json.loads(shared.read_text(encoding='utf-8')):
                        raise ValueError('Paired model initialization or source differs')
                else:
                    atomic_json(shared, record)
                if local.exists() and record != json.loads(local.read_text(encoding='utf-8')):
                    raise ValueError('Existing local initialization proof differs')
                atomic_json(local, record)
            if any(record.get(key) != value for key, value in identity.items()):
                raise ValueError('Initialization provenance changed on resume')
            packet[0] = dict(record=record)
        except Exception as error:
            packet[0] = dict(error=f'{type(error).__name__}: {error}')
    if distributed:
        dist.broadcast_object_list(packet, src=0)
    if 'error' in packet[0]:
        raise ValueError(packet[0]['error'])
    args.initialization_sha256 = packet[0]['record']['model_sha256']
    print('Paired initialization SHA256:', args.initialization_sha256)


def recover_final_evaluation(log, args, stats, n_parameters):
    """Recover checkpoint-before-evaluation interruption at the fixed pause."""
    if not args.pause_after_epochs or args.start_epoch != args.pause_after_epochs:
        return
    text = Path(log).read_text(encoding='utf-8') if Path(log).is_file() else ''
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    epoch = args.start_epoch - 1
    existing = [row for row in rows if row['epoch'] == epoch]
    if len(existing) > 1:
        raise ValueError('Duplicate fixed-epoch evaluation')
    if existing:
        return
    with Path(log).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(epoch=epoch, n_parameters=n_parameters,
            evaluation_recovered_after_resume=True,
            **{f'test_{key}': value for key, value in stats.items()})) + '\n')
