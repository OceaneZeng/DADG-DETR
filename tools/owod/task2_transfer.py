"""Paired Task 2 initialization and phase bookkeeping for the transfer ablation."""
import json

from tools.owod.protocol import file_sha256


def schedule(epochs):
    if epochs == 3:
        return {'increment': [1, 3]}
    if epochs == 50:
        return {'increment': [1, 3, 8, 50], 'finetune': [20]}
    raise ValueError('Transfer requires 3 increment epochs or the full 50 + 20 schedule')


def shared_calibration(model, criterion, configuration, opt, identity):
    """Calibrate once. Both arms read identical pre-promotion tensors and centroids."""
    import torch
    from models.graph_local.routed_adapter import append_stage
    from tools.owod.debug_discovery_revision import prepare_task2
    from tools.owod.gnn_calibration import atomic_torch_save

    path = opt.output / 'common_initialization.pth'
    create = [not path.is_file() if opt.rank == 0 else None]
    if opt.world_size > 1:
        torch.distributed.broadcast_object_list(create, src=0)
    if create[0]:
        layout, means, routes = prepare_task2(model, criterion, configuration, opt.checkpoint)
        if opt.rank == 0:
            atomic_torch_save(dict(diagnostic_only=True, diagnostic_identity=identity, layout=layout,
                                   means=means, routes=routes, model=model.state_dict()), path)
        if opt.world_size > 1:
            torch.distributed.barrier()
    shared = torch.load(path, map_location='cpu', weights_only=False)
    if shared.get('diagnostic_only') is not True or shared.get('diagnostic_identity') != identity:
        raise ValueError('Cached transfer initialization belongs to a different plan')
    if not create[0]:
        append_stage(model, shared['layout'])
    model.load_state_dict(shared['model'], strict=True)
    model.known_ids = criterion.known_ids = criterion.base.active_class_ids = list(configuration.owod_known_class_ids)
    return shared['layout'], shared['means'], shared['routes']


def promote_arm(model, layout, means, routes, arm):
    import torch
    from models.graph_local.routed_adapter import RiskRoutedStage
    from tools.owod.task2_learning_probe import state_sha256

    if arm not in ('inherit', 'scratch'):
        raise ValueError('Unknown transfer arm')
    # Both arms first install the exact same supervised router and keep the
    # ordinary Kaiming-down/zero-up initialization in all new class adapters.
    model.transfer_enabled = False
    model.promote(means, layout, routes)
    common = state_sha256(model)
    model.transfer_enabled = arm == 'inherit'
    if model.transfer_enabled:
        model.promote(means, layout, routes)
    matched = {row['class_id'] for row in layout['promotion'] if row['anonymous_slots']}
    excluded = set()
    norms = {cid: [] for cid in layout['class_ids']}
    with torch.no_grad():
        for name, stage in model.named_modules():
            if not isinstance(stage, RiskRoutedStage):
                continue
            if list(stage.class_ids) != layout['class_ids']:
                raise ValueError('Expected exactly the Task 2 class bank')
            for index, (cid, adapter) in enumerate(zip(stage.class_ids, stage.adapters)):
                if cid in matched:
                    excluded.update(f'{name}.adapters.{index}.{factor}.weight' for factor in ('up', 'down'))
                norm = (adapter.up.weight @ adapter.down.weight).norm()
                if not torch.isfinite(norm):
                    raise ValueError(f'Nonfinite transfer mapping for class {cid}')
                norms[cid].append(float(norm))
    return dict(arm=arm, common_initial_sha256=common,
                unchanged_state_sha256=state_sha256(model, exclude=excluded),
                promotion=layout['promotion'], effective_mapping_norms=norms,
                matched_classes=sorted(matched),
                nonzero_classes=[cid for cid, values in norms.items() if any(values)])


def check_transfer(audit, reference=None):
    if audit['arm'] == 'inherit' and not audit['nonzero_classes']:
        raise ValueError('No nonzero discovery mapping was inherited; see transfer.json. No training contrast.')
    if audit['arm'] == 'scratch' and audit['nonzero_classes']:
        raise ValueError('Scratch must start with zero effective class adapter mappings')
    if reference is not None:
        for field in ('common_initial_sha256', 'unchanged_state_sha256', 'matched_classes'):
            if audit[field] != reference[field]:
                raise ValueError(f'Transfer arms differ outside the intended initialization: {field}')
        def matches(report):
            return [{k: row[k] for k in ('class_id', 'anonymous_slots', 'weights')}
                    for row in report['promotion']]
        if matches(audit) != matches(reference):
            raise ValueError('Anonymous matching differs between arms')


def increment_parent(directory, plan, arm):
    """Require the exact finished increment arm before starting balanced finetuning."""
    import torch
    checkpoint = directory / 'checkpoint.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    from tools.owod.train_task2_diagnostic import validate_diagnostic_resume, validate_pair_resume
    validate_diagnostic_resume(payload, plan, 7)
    validate_pair_resume(payload, 'soft', 'frozen', arm)
    args = payload['args']
    end = plan['epochs'] - 1
    marker = json.loads((directory / 'training_complete.json').read_text())
    if (payload['epoch'] != end or marker['epoch'] != end or args.owod_phase != 'increment'
            or args.owod_stage != 1 or args.discovery_no_transfer != (arm == 'scratch')
            or not (directory / f'evaluation_epoch_{end + 1}_complete.json').is_file()):
        raise ValueError('Expected the completed increment of the same transfer arm')
    return checkpoint, file_sha256(checkpoint), args.graph_adapter_specs


def report(directory, milestones):
    from tools.owod.train_task2_diagnostic import dump
    fields = ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'unknown_precision',
              'unknown_predictions', 'wi', 'a_ose')
    result = dict(diagnostic_only=True, ap_convention='VOC2007 11-point AP50; fractions',
                  initialization={}, phases={})
    for arm in ('inherit', 'scratch'):
        result['initialization'][arm] = json.loads((directory / arm / 'transfer.json').read_text())
    for phase, epochs in milestones.items():
        root = directory if phase == 'increment' else directory / phase
        records = {}
        for epoch in ([0] if phase == 'increment' else []) + epochs:
            filename = 'initial_metrics.json' if epoch == 0 else f'metrics_epoch_{epoch}.json'
            row = {}
            for arm in ('inherit', 'scratch'):
                metrics = json.loads((root / arm / filename).read_text())
                row[arm] = {k: metrics['owod_' + k] for k in fields}
            row['inherit_minus_scratch'] = {k: row['inherit'][k] - row['scratch'][k] for k in fields}
            records[epoch] = row
        result['phases'][phase] = records
    dump(directory / 'comparison.json', result)
    return result
