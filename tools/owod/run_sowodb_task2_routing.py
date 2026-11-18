"""Full-data, paired v7 hard/soft Task 2 diagnostic, with read-only reference."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.protocol import file_sha256
from tools.owod.task2_parent_provenance import verify_parent_inputs, verify_reference
from tools.owod.train_task2_diagnostic import diagnostic_identity, dump, run_policy

EPOCHS = 8
EVALUATION_EPOCHS = (3, 8)


def evaluation_epochs(epochs):
    if type(epochs) is not int or epochs not in (3, 8):
        raise ValueError('Paired diagnostic budget must be 3 or 8 epochs')
    return (3,) if epochs == 3 else EVALUATION_EPOCHS


def output_directory(comparison, epochs):
    paired_variants(comparison)
    if comparison == 'transfer':
        from tools.owod.task2_transfer import schedule
        schedule(epochs)
    else:
        evaluation_epochs(epochs)
    return Path(f'output/sowodb-task2-v7-{comparison}-{epochs}epochs-2gpu/results')


def routing_identity(manifest, checkpoint, gnn, proof, reference, world_size, gpu_selection, *, epochs=EPOCHS):
    milestones = evaluation_epochs(epochs)
    identity = diagnostic_identity(manifest, checkpoint, gnn, world_size, epochs=epochs)
    suite = 'sowodb-task2-v7-routing-v2' if epochs == 8 else 'sowodb-task2-v7-routing-3epochs-v1'
    identity.update(suite=suite, policies=[7],
                    evaluation_epochs=list(milestones),
                    variants=['hard', 'soft'], localization='frozen',
                    parent_provenance=proof, reference=reference, probe_images_per_class=12,
                    cuda_visible_devices=gpu_selection)
    return identity


def paired_report(hard, soft, *, epoch=None, epochs=EPOCHS):
    milestones = evaluation_epochs(epochs)
    epoch = epochs if epoch is None else epoch
    if epoch not in milestones:
        raise ValueError('Report requires a predeclared evaluation epoch')
    metrics = ('owod_previous_ap50', 'owod_current_ap50', 'owod_known_ap50', 'owod_u_recall',
               'owod_unknown_precision', 'owod_unknown_predictions', 'owod_wi', 'owod_a_ose')
    return dict(diagnostic_only=True, epochs=epochs, evaluation_epoch=epoch,
                evaluation_epochs=list(milestones), interim=epoch < epochs, finetune=False,
                ap_convention='VOC2007 11-point AP50; AP/recall/precision in [0,1]',
                hard={k: hard[k] for k in metrics if k in hard},
                soft={k: soft[k] for k in metrics if k in soft},
                soft_minus_hard={k: soft[k] - hard[k] for k in metrics if k in hard and k in soft},
                note=f'Predeclared epoch {epoch} results; final comparison is epoch {epochs}. '
                     'Increment-only diagnostic, '
                     'not formal Task 2 or evidence of successful Task 1 unknown-to-new transfer.')


def paired_variants(comparison):
    if comparison == 'transfer':
        return (('inherit', 'soft', 'frozen'), ('scratch', 'soft', 'frozen'))
    if comparison == 'routing':
        return (('hard', 'hard', 'frozen'), ('soft', 'soft', 'frozen'))
    if comparison == 'localization':
        return (('frozen', 'soft', 'frozen'), ('bbox-last', 'soft', 'bbox-last'))
    if comparison == 'unknown':
        return (('baseline', 'soft', 'frozen'), ('protected', 'soft', 'frozen'))
    raise ValueError('Unknown paired comparison')


def localization_identity(*args, epochs=EPOCHS):
    identity = routing_identity(*args, epochs=epochs)
    suite = 'sowodb-task2-v7-localization-v1' if epochs == 8 else 'sowodb-task2-v7-localization-3epochs-v1'
    identity.update(suite=suite,
                    variants=['frozen', 'bbox-last'], routing='soft',
                    localization=['frozen', 'bbox-last'])
    return identity


def localization_report(frozen, bbox_last, *, epoch=None, epochs=EPOCHS):
    report = paired_report(frozen, bbox_last, epoch=epoch, epochs=epochs)
    report['frozen'] = report.pop('hard')
    report['bbox_last'] = report.pop('soft')
    report['bbox_last_minus_frozen'] = report.pop('soft_minus_hard')
    report.update(comparison='localization', routing='soft')
    return report


def unknown_identity(*args, epochs=3):
    identity = routing_identity(*args, epochs=epochs)
    identity.update(suite=f'sowodb-task2-v7-unknown-protection-{epochs}epochs-v1',
                    variants=['baseline', 'protected'], routing='soft', localization='frozen',
                    negative_policies={'baseline': 'dense', 'protected': 'cross-view-protected'},
                    protection=dict(saliency_both_views=.5, consistency='same as pseudo',
                                    max_extra='discovery_top_k', max_pair_iou=.7,
                                    max_released_gt_iou=.3, teacher_required=True),
                    changed_factor='objectness/router negatives on bounded unselected candidates')
    return identity


def unknown_report(baseline, protected, *, epoch=None, epochs=3):
    report = paired_report(baseline, protected, epoch=epoch, epochs=epochs)
    report['baseline'] = report.pop('hard')
    report['protected'] = report.pop('soft')
    report['protected_minus_baseline'] = report.pop('soft_minus_hard')
    report.update(comparison='unknown', routing='soft', localization='frozen',
                  negative_policies={'baseline': 'dense', 'protected': 'cross-view-protected'},
                  desired_direction={'owod_u_recall': 'higher', 'owod_wi': 'lower',
                                     'owod_a_ose': 'lower', 'owod_unknown_precision': 'higher'},
                  note='Fixed final-budget increment-only screening; compare all APs and unknown '
                       'precision/count as well as U-Recall, WI and A-OSE. No automatic efficacy claim.')
    return report


def write_reports(directory, comparison, epochs):
    variants = paired_variants(comparison)
    reporter = {'routing': paired_report, 'localization': localization_report, 'unknown': unknown_report}[comparison]
    comparisons = {}
    for epoch in evaluation_epochs(epochs):
        metrics = [json.loads((directory / name / f'metrics_epoch_{epoch}.json').read_text())
                   for name, _, _ in variants]
        comparisons[epoch] = reporter(*metrics, epoch=epoch, epochs=epochs)
        dump(directory / f'comparison_epoch_{epoch}.json', comparisons[epoch])
    report = comparisons[epochs]
    if epochs == 8:
        report['change_from_epoch_3_to_8'] = {
            variant: {key: value - comparisons[3][variant][key]
                      for key, value in report[variant].items() if key in comparisons[3][variant]}
            for variant in (('hard', 'soft') if comparison == 'routing' else
                            ('frozen', 'bbox_last') if comparison == 'localization' else
                            ('baseline', 'protected'))}
    if comparison == 'unknown':
        initial_metrics = [json.loads((directory / name / 'initial_metrics.json').read_text())
                           for name, _, _ in variants]
        initial = unknown_report(*initial_metrics, epochs=epochs)
        report['initial'] = {name: initial[name] for name, _, _ in variants}
        report['change_from_initial'] = {
            name: {key: value - initial[name][key] for key, value in report[name].items()
                   if key in initial[name]} for name, _, _ in variants}
        activity_fields = ('discovery_pseudo_count', 'discovery_protected_count',
                           'discovery_protected_fraction', 'discovery_proposal_negative_fraction')
        report['training_activity'] = {
            name: [dict(epoch=epoch + 1, **{key: value for key, value in json.loads(
                (directory / name / f'epoch_{epoch}.json').read_text())['train'].items()
                if key in activity_fields}) for epoch in range(epochs)]
            for name, _, _ in variants}
    dump(directory / 'comparison.json', report)
    return report


def run_paired_variants(opt, identity, proof, comparison):
    variants = paired_variants(comparison)
    for index, (name, routing, localization) in enumerate(variants):
        negative_policy = identity.get('negative_policies', {}).get(name, 'dense')
        print(f'v7 Task 2 paired experiment: {name}, {opt.epochs} increment epochs, '
              f'{routing} routing, {localization} localization, {negative_policy} negatives', flush=True)
        run_policy(opt, identity, 7, variant=routing, provenance=proof,
                   localization=localization, run_name=name if comparison != 'routing' else None,
                   negative_policy=negative_policy,
                   initialization_reference=variants[0][0] if index else None)


def transfer_identity(manifest, checkpoint, gnn, proof, reference, world_size, gpu_selection, *, epochs):
    from tools.owod.task2_transfer import schedule
    phases = schedule(epochs)
    identity = diagnostic_identity(manifest, checkpoint, gnn, world_size, epochs=epochs)
    identity.update(suite='sowodb-task2-transfer-v1', policies=[7], comparison='transfer',
                    phase='increment', milestones=phases, evaluation_epochs=phases['increment'],
                    variants=['inherit', 'scratch'], routing='soft', localization='frozen',
                    negative_policy='dense', parent_provenance=proof, cuda_visible_devices=gpu_selection)
    return identity


def main(*, comparison='routing'):
    variants = paired_variants(comparison)
    parser = argparse.ArgumentParser(description={
        'routing': __doc__,
        'localization': 'Paired soft/frozen vs soft/bbox-last Task 2 diagnostic; fresh Task 1 parents.',
        'unknown': 'Paired dense vs cross-view-protected unknown supervision; fresh, identical Task 1 parents.',
        'transfer': 'Matched discovery snapshot vs ordinary class adapter initialization in Task 2.',
    }[comparison])
    parser.add_argument('--manifest', type=Path, default=Path('data/owod/s-owodb/split_manifest.json'))
    parent = Path('output/tree-detr-s-v5-full/s-owodb/seed_42')
    parser.add_argument('--checkpoint', type=Path, default=parent / 'stage_0/increment/checkpoint0049.pth')
    parser.add_argument('--gnn', type=Path, default=parent / 'gnn/stage_0/gnn_stage0.pt')
    needs_reference = comparison in ('routing', 'localization')
    if needs_reference:
        previous = Path('output/task2-v5-v7-diagnostic-3epochs-2gpu/results')
        parser.add_argument('--reference-checkpoint', type=Path, default=previous / 'v7/checkpoint.pth')
        parser.add_argument('--reference-plan', type=Path, default=previous / 'diagnostic_plan.json')
    parser.add_argument('--epochs', type=int, choices=(3, 50) if comparison == 'transfer' else (3, 8),
                        default=3 if not needs_reference else EPOCHS)
    parser.add_argument('--output', type=Path,
                        help='Defaults to a separate comparison/epoch-budget directory')
    parser.add_argument('--preflight', action='store_true', help='CPU-only provenance checks; writes no experiment state')
    parser.add_argument('--resume', action='store_true')
    opt = parser.parse_args()
    if opt.output is None:
        opt.output = output_directory(comparison, opt.epochs)
    fields = ['manifest', 'checkpoint', 'gnn', 'output']
    if needs_reference:
        fields.extend(['reference_checkpoint', 'reference_plan'])
    for field in fields:
        setattr(opt, field, getattr(opt, field).resolve())
    import torch
    torch.set_num_threads(2)
    proof = verify_parent_inputs(opt.manifest, opt.checkpoint, opt.gnn)
    # The new supervision pair needs only the verified formal Task 1 parent.
    # Historical diagnostic weights are not an initialization or prerequisite.
    reference = (verify_reference(opt.reference_checkpoint, opt.reference_plan, proof)
                 if needs_reference else None)
    if comparison == 'transfer':
        from tools.owod.task2_transfer import schedule
        phases = schedule(opt.epochs)
    else:
        phases = {'increment': list(evaluation_epochs(opt.epochs))}
    if opt.preflight:
        print(json.dumps(dict(parent_provenance=proof, reference=reference,
                              variants=[name for name, _, _ in variants], comparison=comparison, epochs=opt.epochs,
                              negative_policies=({'baseline': 'dense', 'protected': 'cross-view-protected'}
                                                 if comparison == 'unknown' else None),
                              evaluation_epochs=phases['increment'], effective_batch_size=2,
                              phases=phases, output=str(opt.output), finetune='finetune' in phases), indent=2))
        return
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.rank = int(os.environ.get('RANK', 0))
    opt.gpu = int(os.environ.get('LOCAL_RANK', 0))
    if opt.world_size != 2 or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        parser.error('This paired experiment requires two GPUs via torchrun, batch 1 per rank')
    torch.cuda.set_device(opt.gpu)
    from datetime import timedelta
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30))
    # LOCAL_SIZE is consumed by the existing COCO dataset helpers.
    os.environ['LOCAL_SIZE'] = '2'
    identity_builder = {'routing': routing_identity, 'localization': localization_identity,
                        'unknown': unknown_identity, 'transfer': transfer_identity}[comparison]
    identity = identity_builder(opt.manifest, opt.checkpoint, opt.gnn, proof, reference,
                                opt.world_size, os.environ.get('CUDA_VISIBLE_DEVICES', ''), epochs=opt.epochs)
    import torchvision
    import MultiScaleDeformableAttention as extension
    identity['runtime'] = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                               cuda=torch.version.cuda, extension_sha256=file_sha256(Path(extension.__file__)),
                               devices=[dict(name=torch.cuda.get_device_name(i),
                                             capability=torch.cuda.get_device_capability(i)) for i in range(2)])
    # JSON roundtrip keeps tuple-valued runtime fields stable on resume.
    identity = json.loads(json.dumps(identity))
    opt.output.mkdir(parents=True, exist_ok=True)
    plan_path = opt.output / 'diagnostic_plan.json'
    if opt.rank == 0:
        if plan_path.exists():
            if not opt.resume or json.loads(plan_path.read_text()) != identity:
                raise ValueError('Resume requires exactly the same code, inputs, GPU selection and plan')
        elif opt.resume or any(opt.output.iterdir()):
            raise ValueError('First run requires an empty result directory, without --resume')
        dump(plan_path, identity)
        dump(opt.output / 'parent_provenance.json', proof)
    torch.distributed.barrier()
    if reference is not None:
        from tools.owod.task2_learning_probe import evaluate_reference
        evaluate_reference(opt, identity)
    gc.collect()
    torch.cuda.empty_cache()
    run_paired_variants(opt, identity, proof, comparison)
    if comparison == 'transfer' and 'finetune' in phases:
        from tools.owod.task2_transfer import increment_parent, report as transfer_report
        if opt.rank == 0:
            transfer_report(opt.output, {'increment': phases['increment']})
        for arm, routing, localization in variants:
            checkpoint, digest, specs = increment_parent(opt.output / arm, identity, arm)
            ft_identity = dict(identity, phase='finetune', finetune=True, arm=arm, increment_sha256=digest,
                               epochs=phases['finetune'][-1], evaluation_epochs=phases['finetune'],
                               increment_adapter_specs=specs)
            run_policy(opt, ft_identity, 7, variant=routing, provenance=proof, localization=localization,
                       run_name=arm, increment_checkpoint=checkpoint, initialization_reference=None)
    inputs = [(opt.checkpoint, proof['checkpoint_sha256']), (opt.gnn, proof['gnn_sha256']),
              (opt.manifest, proof['canonical_manifest_sha256'])]
    if reference is not None:
        inputs.append((opt.reference_checkpoint, reference['checkpoint_sha256']))
    for path, expected in inputs:
        if file_sha256(path) != expected:
            raise ValueError(f'Input changed during experiment: {path}')
    if opt.rank == 0:
        if comparison == 'transfer':
            from tools.owod.task2_transfer import report as transfer_report
            report = transfer_report(opt.output, phases)
        else:
            report = write_reports(opt.output, comparison, opt.epochs)
        print(json.dumps(report, indent=2), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
