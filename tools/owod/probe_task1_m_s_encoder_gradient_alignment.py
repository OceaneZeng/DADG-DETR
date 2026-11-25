"""Compare frozen Task 1 detection and small-box rank gradients on encoder memory."""

import argparse
from pathlib import Path
from statistics import median
import subprocess

import torch

from models.graph_local.instance_proposals import geometry_losses
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_refinement_saliency_training import DETACHED_OUTPUT, QUALITY_OUTPUT
from tools.owod.screening_data import screening_files, write_json


OUTPUT = QUALITY_OUTPUT.with_name('task1-m-s-encoder-gradient-alignment')
SOURCES = {
    'full': (QUALITY_OUTPUT, '8ac23ca0c1eb44c2b71ef8f73fd31f7f400585e5db13be76848ccd2182516292'),
    'detached': (DETACHED_OUTPUT, 'b7b4f5f78bc7a676880e79cee941550259689f4c03df175abe946642e9ce9aab'),
}
CHECKPOINTS = {
    'm-owodb': {
        'full': '9d5fe319f71c9ae8ecc445b024dca309293b2e45e2c10b6829e52b85581a6a04',
        'detached': '62709f3733993683729fbc5b14240fb00af85ad432ecfea89e7528d6bc4539a9',
    },
    's-owodb': {
        'full': '1cb87265dac80851d17f22d0e3cd4e92d170c686f7fa1f59a0f79c2c69de211f',
        'detached': '4dd72d434ce03e97327acc52067515c35ec1e9e0b2f9de1482d90665ef84b838',
    },
}
PREDICTIONS = ('pred_logits', 'pred_boxes', 'pred_objectness', 'pred_novelty')


def prepare():
    parents = {}
    jobs = {protocol: {'arms': {}} for protocol in CHECKPOINTS}
    for arm, (folder, comparison_sha) in SOURCES.items():
        comparison = verified(folder / 'comparison.json', comparison_sha)
        if comparison['status'] != 'complete':
            raise ValueError(f'Incomplete source experiment: {folder}')
        parents[arm] = dict(comparison_sha256=comparison_sha, plan_sha256=comparison['plan_sha256'])
        training = verified(folder / 'plan.json', comparison['plan_sha256'])
        for protocol, job in jobs.items():
            source = training['jobs'][protocol]
            command = source['command']
            screen = Path(command[command.index('--owod-screening') + 1])
            provenance = dict(manifest_sha256=source['manifest_sha256'], screen=str(screen),
                              screen_sha256=source['screen_sha256'])
            if job['arms'] and any(job[key] != value for key, value in provenance.items()):
                raise ValueError(f'Unpaired screen or manifest: {protocol}/{arm}')
            job.update(provenance)
            checkpoint = folder / protocol / 'increment' / 'checkpoint.pth'
            digest = CHECKPOINTS[protocol][arm]
            if file_sha256(checkpoint) != digest:
                raise ValueError(f'Changed terminal checkpoint: {checkpoint}')
            job['arms'][arm] = dict(checkpoint=str(checkpoint), sha256=digest)
    for protocol, job in jobs.items():
        path = MANIFESTS[protocol]
        manifest = verified(path, job['manifest_sha256'])
        screen = Path(job['screen'])
        if file_sha256(screen) != job['screen_sha256']:
            raise ValueError(f'Changed released screen: {screen}')
        _, annotations = screening_files(path, screen, 0)
        training = annotations['increment_train']
        job.update(classes=manifest['stages'][0]['classes'], training=str(training),
                   training_sha256=file_sha256(training), image_root=str(resolve_image_root(path, manifest)))
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                seed=42, images=256, batch_size=1, small_area=32**2, parents=parents, jobs=jobs,
                minimum_rank_active=32, minimum_s_excess_conflict_pp=15.,
                paper_comparable=False, scope='Frozen eval memory leaf, dense weighted DETR losses and raw rank.')


def gradient_row(model, criterion, samples, target, *, verify_forward):
    encoder = model.detr.transformer.encoder
    leaves = []

    def replace_memory(_encoder, _inputs, memory):
        leaf = memory.detach().requires_grad_()
        leaves.append(leaf)
        return leaf

    if verify_forward:
        with torch.no_grad():
            native = model(samples)

    handle = encoder.register_forward_hook(replace_memory)
    try:
        outputs = model(samples)
    finally:
        handle.remove()
    memory, = leaves
    equality = {}
    if verify_forward:
        for key in PREDICTIONS:
            torch.testing.assert_close(outputs[key], native[key], rtol=1e-6, atol=1e-6)
            equality[key] = float((outputs[key].detach() - native[key]).abs().max())

    keep = torch.isin(target['labels'], target['labels'].new_tensor(criterion.known_ids))
    if not bool(keep.all()):
        raise ValueError('Screen replay encountered an unreleased target')
    released = {key: target[key][keep] for key in ('labels', 'boxes', 'source_box_area')}
    prepared = dict(outputs)
    prepared.pop('query_label_weights', None)
    prepared['aux_outputs'] = [dict(layer) for layer in outputs.get('aux_outputs', [])]
    for layer in prepared['aux_outputs']:
        layer.pop('query_label_weights', None)

    losses = criterion.base(prepared, [released])
    weights = criterion.base.weight_dict
    classification = sum(value * weights[key] for key, value in losses.items()
                         if key.startswith('loss_ce') and key in weights)
    box = sum(value * weights[key] for key, value in losses.items()
              if key.startswith(('loss_bbox', 'loss_giou')) and key in weights)
    geometry = outputs['geometry']
    if 'rank_logits' in geometry or geometry['memory'] is not memory:
        raise ValueError('Proposal rank must use the same unscaled eval memory leaf')
    rank = geometry_losses(geometry, [released], supervise_quality=True)['loss_small_box_quality_rank']

    class_grad, = torch.autograd.grad(classification, memory, retain_graph=True)
    box_grad, = torch.autograd.grad(box, memory, retain_graph=True)
    rank_grad, = torch.autograd.grad(rank, memory)
    if not all(bool(value.isfinite().all()) for value in (classification, box, rank, class_grad, box_grad, rank_grad)):
        raise ValueError(f'Nonfinite loss or gradient at image {int(target["image_id"])}')
    support = rank_grad.ne(0)
    rank_norm = float(torch.linalg.vector_norm(rank_grad))
    row = dict(image_id=int(target['image_id']), released_boxes=int(keep.sum()),
               released_small_boxes=int((released['source_box_area'] < 32**2).sum()),
               memory_elements=memory.numel(), rank_support_elements=int(support.sum()),
               rank_support_cells=int(support.any(dim=-1).sum()), rank_loss=float(rank.detach()), rank_norm=rank_norm)
    if verify_forward:
        row['native_forward_max_abs'] = equality
    for name, gradient in (('classification', class_grad), ('box', box_grad)):
        full_norm = float(torch.linalg.vector_norm(gradient))
        supported = gradient[support]
        supported_norm = float(torch.linalg.vector_norm(supported))
        # Zero rank components make the numerator identical for both scopes.
        dot = float((supported.double() * rank_grad[support].double()).sum())
        row[name] = {
            scope: dict(dot=dot, norm=norm, cosine=dot / (norm * rank_norm) if norm and rank_norm else None)
            for scope, norm in (('rank_support', supported_norm), ('all_memory', full_norm))
        }
    return row


def summarize(rows):
    active = [row for row in rows if row['rank_support_elements']]
    result = dict(images=len(rows), rank_active=len(active),
                  released_boxes=sum(row['released_boxes'] for row in rows),
                  released_small_boxes=sum(row['released_small_boxes'] for row in rows),
                  native_forward_max_abs=rows[0]['native_forward_max_abs'])
    for loss in ('classification', 'box'):
        result[loss] = {}
        for scope in ('rank_support', 'all_memory'):
            pairs = [row[loss][scope] for row in active]
            cosines = [pair['cosine'] for pair in pairs if pair['cosine'] is not None]
            negative = sum(pair['dot'] < 0 for pair in pairs)
            result[loss][scope] = dict(negative=negative, cosine_active=len(cosines),
                                       negative_fraction=negative / len(pairs) if pairs else None,
                                       zero_dot=sum(pair['dot'] == 0 for pair in pairs),
                                       median_dot=median(pair['dot'] for pair in pairs) if pairs else None,
                                       median_cosine=median(cosines) if cosines else None)
    return result


def evaluate(protocol, job, plan, *, preflight):
    device = torch.device('cuda')
    checkpoint = Path(job['arms']['full']['checkpoint'])
    loader, order = training_loader(checkpoint, Path(job['training']), Path(job['image_root']), job, plan,
                                    source_box_area=True)
    models = {}
    for arm, endpoint in job['arms'].items():
        model, criterion = load_detector(None, Path(endpoint['checkpoint']), device)
        proposals = model.detr.transformer.instance_proposals
        if (criterion.known_ids != job['classes'] or not criterion.supervise_small_box_quality
                or proposals.rank_feature_grad_scale != float(arm == 'full')):
            raise ValueError(f'Unexpected rank arm settings: {protocol}/{arm}')
        models[arm] = model.requires_grad_(False).eval(), criterion.eval()

    rows = {arm: [] for arm in models}
    for step, (samples, targets) in enumerate(loader, 1):
        target, = targets
        target = {key: target[key].to(device) for key in ('image_id', 'labels', 'boxes', 'source_box_area')}
        samples = samples.to(device)
        for arm, (model, criterion) in models.items():
            rows[arm].append(gradient_row(model, criterion, samples, target, verify_forward=step == 1))
        if step == 1 or step % 16 == 0 or step == len(loader):
            active = {arm: sum(row['rank_support_elements'] > 0 for row in values) for arm, values in rows.items()}
            print(f'{protocol} {step}/{len(loader)} paired images, rank-active {active}', flush=True)
        if preflight:
            break
    count = 1 if preflight else plan['images']
    if any(len({row['image_id'] for row in values}) != count for values in rows.values()):
        raise ValueError(f'Incomplete or duplicate released image replay: {protocol}')
    return dict(order_sha256=order, rows=rows, summaries={arm: summarize(values) for arm, values in rows.items()})


def compare(protocols, plan):
    decisions = {}
    for arm in SOURCES:
        m, s = (protocols[protocol][arm] for protocol in ('m-owodb', 's-owodb'))
        enough = min(m['rank_active'], s['rank_active']) >= plan['minimum_rank_active']
        m_cls, s_cls = (report['classification']['rank_support'] for report in (m, s))
        delta = (s_cls['negative_fraction'] - m_cls['negative_fraction']) * 100 if m['rank_active'] and s['rank_active'] else None
        cosine = s_cls['median_cosine']
        decisions[arm] = dict(enough_rank_active=enough, s_excess_conflict_pp=delta, s_median_cosine=cosine,
                              passed=enough and delta >= plan['minimum_s_excess_conflict_pp']
                              and cosine is not None and cosine < 0)
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the paired frozen gradient diagnostic')
    plan = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    preflight = None
    if not args.preflight:
        preflight = read_json(OUTPUT / 'preflight.json')
        if preflight['status'] != 'complete' or preflight['plan_sha256'] != digest:
            raise ValueError('Expected completed preflight for this pinned plan')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        report = evaluate(protocol, job, plan, preflight=args.preflight)
        if preflight is not None:
            first = read_json(OUTPUT / f'preflight-{protocol}.json')
            if first['plan_sha256'] != digest or first['order_sha256'] != report['order_sha256']:
                raise ValueError(f'Changed preflight image order: {protocol}')
            for arm, rows in report['rows'].items():
                keys = ('image_id', 'released_boxes', 'released_small_boxes', 'memory_elements')
                if any(rows[0][key] != first['rows'][arm][0][key] for key in keys):
                    raise ValueError(f'Changed first released image: {protocol}/{arm}')
        label = f'preflight-{protocol}' if args.preflight else protocol
        write_json(OUTPUT / f'{label}.json', dict(plan_sha256=digest, **report))
        protocols[protocol] = report['summaries']
    if args.preflight:
        write_json(OUTPUT / 'preflight.json', dict(status='complete', plan_sha256=digest, protocols=protocols))
        return
    decisions = compare(protocols, plan)
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               decisions=decisions, passed=all(row['passed'] for row in decisions.values())))


if __name__ == '__main__':
    main()
