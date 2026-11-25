"""Trace collateral changes from fixed proposal replacements in the Task 1 decoder."""

from collections import Counter
from contextlib import ExitStack
from functools import partial
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_prescreen_replacement import replace_cells
from tools.owod.probe_task1_m_s_refinement_fine_rank import MINED_OUTPUT, MINED_SHA, mined_selector
from tools.owod.probe_task1_m_s_refinement_fixed_injection import match_new_seeds, record
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_mined_seed_holdout import OUTPUT as HOLDOUT_OUTPUT, holdout_loader
from tools.owod.probe_task1_m_s_refinement_top_known import TopKnownPerQuery
from tools.owod.probe_task1_m_s_refinement_trajectory import hits
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256
from tools.owod.run_cross_attention_study import pin
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = HOLDOUT_OUTPUT.with_name('task1-m-s-reservation-attention')
HOLDOUT_SHA = '15d7e2827d896d94d189e6655ed616d8f89fc3f8176d60bbb47865d181c149c1'


def prepare():
    prior = verified(HOLDOUT_OUTPUT / 'comparison.json', HOLDOUT_SHA)
    plan = verified(HOLDOUT_OUTPUT / 'plan.json', prior['plan_sha256'])
    mined = verified(MINED_OUTPUT / 'comparison.json', MINED_SHA)
    if (prior['status'] != 'complete' or prior['passed'] or not mined['passed'] or plan['mined_sha256'] != MINED_SHA
            or any(not row['rank_improved'] or row['passed'] for row in prior['protocols'].values())):
        raise ValueError('Expected failed independent decoder gates with improved paired seed ranking')
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                holdout_sha256=HOLDOUT_SHA, holdout_plan_sha256=prior['plan_sha256'],
                factor='Replay native self-attention outputs for untouched decoder query rows, in all six layers. '
                       'Keep the same replaced cells, live injected rows, graph readout and Top-50 postprocessing.',
                advance='On both protocols, recover at least half the original paired known losses, retain at least '
                        'half the net small-box geometry gain, and keep all/small known matches at native counts or higher.')
    return plan, prior['protocols'], mined['protocols']


def save_attention(store, layer, _module, _inputs, result):
    if layer in store:
        raise ValueError('Expected one decoder pass per view')
    store[layer] = result[0]


def restore_attention(value, changed, _module, _inputs, result):
    return torch.where(changed.T[..., None], result[0], value), result[1]


def known_matches(detection, truth, labels):
    return ((box_iou(detection['boxes'], truth)[0] > .5) & (detection['labels'][:, None] == labels)
            & detection['scores'].isfinite()[:, None])


def retained_stages(post, protected, witnesses, losses, small):
    # The witnesses are original known matches that still have their native boxes and class logits.
    raw = TopKnownPerQuery.candidate_scores(post, protected)[0].flatten(1)
    nms = post.scores_before_competition(protected)[0].flatten(1)
    competition = post.candidate_scores(protected)[0].flatten(1)
    for batch, (match, indices) in enumerate(witnesses):
        lost = losses[batch]
        if not lost.any():
            continue
        survived = [(match[:, lost] & values[batch, indices].isfinite()[:, None]).any(dim=0)
                    for values in (raw, nms, competition)]
        if not survived[0].all():
            raise ValueError('A retained native match lost its class before known NMS')
        stage = (~survived[1], survived[1] & ~survived[2], survived[2])
        yield stage, small[batch][lost]


@torch.inference_mode()
def evaluate(model, loader, protocol, job, plan, selector, reference):
    detector = model.detr
    proposal = detector.transformer.instance_proposals
    attention = [layer.self_attn for layer in detector.transformer.decoder.layers]
    if len(attention) != 6 or getattr(detector, 'cascade_decoder', False):
        raise ValueError('Expected the pinned six-layer refinement decoder')
    known = job['classes']
    columns = len(known) + 1
    post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
    views = ('native', 'replaced', 'protected')
    totals = {group: {name: Counter() for name in views} for group in ('all', 'small')}
    attribution = {group: Counter() for group in totals}
    transfer = {name: Counter() for name in views[1:]}
    changed_rows = Counter()
    parity = Counter()
    images = 0

    for step, (samples, targets) in enumerate(loader, 1):
        samples = samples.to('cuda')
        captured = {}
        with ExitStack() as handles:
            for layer, module in enumerate(attention):
                handles.enter_context(module.register_forward_hook(partial(save_attention, captured, layer)))
            native = model(samples)
        unit = native['pred_boxes'].new_ones((len(targets), 2))
        native_detections = post(native, unit, include_top_indices=True)
        picks = {batch: pick for batch, detection in enumerate(native_detections)
                 if (pick := selector(native, batch, detection, columns)) is not None}
        labels = [target['labels'].cuda() for target in targets]
        truths = [box_cxcywh_to_xyxy(target['boxes'].cuda()) for target in targets]
        small = [target['source_box_area'].cuda() < plan['small_area'] for target in targets]
        outputs = {name: native for name in views}
        if picks:
            replacements = {batch: entry[1] for batch, entry in picks.items()}
            positions = {batch: entry[0] for batch, entry in picks.items()}
            hook = partial(replace_cells, replacements=replacements, positions=positions)
            changed = torch.zeros_like(native['pred_boxes'][..., 0], dtype=torch.bool)
            expected = native['geometry']['chosen'].clone()
            for batch, (slots, cells, _) in picks.items():
                changed[batch, slots] = True
                expected[batch, slots] = cells
            with proposal.register_forward_hook(hook):
                outputs['replaced'] = model(samples)
                with ExitStack() as handles:
                    for layer, module in enumerate(attention):
                        handles.enter_context(module.register_forward_hook(partial(restore_attention, captured[layer], changed)))
                    outputs['protected'] = model(samples)

            assigned = {batch: match_new_seeds(native, batch, slots, cells, truths[batch], small[batch])
                        for batch, (slots, cells, _) in picks.items()}
            placed = sum(len(slots) for slots in positions.values())
            for name in views[1:]:
                geometry = outputs[name]['geometry']
                if not torch.equal(geometry['active'], native['geometry']['active']) or not torch.equal(geometry['chosen'], expected):
                    raise ValueError('Changed replacement identity or activation budget')
                transfer[name]['replacement_slots'] += placed
                record(outputs[name], assigned, truths, post, columns, transfer[name])

            for key in ('pred_boxes', 'pred_logits'):
                value = outputs['protected'][key][~changed]
                original = native[key][~changed]
                torch.testing.assert_close(value, original, rtol=1e-6, atol=1e-6)
                parity[key] = max(parity[key], float((value - original).abs().max()))
            revised = (outputs['replaced']['pred_boxes'] != native['pred_boxes']).any(dim=-1)
            retained = ~changed
            changed_rows.update(all=int(revised.sum()), images=int(revised.any(dim=1).sum()),
                                retained=int((revised & retained).sum()), injected=placed)
            for key in ('pred_novelty', 'pred_objectness'):
                changed_rows[key] += int(((outputs['protected'][key] != native[key]) & retained).sum())

        detections = {name: post(outputs[name], unit, include_top_indices=True) for name in views[1:]}
        detections['native'] = native_detections
        witnesses, losses = [], []
        for batch, (truth, label, is_small) in enumerate(zip(truths, labels, small, strict=True)):
            matches = {name: known_matches(detections[name][batch], truth, label) for name in views}
            emitted = {name: matrix.any(dim=0) for name, matrix in matches.items()}
            geometry = {name: hits(outputs[name]['pred_boxes'][batch], truth).any(dim=0) for name in views}
            replaced_lost = emitted['native'] & ~emitted['replaced']
            protected_lost = emitted['native'] & ~emitted['protected']
            indices = detections['native'][batch]['top_indices']
            if batch in picks and matches['native'][torch.isin(indices // columns, positions[batch])].any():
                raise ValueError('Replacement touched a native known matched query')
            witnesses.append((matches['native'], indices))
            losses.append(protected_lost)
            for group, mask in (('all', torch.ones_like(is_small)), ('small', is_small)):
                for name in views:
                    totals[group][name].update(gt=int(mask.sum()), geometry=int((geometry[name] & mask).sum()),
                        gained=int((~geometry['native'] & geometry[name] & mask).sum()),
                        lost=int((geometry['native'] & ~geometry[name] & mask).sum()),
                        known_tp=int((emitted[name] & mask).sum()),
                        known_lost=int((emitted['native'] & ~emitted[name] & mask).sum()),
                        known_gained=int((~emitted['native'] & emitted[name] & mask).sum()))
                attribution[group].update(restored=int((replaced_lost & ~protected_lost & mask).sum()),
                    persistent=int((replaced_lost & protected_lost & mask).sum()),
                    newly_lost=int((~replaced_lost & protected_lost & mask).sum()))
        if any(loss.any() for loss in losses):
            for stages, lost_small in retained_stages(post, outputs['protected'], witnesses, losses, small):
                for stage, mask in zip(('known_nms', 'unknown_overlap', 'top50'), stages, strict=True):
                    attribution['all'][stage] += int(mask.sum())
                    attribution['small'][stage] += int((mask & lost_small).sum())
        images += len(targets)
        if step == 1 or step % 32 == 0 or step == len(loader):
            print(f'{protocol} {step}/{len(loader)} batches, {changed_rows["injected"]} fixed replacements', flush=True)

    if (images != plan['images'] or transfer['replaced'] != reference['transfer']
            or changed_rows['images'] != reference['changes']['images']):
        raise ValueError(f'Changed independent holdout replay: {protocol}')
    if changed_rows['all'] != reference['changes']['queries']:
        raise ValueError(f'Changed unrestricted geometry rows: {protocol}')
    for group, rows in totals.items():
        original, changed, protected = (rows[name] for name in views)
        reproduced = dict(gt=original['gt'], final=original['geometry'], gained=changed['gained'], lost=changed['lost'],
                          revised=changed['geometry'], native_known_tp=original['known_tp'], revised_known_tp=changed['known_tp'])
        moves = attribution[group]
        if (reproduced != reference[group] or moves['restored'] + moves['persistent'] != changed['known_lost']
                or moves['persistent'] + moves['newly_lost'] != protected['known_lost']
                or sum(moves[key] for key in ('known_nms', 'unknown_overlap', 'top50')) != protected['known_lost']):
            raise ValueError(f'Changed prior totals or inconsistent lost-target attribution: {protocol}/{group}')
    small = totals['small']
    geometry_gain = small['replaced']['geometry'] - small['native']['geometry']
    passed = (2 * attribution['all']['restored'] >= totals['all']['replaced']['known_lost']
              and 2 * (small['protected']['geometry'] - small['native']['geometry']) >= geometry_gain
              and all(rows['protected']['known_tp'] >= rows['native']['known_tp'] for rows in totals.values()))
    return dict(groups=totals, transfer=transfer, changed_rows=changed_rows, retained_max_abs=parity,
                known_losses=attribution, passed=passed)


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen attention replay')
    plan, holdout, mined = prepare()
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        loader, order = holdout_loader(protocol, job, plan)
        if order != plan['holdout_order_sha256'][protocol]:
            raise ValueError(f'Changed independent holdout images: {protocol}')
        model, _ = load_detector(None, Path(job['checkpoint']), torch.device('cuda'))
        selector = mined_selector(mined[protocol]['ranker'])
        report = evaluate(model.eval(), loader, protocol, job, plan, selector, holdout[protocol]['mined'])
        write_json(OUTPUT / f'{protocol}.json', dict(plan_sha256=digest, order_sha256=order, report=report))
        protocols[protocol] = report
        del model
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['passed'] for row in protocols.values())))


if __name__ == '__main__':
    main()
