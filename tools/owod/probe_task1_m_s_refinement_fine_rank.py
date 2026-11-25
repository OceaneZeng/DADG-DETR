"""Isolate learned fine-level candidate ranking at the same released seed budget."""

from collections import Counter
from pathlib import Path
import argparse
from statistics import median
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_selector import L2, NEGATIVE_COST, fit_selector, source_ids
from tools.owod.probe_task1_m_s_refinement_dense_oracle import positive_pairs
from tools.owod.probe_task1_m_s_refinement_fixed_injection import (
    MAX_REPLACEMENTS, NATIVE_OUTPUT, choose_native, fine_candidates, run_injection,
)
from tools.owod.probe_task1_m_s_refinement_known_dedup import CompeteWithEmittedUnknown
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_m_s_refinement_trajectory import hits
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = NATIVE_OUTPUT.with_name('task1-m-s-refinement-fine-seed-rank')
MINED_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-fine-seed-mining')
DECODER_OUTPUT = OUTPUT.with_name('task1-m-s-refinement-mined-seed-decoder')
MINED_SHA = '42f34515a38ed5adcabaef50cb75090e4110322cf5af719edf5806e7226188ce'
RANK_SHA = '5036b6c15283f3ea603a862a684baa50e5c9592c645948d89f2a58cd48012759'
RESERVATION_SHA = '1e8b6b879f13943ba41338595391500d7158869f55143cdf1be62b6d81451bd3'
POSITIVES_PER_TARGET = 4
NEGATIVES_PER_IMAGE = 16


def missed_boxes(outputs, batch, target, small_area):
    truth = box_cxcywh_to_xyxy(target['boxes'].cuda())
    small = target['source_box_area'].cuda() < small_area
    state = outputs['geometry']
    chosen = state['chosen'][batch, state['active'][batch]]
    seed = hits(state['boxes'][batch, chosen], truth).any(dim=0)
    final = hits(outputs['pred_boxes'][batch], truth).any(dim=0)
    return truth[small & ~seed & ~final]


def features(geometry, batch, cells):
    boxes = geometry['boxes'][batch, cells]
    anchors = geometry['anchors'][batch, cells]
    extent = boxes[:, 2:].clamp_min(1e-6)
    return torch.cat((geometry['memory'][batch, cells], geometry['logits'][batch, cells, None],
                      boxes[:, :2], extent.log(), (boxes[:, :2] - anchors[:, :2]) / extent), dim=1).float()


def readout_score(params):
    mean, scale, weight = (torch.tensor(params[key], device='cuda') for key in ('mean', 'scale', 'weight'))

    def score(values):
        return ((values - mean) / scale) @ weight + params['bias']

    return score


def mined_selector(params):
    score = readout_score(params)

    def select(native, batch, detections, columns):
        geometry = native['geometry']
        values = features(geometry, batch, fine_candidates(geometry, batch))
        return choose_native(native, batch, detections, columns, scores=score(values))

    return select


def replay_mined_decoder(reservation):
    result = verified(MINED_OUTPUT / 'comparison.json', MINED_SHA)
    plan = verified(MINED_OUTPUT / 'plan.json', result['plan_sha256'])
    if (result['status'] != 'complete' or not result['passed'] or plan['reservation_sha256'] != RESERVATION_SHA
            or plan['rank_sha256'] != RANK_SHA or set(result['protocols']) != set(plan['jobs'])):
        raise ValueError('Expected successful paired released ranking with the original reservation')
    selectors = {key: mined_selector(row['ranker']) for key, row in result['protocols'].items()}
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                mined_sha256=MINED_SHA,
                expected_reservations={key: row['arms']['readout'] for key, row in result['protocols'].items()},
                factor='Use the pinned mined ranker per protocol for the fixed reservation and rerun the decoder. '
                       'Compare released small localization and class-correct Top-50 to the unmodified detector.',
                advance='Require increased small final coverage on S, no loss on M, and no decline in small or '
                        'all class-correct Top-50 matches on either protocol. Trace actual same-slot emissions.')
    run_injection(DECODER_OUTPUT, plan, reservation, native_selection=True, selectors=selectors)


@torch.no_grad()
def fit_examples(model, loader, plan, mine_score=None):
    samples_out, labels_out = [], []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        geometry = outputs['geometry']
        for batch, target in enumerate(targets):
            cells = fine_candidates(geometry, batch)
            truth = missed_boxes(outputs, batch, target, plan['small_area'])
            iou = box_iou(box_cxcywh_to_xyxy(geometry['boxes'][batch, cells]), truth)[0]
            top_iou, indices = iou.topk(min(POSITIVES_PER_TARGET, len(cells)), dim=0)
            positive = indices[top_iou > .5].unique()
            negatives = torch.where(~(iou > .5).any(dim=1))[0]
            priority = geometry['logits'][batch, cells[negatives]].argsort(descending=True, stable=True)
            negative = negatives[priority[:NEGATIVES_PER_IMAGE]]
            if mine_score is not None:
                scores = mine_score(features(geometry, batch, cells[negatives]))
                hard = scores.argsort(descending=True, stable=True)[:NEGATIVES_PER_IMAGE]
                negative = torch.cat((negative, negatives[hard])).unique()
            picked = torch.cat((positive, negative))
            if len(picked):
                values = features(geometry, batch, cells[picked])
                samples_out.append(values.cpu())
                labels_out.append((torch.arange(len(picked)) < len(positive)).float())
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Fitting rank features {step}/{len(loader)} batches', flush=True)
    if observed != plan['fit_images'] or not samples_out:
        raise ValueError('Incomplete released rank fitting replay')
    return torch.cat(samples_out), torch.cat(labels_out)


def false_negative_ranks(iou, picked, quality):
    # Rank only negatives, in exactly the population used to draw the original fitting examples.
    negative = ~(iou > .5).any(dim=1)
    indices = torch.where(negative)[0]
    order = indices[quality[indices].argsort(descending=True, stable=True)]
    ranks = torch.zeros_like(quality, dtype=torch.long)
    ranks[order] = torch.arange(1, len(order) + 1, device=order.device)
    return ranks[picked[negative[picked]]].tolist()


@torch.inference_mode()
def measure(model, loader, post, params, known, plan, *, trace=False):
    keys = ('native', 'readout', 'label_rank')
    arms = {key: Counter(placed=0, matched=0) for key in keys}
    counts = Counter()
    observed = 0
    device = torch.device('cuda')
    score = readout_score(params)
    false_ranks = {'all_images': [], 'reachable_images': []}
    columns = len(known) + 1
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to(device))
        geometry = outputs['geometry']
        scores = post(outputs, outputs['pred_boxes'].new_ones((len(targets), 2)), include_top_indices=True)
        for batch, (target, row) in enumerate(zip(targets, scores, strict=True)):
            cells = fine_candidates(geometry, batch)
            truth = missed_boxes(outputs, batch, target, plan['small_area'])
            iou = box_iou(box_cxcywh_to_xyxy(geometry['boxes'][batch, cells]), truth)[0]
            readout = score(features(geometry, batch, cells))
            oracle = iou.amax(dim=1) if len(truth) else iou.new_full((len(cells),), -torch.inf)
            oracle = oracle.masked_fill(oracle <= .5, -torch.inf)
            _, reachable = positive_pairs(iou)
            counts.update(misses=len(truth), pool_reachable=int((iou > .5).any(dim=0).sum()),
                          two_cell_upper_bound=min(len(reachable), MAX_REPLACEMENTS))
            for key, priority in zip(keys, (None, readout, oracle), strict=True):
                reserve = choose_native(outputs, batch, row, columns, scores=priority)
                if reserve is not None:
                    picked = reserve[1]
                    indices = torch.searchsorted(cells, picked)
                    _, matched = positive_pairs(iou[indices])
                    arms[key].update(placed=len(picked), matched=len(matched))
                    if trace and key == 'readout':
                        ranks = false_negative_ranks(iou, indices, geometry['logits'][batch, cells])
                        false_ranks['all_images'].extend(ranks)
                        if len(reachable):
                            false_ranks['reachable_images'].extend(ranks)
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete released fine ranking replay')
    result = dict(arms=arms, capacity=counts, improved=arms['readout']['matched'] > arms['native']['matched'])
    if trace:
        result['false_ranks'] = {
            key: dict(total=len(ranks), outside=sum(rank > NEGATIVES_PER_IMAGE for rank in ranks),
                      median=median(ranks) if ranks else None)
            for key, ranks in false_ranks.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--mine-negatives', action='store_true')
    mode.add_argument('--decoder-replay', action='store_true')
    args = parser.parse_args()
    mining = args.mine_negatives
    output = MINED_OUTPUT if mining else OUTPUT
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for released fine-level ranking')
    source = verified(NATIVE_OUTPUT / 'comparison.json', RESERVATION_SHA)
    plan = verified(NATIVE_OUTPUT / 'plan.json', source['plan_sha256'])
    if source['status'] != 'complete' or all(row['groups']['passed'] for row in source['protocols'].values()):
        raise ValueError('Expected the failed native fine-level reservation')
    if args.decoder_replay:
        replay_mined_decoder(source)
        return
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                reservation_sha256=RESERVATION_SHA, positives_per_target=POSITIVES_PER_TARGET,
                negatives_per_image=NEGATIVES_PER_IMAGE, l2=L2, negative_cost=NEGATIVE_COST,
                factor='Fit a linear rank readout over frozen encoder features, raw quality and box geometry. '
                       'Use accurate fine cells for released misses and high raw-score negatives. Change only '
                       'the fine-cell order under the same native protection, suppression and slot budget.',
                advance='Require increased correct replacements in both protocols before replaying the decoder.')
    if mining:
        previous = verified(OUTPUT / 'comparison.json', RANK_SHA)
        previous_plan = verified(OUTPUT / 'plan.json', previous['plan_sha256'])
        if previous['status'] != 'complete' or previous['passed'] or any(
                plan[key] != value for key, value in previous_plan.items() if key != 'commit'):
            raise ValueError('Expected the original failed fine-rank plan and matched inputs')
        plan.update(rank_sha256=RANK_SHA, additional_negatives_per_image=NEGATIVES_PER_IMAGE,
                    factor='Trace raw negative ranks of prior readout replacements. Per protocol, refit once '
                           'if over half the selected false cells on pool-reachable images fall outside the '
                           'raw top 16 negatives. Fit on the union of raw and readout top 16 negatives.')
    pin(output / 'plan.json', plan)
    digest = file_sha256(output / 'plan.json')
    protocols = {}
    device = torch.device('cuda')
    for protocol, job in plan['jobs'].items():
        dest = output / f'{protocol}.json'
        if not dest.exists():
            path = MANIFESTS[protocol]
            manifest = verified(path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed released endpoint: {protocol}')
            inputs = (checkpoint, Path(job['training']), resolve_image_root(path, manifest), job)
            replay, order = training_loader(*inputs, plan, source_box_area=True)
            fit, fit_order = training_loader(*inputs, dict(plan, images=plan['fit_images']), source_box_area=True,
                                            offset=plan['images'])
            if (order != job['expected_order'] or fit_order != plan['fit_order_sha256'][protocol]
                    or set(source_ids(fit)) & set(source_ids(replay))):
                raise ValueError(f'Changed released fit or replay identities: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            known = job['classes']
            post = CompeteWithEmittedUnknown(known_ids=known, scoring='focal', max_detections=plan['limit'])
            mine_score = None
            extra = {}
            should_fit = True
            if mining:
                prior = previous['protocols'][protocol]
                params = prior['ranker']
                print(f'Tracing {protocol} frozen fine rank', flush=True)
                replay_state = replay.generator.get_state()
                report = measure(model, replay, post, params, known, plan, trace=True)
                replay.generator.set_state(replay_state)
                if report['arms'] != prior['arms'] or report['capacity'] != prior['capacity']:
                    raise ValueError(f'Changed prior released ranking replay: {protocol}')
                ranks = report['false_ranks']['reachable_images']
                should_fit = 2 * ranks['outside'] > ranks['total']
                extra.update(trace=report['false_ranks'], original_arms=prior['arms'], refitted=should_fit)
                mine_score = readout_score(params)
                print(f'{protocol} negative ranks {extra["trace"]}, refit={should_fit}', flush=True)
            if should_fit:
                print(f'Fitting {protocol} released fine rank', flush=True)
                params = fit_selector(*fit_examples(model, fit, plan, mine_score))
                if mining and params['positive'] != prior['ranker']['positive']:
                    raise ValueError(f'Changed released fitting positives: {protocol}')
                print(f'Comparing {protocol} frozen fine rank', flush=True)
                report = measure(model, replay, post, params, known, plan)
                if mining and (report['capacity'] != prior['capacity']
                               or report['arms']['label_rank'] != prior['arms']['label_rank']):
                    raise ValueError(f'Changed released replay capacity: {protocol}')
            reference = source['protocols'][protocol]['groups']['transfer']
            if report['arms']['native'] != dict(placed=reference['replacement_slots'], matched=reference['assigned']):
                raise ValueError(f'Changed native reservation on the same replay: {protocol}')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, fit_order_sha256=fit_order,
                                  ranker=params, **extra, **report))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed released fine-rank plan: {dest}')
        keys = ('arms', 'capacity', 'ranker', 'improved')
        if mining:
            keys += ('trace', 'original_arms', 'refitted')
        protocols[protocol] = {key: report[key] for key in keys}
    write_json(output / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols,
                                               passed=all(row['improved'] for row in protocols.values())))


if __name__ == '__main__':
    main()
