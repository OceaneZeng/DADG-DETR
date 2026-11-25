"""Fit paired novelty heads on one pass of shared frozen S Task 1 features."""

import copy
import math
import subprocess
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from models.graph_local.discovery import (
    balanced_bce, duplicate_known_queries, matched_known_ranking_loss, proposal_supervision,
)
from models.graph_local.discovery_candidates import flip_samples, mine_candidates
from tools.owod.diagnose_learning import make_loader
from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task1_half_rank_quota import MIDPOINT_SHA, RETRY
from tools.owod.probe_s_task1_novelty_head_transfer import TransferredNoveltyPostprocess, require_one_gpu, run
from tools.owod.probe_s_task1_novelty_rank_remap import OUTPUT as REMAP
from tools.owod.probe_s_task1_rank_independent import PROTOCOL
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.run_task1_m_s_released_small_extent import annotation_digest
from tools.owod.screening_data import screening_files, select_images, source_ids, subset, write_json
from util.training_runtime import atomic_torch_save


OUTPUT = REMAP.with_name('task1-s-frozen-novelty-fit')
REMAP_SHA = '01a6a628420b9ced5efa3a4389b230580ddf90564ed7d339a3ae4ccb9cbd8dcb'
ARMS = ('bce', 'bce_rank')
TRAIN_IMAGES, TRAIN_BATCH, SEED = 2048, 4, 42
FIT_LR, FIT_WEIGHT_DECAY = .0002, .0001


def prepare():
    remap = verified(REMAP / 'comparison.json', REMAP_SHA)
    prior = verified(REMAP / 'plan.json', remap['plan_sha256'])
    half = verified(RETRY / 'comparison.json', MIDPOINT_SHA)
    original = verified(RETRY / 'plan.json', half['plan_sha256'])
    control = prior['arms']['control']
    if (remap['status'] != 'complete' or remap['passed'] or half['status'] != 'complete'
            or half['checkpoint_sha256'] != control['sha256']
            or file_sha256(Path(control['checkpoint'])) != control['sha256']
            or prior['manifest_sha256'] != original['manifest_sha256']):
        raise ValueError('Expected the verified frozen half-gradient endpoint')
    manifest_path = MANIFESTS[PROTOCOL]
    manifest = verified(manifest_path, prior['manifest_sha256'])
    command = original['command']
    screen_path = Path(command[command.index('--owod-screening') + 1])
    if file_sha256(screen_path) != original['screen_sha256']:
        raise ValueError('Changed original training screen')
    _, paths = screening_files(manifest_path, screen_path, 0)
    source_path = paths['increment_train']
    payload = read_json(source_path)
    known_ids = manifest['stages'][0]['classes']
    excluded = source_ids(verified(Path(prior['holdout']), prior['holdout_sha256']))
    for entry in prior['folds'].values():
        excluded |= source_ids(verified(Path(entry['path']), entry['sha256']))
    names = select_images(payload, TRAIN_IMAGES, known_ids, SEED, excluded)
    training = subset(payload, names, set(known_ids))
    train_input = OUTPUT / 'train.json'
    keys = ('holdout', 'holdout_sha256', 'manifest_sha256', 'batch_size', 'limit', 'min_known_ap50',
            'folds', 'fold_native', 'max_wi_increase', 'max_aose_fraction')
    plan = dict(**{key: prior[key] for key in keys}, control=control, remap_sha256=REMAP_SHA,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                paper_comparable=False, screen_sha256=original['screen_sha256'],
                screen_train_sha256=file_sha256(source_path), subset_sha256=annotation_digest(training),
                input=str(train_input), train_images=TRAIN_IMAGES, train_batch=TRAIN_BATCH, seed=SEED, passes=1,
                lr=FIT_LR, weight_decay=FIT_WEIGHT_DECAY, rank_weights=dict(bce=0., bce_rank=1.),
                teacher='Same frozen half-gradient checkpoint on horizontally flipped images; no EMA or memory update.',
                transforms='Deterministic calibration resize and normalization on the training-only screen subset.')
    pin(OUTPUT / 'plan.json', plan)
    if not train_input.is_file():
        write_json(train_input, training)
    if file_sha256(train_input) != plan['subset_sha256']:
        raise ValueError('Changed frozen-head training subset')
    return plan, manifest


@torch.no_grad()
def frozen_novelty_batch(model, criterion, samples, targets):
    features = []

    def capture(_head, inputs, _output):
        features.append(inputs[0])

    with model.affinity_graph.novelty.register_forward_hook(capture):
        outputs = model(samples)
    embedding, = features
    teacher = model.forward_teacher(flip_samples(samples), purpose='discovery')
    indices = criterion.matcher(outputs, targets)
    candidates = mine_candidates(outputs, targets, indices, criterion.known_ids, teacher,
                                 criterion.top_k, criterion.saliency_threshold, criterion.consistency_threshold,
                                 saliency_top_points=criterion.saliency_top_points,
                                 saliency_match_count=criterion.saliency_match_count)
    pseudo, confidence = candidates['pseudo'], candidates['weights']
    positive = torch.zeros_like(pseudo)
    for image, (queries, _) in enumerate(indices):
        positive[image, queries] = True
    foreground, weights, _ = proposal_supervision(positive, pseudo, confidence)
    duplicate, known_logits = duplicate_known_queries(outputs, targets, foreground, criterion.known_ids)
    return embedding, pseudo, weights, positive, duplicate, known_logits


def fit(plan, manifest, digest):
    device = torch.device('cuda')
    model, criterion = load_detector(None, Path(plan['control']['checkpoint']), device)
    model.eval().requires_grad_(False)
    if criterion.known_ids != manifest['stages'][0]['classes'] or criterion.background_negative:
        raise ValueError('Unexpected frozen baseline novelty supervision')
    heads = {name: copy.deepcopy(model.affinity_graph.novelty).requires_grad_(True) for name in ARMS}
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
                  for name, head in heads.items()}
    root = resolve_image_root(MANIFESTS[PROTOCOL], manifest)
    dataset, loader = make_loader(root, plan['input'], plan['train_batch'], 2, shuffle=True, seed=plan['seed'])
    if len(dataset) != plan['train_images']:
        raise ValueError('Changed frozen-head fitting budget')
    counts = Counter(known=0, pseudo=0, duplicate=0)
    totals = Counter(bce=0., bce_rank=0., rank=0.)
    for step, (samples, targets) in enumerate(loader, 1):
        released = [{key: target[key].to(device) for key in ('boxes', 'labels')} for target in targets]
        embedding, pseudo, weights, positive, duplicate, known_logits = frozen_novelty_batch(
            model, criterion, samples.to(device), released)
        for name, head in heads.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            logits = head(embedding).squeeze(-1)
            loss = balanced_bce(logits, pseudo.float(), weights, normalization=criterion.pseudo_normalization)
            if plan['rank_weights'][name]:
                log_ratio = F.logsigmoid(logits[duplicate]) - F.logsigmoid(known_logits[duplicate])
                rank = matched_known_ranking_loss(log_ratio)
                loss = loss + plan['rank_weights'][name] * rank
                totals['rank'] += float(rank.detach())
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite frozen-head loss on step {step}: {name}')
            totals[name] += value
            loss.backward()
            optimizer.step()
        counts.update(known=int(positive.sum()), pseudo=int((pseudo & (weights > 0)).sum()),
                      duplicate=int(duplicate.sum()))
        if step == 1 or step % 32 == 0 or step == len(loader):
            print(f'Epoch: [0] [{step}/{len(loader)}] iters, supervision {dict(counts)}', flush=True)
    if not all(counts.values()):
        raise ValueError(f'Insufficient frozen-head supervision: {dict(counts)}')
    fitted = {}
    for name, head in heads.items():
        dest = OUTPUT / f'{name}.pth'
        state = {f'affinity_graph.novelty.{key}': value.detach().cpu() for key, value in head.state_dict().items()}
        if any(not value.isfinite().all() for value in state.values()):
            raise ValueError(f'Nonfinite fitted novelty layer: {name}')
        atomic_torch_save(dict(model=state, plan_sha256=digest, arm=name), dest)
        fitted[name] = dict(checkpoint=str(dest), sha256=file_sha256(dest))
    write_json(OUTPUT / 'fit.json', dict(plan_sha256=digest, steps=len(loader), counts=dict(counts), arms=fitted,
                                        average_losses={key: value / len(loader) for key, value in totals.items()}))


def main():
    require_one_gpu()
    plan, manifest = prepare()
    digest = file_sha256(OUTPUT / 'plan.json')
    fit_path = OUTPUT / 'fit.json'
    if not fit_path.is_file():
        print('Fitting paired novelty heads on frozen S Task 1 training inputs', flush=True)
        fit(plan, manifest, digest)
    fit_record = read_json(fit_path)
    if (fit_record['plan_sha256'] != digest or fit_record['steps'] != plan['train_images'] // plan['train_batch']
            or set(fit_record['arms']) != set(ARMS) or not all(fit_record['counts'].values())):
        raise ValueError('Unexpected paired frozen-head fitting record')
    outcomes = {}
    for name, endpoint in fit_record['arms'].items():
        if file_sha256(Path(endpoint['checkpoint'])) != endpoint['sha256']:
            raise ValueError(f'Changed fitted novelty layer: {name}')
        output = OUTPUT / name
        arm_plan = dict(plan, arms=dict(control=plan['control'], candidate=endpoint), arm=name,
                        parent_plan_sha256=digest, fit_sha256=file_sha256(fit_path))
        pin(output / 'plan.json', arm_plan)
        print(f'Evaluating frozen S fitted head: {name}', flush=True)
        run(arm_plan, output, TransferredNoveltyPostprocess)
        result = read_json(output / 'comparison.json')
        if result['plan_sha256'] != file_sha256(output / 'plan.json') or result['donor_sha256'] != endpoint['sha256']:
            raise ValueError(f'Unexpected evaluation record: {name}')
        outcomes[name] = result
    bce, rank = (outcomes[name] for name in ARMS)
    paired = dict(released_known_ap50=rank['released']['transferred'] - bce['released']['transferred'])
    for fold, entry in rank['folds'].items():
        native_bce = bce['folds'][fold]['metrics']['transferred']
        paired[fold] = {key: val - native_bce[key] for key, val in entry['metrics']['transferred'].items()}
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, fit_sha256=file_sha256(fit_path),
                                               outcomes=outcomes, paired_delta=paired, passed=rank['passed']))


if __name__ == '__main__':
    main()
