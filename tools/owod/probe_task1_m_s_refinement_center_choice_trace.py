"""Compare frozen pointwise and query-level center choices on released images."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_quality_folds import GROUP_FOLD_OUTPUT, GROUP_PREFLIGHT_SHA
from tools.owod.probe_task1_m_s_refinement_center_selector import (
    GROUP_OUTPUT, RANK_COMPARISON_SHA, RANK_OUTPUT, center_labels, correction_logits, refine_learned_boxes,
)
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json


OUTPUT = GROUP_OUTPUT.with_name('task1-m-s-refinement-center-choice-trace')
FOLDS_SHA = '60aa9853e997715ff93a022e818e8ce0cbc1726e080b60e69b415b717e231020'


@torch.inference_mode()
def trace(model, loader, plan, scorers):
    counts = {name: Counter() for name in scorers}
    events = []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        options = {}

        def capture(batch, queries, centers, quality):
            options[batch] = (queries, centers, quality)

        refine_learned_boxes(outputs, center_only=True, observe_centers=capture)
        for batch, target in enumerate(targets):
            if batch not in options:
                continue
            queries, centers, quality = options[batch]
            candidates = quality.isfinite()
            rows, columns = torch.where(candidates)
            good, relevant = center_labels(outputs, target, batch, queries[rows], centers[rows, columns],
                                           plan['small_area'])
            can_gain = torch.zeros_like(candidates)
            can_harm = torch.zeros_like(candidates)
            can_gain[rows, columns] = good
            can_harm[rows, columns] = relevant & ~good
            useful = can_gain.any(dim=1)
            affected = (can_gain | can_harm).any(dim=1)
            idx = torch.arange(len(queries), device=queries.device)
            outcomes = {}
            for name, score in scorers.items():
                value = score(outputs, batch, queries[rows], centers[rows, columns], quality[rows, columns])
                logits = torch.full_like(quality, -torch.inf)
                logits[rows, columns] = value
                best, selected = logits.max(dim=1)
                move = best > 0
                beneficial = move & can_gain[idx, selected]
                harmful = move & can_harm[idx, selected]
                neutral = move & ~beneficial & ~harmful
                actually_changed = move & (centers[idx, selected] != final[batch, queries, :2]).any(dim=1)
                counts[name].update(oracle_positive=int(useful.sum()), changed_queries=int(actually_changed.sum()),
                                    beneficial=int(beneficial.sum()), harmful=int(harmful.sum()),
                                    neutral=int((neutral & actually_changed).sum()),
                                    missed_positive=int((useful & ~beneficial).sum()),
                                    stayed_with_safe_option=int((useful & ~move).sum()))
                outcomes[name] = (best, beneficial, harmful, move, logits.masked_fill(~can_gain, -torch.inf).amax(dim=1))
            for row in torch.where(affected)[0].tolist():
                choices = {}
                for name, (best, beneficial, harmful, move, safe_logit) in outcomes.items():
                    choices[name] = dict(logit=float(best[row]), move=bool(move[row]),
                                         beneficial=bool(beneficial[row]), harmful=bool(harmful[row]),
                                         best_safe_logit=float(safe_logit[row]) if useful[row] else None)
                events.append(dict(image=int(target['image_id']), query=int(queries[row]),
                                   beneficial_candidates=int(can_gain[row].sum()), harmful_candidates=int(can_harm[row].sum()),
                                   choices=choices))
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete fixed released replay')
    return counts, events


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    folds = verified(GROUP_FOLD_OUTPUT / 'comparison.json', FOLDS_SHA)
    if folds['status'] != 'complete' or folds['passed']:
        raise ValueError('Expected the completed query choice fold screen')
    sources = {'pointwise': verified(RANK_OUTPUT / 'comparison.json', RANK_COMPARISON_SHA),
               'query_choice': verified(GROUP_OUTPUT / 'comparison.json', GROUP_PREFLIGHT_SHA)}
    plan = verified(GROUP_OUTPUT / 'plan.json', sources['query_choice']['plan_sha256'])
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                query_choice_folds_sha256=FOLDS_SHA, query_choice_sha256=GROUP_PREFLIGHT_SHA,
                pointwise_sha256=RANK_COMPARISON_SHA,
                factor='Pair the frozen choice scores for pointwise and per-query fits. Record safe moves '
                       'and harmful moves at the original zero-logit threshold.')
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    device = torch.device('cuda')
    protocols = {}
    for protocol, job in plan['jobs'].items():
        dest = OUTPUT / f'{protocol}.json'
        if not dest.exists():
            manifest_path = MANIFESTS[protocol]
            manifest = verified(manifest_path, job['manifest_sha256'])
            checkpoint = Path(job['checkpoint'])
            if file_sha256(checkpoint) != job['checkpoint_sha256']:
                raise ValueError(f'Changed matched-count endpoint: {protocol}')
            loader, order = training_loader(checkpoint, Path(job['training']), resolve_image_root(manifest_path, manifest),
                                            job, plan, source_box_area=True)
            if order != job['expected_order']:
                raise ValueError(f'Changed released replay order: {protocol}')
            model, _ = load_detector(None, checkpoint, device)
            model.eval()
            refined_quality = model.detr.transformer.instance_proposals.refined_quality
            scorers = {name: correction_logits(source['protocols'][protocol]['selector'], job['classes'],
                                                device, refined_quality) for name, source in sources.items()}
            print(f'Evaluating {protocol} frozen query choices', flush=True)
            summary, events = trace(model, loader, plan, scorers)
            for name, source in sources.items():
                if summary[name]['changed_queries'] != source['protocols'][protocol]['changes']['queries']:
                    raise ValueError(f'Changed frozen selector choices: {protocol}/{name}')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, summary=summary, events=events))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen choice plan: {dest}')
        protocols[protocol] = report['summary']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
