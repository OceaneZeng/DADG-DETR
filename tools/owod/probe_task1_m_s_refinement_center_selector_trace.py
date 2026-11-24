"""Pair released target matches for native, midpoint and selected query centers."""

from collections import Counter
from pathlib import Path
import subprocess

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_task1_m_s_refinement_center_selector import (
    CENTER_COMPARISON_SHA, CENTER_OUTPUT, OUTPUT as SELECTOR_OUTPUT, correction_logits, refine_learned_boxes,
)
from tools.owod.probe_task1_m_s_refinement_pseudo_scores import training_loader
from tools.owod.probe_task1_paired_discovery import verified
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_task1_m_s_iterative_refinement import MANIFESTS
from tools.owod.screening_data import write_json
from util.box_ops import box_cxcywh_to_xyxy, box_iou


OUTPUT = SELECTOR_OUTPUT.with_name('task1-m-s-refinement-center-selector-trace')
SELECTOR_SHA = '9a32eb2af9fcf9988b341baf6893f1c75feb44b5c373928b5046f67d95d1f6a3'


@torch.inference_mode()
def trace(model, loader, plan, score):
    counts = {name: Counter() for name in ('all', 'small')}
    events = []
    observed = 0
    for step, (samples, targets) in enumerate(loader, 1):
        outputs = model(samples.to('cuda'))
        final = outputs['pred_boxes']
        margins = final.new_full(final.shape[:2], -torch.inf)

        def record(batch, queries, center, quality):
            margins[batch, queries] = score(outputs, batch, queries, center, quality)
            return torch.ones_like(queries, dtype=torch.bool)

        midpoints = refine_learned_boxes(outputs, center_only=True, choose_centers=record)
        for batch, target in enumerate(targets):
            gt = box_cxcywh_to_xyxy(target['boxes'].cuda())
            small = target['source_box_area'].cuda() < plan['small_area']
            old, new = (box_iou(box_cxcywh_to_xyxy(boxes[batch]), gt)[0] > .5
                        for boxes in (final, midpoints))
            selected = torch.where(margins[batch, :, None] > 0, new, old)
            safe = ~(old & ~new).any(dim=1)
            protected = torch.where(safe[:, None], new, old)
            base, full, gated, preserved = (matches.any(dim=0) for matches in (old, new, selected, protected))
            if not (preserved | ~base).all():
                raise ValueError('The annotation-only protected replay lost a native hit')
            for name, mask in (('all', torch.ones_like(small)), ('small', small)):
                counts[name].update(gt=int(mask.sum()), final=int((base & mask).sum()),
                                    midpoint_gained=int((full & ~base & mask).sum()),
                                    midpoint_lost=int((base & ~full & mask).sum()),
                                    selected_gained=int((gated & ~base & mask).sum()),
                                    selected_lost=int((base & ~gated & mask).sum()),
                                    protected_gained=int((preserved & ~base & mask).sum()),
                                    midpoint_gain_refused=int((full & ~base & ~gated & mask).sum()))
            changed_matches = (old ^ new).any(dim=0) & small
            for pos in torch.where(changed_matches)[0].tolist():
                query_ids = torch.where(old[:, pos] | new[:, pos])[0]
                event = dict(image=int(target['image_id']), target_index=pos, native=bool(base[pos]),
                             midpoint=bool(full[pos]), selected=bool(gated[pos]), protected=bool(preserved[pos]),
                             queries=[])
                for query in query_ids.tolist():
                    margin = float(margins[batch, query])
                    event['queries'].append(dict(query=query, native=bool(old[query, pos]),
                                                midpoint=bool(new[query, pos]), selected=bool(selected[query, pos]),
                                                no_native_hit_lost=bool(safe[query]),
                                                logit=margin if margin != -torch.inf else None))
                events.append(event)
        observed += len(targets)
        if step % 64 == 0 or step == len(loader):
            print(f'Inference {step}/{len(loader)} batches', flush=True)
    if observed != plan['images']:
        raise ValueError('Incomplete fixed released replay')
    return counts, events


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3090':
        raise ValueError('Use one RTX 3090 for the frozen released replay')
    source = verified(SELECTOR_OUTPUT / 'comparison.json', SELECTOR_SHA)
    midpoint = verified(CENTER_OUTPUT / 'comparison.json', CENTER_COMPARISON_SHA)
    plan = verified(SELECTOR_OUTPUT / 'plan.json', source['plan_sha256'])
    plan.update(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                selector_comparison_sha256=SELECTOR_SHA,
                factor='Pair target identities and matching queries at the native, ungated midpoint, and frozen '
                       'selector endpoints. A separate annotation-only diagnostic accepts every midpoint '
                       'that preserves all native hits of its query, including non-small released targets.')
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
            scorer = correction_logits(source['protocols'][protocol]['selector'], job['classes'], device)
            print(f'Evaluating {protocol} paired selector decisions', flush=True)
            summary, events = trace(model, loader, plan, scorer)
            for name, row in summary.items():
                for prefix, reference in (('midpoint', midpoint), ('selected', source)):
                    expected = reference['protocols'][protocol]['groups'][name]
                    if any(row[field] != expected[field] for field in ('gt', 'final')) or any(
                            row[f'{prefix}_{field}'] != expected[field] for field in ('gained', 'lost')):
                        raise ValueError(f'Changed frozen coverage: {protocol}/{name}/{prefix}')
            write_json(dest, dict(plan_sha256=digest, order_sha256=order, summary=summary, events=events))
            del model
        report = read_json(dest)
        if report['plan_sha256'] != digest:
            raise ValueError(f'Changed frozen trace plan: {dest}')
        protocols[protocol] = report['summary']
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, protocols=protocols))


if __name__ == '__main__':
    main()
