"""Evidence and paired shortlist measurements for increment-stage quality adaptation."""
from hashlib import sha256
from pathlib import Path

import torch

from tools.owod.gnn_calibration import load_detector
from tools.owod.probe_s_task2_proposal_stages import OUTPUT as STAGES, trace
from tools.owod.protocol import file_sha256, read_json
from tools.owod.screening_data import write_json


def quality_digest(path):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)['model']
    keys = sorted(k for k in state if '.instance_proposals.quality.' in k)
    if not keys:
        raise ValueError('Checkpoint lacks the proposal quality MLP')
    digest = sha256()
    for key in keys:
        digest.update(key.encode())
        digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def add_evidence(plan, parent):
    result = read_json(STAGES / 'comparison.json')
    source = read_json(STAGES / 'plan.json')
    result_sha = file_sha256(STAGES / 'comparison.json')
    reference_sha = file_sha256(STAGES / 'increment.json')
    if (result_sha != '510173830230479750ac142f3c5db84d2ba3d9d38d2c2d724a7f8067465da707'
            or reference_sha != '0f17c85a939ce4466ad1d8ffa0d98b4bd3727df416a87d32d29b2b46f3460beb'
            or result['plan_sha256'] != file_sha256(STAGES / 'plan.json') or result['priority'] != ['outside_shortlist']
            or source['endpoints'] != plan['control'] or source['annotation_sha256'] != plan['annotation_sha256']):
        raise ValueError('Changed source stage trace, controls or release-only annotations')
    baseline = quality_digest(parent)
    if any(quality_digest(e['path']) != baseline for e in plan['control'].values()):
        raise ValueError('Historical control quality head differs from the parent')
    plan.update(source_stages_sha256=result_sha, source_increment_trace_sha256=reference_sha,
                parent_quality_sha256=baseline,
                intervention='During increment, additionally train only the proposal quality MLP with the existing '
                             'objective and the base learning rate. Freeze its inherited weights during finetune.')
    plan['criterion'] += ' Also require more unique Current targets covered by the increment shortlist than in the frozen control.'


def compare_shortlist(path, output, loader, manifest, plan):
    trace_path = output / 'increment-proposals.json'
    digest = file_sha256(output / 'plan.json')
    checkpoint_sha = file_sha256(path)
    if not trace_path.is_file():
        classes = set(manifest['stages'][1]['classes'])
        anns = {i: a for i, a in loader.dataset.coco.anns.items() if a['category_id'] in classes and not a.get('difficult', 0)}
        model, _ = load_detector(None, Path(path), torch.device('cuda'))
        print('Measuring increment proposal coverage on the paired holdout', flush=True)
        hits = trace(model, loader, anns, manifest['evaluation']['iou_threshold'])
        write_json(trace_path, dict(checkpoint_sha256=checkpoint_sha, plan_sha256=digest, hits=hits))
    actual = read_json(trace_path)
    if (actual['checkpoint_sha256'] != checkpoint_sha or actual['plan_sha256'] != digest
            or file_sha256(STAGES / 'increment.json') != plan['source_increment_trace_sha256']):
        raise ValueError('Proposal trace identity mismatch')
    reference = read_json(STAGES / 'increment.json')['hits']
    rows = {}
    for stage in ('dense', 'shortlist', 'selected', 'final'):
        new, old = set(actual['hits'][stage]), set(reference[stage])
        rows[stage] = dict(candidate=len(new), control=len(old), gained=len(new-old), lost=len(old-new))
    if rows['dense']['gained'] or rows['dense']['lost']:
        raise ValueError('Changing the quality head should preserve the dense box pool')
    return rows
