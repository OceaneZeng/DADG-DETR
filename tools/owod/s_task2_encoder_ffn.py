"""Track the encoder parameters and the in-sample AP of the S increment FFN study."""
from hashlib import sha256
from pathlib import Path

import torch

from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_s_task2_duration import MANIFEST


def encoder_signatures(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    prefix = 'detr.transformer.encoder.'
    last = f'{prefix}layers.{payload["args"].enc_layers - 1}.'
    selected = {last + name + '.' + field for name in ('linear1', 'linear2') for field in ('weight', 'bias')}
    state = payload['model']
    if not selected <= state.keys():
        raise ValueError('Checkpoint lacks the final encoder FFN tensors')
    digests = {'ffn': sha256(), 'other_encoder': sha256()}
    for name in sorted(k for k in state if k.startswith(prefix)):
        group = 'ffn' if name in selected else 'other_encoder'
        digests[group].update(name.encode())
        digests[group].update(state[name].contiguous().numpy().tobytes())
    return dict(ffn=digests['ffn'].hexdigest(), other_encoder=digests['other_encoder'].hexdigest(),
                parameters=sum(state[k].numel() for k in selected))


def register_evidence(plan, output, parent):
    fit = output.with_name('s-instance-task2-alternate-pool-fit')
    training = output.with_name('s-instance-task2-increment-alternate-images')
    result, source = read_json(fit/'comparison.json'), read_json(fit/'plan.json')
    prior = read_json(training/'plan.json')
    ref = read_json(fit/'repeat-A.json')
    if (file_sha256(fit/'comparison.json') != '61b846e9a4743b09623700ea80120e19a04dea83d3cb36edc3c1fae3bb6dff49'
            or result['plan_sha256'] != file_sha256(fit/'plan.json') or result['b_specific'] or not result['loss_on_a']
            or source['source_sha256'] != file_sha256(training/'comparison.json')
            or source['source_plan_sha256'] != file_sha256(training/'plan.json') or prior['identity'] != plan['identity']
            or source['checkpoints']['repeat'] != plan['control']['increment']
            or ref['checkpoint_sha256'] != plan['control']['increment']['sha256']
            or ref['plan_sha256'] != result['plan_sha256'] or ref['annotation_sha256'] != source['annotations']['A']
            or file_sha256(fit/'A-annotations.json') != ref['annotation_sha256']
            or prior['annotation_sha256'] != plan['annotation_sha256']
            or file_sha256(Path(prior['independent_annotation'])) != prior['independent_annotation_sha256']):
        raise ValueError('Changed A/A/A control, earlier fit evidence or held-out annotations')
    baseline = encoder_signatures(parent)
    for phase in ('increment', 'finetune'):
        if encoder_signatures(Path(plan['control'][phase]['path'])) != baseline:
            raise ValueError('Expected the original control to retain the parent encoder')
    plan.update(parent_encoder=baseline, source_fit_sha256=file_sha256(fit/'comparison.json'),
                train_a_annotation=str(fit/'A-annotations.json'), train_a_annotation_sha256=ref['annotation_sha256'],
                train_a_reference_sha256=file_sha256(fit/'repeat-A.json'), train_a_reference=result['arms']['repeat']['A'],
                intervention='Unfreeze only the last encoder linear1 and linear2 weights and biases during increment. '
                             'Use the original base learning rate. Freeze these inherited parameters in balanced finetuning.',
                primary='Finetune native Current AP50 on both fixed 1024-image release-source holdouts.')
    for key in ('independent_annotation', 'independent_order_sha256', 'independent_annotation_sha256'):
        plan[key] = prior[key]
    plan['criterion'] += (' Apply every gate independently on both slices. Separately report whether Current AP '
                          'on the original 1536 training images rises by at least 1 pp at the increment endpoint.')


def check_encoder(path, phase, plan, report):
    from tools.owod.run_s_task2_finetune_half_lr import optimizer_groups

    sig = encoder_signatures(path)
    base = plan['parent_encoder']
    ref = Path(plan['control'][phase]['path']).parent
    expected_extra = base['parameters'] if phase == 'increment' else 0
    count = lambda log: sum(size for _, size in optimizer_groups(log))
    if (sig['parameters'] != base['parameters'] or sig['other_encoder'] != base['other_encoder']
            or count(path.parent/'train.log') - count(ref/'train.log') != expected_extra
            or sig['ffn'] == base['ffn']):
        raise ValueError('Unexpected encoder changes or optimizer parameter count')
    if phase == 'finetune' and sig != report['phases']['increment']['encoder']:
        raise ValueError('Balanced finetuning changed the transferred encoder parameters')
    report['phases'][phase]['encoder'] = sig


def evaluate_training_pool(path, plan, output, manifest):
    from tools.owod.run_s_task2_increment_giou import evaluate

    annotation = Path(plan['train_a_annotation'])
    order = [im['owod_source_id'] for im in read_json(annotation)['images']]
    if len(order) != 1536 or file_sha256(annotation) != plan['train_a_annotation_sha256']:
        raise ValueError('Changed in-sample diagnostic images')
    ds, loader = make_loader(resolve_image_root(MANIFEST, manifest), annotation, 2, 2)
    view = evaluate('increment on training pool A', path, plan, ds, loader, manifest, order, proxies=False,
                    output=output, result_path=output/'increment-train-A.json')
    metrics = view['released']['metrics']
    ref = plan['train_a_reference']
    deltas = {k: metrics[k] - ref[k] for k in ('current_ap50', 'previous_ap50', 'known_ap50')}
    raw = view['raw_current_box_recall_iou50']
    if raw['gt'] != ref['raw_current']['gt']:
        raise ValueError('Training pool coverage uses a different GT population')
    return dict(metrics=metrics, delta=deltas, raw_hits_gained=raw['hits'] - ref['raw_current']['hits'],
                fit_gain=deltas['current_ap50'] >= .01)
