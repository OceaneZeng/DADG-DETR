"""Selection and validation for the increment low-frequency class weighting experiment."""
from pathlib import Path

from tools.owod.protocol import file_sha256, read_json


def configure_tail(plan, output, training_file):
    coverage = output.with_name('s-instance-task2-increment-match-coverage')
    audit = read_json(coverage/'3epochs.json')
    audit_plan = read_json(coverage/'plan.json')
    source = output.with_name('s-instance-task2-crop-ap-independent')
    independent = read_json(source/'plan.json')
    comparison = read_json(source/'comparison.json')
    original = output.with_name('s-instance-task2-crop-ap-anchors')
    if (file_sha256(coverage/'3epochs.json') != 'c69f13afe40dbc50bbba6eb76821b27ceced9a85e5d2ef631fa77f7b71aec3fe'
            or audit['plan_sha256'] != file_sha256(coverage/'plan.json')
            or file_sha256(training_file) != audit_plan['annotation_sha256']
            or audit['checkpoint_sha256'] != plan['control']['increment']['sha256']
            or file_sha256(source/'comparison.json') != 'f5a22691784962512a5f5d28e81a04ddb61cb702e74f0dfc5955011a711af29b'
            or comparison['replicated'] or comparison['plan_sha256'] != file_sha256(source/'plan.json')
            or independent['parent_attribution_sha256'] != file_sha256(original/'comparison.json')
            or independent['original_annotation_sha256'] != plan['annotation_sha256']
            or file_sha256(Path(independent['annotation'])) != independent['annotation_sha256']):
        raise ValueError('Changed training class counts, weights or held-out evidence')
    counts = {int(label): row['gt'] for label, row in audit['classes'].items()}
    tail = sorted(counts, key=lambda c: (counts[c], c))[:len(counts)//4]
    for folder in (original, source):
        for key in independent['endpoints']:
            stats = read_json(folder/(key+'.json'))['view']['current_class_stats']
            if any(stats[str(c)]['gt'] <= 0 or stats[str(c)]['tp'] for c in tail):
                raise ValueError('Expected supported low-frequency classes with zero TP across both slices')
    plan['jobs']['increment'].extend(['--screen-tail-positive-ids', *map(str, tail)])
    plan.update(tail_class_ids=tail, increment_class_counts={str(c): n for c, n in counts.items()}, training_annotation_sha256=audit_plan['annotation_sha256'],
                training_audit_sha256=file_sha256(coverage/'3epochs.json'),
                independent_comparison_sha256=file_sha256(source/'comparison.json'),
                independent_annotation=independent['annotation'], independent_annotation_sha256=independent['annotation_sha256'],
                independent_order_sha256=independent['eval_order_sha256'],
                intervention='Double only the matched focal positives of the least frequent floor(K/4) current classes '
                             'during increment. All counts are from the original increment training images.',
                primary='Final native focal Current AP50 on both fixed 1024-image training-source validation sets.')
    plan['criterion'] += (' Apply every known and open-set gate to each slice separately. Require at least two of the '
                          'same weighted classes to gain true positives over the paired control on both slices at finetune.')


def gained_classes(candidate, control, tail):
    return {label for label in tail if candidate[str(label)]['tp'] > control[str(label)]['tp']}
