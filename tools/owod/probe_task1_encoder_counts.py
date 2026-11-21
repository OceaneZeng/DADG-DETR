"""Compare encoder rankings after matching actual encoder-seeded slots in each view."""
from pathlib import Path

from tools.owod import probe_task1_discovery_gates as gates
from tools.owod import probe_task1_encoder_levels as levels
from tools.owod import probe_task1_view_boxes as paired
from tools.owod.diagnose_learning import make_loader
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_instance_pilot import MANIFEST, ROOT
from tools.owod.screening_data import write_json


OUTPUT = ROOT / 'output/m-instance-task1-encoder-counts'
ARMS = ('global', 'within_level')


def verify_prefixes(saved, originals, arm):
    for image, row in saved.items():
        original_ids = originals[arm][image]['encoder_ids']
        counts = tuple(originals[name][image]['active'] for name in ARMS)
        if ((row['student_global'], row['student_within']) != counts
                or row['encoder_ids'] != original_ids[:min(counts)] or row['active'] != min(counts)):
            raise ValueError(f'{arm}: encoder IDs do not match the archived prefix at {image}')


def main():
    parent = levels.OUTPUT
    plan, manifest, old_report, large = paired.paired_plan(
        parent, 'Both rankings compute their original NMS lists on each view. Retain the leading '
        'min(global count, within-level count) cells from the assigned ranking in each view separately. '
        'Verify student encoder IDs against archived prefixes and teacher counts across paired forwards.',
        'Both seeds: equal per-image student and teacher encoder counts and original anonymous output quotas, '
        'higher final new-class coverage and soft hits than matched global on holdout1024, gain >=5 pp over '
        'the original global hard control there, <=1 pp drop in known raw coverage versus matched global, '
        'and no matched soft-hit regression on reference128.', arms=ARMS)
    pin(OUTPUT / 'plan.json', plan)
    digest = file_sha256(OUTPUT / 'plan.json')
    root = resolve_image_root(MANIFEST, manifest)
    rows, passed = {}, {}
    for seed, job in plan['jobs'].items():
        model, criterion, ema = gates.frozen_miner(Path(job['parent']), manifest)
        student = model.detr.transformer.instance_proposals
        teacher = ema.detr.transformer.instance_proposals
        student.match_encoder_counts = teacher.match_encoder_counts = True
        teacher_budget = []

        def record_teacher(module, inputs, output):
            state = output[-1]
            teacher_budget[:] = zip(state['natural_counts'], state['active'].sum(1).tolist())

        hook = teacher.register_forward_hook(record_teacher)
        rows[seed] = {}
        try:
            for label, annotation in [('reference128', Path(job['small'])), ('holdout1024', large)]:
                originals = {arm: read_json(parent / f'seed{seed}-{label}-{arm}.json')['images'] for arm in ARMS}
                control = originals['global']
                matched = {}
                rows[seed][label] = {}
                for arm in ARMS:
                    student.rank_within_level = teacher.rank_within_level = arm == 'within_level'
                    path = OUTPUT / f'seed{seed}-{label}-{arm}.json'
                    if not path.exists():
                        _, loader = make_loader(root, annotation, plan['batch_size'], 2)
                        print(f'Seed {seed}, {label}, {arm}: matched encoder counts', flush=True)
                        result = levels.evaluate(model, criterion, ema, loader, manifest['stages'][1]['classes'],
                                                 originals['global'], control, teacher_budget=teacher_budget)
                        write_json(path, dict(plan_sha256=digest, arm=arm, **result))
                    saved = read_json(path)
                    if saved['arm'] != arm or saved['plan_sha256'] != digest or set(saved['images']) != set(control):
                        raise ValueError(f'Unexpected paired endpoint: {seed}, {label}, {arm}')
                    verify_prefixes(saved['images'], originals, arm)
                    matched[arm] = saved['images']
                    rows[seed][label][arm] = saved['counts']
                    control = saved['images']
                budget_fields = ('active', 'teacher_active', 'teacher_global', 'teacher_within', 'queries', 'quota')
                if any(matched['global'][im][k] != matched['within_level'][im][k]
                       for im in matched['global'] for k in budget_fields):
                    raise ValueError(f'Unpaired view budgets: {seed}, {label}')
        finally:
            hook.remove()
        big, small = rows[seed]['holdout1024'], rows[seed]['reference128']
        g, w = big['global'], big['within_level']
        hard = old_report['rows'][seed]['holdout1024']['global']['hard_hits']
        full_quotas = all(row['proposals'] == row['quota'] for sliced in rows[seed].values() for row in sliced.values())
        passed[seed] = (full_quotas and w['raw_final'] > g['raw_final'] and w['soft_hits'] > g['soft_hits']
                        and 100 * (w['soft_hits'] - hard) >= 5 * g['gt']
                        and 100 * (g['known_final'] - w['known_final']) <= g['known_gt']
                        and small['within_level']['soft_hits'] >= small['global']['soft_hits'])
        del model, criterion, ema, student, teacher
    write_json(OUTPUT / 'comparison.json', dict(status='complete', plan_sha256=digest, rows=rows,
                                               passed_by_seed=passed, passes_screen=all(passed.values())))
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
