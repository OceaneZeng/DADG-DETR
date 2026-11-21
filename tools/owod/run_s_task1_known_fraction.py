"""Transfer the locked M single-view rule to an independently trained S Task 1 parent."""
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys

import torch

from tools.owod import prepare_benchmarks, run_instance_pilot as pilot
from tools.owod import probe_task1_known_fraction as fraction
from tools.owod import probe_task1_known_queries as independent
from tools.owod.probe_task1_encoder_transfer import identities
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.run_cross_attention_study import pin
from tools.owod.run_stronger_parent_study import PARENT_EPOCHS, PARENT_IMAGES, run_job
from tools.owod.screening_data import prepare_screen, screening_files, source_ids, write_json


OUTPUT = pilot.ROOT / 'output/s-instance-task1-known-fraction'
MANIFEST = pilot.ROOT / 'data/owod/s-owodb/split_manifest.json'
M_PLAN_SHA256 = '12f7dc8cf3ccc4b29c2335f734602fe656f70307f6f1a6adc474d5c285725aad'


def proxy_data(manifest, origins):
    root = resolve_image_root(MANIFEST, manifest)
    images, boxes, hashes = prepare_benchmarks.load_coco(root)
    actual = {Path(p).name: digest for p, digest in hashes.items()}
    expected = {Path(p).name: digest for p, digest in manifest['source_hashes'].items()}
    if actual != expected:
        raise ValueError('COCO proxy source differs from S manifest')
    selected = {name: (images[name], boxes[name]) for name in origins}
    return prepare_benchmarks.subset(origins, manifest['stages'][1]['active_classes'], selected)


def main():
    source = fraction.OUTPUT / 'plan.json'
    if file_sha256(source) != M_PLAN_SHA256:
        raise ValueError('Expected the locked M calibration and selection rule')
    calibration = read_json(source)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if devices != [calibration['gpu']] * 2 or str(torch.__version__) != calibration['torch_version']:
        raise ValueError('Expected the paired two-3090 training environment')
    screen = OUTPUT / 'data/screening.json'
    train = pilot.command('off', 'task1', output=OUTPUT, screen=screen, epochs=PARENT_EPOCHS,
                          seed=42, manifest=MANIFEST)
    plan = dict(protocol='s-owodb', seed=42, task1=train, train_images=PARENT_IMAGES, epochs=PARENT_EPOCHS,
                manifest_sha256=file_sha256(MANIFEST), backbone_sha256=file_sha256(pilot.WEIGHTS),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
                devices=os.environ['CUDA_VISIBLE_DEVICES'], gpus=devices, torch_version=str(torch.__version__),
                m_plan_sha256=M_PLAN_SHA256, selection_seed=49, per_source=512,
                paper_comparable=False, **{k: calibration[k] for k in ('rule', 'criterion', 'numerator', 'denominator')})
    pin(OUTPUT / 'plan.json', plan)
    pilot.run([sys.executable, str(pilot.ROOT / 'tools/owod/manage_data.py'), 'verify',
               '--benchmark', 's-owodb', '--output-root', str(MANIFEST.parent.parent)])
    prepare_screen(MANIFEST, screen.parent, PARENT_IMAGES, 128, 42, task2_images=pilot.TASK2_IMAGES)
    manifest, files = screening_files(MANIFEST, screen, 0)
    blocked = set(manifest['evaluation']['image_source_ids']) | set(read_json(screen)['heldout_source_ids'])
    blocked.update(source_ids(read_json(files['increment_train'])))
    origins = independent.fresh_origins(manifest, blocked, plan['selection_seed'])
    selection = dict(excluded_sha256=identities(blocked), origins=origins, screen_sha256=file_sha256(screen))
    pin(OUTPUT / 'selection/inputs.json', selection)

    checkpoint, args = run_job(train, selection['screen_sha256'])
    if args.owod_protocol != 's-owodb' or args.discovery_version != 10 or args.paper_comparable_protocol:
        raise ValueError('Expected the diagnostic S v10 endpoint')
    evaluation = OUTPUT / 'evaluation'
    data = proxy_data(manifest, origins)
    sources = {im['id']: origins[im['owod_source_id']] for im in data['images']}
    encoded = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
    job = dict(parent=str(checkpoint), parent_sha256=file_sha256(checkpoint))
    inputs = dict(jobs={'42': job}, batch_size=2, plan_sha256=file_sha256(OUTPUT / 'plan.json'),
                  fresh_sha256=sha256(encoded).hexdigest(),
                  selection_sha256=file_sha256(OUTPUT / 'selection/inputs.json'))
    pin(evaluation / 'plan.json', inputs)
    path = evaluation / 'fresh1024.json'
    if not path.exists():
        write_json(path, data)
    if file_sha256(path) != inputs['fresh_sha256']:
        raise ValueError('Changed S evaluation data')
    choose = partial(fraction.fraction_picks, numerator=plan['numerator'], denominator=plan['denominator'])
    rows = independent.run_endpoints(evaluation, inputs, manifest, sources, extra_selection=choose,
                                     manifest_path=MANIFEST)
    report = fraction.comparison(rows, inputs['plan_sha256'])
    report.update(evaluation_plan_sha256=file_sha256(evaluation / 'plan.json'), paper_comparable=False,
                  parent_sha256=job['parent_sha256'],
                  native_author_endpoint=pilot.metrics(checkpoint.parent, PARENT_EPOCHS - 1))
    write_json(OUTPUT / 'comparison.json', report)
    print('Comparison saved:', OUTPUT / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
