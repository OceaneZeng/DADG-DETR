"""Shared preflight: official task phases, labels, data hashes, and checkpoint lineage."""
from pathlib import Path
import math

from tools.owod.protocol import file_sha256, stage_files, RETIRED_MESSAGE


def configure_run(args):
    ramp_epochs = getattr(args, 'discovery_pseudo_ramp_epochs', 0)
    if ramp_epochs < 0 or (ramp_epochs and not getattr(args, 'owod_discovery', False)):
        raise ValueError('discovery_pseudo_ramp_epochs must be nonnegative and requires discovery')
    if getattr(args, 'screen_no_distill', False) and not (
            getattr(args, 'owod_screening', '') and getattr(args, 'screen_branch_checkpoint', '')
            and args.owod_stage == 1 and getattr(args, 'owod_discovery', False)):
        raise ValueError('screen-no-distill requires shared-parent discovery screening Task 2')
    if getattr(args, 'owod_discovery', False):
        from models.graph_local.discovery import DISCOVERY_VERSION, DISCOVERY_RAMP_VERSION
        args.discovery_pseudo_ramp_epochs = ramp_epochs
        args.discovery_version = DISCOVERY_RAMP_VERSION if ramp_epochs else DISCOVERY_VERSION
        args.graph_routed_adapter = True
    requested = bool(args.owod_manifest or args.owod_known_class_ids is not None or args.paper_baseline
                     or getattr(args, 'graph_routed_adapter', False))
    if not requested:
        return
    if not args.owod_manifest or args.owod_stage is None:
        raise ValueError('OWOD experiments require an author-release manifest and stage')
    manifest, files = stage_files(Path(args.owod_manifest), args.owod_stage)
    recipe = getattr(args, 'owod_recipe', None)
    if recipe not in ('ore-paper', 'ow-detr-paper'):
        raise ValueError('OWOD runs must explicitly select --owod-recipe ore-paper or ow-detr-paper')
    if recipe == 'ore-paper' and manifest['protocol'] != 'm-owodb':
        raise ValueError('ORE 2021 does not define an S-OWODB recipe')
    phase = args.owod_phase
    if phase == 'finetune' and args.owod_stage == 0:
        raise ValueError('Task 1 has no incremental balanced-finetuning phase')
    expected_train = files['increment_train' if phase == 'increment' else 'train']
    for field, expected in [('train_ann', expected_train), ('val_ann', files['full_val'])]:
        supplied = getattr(args, field, '')
        if supplied and file_sha256(Path(supplied)) != file_sha256(expected):
            raise ValueError(f'{field} does not match the official {phase} phase; {RETIRED_MESSAGE}')
        setattr(args, field, str(expected))
    record = manifest['stages'][args.owod_stage]
    previous = [c for c in manifest['category_order_source_ids'] if
                c in set(record['active_classes']) - set(record['classes'])]
    for field, values in [('owod_known_class_ids', record['active_classes']),
                          ('owod_current_class_ids', record['classes']),
                          ('owod_previous_class_ids', previous)]:
        supplied = getattr(args, field)
        if supplied is not None and set(supplied) != set(values):
            raise ValueError(f'{field} differs from the author-defined stage')
        setattr(args, field, values)
    image_root = manifest.get('image_root')
    if not image_root:
        raise ValueError('Author manifest must record image_root; re-register with --image-root')
    args.owod_image_root = str(Path(image_root).resolve())
    if not Path(args.owod_image_root).is_dir():
        raise FileNotFoundError(f'Benchmark images missing: {args.owod_image_root}')
    if args.lightweight or args.no_augmentation or args.replay_sampling_fraction:
        raise ValueError('Old lightweight/overfit/custom replay-exposure settings are retired for OWOD')
    if args.owod_stage > 0 and not (args.pretrained or args.resume):
        raise ValueError('Incremental tasks must continue the preceding task checkpoint')
    if args.owod_stage == 0 and args.pretrained:
        raise ValueError('Task 1 cannot load a COCO detector checkpoint; use backbone-only pretraining')
    if manifest['protocol'] == 's-owodb' and not args.backbone_weights:
        raise ValueError('S-OWODB original OW-DETR setting requires --backbone-weights with DINO ResNet-50 weights')
    args.owod_manifest_sha256 = file_sha256(Path(args.owod_manifest))
    args.backbone_weights_sha256 = file_sha256(Path(args.backbone_weights)) if args.backbone_weights else None
    args.owod_evaluation = manifest['evaluation']
    args.owod_protocol = manifest['protocol']
    args.owod_max_detections = 50 if (manifest['protocol'] == 's-owodb'
                                   or getattr(args, 'owod_recipe', None) == 'ow-detr-paper') else 100
    args.paper_comparable_protocol = True
    if not getattr(args, 'graph_routed_adapter', False) and not getattr(args, 'paper_baseline', ''):
        args.control_postprocess_version = 'known-sigmoid-topk-v2'
    if getattr(args, 'screen_branch_checkpoint', ''):
        if not getattr(args, 'owod_screening', '') or args.owod_stage != 1:
            raise ValueError('Shared-parent branching is restricted to screening Task 2')
        args.screen_branch_checkpoint_sha256 = file_sha256(Path(args.screen_branch_checkpoint))
        args.screen_branch_gnn_sha256 = file_sha256(Path(args.graph_gnn_checkpoint))
        if args.owod_phase == 'increment' and (
                not args.pretrained or file_sha256(Path(args.pretrained)) != args.screen_branch_checkpoint_sha256):
            raise ValueError('Task 2 must initialize from the pinned common parent')
    if getattr(args, 'owod_screening', ''):
        from tools.owod.screening_data import screening_files
        from tools.owod.protocol import read_json
        _, screen_files = screening_files(Path(args.owod_manifest), Path(args.owod_screening), args.owod_stage)
        screen = read_json(Path(args.owod_screening))
        if args.seed != screen['seed'] or not getattr(args, 'owod_discovery', False):
            raise ValueError('Screen requires discovery method and the registered paired seed')
        args.owod_screening_sha256 = file_sha256(Path(args.owod_screening))
        args.train_ann = str(screen_files['increment_train' if phase == 'increment' else 'train'])
        args.val_ann = str(screen_files['full_val'])
        args.paper_comparable_protocol = False
        args.owod_evaluation = {**manifest['evaluation'], 'known_only_screening': True,
                               'image_source_ids': screen['heldout_source_ids']}
    if getattr(args, 'owod_discovery', False):
        if (args.discovery_capacity < 1 or args.discovery_top_k < 1 or args.discovery_warmup < 0
                or not math.isfinite(args.discovery_min_support) or args.discovery_min_support <= 0
                or any(not math.isfinite(value) or not 0 < value <= 1 for value in
                       (args.discovery_transfer_threshold, args.discovery_saliency_threshold,
                        args.discovery_consistency_threshold))
                or args.teacher_completion):
            raise ValueError('Invalid discovery configuration or unsupported external teacher combination')
        if args.owod_stage > 0 and not args.discovery_no_keep:
            if not args.pretrained:
                raise ValueError('Discovery preservation requires its preceding --pretrained checkpoint, also on resume')
            args.old_class_distillation = True
            args.distill_old_class_ids = list(previous)
        if getattr(args, 'screen_no_distill', False):
            args.old_class_distillation = False
            args.distill_old_class_ids = []
    if getattr(args, 'graph_routed_adapter', False):
        if (args.with_tree or args.paper_baseline or args.neighbor_scoped_lora
                or args.reset_classifier or args.frozen_weights or args.masks
                or args.two_stage or args.off_projection_coef):
            raise ValueError('Graph-routed adapters require the plain one-stage detector without legacy LoRA')
        if (args.graph_adapter_rank < 1 or not 1 <= args.graph_adapter_last_n <= args.dec_layers
                or args.graph_sketch_max_images < 1
                or not math.isfinite(args.graph_risk_beta) or args.graph_risk_beta < 0
                or not math.isfinite(args.graph_route_loss_coef) or args.graph_route_loss_coef <= 0):
            raise ValueError('Invalid graph adapter method hyperparameters')
        if args.owod_stage > 0 and not args.resume and args.owod_phase == 'increment':
            if not Path(args.graph_gnn_checkpoint).is_file():
                raise ValueError('New task requires a calibrated preceding-stage --graph-gnn-checkpoint')


def validate_checkpoint(args, checkpoint, resume=False):
    if not getattr(args, 'owod_manifest', ''):
        return
    saved = checkpoint.get('args')
    shared_parent = bool(getattr(args, 'screen_branch_checkpoint_sha256', None))
    branching = (shared_parent and not resume and args.owod_stage == 1
                 and args.owod_phase == 'increment' and getattr(saved, 'owod_stage', None) == 0
                 and getattr(saved, 'owod_phase', None) == 'increment'
                 and not getattr(saved, 'graph_adapter_specs', []))
    if shared_parent and (not getattr(args, 'owod_screening_sha256', None) or args.owod_stage != 1
                          or not getattr(args, 'owod_discovery', False)
                          or not getattr(args, 'graph_routed_adapter', False)):
        raise ValueError('Shared-parent branching is restricted to screening Task 2')
    if not branching:
        if bool(getattr(saved, 'screen_no_distill', False)) != bool(getattr(args, 'screen_no_distill', False)):
            raise ValueError('Checkpoint cannot change screen_no_distill within Task 2')
        for field in ('screen_branch_checkpoint_sha256', 'screen_branch_gnn_sha256'):
            if getattr(saved, field, None) != getattr(args, field, None):
                raise ValueError(f'Shared-parent checkpoint chain mismatch: {field}')
    elif (not args.pretrained or file_sha256(Path(args.pretrained)) != args.screen_branch_checkpoint_sha256
          or getattr(saved, 'seed', None) != args.seed):
        raise ValueError('Shared Task 1 checkpoint or seed mismatch')
    if getattr(saved, 'owod_screening_sha256', None) != getattr(args, 'owod_screening_sha256', None):
        raise ValueError('Cannot mix screening and full benchmark checkpoints or different screening data')
    if (saved is None or getattr(saved, 'owod_manifest_sha256', None) != args.owod_manifest_sha256):
        raise ValueError('Checkpoint belongs to an old/custom/different benchmark manifest')
    if getattr(saved, 'owod_recipe', None) != args.owod_recipe:
        raise ValueError('Training recipe changed within the checkpoint chain')
    if bool(getattr(saved, 'activation_checkpointing', False)) != bool(getattr(args, 'activation_checkpointing', False)):
        raise ValueError('Activation checkpointing changed within the checkpoint chain')
    if bool(getattr(saved, 'owod_discovery', False)) != bool(getattr(args, 'owod_discovery', False)):
        raise ValueError('Cannot switch discovery architecture within the checkpoint chain')
    if getattr(args, 'owod_discovery', False):
        if getattr(saved, 'discovery_scoring', 'hierarchical') != getattr(args, 'discovery_scoring', 'hierarchical'):
            raise ValueError('Discovery checkpoint cannot change discovery_scoring')
        if getattr(saved, 'discovery_pseudo_ramp_epochs', 0) != getattr(args, 'discovery_pseudo_ramp_epochs', 0):
            raise ValueError('Discovery checkpoint cannot change discovery_pseudo_ramp_epochs')
        for field in ('discovery_version', 'discovery_no_message',
                      'discovery_no_transfer', 'discovery_no_keep', 'discovery_capacity', 'discovery_min_support',
                      'discovery_transfer_threshold', 'discovery_warmup', 'discovery_top_k',
                      'discovery_saliency_threshold', 'discovery_consistency_threshold',
                      'distill_class_coef', 'distill_bbox_coef', 'distill_score_threshold', 'distill_max_queries'):
            if getattr(saved, field, None) != getattr(args, field, None):
                raise ValueError(f'Discovery checkpoint cannot change {field}')
    if bool(getattr(saved, 'graph_routed_adapter', False)) != bool(getattr(args, 'graph_routed_adapter', False)):
        raise ValueError('Cannot switch graph adapter method within the checkpoint chain')
    if getattr(args, 'graph_routed_adapter', False):
        if not branching and getattr(saved, 'graph_routing', 'hard') != getattr(args, 'graph_routing', 'hard'):
            raise ValueError('Graph adapter checkpoint cannot change graph_routing')
        if not branching and getattr(saved, 'graph_localization', 'frozen') != getattr(args, 'graph_localization', 'frozen'):
            raise ValueError('Graph adapter checkpoint cannot change graph_localization')
        for field in ('graph_adapter_rank', 'graph_adapter_last_n', 'graph_risk_beta',
                      'graph_route_loss_coef', 'graph_sketch_max_images'):
            if getattr(saved, field, None) != getattr(args, field, None):
                raise ValueError(f'Graph adapter checkpoint cannot change {field}')
        specs = getattr(saved, 'graph_adapter_specs', [])
        if len(specs) != saved.owod_stage:
            raise ValueError('Graph checkpoint has an incomplete persistent adapter bank')
        manifest, _ = stage_files(Path(args.owod_manifest), saved.owod_stage)
        for index, spec in enumerate(specs, start=1):
            record = manifest['stages'][index]
            old = set(record['active_classes']) - set(record['classes'])
            if (spec['stage'] != index or set(spec['class_ids']) != set(record['classes'])
                    or set(spec['old_ids']) != old
                    or spec['rank'] != args.graph_adapter_rank
                    or spec['last_n'] != args.graph_adapter_last_n
                    or spec['risk_beta'] != args.graph_risk_beta
                    or spec.get('routing', 'hard') != getattr(args, 'graph_routing', 'hard')):
                raise ValueError('Graph checkpoint adapter layout differs from the official task chain')
    stage = args.owod_stage if resume or args.owod_phase == 'finetune' else args.owod_stage - 1
    phase = args.owod_phase if resume else ('increment' if args.owod_phase == 'finetune'
                                          or stage == 0 else 'finetune')
    if getattr(saved, 'owod_stage', None) != stage or getattr(saved, 'owod_phase', None) != phase:
        raise ValueError(f'Expected checkpoint from stage_{stage}/{phase}')
    if not resume and checkpoint.get('epoch') != getattr(saved, 'epochs', 0) - 1:
        raise ValueError('Stage transition requires the completed preceding phase checkpoint')
    if getattr(saved, 'backbone_weights_sha256', None) != getattr(args, 'backbone_weights_sha256', None):
        raise ValueError('Backbone pretraining changed within the checkpoint chain')
    if resume:
        for field in ('epochs', 'lr', 'lr_backbone', 'lr_drop', 'batch_size', 'weight_decay',
                      'seed', 'sgd', 'backbone', 'num_classes', 'num_queries', 'world_size'):
            if getattr(saved, field, None) != getattr(args, field, None):
                raise ValueError(f'Strict benchmark resume cannot change {field}')
