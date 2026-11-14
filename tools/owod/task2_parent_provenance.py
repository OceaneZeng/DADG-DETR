"""Exact manifest equivalence for isolated diagnostics, never formal resume."""
import copy
import hashlib
import json
import posixpath
from pathlib import PurePosixPath

from tools.owod.protocol import file_sha256, read_json, stage_files


def legacy_manifest_bytes(manifest, saved):
    """Reproduce manage_data's pre-canonical schema-3 serialization exactly.

    Its stage paths were already relative after atomic publication. Only the
    annotation/image/raw-source roots became portable in the canonical file;
    the two embedded-split fields were added at its end. An exact SHA match,
    rather than a list of ignored fields, is required by the caller.
    """
    old_path = PurePosixPath(saved.owod_manifest)
    if not old_path.is_absolute() or not manifest.get('splits') or not manifest.get('split_hashes'):
        raise ValueError('Expected a canonical manifest and an absolute original Linux manifest path')
    base = str(old_path.parent)
    result = copy.deepcopy(manifest)
    result.pop('splits')
    result.pop('split_hashes')
    result['annotation_root'] = posixpath.normpath(posixpath.join(base, result['annotation_root']))
    result['image_root'] = posixpath.normpath(posixpath.join(base, result['image_root']))
    if result['image_root'] != saved.owod_image_root:
        raise ValueError('Reconstructed original image root differs from the Task 1 checkpoint')
    result['source_hashes'] = {
        posixpath.normpath(posixpath.join(base, key)): value
        for key, value in result['source_hashes'].items()
    }
    return (json.dumps(result, indent=2) + '\n').encode('utf-8')


def manifest_equivalence(manifest_path, saved):
    manifest = read_json(manifest_path)
    current = file_sha256(manifest_path)
    original = getattr(saved, 'owod_manifest_sha256', None)
    if not original:
        raise ValueError('Parent has no manifest identity')
    if current == original:
        proof = 'identical_manifest'
    else:
        reconstructed = hashlib.sha256(legacy_manifest_bytes(manifest, saved)).hexdigest()
        if reconstructed != original:
            raise ValueError('Cannot reproduce the original manifest SHA-256 exactly; stopping')
        proof = 'exact_precanonical_serialization_sha256'
    if (manifest['protocol'] != 's-owodb' or saved.owod_protocol != 's-owodb'
            or manifest['evaluation'] != saved.owod_evaluation
            or manifest['stages'][0]['active_classes'] != saved.owod_known_class_ids):
        raise ValueError('Parent protocol, evaluation or class order changed')
    return dict(diagnostic_only=True, proof=proof, canonical_manifest_sha256=current,
                original_manifest_sha256=original,
                stage_annotation_sha256=[record['sha256'] for record in manifest['stages']])


def verify_parent_inputs(manifest_path, checkpoint_path, gnn_path):
    """CPU preflight; no saved Namespace, weight, manifest or GNN is edited."""
    import torch
    parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved = parent['args']
    if (parent.get('diagnostic_only') or getattr(saved, 'owod_screening', '')
            or getattr(saved, 'owod_screening_sha256', None)
            or saved.owod_stage != 0 or saved.owod_phase != 'increment'
            or saved.epochs != 50 or parent.get('epoch') != 49
            or getattr(saved, 'discovery_version', None) != 5
            or not saved.owod_discovery or not saved.graph_routed_adapter
            or getattr(saved, 'graph_adapter_specs', [])
            or saved.graph_routing != 'hard' or saved.graph_localization != 'frozen'
            or saved.discovery_scoring != 'hierarchical' or saved.seed != 42):
        raise ValueError('Requires the completed, full-data, non-diagnostic v5 Task 1 epoch 50')
    proof = manifest_equivalence(manifest_path, saved)
    # Validate all 16 phase files against the canonical author lists and hashes.
    for stage in range(4):
        stage_files(manifest_path, stage)
    checkpoint_hash = file_sha256(checkpoint_path)
    gnn = torch.load(gnn_path, map_location='cpu', weights_only=False)
    extra = gnn.get('extra', {})
    expected = dict(manifest_sha256=proof['original_manifest_sha256'],
                    checkpoint_sha256=checkpoint_hash, stage=0,
                    annotation_sha256=proof['stage_annotation_sha256'][0]['train'],
                    production_ready=True, uses_validation_labels=False,
                    supervision='empirical_train_loss_increase',
                    last_decoder_layers=saved.graph_adapter_last_n)
    if any(extra.get(key) != value for key, value in expected.items()):
        raise ValueError('Task 1 GNN provenance does not match the unchanged parent/data')
    if (set(extra.get('training_classes', [])) != set(saved.owod_known_class_ids)
            or extra.get('screening_sha256') is not None):
        raise ValueError('GNN must use the full-data Task 1 known classes only')
    return dict(**proof, checkpoint_sha256=checkpoint_hash, gnn_sha256=file_sha256(gnn_path))


def legacy_validation_arguments(args, parent, manifest_path, proof):
    """Validate the intact parent in its original identity before a new fork.

    This copy is used only for strict parent/GNN validation and fresh Task 2
    initialization. Training uses the canonical hash; formal guards are intact.
    """
    from tools.owod.strict_setting import validate_checkpoint
    repeated = manifest_equivalence(manifest_path, parent['args'])
    if any(proof.get(key) != value for key, value in repeated.items()):
        raise ValueError('Diagnostic manifest equivalence proof changed')
    result = copy.deepcopy(args)
    result.owod_manifest_sha256 = repeated['original_manifest_sha256']
    validate_checkpoint(result, parent)
    return result


def verify_reference(checkpoint_path, plan_path, proof):
    """An old v7 diagnostic is read for measurement, never for initialization."""
    import torch
    plan = read_json(plan_path)
    for key in ('checkpoint_sha256', 'gnn_sha256'):
        if plan.get(key) != proof[key]:
            raise ValueError('Reference diagnostic belongs to another Task 1/GNN')
    if plan.get('manifest_sha256') != proof['original_manifest_sha256']:
        raise ValueError('Reference diagnostic has another original manifest')
    reference = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved = reference['args']
    if (reference.get('diagnostic_only') is not True or reference.get('policy') != 7
            or reference.get('epoch') != 2 or reference.get('diagnostic_identity') != plan
            or saved.owod_stage != 1 or saved.owod_phase != 'increment'
            or saved.discovery_version != 7 or saved.graph_routing != 'hard'
            or saved.graph_localization != 'frozen' or saved.discovery_scoring != 'hierarchical'
            or saved.owod_manifest_sha256 != proof['original_manifest_sha256']
            or saved.world_size != 2 or saved.batch_size != 1):
        raise ValueError('Expected the completed two-GPU v7 hard/frozen three-epoch reference')
    # Canonical class/evaluation validation is repeated by the caller's probe.
    del reference
    return dict(checkpoint_sha256=file_sha256(checkpoint_path), plan_sha256=file_sha256(plan_path),
                role='read_only_reference_not_training_parent')
