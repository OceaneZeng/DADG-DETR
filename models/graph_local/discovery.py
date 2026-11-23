"""Open-world discovery, persistent anonymous memory, and supervised promotion."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from models.deformable_detr import build as build_base, PostProcess
from models.graph_local.lora import target_decoder_layers
from .routed_adapter import ClassAdapter
from .discovery_memory import PrototypeMemory, AffinityGraph, initialize_from_deltas
from .discovery_candidates import mine_candidates
from util import box_ops
from models.segmentation import sigmoid_focal_loss
from util.misc import get_world_size, is_dist_avail_and_initialized
from util.training_runtime import gather_evidence, foreach_ema_parameters, copy_model_buffers
from util.runtime_profile import scope

DISCOVERY_VERSION = 5
DISCOVERY_RAMP_VERSION = 6
DISCOVERY_LOSS_VERSION = 7
DISCOVERY_STABILITY_VERSION = 8
DISCOVERY_BACKGROUND_VERSION = 9


def scale_gradient(value, scale):
    """Keep the forward value; scale only this branch's input gradient."""
    return value if scale == 1. else value.detach() + scale * (value - value.detach())


def known_unknown_log_ratio(logits, novelty, labels, known_ids, scoring='hierarchical'):
    """log(S_unknown / S_true_known) on released, matched GT slots.

    Objectness cancels for the same slot. Work in log space so uncertain
    classifiers and saturated novelty still receive finite, useful gradients.
    This does not rank genuine unknown objects below known objects elsewhere.
    """
    if scoring == 'hierarchical':
        columns = (labels[:, None] == labels.new_tensor(known_ids)).long().argmax(-1)
        log_known = logits[..., known_ids].log_softmax(-1).gather(1, columns[:, None]).squeeze(1)
        return novelty - log_known
    if scoring == 'focal':
        log_known = F.logsigmoid(logits.gather(1, labels[:, None]).squeeze(1))
        return F.logsigmoid(novelty) - log_known
    raise ValueError('Unknown discovery scoring rule')


def matched_known_ranking_loss(log_ratio):
    """Pairwise logistic loss, normalized by global matched-GT count under DDP."""
    count = log_ratio.new_tensor(log_ratio.numel(), dtype=torch.float32)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(count)
        # Clamp the GLOBAL count before dividing by world size. A single GT
        # across two ranks must have the same gradient as a one-process batch.
        count = count.clamp_min(1) / get_world_size()
    else:
        count = count.clamp_min(1)
    return F.softplus(log_ratio).sum() / count


class DiscoveryFFN(nn.Module):
    def __init__(self, dim, rank, capacity):
        super().__init__()
        self.adapter = ClassAdapter(dim, rank)
        self.router = nn.Linear(dim, 3)  # background, known, anonymous object
        self.register_buffer('snapshot_down', torch.zeros(capacity, rank, dim))
        self.register_buffer('snapshot_up', torch.zeros(capacity, dim, rank))
        self.last_logits = None
        self.last_inputs = None

    def forward(self, features):
        logits = self.router(features.detach())
        self.last_logits = logits
        self.last_inputs = features.detach()
        return self.adapter(features) * logits.softmax(-1)[..., 2:3]

    @torch.no_grad()
    def snapshot(self, slots):
        for slot in slots:
            self.snapshot_down[slot].copy_(self.adapter.down.weight)
            self.snapshot_up[slot].copy_(self.adapter.up.weight)


class DiscoveryDetector(nn.Module):
    def __init__(self, detector, args):
        super().__init__()
        self.detr = detector
        self.known_ids = list(args.owod_known_class_ids)
        self.transfer_enabled = not args.discovery_no_transfer
        self.transfer_threshold = args.discovery_transfer_threshold
        self.min_support = args.discovery_min_support
        self.feature_grad_scale = getattr(args, 'discovery_feature_grad_scale', 1.)
        if not math.isfinite(self.feature_grad_scale) or not 0 <= self.feature_grad_scale <= 1:
            raise ValueError('discovery_feature_grad_scale must be finite and in [0, 1]')
        self.instance_mode = getattr(args, 'instance_mode', 'legacy')
        if self.instance_mode != 'legacy':
            from .instance_proposals import InstanceProposals
            from .instance_transfer import SETTINGS
            if detector.two_stage:
                raise ValueError('Mixed instance proposals require one-stage decoder queries')
            # Isolate initialization so common detector and discovery parameters start identically.
            with torch.random.fork_rng(devices=[]):
                detector.transformer.instance_proposals = InstanceProposals(args.hidden_dim)
                self.instance_projection = nn.Linear(args.hidden_dim, SETTINGS['projection_dim'])
                self.instance_classes = nn.Linear(SETTINGS['projection_dim'], args.num_classes)
            self.transfer_enabled = False
        detector.return_baseline_features = True
        self.memory = PrototypeMemory(args.hidden_dim, args.discovery_capacity)
        self.affinity_graph = AffinityGraph(
            args.hidden_dim, use_messages=not getattr(args, 'discovery_no_message', False))
        for layer in target_decoder_layers(self, args.graph_adapter_last_n):
            layer.discovery_adapter = DiscoveryFFN(args.hidden_dim, args.graph_adapter_rank,
                                                   args.discovery_capacity if self.instance_mode == 'legacy' else 0)

    def forward(self, samples, *, teacher_purpose=None):
        from util.misc import NestedTensor, nested_tensor_from_tensor_list
        if teacher_purpose is not None:
            if self.training or teacher_purpose not in ('distillation', 'discovery'):
                raise ValueError('Teacher forward requires eval mode and a supported purpose')
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        outputs = (self.detr(samples, final_only=True) if teacher_purpose is not None
                   else self.detr(samples))
        if teacher_purpose == 'distillation':
            self._discovery_routes()  # Release transient per-layer references.
            return {key: outputs[key] for key in ('pred_logits', 'pred_boxes')}
        features = outputs.pop('decoder_features')[-1]
        outputs['query_features'] = features
        if self.instance_mode != 'legacy':
            z = F.normalize(self.instance_projection(features), dim=-1)
            outputs['instance_features'] = z
            outputs['instance_logits'] = self.instance_classes(z)
        outputs.update(self.affinity_graph(scale_gradient(features, self.feature_grad_scale), self.memory))
        routes, route_inputs = self._discovery_routes()
        if teacher_purpose is None:
            outputs['discovery_routes'] = routes
            outputs['discovery_inputs'] = route_inputs
        masks = samples.mask
        outputs['image_sizes'] = [(int((~mask).any(1).sum()), int((~mask).any(0).sum())) for mask in masks]
        if teacher_purpose == 'discovery':
            # Keep the graph embedding: cross-view consistency consumes it.
            keys = ['pred_boxes', 'query_features', 'attention_feature', 'padded_size',
                    'image_sizes', 'discovery_embedding']
            if self.instance_mode != 'legacy':
                keys.append('instance_features')
            return {key: outputs[key] for key in keys}
        return outputs

    def _discovery_routes(self):
        routes = []
        route_inputs = []
        for module in self.modules():
            if isinstance(module, DiscoveryFFN):
                routes.append(module.last_logits)
                route_inputs.append(module.last_inputs)
                module.last_logits = None
                module.last_inputs = None
        return routes, route_inputs

    @torch.no_grad()
    def forward_teacher(self, samples, *, purpose='discovery'):
        return self(samples, teacher_purpose=purpose)

    def train_discovery(self):
        self.affinity_graph.requires_grad_(True)
        for module in self.modules():
            if isinstance(module, DiscoveryFFN):
                module.requires_grad_(True)

    @torch.no_grad()
    def observe_training(self, evidence):
        if evidence is None:
            return
        packed = gather_evidence(evidence)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed or torch.distributed.get_rank() == 0:
            features, labels, weights = [tensor.to(self.memory.centers.device) for tensor in packed]
            if any(label >= 0 and label not in self.known_ids for label in labels.cpu().tolist()):
                raise ValueError('Prototype update received unreleased semantic labels')
            slots = self.memory.observe(features, labels, weights)
            for module in self.modules():
                if isinstance(module, DiscoveryFFN) and self.instance_mode == 'legacy':
                    module.snapshot(slots)
        if distributed:
            # All model buffers (including anonymous snapshots) remain identical.
            for buffer in self.memory.buffers():
                torch.distributed.broadcast(buffer, 0)
            for module in self.modules():
                if isinstance(module, DiscoveryFFN) and self.instance_mode == 'legacy':
                    torch.distributed.broadcast(module.snapshot_down, 0)
                    torch.distributed.broadcast(module.snapshot_up, 0)

    @torch.no_grad()
    def promote(self, class_means, spec, route_means=None):
        """Called once after adding this task's adapters, using current train GT."""
        reports = []
        valid = self.memory.counts[self.memory.known_slots:] >= self.min_support
        for class_id, feature in class_means.items():
            if route_means is not None:
                for layer_index, layer in enumerate(target_decoder_layers(self, spec['last_n'])):
                    centroid = route_means[class_id][layer_index]
                    branch = layer.graph_adapter.stages[-1]
                    row = spec['class_ids'].index(class_id)
                    branch.router.weight[row].copy_(4 * centroid / centroid.square().sum().clamp_min(1e-6))
                    branch.router.bias[row] = -2.
            if self.instance_mode != 'legacy':
                continue  # The class router is initialized above; v10 learns from instance targets.
            similarity = self.affinity_graph.similarities(
                feature[None], self.memory.centers[self.memory.known_slots:])[0]
            selected = torch.where(valid & (similarity >= self.transfer_threshold))[0]
            weights = (similarity[selected] / .1).softmax(0)
            inherited = self.transfer_enabled and bool(selected.numel())
            if inherited:
                for layer in target_decoder_layers(self, spec['last_n']):
                    source = layer.discovery_adapter
                    target = layer.graph_adapter.stages[-1].adapters[spec['class_ids'].index(class_id)]
                    deltas = [source.snapshot_up[i] @ source.snapshot_down[i] for i in selected]
                    initialize_from_deltas(target, deltas, weights)
            reports.append({'class_id': class_id, 'anonymous_slots': selected.tolist(),
                            'weights': weights.tolist(), 'inherited': inherited})
        # Keep mixed prototypes: semantic identity is not assigned to entire clusters.
        spec['promotion'] = reports
        if route_means is not None:
            spec['router_initialization'] = 'current_train_GT_pre_FFN_centroids_scale_4'
        return reports


def pseudo_supervision_scale(epoch, warmup, ramp_epochs):
    """Phase-local schedule, reconstructed from the saved epoch on resume."""
    if epoch < warmup:
        return 0.
    return min(1., (epoch - warmup + 1) / ramp_epochs) if ramp_epochs else 1.


def balanced_bce(logits, targets, weights, contribution_scale=None, normalization='weight-sum'):
    if normalization not in ('weight-sum', 'count'):
        raise ValueError('Unknown discovery BCE normalization')
    values = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if contribution_scale is not None:
        # Scale contributions AFTER fixing normalization. Scaling the weights
        # in both numerator and denominator would cancel for a pseudo group.
        values = values * contribution_scale
    parts = []
    for mask in (targets > .5, targets <= .5):
        effective = weights * mask
        if effective.sum() > 0:
            # Count only supervised entries. Confidence stays in the numerator
            # in v7, so low-confidence pseudo evidence reduces the whole group.
            denominator = (effective.sum() if normalization == 'weight-sum'
                           else ((weights > 0) & mask).sum())
            parts.append((values * effective).sum() / denominator)
    return torch.stack(parts).mean() if parts else logits.sum() * 0


def proposal_focal_loss(logits, foreground, weights, contribution_scale=None, alpha=.25):
    """Dense selected-slot focal loss, normalized by DDP mean positive count.

    Positive counts, rather than confidence sums or separate positive/negative
    group means, fix the normalization. Pseudo confidence and the phase ramp
    therefore remain actual contribution multipliers, including on empty ranks.
    """
    normalizer = foreground.sum().detach().to(dtype=torch.float32)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(normalizer)
        normalizer /= get_world_size()
    effective = weights if contribution_scale is None else weights * contribution_scale
    return sigmoid_focal_loss(
        logits[..., None], foreground[..., None].to(logits.dtype), normalizer.clamp_min(1),
        alpha=alpha, gamma=2, query_weights=effective) * logits.shape[1]


def proposal_supervision(positive, pseudo, confidence):
    """Supervise detection slots, not exhaustive foreground regions.

    The GT matcher and deduplicated pseudo miner select positives; every other
    slot is a negative objectness/router example. This trades possible missed
    unknowns for learning to reject unselected and duplicate proposals.
    Novelty is conditional on a selected object, so negatives do not teach it
    that background is a known semantic object.
    """
    foreground = positive | pseudo
    foreground_weights = torch.where(positive, torch.ones_like(confidence),
                                     torch.where(pseudo, confidence, torch.zeros_like(confidence)))
    proposal_weights = torch.where(foreground, foreground_weights, torch.ones_like(confidence))
    return foreground, foreground_weights, proposal_weights


class DiscoveryCriterion(nn.Module):
    def __init__(self, base, model, args):
        super().__init__()
        self.base = base
        self.matcher = base.matcher
        self.weight_dict = base.weight_dict
        self.known_ids = list(args.owod_known_class_ids)
        self.warmup = args.discovery_warmup
        self.pseudo_ramp_epochs = getattr(args, 'discovery_pseudo_ramp_epochs', 0)
        self.objectness_loss = getattr(args, 'discovery_objectness_loss', 'balanced-bce')
        self.pseudo_normalization = getattr(args, 'discovery_pseudo_normalization', 'weight-sum')
        self.known_rank_coef = getattr(args, 'discovery_known_rank_coef', 0.)
        self.feature_grad_scale = getattr(args, 'discovery_feature_grad_scale', 1.)
        self.scoring = getattr(args, 'discovery_scoring', 'hierarchical')
        self.background_negative = getattr(args, 'discovery_background_negative', False)
        if not math.isfinite(self.known_rank_coef) or self.known_rank_coef < 0:
            raise ValueError('discovery_known_rank_coef must be finite and nonnegative')
        if not math.isfinite(self.feature_grad_scale) or not 0 <= self.feature_grad_scale <= 1:
            raise ValueError('discovery_feature_grad_scale must be finite and in [0, 1]')
        self.stability = self.known_rank_coef > 0 or self.feature_grad_scale != 1.
        if self.objectness_loss not in ('balanced-bce', 'focal'):
            raise ValueError('Invalid discovery_objectness_loss')
        if self.pseudo_normalization not in ('weight-sum', 'count'):
            raise ValueError('Invalid discovery_pseudo_normalization')
        if self.pseudo_ramp_epochs < 0:
            raise ValueError('discovery_pseudo_ramp_epochs must be nonnegative')
        self.top_k = args.discovery_top_k
        self.saliency_threshold = args.discovery_saliency_threshold
        self.saliency_top_points = getattr(args, 'discovery_saliency_top_points', None)
        self.saliency_match_count = getattr(args, 'discovery_saliency_match_count', False)
        self.consistency_threshold = args.discovery_consistency_threshold
        self.instance_mode = getattr(args, 'instance_mode', 'legacy')
        self.instance_targets = {}
        self.train_instance_projection = getattr(args, 'owod_stage', 0) == 0
        self.supervise_small_box_extent = getattr(args, 'screen_small_box_extent', False)
        self.supervise_small_box_quality = getattr(args, 'screen_small_box_quality_rank', False)
        if self.instance_mode != 'legacy':
            from .instance_transfer import SETTINGS
            self.weight_dict.update(loss_geometry=1., loss_instance_class=.2,
                                    loss_instance_view=.2, loss_instance_align=SETTINGS['alignment_weight'])
        if self.supervise_small_box_extent:
            from .instance_proposals import SMALL_BOX_EXTENT_WEIGHT
            self.weight_dict['loss_small_box_extent'] = SMALL_BOX_EXTENT_WEIGHT
        if self.supervise_small_box_quality:
            from .instance_proposals import SMALL_BOX_QUALITY_WEIGHT
            self.weight_dict['loss_small_box_quality_rank'] = SMALL_BOX_QUALITY_WEIGHT
        self.epoch = 0
        self.pseudo_epoch_offset = getattr(args, 'discovery_pseudo_epoch_offset', 0)
        self.evidence = None
        self.weight_dict.update(loss_objectness=1., loss_unknown=1., loss_discovery_route=1.,
                                loss_discovery_consistency=1., loss_affinity=.1)
        if self.known_rank_coef:
            self.weight_dict['loss_discovery_rank'] = self.known_rank_coef

    def forward(self, outputs, targets):
        # Unknown GT in full evaluation annotations is never visible to losses.
        released = []
        for target in targets:
            keep = torch.isin(target['labels'], target['labels'].new_tensor(self.known_ids))
            filtered = {**target, 'labels': target['labels'][keep], 'boxes': target['boxes'][keep]}
            if 'source_objects' in target:
                filtered['source_objects'] = target['source_objects'][keep]
            if self.supervise_small_box_extent or self.supervise_small_box_quality:
                filtered['source_box_area'] = target['source_box_area'][keep]
            released.append(filtered)
        targets = released
        indices = self.matcher(outputs, targets)
        teacher = outputs.get('discovery_teacher') if self.training and self.epoch >= self.warmup else None
        with scope('candidates'):
            candidates = mine_candidates(outputs, targets, indices, self.known_ids, teacher,
                                         self.top_k, self.saliency_threshold, self.consistency_threshold,
                                         saliency_top_points=self.saliency_top_points,
                                         saliency_match_count=self.saliency_match_count)
        pseudo, background, confidence = (candidates[k] for k in ('pseudo', 'background', 'weights'))
        positive = torch.zeros_like(pseudo)
        semantic = torch.full_like(pseudo, -1, dtype=torch.long)
        for batch, (queries, matches) in enumerate(indices):
            positive[batch, queries] = True
            semantic[batch, queries] = targets[batch]['labels'][matches]
        foreground, foreground_weights, proposal_weights = proposal_supervision(
            positive, pseudo, confidence)
        # Only two-view low-saliency matches are trusted negatives. In warmup,
        # evaluation, or without an EMA view, the legacy foreground loss applies.
        trusted_background = background & ~foreground
        if not self.background_negative or teacher is None:
            trusted_background = torch.zeros_like(background)
        novelty_weights = foreground_weights + trusted_background.to(confidence.dtype)
        pseudo_scale = (pseudo_supervision_scale(self.epoch, self.warmup, self.pseudo_ramp_epochs)
                        if self.training else 1.)
        contribution_scale = (torch.where(pseudo, confidence.new_tensor(pseudo_scale), 1.)
                              if self.pseudo_ramp_epochs else None)
        # Known-class membership and foreground existence are different targets.
        # Keep standard one-to-one DETR supervision over ALL queries for released
        # classes. An unmatched query is a negative for those classes, not a
        # claim that its region contains no (possibly unknown) object. Masking
        # this loss with discovery trust removes duplicate/background suppression.
        # Novelty optionally includes trusted background; objectness and routing learn
        # one-to-one proposal selection, including unselected-query negatives.
        prepared = dict(outputs)
        prepared.pop('query_label_weights', None)
        prepared['aux_outputs'] = []
        for layer in outputs.get('aux_outputs', []):
            layer = dict(layer)
            layer.pop('query_label_weights', None)
            prepared['aux_outputs'].append(layer)
        losses = self.base(prepared, targets)
        if self.objectness_loss == 'focal':
            losses['loss_objectness'] = proposal_focal_loss(
                outputs['pred_objectness'], foreground, proposal_weights, contribution_scale,
                alpha=self.base.focal_alpha)
        else:
            losses['loss_objectness'] = balanced_bce(outputs['pred_objectness'], foreground.float(),
                                                    proposal_weights, contribution_scale)
        losses['loss_unknown'] = balanced_bce(outputs['pred_novelty'], pseudo.float(),
                                             novelty_weights, contribution_scale,
                                             normalization=self.pseudo_normalization)
        if self.stability:
            matched_logits = outputs['pred_logits'][positive]
            matched_novelty = outputs['pred_novelty'][positive]
            matched_labels = semantic[positive]
            log_ratio = known_unknown_log_ratio(matched_logits, matched_novelty,
                                                matched_labels, self.known_ids, self.scoring)
            if self.known_rank_coef:
                # Task 4 has no unknown output or unknown-ranking objective.
                losses['loss_discovery_rank'] = (matched_known_ranking_loss(log_ratio)
                    if len(self.known_ids) < 80 else log_ratio.sum() * 0)
            with torch.no_grad():
                obj = outputs['pred_objectness'].sigmoid()
                novelty = matched_novelty.sigmoid() if len(self.known_ids) < 80 else torch.zeros_like(matched_novelty)
                gt_logits = matched_logits.gather(1, matched_labels[:, None]).squeeze(1)
                if self.scoring == 'hierarchical':
                    columns = (matched_labels[:, None] == matched_labels.new_tensor(self.known_ids)).long().argmax(-1)
                    conditional = matched_logits[..., self.known_ids].softmax(-1).gather(1, columns[:, None]).squeeze(1)
                    known_score = obj[positive] * (1 - novelty) * conditional
                else:
                    known_score = obj[positive] * gt_logits.sigmoid()
                losses.update(discovery_known_count=positive.sum().float(),
                    discovery_known_rank_violations=((log_ratio >= 0).sum().float()
                                                     if len(self.known_ids) < 80 else obj.new_zeros(())),
                    discovery_known_score_sum=known_score.sum(),
                    discovery_known_unknown_score_sum=(obj[positive] * novelty).sum(),
                    discovery_known_objectness_sum=obj[positive].sum(),
                    discovery_known_novelty_sum=novelty.sum(),
                    discovery_unselected_count=(~foreground).sum().float(),
                    discovery_unselected_objectness_sum=obj[~foreground].sum())
        route_target = torch.zeros_like(semantic)
        route_target[positive] = 1
        route_target[pseudo] = 2
        route_terms = []
        for logits in outputs['discovery_routes']:
            values = F.cross_entropy(logits.transpose(1, 2), route_target, reduction='none')
            if contribution_scale is not None:
                values = values * contribution_scale
            route_terms.append((values * proposal_weights).sum() / proposal_weights.sum().clamp_min(1))
        losses['loss_discovery_route'] = torch.stack(route_terms).mean()
        consistency = outputs['query_features'].sum() * 0
        if teacher is not None:
            maps = candidates['matches']
            batch_ids = torch.arange(pseudo.shape[0], device=pseudo.device)[:, None]
            teacher_features = teacher['discovery_embedding'][batch_ids, maps.clamp_min(0)].detach()
            feature_cost = 1 - F.cosine_similarity(outputs['discovery_embedding'], teacher_features, dim=-1)
            teacher_boxes = candidates['teacher_boxes'][batch_ids, maps.clamp_min(0)].detach()
            box_cost = (scale_gradient(outputs['pred_boxes'], self.feature_grad_scale)
                        - teacher_boxes).abs().sum(-1)
            consistency = ((feature_cost + box_cost) * confidence).sum() / confidence.sum().clamp_min(1)
        losses['loss_discovery_consistency'] = consistency * pseudo_scale if self.pseudo_ramp_epochs else consistency
        if self.instance_mode != 'legacy':
            from .instance_proposals import geometry_losses
            from .instance_transfer import alignment_loss
            z = outputs['instance_features']
            scores = outputs['instance_logits'][positive][:, self.known_ids]
            labels = semantic[positive]
            columns = (labels[:, None] == labels.new_tensor(self.known_ids)).long().argmax(-1)
            losses['loss_instance_class'] = (F.cross_entropy(scores, columns) if len(labels) and self.train_instance_projection
                                             else scores.sum()*0)
            losses.update(geometry_losses(outputs['geometry'], targets,
                                          supervise_extent=self.supervise_small_box_extent,
                                          supervise_quality=self.supervise_small_box_quality and self.training))
            if len(labels):
                from .instance_transfer import paired_iou
                gt_boxes = torch.cat([t['boxes'][g] for t, (_,g) in zip(targets, indices)])
                # Keep the matcher's order for both operands, independently of query indexing order.
                refined = torch.cat([outputs['pred_boxes'][i,q] for i,(q,_) in enumerate(indices)])
                quality = torch.cat([outputs['geometry_quality'][i,q] for i,(q,_) in enumerate(indices)])
                losses['loss_geometry'] += F.binary_cross_entropy(
                    quality.clamp(1e-6,1-1e-6), paired_iou(refined.detach(),gt_boxes).clamp(0,1))
            losses['loss_instance_view'] = z.sum()*0
            if teacher is not None:
                teacher_z = teacher['instance_features'][batch_ids, maps.clamp_min(0)].detach()
                cost = 1 - F.cosine_similarity(z, teacher_z, dim=-1)
                losses['loss_instance_view'] = (cost*confidence).sum()/confidence.sum().clamp_min(1.)
            losses['loss_instance_align'], losses['discovery_alignment_count'] = (
                alignment_loss(z, targets, indices, self.instance_targets)
                if self.training and self.instance_targets else (z.sum()*0, z.new_zeros(())))
        affinity = outputs['affinities']
        anchor = positive & (affinity.amax(-1) > -100)
        # The correct known prototype must exist; invalid anchors cannot be targets.
        anchor &= affinity.gather(-1, semantic.clamp_min(0)[..., None]).squeeze(-1) > -100
        losses['loss_affinity'] = (F.cross_entropy(affinity[anchor] / .1, semantic[anchor])
                                   if anchor.any() else affinity.sum() * 0)
        losses['discovery_pseudo_count'] = pseudo.sum().float()
        losses['discovery_pseudo_per_image'] = pseudo.sum().float() / pseudo.shape[0]
        losses['discovery_pseudo_weight_sum'] = confidence.sum()
        losses['discovery_pseudo_scale'] = confidence.new_tensor(pseudo_scale)
        # v5 ignores unselected slots for conditional novelty only. Classification,
        # objectness and routing all receive dense proposal supervision.
        losses['discovery_ignored_fraction'] = (novelty_weights == 0).float().mean()
        losses['discovery_known_supervised_fraction'] = confidence.new_ones(())
        losses['discovery_objectness_supervised_fraction'] = (proposal_weights > 0).float().mean()
        losses['discovery_novelty_supervised_fraction'] = (novelty_weights > 0).float().mean()
        losses['discovery_novelty_background_fraction'] = trusted_background.float().mean()
        losses['discovery_proposal_negative_fraction'] = (~foreground).float().mean()
        losses['discovery_background_fraction'] = background.float().mean()
        losses['discovery_positive_fraction'] = positive.float().mean()
        if self.training:
            accepted = positive | pseudo
            self.evidence = (outputs['query_features'][accepted].detach(), semantic[accepted].detach(),
                             foreground_weights[accepted].detach())
        else:
            self.evidence = None
        return losses


class DiscoveryPostProcess(PostProcess):
    def __init__(self, *args, scoring='hierarchical', **kwargs):
        super().__init__(*args, **kwargs)
        if scoring not in ('hierarchical', 'focal', 'focal_class_only'):
            raise ValueError('Unknown discovery scoring rule')
        self.scoring = scoring

    def candidate_scores(self, outputs):
        # Hierarchical is the existing reference. Focal uses independent class
        # scores consistent with sigmoid training; they need not sum to one.
        logits = outputs['pred_logits'][..., self.known_ids]
        known = logits.softmax(-1) if self.scoring == 'hierarchical' else logits.sigmoid()
        objectness = outputs['pred_objectness'].sigmoid()
        novelty = outputs['pred_novelty'].sigmoid() if len(self.known_ids) < 80 else torch.zeros_like(objectness)
        unknown = objectness * novelty
        # Focal heads learn independent class events. Preserve that link and
        # do not suppress a supervised known class a second time via novelty.
        # The diagnostic class-only view omits the known objectness factor.
        known_mass = objectness * (1 - novelty) if self.scoring == 'hierarchical' else objectness
        probabilities = known if self.scoring == 'focal_class_only' else known * known_mass[..., None]
        categories = torch.tensor(self.known_ids, device=known.device)
        if len(self.known_ids) < 80:
            probabilities = torch.cat([probabilities, unknown[..., None]], -1)
            categories = torch.cat([categories, categories.new_tensor([91])])
        return probabilities, categories, unknown

    @torch.no_grad()
    def forward(self, outputs, target_sizes, *, include_top_indices=False):
        probabilities, categories, unknown = self.candidate_scores(outputs)
        scores = probabilities.flatten(1)
        values, positions = scores.topk(min(self.max_detections, scores.shape[1]), dim=1)
        queries = positions // len(categories)
        labels = categories[positions % len(categories)]
        boxes = torch.gather(box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes']), 1,
                             queries[..., None].expand(-1, -1, 4))
        h, w = target_sizes.unbind(1)
        boxes *= torch.stack([w, h, w, h], -1)[:, None]
        unknown = unknown.gather(1, queries)
        results = [dict(scores=s, labels=l, boxes=b, unknown_scores=u,
                        unknown_mask=u >= self.unknown_threshold) for s, l, b, u in zip(values, labels, boxes, unknown)]
        if include_top_indices:
            for result, indices in zip(results, positions):
                result['top_indices'] = indices
        return results


class UnknownRefillScore(DiscoveryPostProcess):
    """Use a repeated known detection's slot for a distinct unknown query."""

    def __init__(self, *args, protected_ids=(), min_same_class_iou=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.protected_ids = tuple(protected_ids)
        self.min_same_class_iou = min_same_class_iou

    def same_class_overlap(self, predictions, queries, classes, earlier):
        boxes = box_ops.box_cxcywh_to_xyxy(predictions)
        boxes = boxes.gather(1, queries[..., None].expand(-1, -1, 4))
        lower, upper = boxes[..., :2], boxes[..., 2:]
        intersection = (torch.minimum(upper[:, :, None], upper[:, None, :])
                        - torch.maximum(lower[:, :, None], lower[:, None, :])).clamp_min(0).prod(dim=-1)
        area = (upper - lower).clamp_min(0).prod(dim=-1)
        union = area[:, :, None] + area[:, None, :] - intersection
        compatible = (earlier & (classes[:, :, None] == classes[:, None, :])
                      & (queries[:, :, None] != queries[:, None, :]))
        overlaps = (union > 0) & (intersection >= self.min_same_class_iou * union)
        return (compatible & overlaps).any(dim=-1)

    def candidate_scores(self, outputs):
        scores, categories, unknown = super().candidate_scores(outputs)
        rows, query_count, class_count = scores.shape
        if len(self.known_ids) + 1 != class_count or int(categories[-1]) != 91:
            raise ValueError('Expected an active unknown output')
        native = scores.flatten(1)
        values, indices = native.topk(self.max_detections, dim=1)
        queries = indices // class_count
        positions = torch.arange(self.max_detections, device=scores.device)
        earlier = positions[:, None] > positions[None, :]
        repeated = ((queries[..., None] == queries[:, None, :]) & earlier).any(dim=-1)
        classes = indices % class_count
        eligible = repeated & (classes != class_count - 1)
        if self.protected_ids:
            eligible &= ~torch.isin(categories[classes], categories.new_tensor(self.protected_ids))
        if self.min_same_class_iou is not None:
            eligible &= self.same_class_overlap(outputs['pred_boxes'], queries, classes, earlier)
        drop = torch.where(eligible, positions, -1).max(dim=1).values
        used = torch.zeros((rows, query_count), dtype=torch.bool, device=scores.device)
        used.scatter_(1, queries, True)
        refill_scores, refill_queries = unknown.masked_fill(used, -torch.inf).max(dim=1)
        refill = drop.ge(0) & refill_scores.isfinite()
        selected = torch.full_like(native, -torch.inf).scatter_(1, indices, values)
        batch = torch.arange(rows, device=scores.device)[refill]
        selected[batch, indices[batch, drop[batch]]] = -torch.inf
        selected[batch, refill_queries[batch] * class_count + class_count - 1] = refill_scores[batch]
        return selected.reshape_as(scores), categories, unknown


def build_discovery_postprocessor(args):
    options = dict(
        known_ids=args.owod_known_class_ids, max_detections=args.owod_max_detections,
        unknown_threshold=args.unknown_threshold,
        scoring=getattr(args, 'discovery_scoring', 'hierarchical'))
    if getattr(args, 'task2_protected_refill', False):
        min_same_class_iou = .70 if getattr(args, 'task2_refill_same_class_overlap', False) else None
        return UnknownRefillScore(protected_ids=args.owod_current_class_ids,
                                  min_same_class_iou=min_same_class_iou, **options)
    return DiscoveryPostProcess(**options)


def build(args):
    base, criterion, _ = build_base(args)
    model = DiscoveryDetector(base, args)
    criterion = DiscoveryCriterion(criterion, model, args)
    return model, criterion, {'bbox': build_discovery_postprocessor(args)}


def make_ema(model):
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    teacher.register_buffer('ema_update_count', torch.zeros((), dtype=torch.long,
                                                         device=next(model.parameters()).device))
    return teacher


@torch.no_grad()
def update_ema(teacher, model, decay=.999):
    # A fixed .999 EMA retains 74% random initialization after 300 updates.
    # Bias-free startup averages actual student iterates, then reaches .999.
    updates = int(teacher.ema_update_count)
    decay = min(decay, updates / (updates + 1))
    foreach_ema_parameters(teacher, model, decay)
    copy_model_buffers(teacher, model)
    teacher.ema_update_count.add_(1)
