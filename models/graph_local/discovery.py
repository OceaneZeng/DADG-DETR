"""Open-world discovery, persistent anonymous memory, and supervised promotion."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

from models.deformable_detr import build as build_base, PostProcess
from models.graph_local.lora import target_decoder_layers
from .routed_adapter import ClassAdapter
from .discovery_memory import PrototypeMemory, AffinityGraph, initialize_from_deltas
from .discovery_candidates import mine_candidates
from util import box_ops
from util.training_runtime import gather_evidence, foreach_ema_parameters, copy_model_buffers

DISCOVERY_VERSION = 5
DISCOVERY_RAMP_VERSION = 6


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
        detector.return_baseline_features = True
        self.memory = PrototypeMemory(args.hidden_dim, args.discovery_capacity)
        self.affinity_graph = AffinityGraph(
            args.hidden_dim, use_messages=not getattr(args, 'discovery_no_message', False))
        for layer in target_decoder_layers(self, args.graph_adapter_last_n):
            layer.discovery_adapter = DiscoveryFFN(args.hidden_dim, args.graph_adapter_rank,
                                                   args.discovery_capacity)

    def forward(self, samples):
        from util.misc import NestedTensor, nested_tensor_from_tensor_list
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        outputs = self.detr(samples)
        features = outputs.pop('decoder_features')[-1]
        outputs['query_features'] = features
        outputs.update(self.affinity_graph(features, self.memory))
        routes = []
        route_inputs = []
        for module in self.modules():
            if isinstance(module, DiscoveryFFN):
                routes.append(module.last_logits)
                route_inputs.append(module.last_inputs)
                module.last_logits = None
                module.last_inputs = None
        outputs['discovery_routes'] = routes
        outputs['discovery_inputs'] = route_inputs
        masks = samples.mask
        outputs['image_sizes'] = [(int((~mask).any(1).sum()), int((~mask).any(0).sum())) for mask in masks]
        return outputs

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
                if isinstance(module, DiscoveryFFN):
                    module.snapshot(slots)
        if distributed:
            # All model buffers (including anonymous snapshots) remain identical.
            for buffer in self.memory.buffers():
                torch.distributed.broadcast(buffer, 0)
            for module in self.modules():
                if isinstance(module, DiscoveryFFN):
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


def balanced_bce(logits, targets, weights, contribution_scale=None):
    values = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if contribution_scale is not None:
        # Scale contributions AFTER fixing normalization. Scaling the weights
        # in both numerator and denominator would cancel for a pseudo group.
        values = values * contribution_scale
    parts = []
    for mask in (targets > .5, targets <= .5):
        effective = weights * mask
        if effective.sum() > 0:
            parts.append((values * effective).sum() / effective.sum())
    return torch.stack(parts).mean() if parts else logits.sum() * 0


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
        if self.pseudo_ramp_epochs < 0:
            raise ValueError('discovery_pseudo_ramp_epochs must be nonnegative')
        self.top_k = args.discovery_top_k
        self.saliency_threshold = args.discovery_saliency_threshold
        self.consistency_threshold = args.discovery_consistency_threshold
        self.epoch = 0
        self.evidence = None
        self.weight_dict.update(loss_objectness=1., loss_unknown=1., loss_discovery_route=1.,
                                loss_discovery_consistency=1., loss_affinity=.1)

    def forward(self, outputs, targets):
        # Unknown GT in full evaluation annotations is never visible to losses.
        targets = [{**target, 'labels': target['labels'][torch.isin(target['labels'],
                    target['labels'].new_tensor(self.known_ids))],
                    'boxes': target['boxes'][torch.isin(target['labels'],
                    target['labels'].new_tensor(self.known_ids))]} for target in targets]
        indices = self.matcher(outputs, targets)
        teacher = outputs.get('discovery_teacher') if self.training and self.epoch >= self.warmup else None
        candidates = mine_candidates(outputs, targets, indices, self.known_ids, teacher,
                                     self.top_k, self.saliency_threshold, self.consistency_threshold)
        pseudo, background, confidence = (candidates[k] for k in ('pseudo', 'background', 'weights'))
        positive = torch.zeros_like(pseudo)
        semantic = torch.full_like(pseudo, -1, dtype=torch.long)
        for batch, (queries, matches) in enumerate(indices):
            positive[batch, queries] = True
            semantic[batch, queries] = targets[batch]['labels'][matches]
        foreground, foreground_weights, proposal_weights = proposal_supervision(
            positive, pseudo, confidence)
        pseudo_scale = (pseudo_supervision_scale(self.epoch, self.warmup, self.pseudo_ramp_epochs)
                        if self.training else 1.)
        contribution_scale = (torch.where(pseudo, confidence.new_tensor(pseudo_scale), 1.)
                              if self.pseudo_ramp_epochs else None)
        # Known-class membership and foreground existence are different targets.
        # Keep standard one-to-one DETR supervision over ALL queries for released
        # classes. An unmatched query is a negative for those classes, not a
        # claim that its region contains no (possibly unknown) object. Masking
        # this loss with discovery trust removes duplicate/background suppression.
        # Novelty uses selected foreground only; objectness and routing learn
        # one-to-one proposal selection, including unselected-query negatives.
        prepared = dict(outputs)
        prepared.pop('query_label_weights', None)
        prepared['aux_outputs'] = []
        for layer in outputs.get('aux_outputs', []):
            layer = dict(layer)
            layer.pop('query_label_weights', None)
            prepared['aux_outputs'].append(layer)
        losses = self.base(prepared, targets)
        losses['loss_objectness'] = balanced_bce(outputs['pred_objectness'], foreground.float(),
                                                proposal_weights, contribution_scale)
        losses['loss_unknown'] = balanced_bce(outputs['pred_novelty'], pseudo.float(),
                                             foreground_weights, contribution_scale)
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
            box_cost = (outputs['pred_boxes'] - teacher_boxes).abs().sum(-1)
            consistency = ((feature_cost + box_cost) * confidence).sum() / confidence.sum().clamp_min(1)
        losses['loss_discovery_consistency'] = consistency * pseudo_scale if self.pseudo_ramp_epochs else consistency
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
        losses['discovery_ignored_fraction'] = (~foreground).float().mean()
        losses['discovery_known_supervised_fraction'] = confidence.new_ones(())
        losses['discovery_objectness_supervised_fraction'] = (proposal_weights > 0).float().mean()
        losses['discovery_novelty_supervised_fraction'] = (foreground_weights > 0).float().mean()
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
        if scoring not in ('hierarchical', 'focal'):
            raise ValueError('Unknown discovery scoring rule')
        self.scoring = scoring

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        # Hierarchical is the existing reference. Focal uses independent class
        # scores consistent with sigmoid training; they need not sum to one.
        logits = outputs['pred_logits'][..., self.known_ids]
        known = logits.softmax(-1) if self.scoring == 'hierarchical' else logits.sigmoid()
        objectness = outputs['pred_objectness'].sigmoid()
        novelty = outputs['pred_novelty'].sigmoid() if len(self.known_ids) < 80 else torch.zeros_like(objectness)
        unknown = objectness * novelty
        # Focal heads learn independent class events. Preserve that link and
        # do not suppress a supervised known class a second time via novelty.
        known_mass = objectness * (1 - novelty) if self.scoring == 'hierarchical' else objectness
        probabilities = known * known_mass[..., None]
        categories = torch.tensor(self.known_ids, device=known.device)
        if len(self.known_ids) < 80:
            probabilities = torch.cat([probabilities, unknown[..., None]], -1)
            categories = torch.cat([categories, categories.new_tensor([91])])
        values, positions = probabilities.flatten(1).topk(min(self.max_detections, probabilities[0].numel()), dim=1)
        queries = positions // len(categories)
        labels = categories[positions % len(categories)]
        boxes = torch.gather(box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes']), 1,
                             queries[..., None].expand(-1, -1, 4))
        h, w = target_sizes.unbind(1)
        boxes *= torch.stack([w, h, w, h], -1)[:, None]
        unknown = unknown.gather(1, queries)
        return [dict(scores=s, labels=l, boxes=b, unknown_scores=u,
                     unknown_mask=u >= self.unknown_threshold) for s, l, b, u in zip(values, labels, boxes, unknown)]


def build(args):
    base, criterion, _ = build_base(args)
    model = DiscoveryDetector(base, args)
    criterion = DiscoveryCriterion(criterion, model, args)
    return model, criterion, {'bbox': DiscoveryPostProcess(
        known_ids=args.owod_known_class_ids, max_detections=args.owod_max_detections,
        unknown_threshold=args.unknown_threshold,
        scoring=getattr(args, 'discovery_scoring', 'hierarchical'))}


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
