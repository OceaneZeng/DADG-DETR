"""Persistent class adapters with query routing and directed interference gates."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .lora import freeze_for_class_ids, target_decoder_layers
from util.runtime_profile import scope


class ClassAdapter(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class RiskRoutedStage(nn.Module):
    """One immutable stage layout; routing never accepts ground-truth labels.

    Router classes are this stage's new classes plus a null route. Old evidence
    is a frozen classifier snapshot. Risk is new-to-old normalized harm, not
    transfer affinity. All branches are evaluated so DDP sees zero gradients
    (rather than unused parameters) for unselected branches.
    """
    def __init__(self, head, spec):
        super().__init__()
        self.class_ids = tuple(spec['class_ids'])
        self.old_ids = tuple(spec['old_ids'])
        if (not self.class_ids or not self.old_ids
                or len(set(self.class_ids)) != len(self.class_ids)
                or len(set(self.old_ids)) != len(self.old_ids)
                or set(self.class_ids).intersection(self.old_ids)
                or any(c < 0 or c >= head.out_features for c in self.class_ids + self.old_ids)):
            raise ValueError('Adapter classes must be disjoint, unique, nonempty seen-class IDs')
        self.beta = float(spec['risk_beta'])
        self.risk_mode = spec.get('risk_mode', 'learned')
        self.compute = spec.get('compute', 'loop')
        if self.risk_mode not in ('learned', 'disabled') or self.compute not in ('loop', 'batched'):
            raise ValueError('Unknown adapter risk or compute mode')
        if self.risk_mode == 'disabled' and self.beta != 0:
            raise ValueError('Disabled risk requires beta=0')
        self.routing = spec.get('routing', 'hard')
        if self.routing not in ('hard', 'soft'):
            raise ValueError('Unknown adapter routing rule')
        dim = head.in_features
        self.adapters = nn.ModuleList([ClassAdapter(dim, spec['rank'])
                                       for _ in self.class_ids])
        self.router = nn.Linear(dim, len(self.class_ids) + 1)
        self.old_evidence = nn.Linear(dim, len(self.old_ids))
        risk = torch.as_tensor(spec['risk'], dtype=torch.float32)
        if (risk.shape != (len(self.class_ids), len(self.old_ids))
                or not torch.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any()
                or not math.isfinite(self.beta) or self.beta < 0 or spec['rank'] <= 0):
            raise ValueError('Invalid directed risk matrix or adapter configuration')
        if self.risk_mode == 'disabled' and risk.count_nonzero():
            raise ValueError('Disabled risk requires an all-zero placeholder matrix')
        self.register_buffer('risk', risk)
        with torch.no_grad():
            self.router.weight[:-1].copy_(head.weight[list(self.class_ids)])
            self.router.bias[:-1].copy_(head.bias[list(self.class_ids)])
            self.router.weight[-1].zero_()
            self.router.bias[-1].zero_()
            self.old_evidence.weight.copy_(head.weight[list(self.old_ids)])
            self.old_evidence.bias.copy_(head.bias[list(self.old_ids)])
        self.old_evidence.requires_grad_(False)
        self.last_logits = None
        self.last_gates = None

    def forward(self, x):
        with scope('class_adapter'):
            return self._forward(x)

    def _forward(self, x):
        # Router supervision should not rewrite the upstream representation.
        logits = self.router(x.detach())
        self.last_logits = logits
        probabilities = logits.softmax(-1)
        selected = (F.one_hot(probabilities.argmax(-1), probabilities.shape[-1])[..., :-1]
                    if self.routing == 'hard' else torch.ones_like(probabilities[..., :-1]))
        gates = probabilities[..., :-1] * selected
        if self.beta:
            old_probability = self.old_evidence(x.detach()).sigmoid()
            # Maximum risk-weighted old evidence avoids dependence on old-class count.
            risk_evidence = (old_probability.unsqueeze(-2) * self.risk).amax(-1)
            gates = gates * torch.exp(-self.beta * risk_evidence)
        self.last_gates = gates.detach()
        if self.compute == 'batched':
            # Keep individual Parameters/state_dict keys and DDP zero gradients.
            # Stack factors, not effective BA matrices, to retain low-rank cost.
            down = torch.stack([adapter.down.weight for adapter in self.adapters])
            up = torch.stack([adapter.up.weight for adapter in self.adapters])
            flat = x.reshape(-1, x.shape[-1])
            hidden = torch.matmul(flat.unsqueeze(0), down.transpose(1, 2))
            updates = torch.bmm(hidden, up.transpose(1, 2))
            weighted = updates * gates.reshape(-1, len(self.adapters)).T.unsqueeze(-1)
            return weighted.sum(0).reshape_as(x)
        updates = torch.stack([adapter(x) for adapter in self.adapters], dim=-2)
        return (updates * gates.unsqueeze(-1)).sum(-2)


class RiskRoutedBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList()

    def forward(self, x):
        # Parallel updates share the same unadapted input at this layer.
        return sum((stage(x) for stage in self.stages), torch.zeros_like(x))


def install_specs(model, specs):
    """Rebuild the full persistent layout before strict checkpoint loading."""
    for spec in specs:
        append_stage(model, spec)


def append_stage(model, spec):
    layers = target_decoder_layers(model, spec['last_n'])
    detector = model.detr if hasattr(model, 'detr') else model
    for layer in layers:
        if not hasattr(layer, 'graph_adapter'):
            layer.graph_adapter = RiskRoutedBank()
        learned = {c for stage in layer.graph_adapter.stages for c in stage.class_ids}
        if learned.intersection(spec['class_ids']):
            raise ValueError('Cannot append a duplicate class adapter')
        stage = RiskRoutedStage(detector.class_embed[-1], spec)
        stage.to(device=layer.linear1.weight.device, dtype=layer.linear1.weight.dtype)
        layer.graph_adapter.stages.append(stage)


def configure_trainable(model, current_ids, known_ids, finetune=False, localization='frozen'):
    """Increment: current bank/rows only. Balanced FT: all banks/known rows."""
    handles, _ = freeze_for_class_ids(model, known_ids if finetune else current_ids)
    if localization not in ('frozen', 'bbox-last'):
        raise ValueError('Unknown localization adaptation policy')
    if localization == 'bbox-last':
        detector = model.detr if hasattr(model, 'detr') else model
        # Only the final coordinate projection (4 x hidden_dim + bias), not
        # the whole detector/head. Shared heads are naturally deduplicated.
        for head in detector.bbox_embed:
            head.layers[-1].requires_grad_(True)
    current = set(current_ids)
    for module in model.modules():
        if isinstance(module, RiskRoutedStage):
            active = finetune or set(module.class_ids).issubset(current)
            module.adapters.requires_grad_(active)
            module.router.requires_grad_(active)
            module.old_evidence.requires_grad_(False)
    return handles


def routing_outputs(model):
    routes = []
    for stage in model.modules():
        if isinstance(stage, RiskRoutedStage):
            if stage.router.weight.requires_grad and stage.last_logits is not None:
                routes.append((stage.last_logits, stage.class_ids))
            stage.last_logits = None  # Do not retain autograd graphs between batches.
    return routes


def mask_inactive_adapter_gradients(model):
    """Skip inactive factors, including AdamW decay and existing momentum.

    DDP first reduces zero gradients normally; global activity then determines
    whether an optimizer should see a branch at all on this step.
    """
    stages = [m for m in model.modules() if isinstance(m, RiskRoutedStage)]
    if not stages:
        return
    flags = torch.cat([(m.last_gates > 0).flatten(0, -2).any(0) for m in stages]).int()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
    # Read global activity once rather than synchronizing each CUDA scalar.
    # Preserve grad=None for globally inactive factors (including AdamW state).
    flags = flags.cpu().tolist()
    offset = 0
    for stage in stages:
        for adapter in stage.adapters:
            if not flags[offset]:
                for parameter in adapter.parameters():
                    parameter.grad = None
            offset += 1


def routing_loss(routes, targets, indices):
    """Matched current labels teach routing; unmatched/other classes teach null."""
    losses = []
    for logits, class_ids in routes:
        desired = torch.full(logits.shape[:2], len(class_ids), dtype=torch.long,
                             device=logits.device)
        for batch, (queries, matches) in enumerate(indices):
            queries = queries.to(logits.device)
            matches = matches.to(logits.device)
            labels = targets[batch]['labels'][matches]
            for column, class_id in enumerate(class_ids):
                desired[batch, queries[labels == class_id]] = column
        # DETR has many empty slots. Balance positive and null averages.
        values = F.cross_entropy(logits.transpose(1, 2), desired, reduction='none')
        positive = desired != len(class_ids)
        parts = [values[mask].mean() for mask in (positive, ~positive) if mask.any()]
        losses.append(torch.stack(parts).mean())
    return torch.stack(losses).mean()
