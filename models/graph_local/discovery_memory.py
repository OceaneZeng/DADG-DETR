"""Anonymous training-only state and a separately learned affinity graph."""
import torch
from torch import nn
from torch.nn import functional as F


class PrototypeMemory(nn.Module):
    def __init__(self, dim, capacity=16, known_slots=91, adaptive=False):
        super().__init__()
        self.capacity = capacity
        self.known_slots = known_slots
        self.register_buffer('centers', torch.zeros(known_slots + capacity, dim))
        self.register_buffer('counts', torch.zeros(known_slots + capacity))
        self.register_buffer('variance', torch.zeros(known_slots + capacity, dim))
        self.adaptive = adaptive
        if adaptive:
            self.enable_adaptive()

    def enable_adaptive(self):
        """Fixed-size recency state, persisted for exact phase/resume continuity."""
        if hasattr(self, 'last_seen'):
            return
        self.adaptive = True
        self.register_buffer('clock', self.counts.new_zeros((), dtype=torch.long))
        self.register_buffer('last_seen', self.counts.new_zeros(self.capacity, dtype=torch.long))

    @torch.no_grad()
    def observe(self, features, labels, weights, threshold=.8):
        assignments = []
        if self.adaptive:
            self.clock.add_(1)
        # Copy predicates once instead of synchronizing CUDA for every known
        # object. Anonymous assignments and running updates remain sequential.
        accepted = (torch.isfinite(features).all(-1) & torch.isfinite(weights) & (weights > 0)).cpu().tolist()
        label_values = labels.cpu().tolist()
        for feature, label, weight, accept in zip(features, label_values, weights, accepted):
            if not accept:
                continue
            if label >= 0:
                if label >= self.known_slots:
                    raise ValueError('Future/unknown semantic IDs are not prototype labels')
                index = int(label)
            else:
                unknown_counts = self.counts[self.known_slots:]
                similarity = F.cosine_similarity(feature[None], self.centers[self.known_slots:])
                similarity[unknown_counts == 0] = -2
                best = int(similarity.argmax())
                if similarity[best] >= threshold:
                    index = self.known_slots + best
                elif (unknown_counts == 0).any():
                    index = self.known_slots + int(torch.where(unknown_counts == 0)[0][0])
                else:
                    if not self.adaptive:
                        # v5: never force unrelated evidence into an occupied slot.
                        continue
                    # Recycle only stale slots; do not merge unrelated clusters.
                    # At most one replacement per slot per 32 observation calls.
                    stale = self.clock - self.last_seen >= 32
                    if not stale.any():
                        continue
                    support = unknown_counts.masked_fill(~stale, float('inf'))
                    best = int(support.argmin())
                    index = self.known_slots + best
                    self.centers[index].zero_()
                    self.counts[index].zero_()
                    self.variance[index].zero_()
                assignments.append(index - self.known_slots)
                if self.adaptive:
                    self.last_seen[index - self.known_slots] = self.clock
            previous = self.counts[index].clone()
            total = previous + weight
            difference = feature - self.centers[index]
            self.centers[index].add_(difference * weight / total)
            self.variance[index].add_(weight * difference * (feature - self.centers[index]))
            self.counts[index] = total
        return sorted(set(assignments))


class AffinityGraph(nn.Module):
    """Query-to-known/anonymous prototype messages; never reinterpret harm scores."""
    def __init__(self, dim, hidden=64, use_messages=True):
        super().__init__()
        self.use_messages = bool(use_messages)
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.objectness = nn.Linear(hidden, 1)
        self.novelty = nn.Linear(hidden, 1)

    def similarities(self, features, centers):
        return F.normalize(self.project(features), dim=-1) @ F.normalize(self.project(centers), dim=-1).T

    def forward(self, features, memory):
        query = F.normalize(self.project(features), dim=-1)
        nodes = F.normalize(self.project(memory.centers.detach()), dim=-1)
        valid = memory.counts > 0
        affinities = (query @ nodes.T).masked_fill(~valid, -1e4)
        attention = (affinities / .1).softmax(-1) * valid
        # Same parameter capacity and heads; the ablation uses only a self-loop.
        context = attention @ nodes if self.use_messages else query
        embedding = self.update(torch.cat([query, context], -1))
        return {'pred_objectness': self.objectness(embedding).squeeze(-1),
                'pred_novelty': self.novelty(embedding).squeeze(-1),
                'affinities': affinities, 'discovery_embedding': F.normalize(embedding, dim=-1)}


@torch.no_grad()
def initialize_from_deltas(adapter, deltas, weights):
    """Mix effective BA mappings, then truncate via SVD (never average factors)."""
    if not deltas:
        return
    delta = sum(d * w for d, w in zip(deltas, weights))
    u, singular, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    rank = adapter.down.out_features
    root = singular[:rank].clamp_min(0).sqrt()
    adapter.up.weight.copy_((u[:, :rank] * root).to(adapter.up.weight))
    adapter.down.weight.copy_((root[:, None] * vh[:rank]).to(adapter.down.weight))
