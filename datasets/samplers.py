# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from codes in torch.utils.data.distributed
# ------------------------------------------------------------------------

import os
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ExposureSampler(DistributedSampler):
    """Use the original per-epoch shuffle and rank split on one of two fixed pools."""

    def __init__(self, dataset, first_names, mode, num_replicas, rank, second_names=None):
        first = set(first_names)
        second = None if second_names is None else set(second_names)
        if second is not None and (second & first or len(second) != len(first) or len(second) != len(second_names)):
            raise ValueError('Explicit exposure pools must be disjoint and equal in size')
        pools = ([], [])
        for index, image_id in enumerate(dataset.ids):
            name = dataset.coco.imgs[image_id]['owod_source_id']
            if name in first:
                pools[0].append(index)
            elif second is None or name in second:
                pools[1].append(index)
        size = len(pools[0])
        if not size or size != len(first) or size != len(pools[1]) or size % num_replicas:
            raise ValueError('Expected two complete equal image pools divisible across ranks')
        super().__init__(pools[0], num_replicas=num_replicas, rank=rank)
        self.pools = pools
        self.mode = mode

    def __iter__(self):
        pool = self.pools[self.epoch % 2] if self.mode == 'alternate' else self.pools[0]
        return (pool[i] for i in super().__iter__())


class ReplayBalancedSampler(Sampler):
    """Sample a fixed replay fraction on every distributed rank each epoch."""

    def __init__(self, dataset, replay_indices, replay_fraction=0.25,
                 num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if not 0.0 < float(replay_fraction) < 1.0:
            raise ValueError("replay_fraction must be between zero and one")
        replay = sorted({int(index) for index in replay_indices})
        current = sorted(set(range(len(dataset))) - set(replay))
        if not replay or not current:
            raise ValueError("balanced replay requires both replay and current samples")
        self.dataset = dataset
        self.replay_indices = replay
        self.current_indices = current
        self.replay_fraction = float(replay_fraction)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.replay_per_rank = max(1, int(round(self.num_samples * self.replay_fraction)))
        self.current_per_rank = self.num_samples - self.replay_per_rank

    @staticmethod
    def _draw(pool, count, generator):
        result = []
        while len(result) < count:
            order = torch.randperm(len(pool), generator=generator).tolist()
            result.extend(pool[index] for index in order)
        return result[:count]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        replay = self._draw(
            self.replay_indices, self.replay_per_rank * self.num_replicas, generator)
        current = self._draw(
            self.current_indices, self.current_per_rank * self.num_replicas, generator)
        start_replay = self.rank * self.replay_per_rank
        start_current = self.rank * self.current_per_rank
        local = (replay[start_replay:start_replay + self.replay_per_rank] +
                 current[start_current:start_current + self.current_per_rank])
        order = torch.randperm(len(local), generator=generator).tolist()
        return iter([local[index] for index in order])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class NodeDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        indices = indices[self.rank // self.num_parts:self.total_size_parts:self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
