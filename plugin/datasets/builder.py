# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmdet.datasets.samplers import GroupSampler
from torch.utils.data import DataLoader

from .samplers import DistributedGroupSampler, DistributedSampler


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Training: shuffled batches. Testing: whole sequences per GPU, in order."""
    rank, world_size = get_dist_info()
    if dist:
        if shuffle:
            sampler = DistributedGroupSampler(
                dataset, samples_per_gpu, num_replicas=world_size, rank=rank, seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # worker seed = num_workers * rank + worker_id + user seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
