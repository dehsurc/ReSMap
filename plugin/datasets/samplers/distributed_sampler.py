# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import numpy as np
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """Test-time sampler that gives each rank whole sequences, in order.

    The BEV memory needs consecutive frames of a scene on the same GPU, so the
    sequence groups (``dataset.flag``) are split round-robin across ranks.
    """

    def __init__(self, dataset=None, num_replicas=None, rank=None, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        flag = self.dataset.flag
        num_groups = len(np.bincount(flag))
        self.sample_idxs = []
        for group_idx in range(num_groups)[self.rank::self.num_replicas]:
            self.sample_idxs.extend(np.where(flag == group_idx)[0].tolist())
        self.num_samples = len(self.sample_idxs)
        self.total_size = len(self.dataset)

    def __iter__(self):
        return iter(self.sample_idxs)
