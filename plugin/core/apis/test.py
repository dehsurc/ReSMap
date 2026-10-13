# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        for _ in range(len(result)):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    """Each rank runs whole sequences; results are gathered on rank 0 through tmpdir."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # avoids a rare deadlock at start-up
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        if rank == 0:
            for _ in range(len(result) * world_size):
                prog_bar.update()
    return collect_results_cpu(results, len(dataset), tmpdir)


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    if rank != 0:
        return None

    # Ranks hold whole sequences, so the parts are concatenated rather than interleaved.
    results = []
    for i in range(world_size):
        results.extend(mmcv.load(osp.join(tmpdir, f'part_{i}.pkl')))
    shutil.rmtree(tmpdir)
    return results[:size]
