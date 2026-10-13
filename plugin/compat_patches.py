"""Shims for running the mmcv 1.x stack on PyTorch >= 2.2. No-op on older torch.

1. mmcv's DDP scatter passes integer device ids to torch's ``_get_stream``,
   which expects a device since torch 2.2.
2. mmcv's DDP eval path reads ``_use_replicated_tensor_module``, which torch
   no longer sets.
"""
import torch


def _patch_mmcv_get_stream():
    try:
        import mmcv.parallel._functions as _mmcv_fn
    except Exception:
        return
    orig = _mmcv_fn._get_stream
    if getattr(orig, '_resmap_compat', False):
        return

    def _get_stream(device):
        if not hasattr(device, 'type'):
            device = torch.device('cuda', int(device))
        return orig(device)

    _get_stream._resmap_compat = True
    _mmcv_fn._get_stream = _get_stream


def _patch_mmcv_ddp_replicated_attr():
    try:
        from mmcv.parallel.distributed import MMDistributedDataParallel as _DDP
    except Exception:
        return
    if not hasattr(_DDP, '_use_replicated_tensor_module'):
        _DDP._use_replicated_tensor_module = False


def _torch_ge_2_2():
    major, minor = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    return (major, minor) >= (2, 2)


def apply():
    if not _torch_ge_2_2():
        return
    _patch_mmcv_get_stream()
    _patch_mmcv_ddp_replicated_attr()


apply()
