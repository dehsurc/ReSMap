import argparse
import os
import os.path as osp
import subprocess
import sys

import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# bit i drops the i-th camera (CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, ...)
CAMERA_DROPS = dict(none=0, front1=1, front3=7, all=63)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ReSMap inference and evaluate vector mAP')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', nargs='?', help='checkpoint file')
    parser.add_argument('--work-dir', help='where submission_vector.json is written')
    parser.add_argument('--camera-drop', choices=list(CAMERA_DROPS), default='none',
                        help='zero these cameras at test time')
    parser.add_argument('--format-only', action='store_true',
                        help='write the submission file without evaluating it')
    parser.add_argument('--result', help='evaluate an existing submission_vector.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override config settings, xxx=yyy')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def evaluate_in_subprocess(config, result):
    # The AP step uses a multiprocessing pool; running it in a process that
    # has not initialised CUDA avoids worker deadlocks.
    subprocess.run([sys.executable, osp.join(osp.dirname(__file__), 'eval_submission.py'),
                    config, result], check=True)


def main():
    args = parse_args()
    if args.result:
        evaluate_in_subprocess(args.config, args.result)
        return
    assert args.checkpoint, 'a checkpoint is required unless --result is given'

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    sys.path.append(os.path.abspath('.'))
    import plugin  # noqa: F401  (registers the ReSMap modules)
    from plugin.core.apis import multi_gpu_test, single_gpu_test
    from plugin.datasets.builder import build_dataloader

    for t in cfg.data.test.pipeline:
        if t['type'] == 'ApplySensorCorruption':
            t['enabled'] = args.camera_drop != 'none'
            t['camera_mask'] = CAMERA_DROPS[args.camera_drop]

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    if args.seed is not None:
        set_random_seed(args.seed)

    work_dir = args.work_dir or osp.join(
        './work_dirs', osp.splitext(osp.basename(args.config))[0], f'test_{args.camera_drop}')
    cfg.data.test.work_dir = work_dir
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed, shuffle=False)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)
    else:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    if rank != 0:
        return
    result = dataset.format_results(outputs, work_dir, save_semantic=dataset.eval_semantic)
    if args.format_only:
        return
    if dataset.eval_semantic:
        print(dataset.evaluate_file(result))
    else:
        # the evaluator reads the GT settings from the config, including --cfg-options
        test_cfg_path = osp.join(work_dir, osp.basename(args.config))
        cfg.dump(test_cfg_path)
        evaluate_in_subprocess(test_cfg_path, result)


if __name__ == '__main__':
    main()
