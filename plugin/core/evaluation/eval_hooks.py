import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return
        from ..apis.test import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(BaseDistEvalHook):

    def _do_evaluate(self, runner):
        # DDP does not sync BN buffers; use rank 0's for evaluation.
        if self.broadcast_bn_buffer:
            for _, module in runner.model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from ..apis.test import multi_gpu_test
        results = multi_gpu_test(runner.model, self.dataloader, tmpdir=tmpdir)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, key_score)
