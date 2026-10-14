from copy import deepcopy
from functools import cached_property
from logging import Logger
from typing import Dict, Optional

import mmcv
import prettytable
import torch
from mmcv import Config
from mmcv.utils import print_log
from mmdet3d.datasets import build_dataloader, build_dataset
from numpy.typing import NDArray

N_WORKERS = 8


class RasterEvaluate(object):
    """Per-class IoU of the BEV semantic mask.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, dataset_cfg: Config, n_workers: int=N_WORKERS):
        self.dataset = build_dataset(dataset_cfg)
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=n_workers, shuffle=False, dist=False)
        self.cat2id = self.dataset.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.n_workers = n_workers

    @cached_property
    def gts(self) -> Dict[str, NDArray]:
        print('collecting gts...')
        gts = {}
        for data in mmcv.track_iter_progress(self.dataloader):
            token = deepcopy(data['img_metas'].data[0][0]['token'])
            gts[token] = deepcopy(data['semantic_mask'].data[0][0])
            del data
        return gts

    def evaluate(self, result_path: str, logger: Optional[Logger]=None) -> Dict[str, float]:
        """Evaluate a submission file, aligned to the GT by sample token."""
        results = mmcv.load(result_path)['results']

        gts_stacked = torch.flip(torch.stack(list(self.gts.values())).bool(), [2])
        preds = []
        for token, gt in self.gts.items():
            pred = torch.zeros((len(self.cat2id), gt.shape[1], gt.shape[2])).bool()
            if token in results and 'semantic_mask' in results[token]:
                semantic_mask = torch.tensor(results[token]['semantic_mask'])
                for label_i in range(gt.shape[0]):
                    pred[label_i] = (semantic_mask == label_i + 1)
            preds.append(pred)
        preds = torch.stack(preds).bool()

        result_dict = {}
        total = 0
        for i in range(gts_stacked.shape[1]):
            category = self.id2cat[i]
            intersect = (preds[:, i] & gts_stacked[:, i]).sum().float().item()
            union = (preds[:, i] | gts_stacked[:, i]).sum().float().item()
            result_dict[category] = intersect / (union + 1e-7)
            total += result_dict[category]
        result_dict['mIoU'] = total / gts_stacked.shape[1]

        categories = list(self.cat2id.keys())
        table = prettytable.PrettyTable([' ', *categories, 'mean'])
        table.add_row(['IoU', *[round(result_dict[cat], 4) for cat in categories],
                       round(result_dict['mIoU'], 4)])
        if logger:
            print_log('\n' + str(table), logger=logger)
            print_log(f'mIoU = {result_dict["mIoU"]:.4f}', logger=logger)
        return result_dict
