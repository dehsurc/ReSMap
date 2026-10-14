import os
import os.path as osp
from time import time

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from .evaluation.raster_eval import RasterEvaluate
from .evaluation.vector_eval import VectorEvaluate
from .map_utils.nuscmap_extractor import NuscMapExtractor


@DATASETS.register_module()
class NuscDataset(Dataset):
    """nuScenes map dataset (adapted from StreamMapNet / MapTracker).

    Args:
        data_root (str): nuScenes root, used to read the HD map.
        ann_file (str): info pkl from tools/data_converter/nuscenes_converter.py.
        cat2id (dict): map class name -> label.
        roi_size (tuple): BEV range in meters.
        meta (dict): written into the submission file.
        pipeline (list): data pipeline.
        interval (int): keep every `interval`-th sample.
        seq_split_num (int): 1 = one group per scene (test), -2 = a single
            group (training), n > 1 = split each scene into n groups.
        work_dir (str): where the submission file is written.
        eval_config (dict): dataset config used to build the evaluation GT.
        test_mode (bool): test mode.
        multi_frame (int): clip length during training (current frame + earlier frames).
        sampling_span (int): earlier frames are drawn from this many preceding samples.
        eval_semantic (bool): evaluate BEV segmentation IoU instead of vector AP.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 cat2id,
                 roi_size,
                 meta,
                 pipeline,
                 interval=1,
                 seq_split_num=1,
                 work_dir=None,
                 eval_config=None,
                 test_mode=False,
                 multi_frame=False,
                 sampling_span=10,
                 eval_semantic=False):
        super().__init__()
        self.ann_file = ann_file
        self.cat2id = cat2id
        self.roi_size = roi_size
        self.meta = meta
        self.interval = interval
        self.seq_split_num = seq_split_num
        self.work_dir = work_dir
        self.eval_config = eval_config
        if eval_config is not None:
            assert test_mode, 'eval_config is valid only in test_mode'
        self.multi_frame = multi_frame
        self.sampling_span = sampling_span
        self.eval_semantic = eval_semantic

        self.load_annotations(ann_file)
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.map_extractor = NuscMapExtractor(data_root, roi_size)

        self._set_sequence_info()
        self._set_sequence_group_flag()

    def load_annotations(self, ann_file):
        start_time = time()
        self.samples = mmcv.load(ann_file)[::self.interval]
        print(f'collected {len(self.samples)} samples in {(time() - start_time):.2f}s')

    def _set_sequence_info(self):
        """Record the scene and the in-scene index of every sample."""
        scene_name2idx = {}
        for idx, sample in enumerate(self.samples):
            scene = sample['scene_name']
            if scene not in scene_name2idx:
                scene_name2idx[scene] = []
                self.samples[idx]['prev'] = -1
            scene_name2idx[scene].append(idx)
        self.scene_name2idx = scene_name2idx

        self.idx2scene = {}
        for scene_name, scene_info in scene_name2idx.items():
            for local_idx, global_idx in enumerate(scene_info):
                self.idx2scene[global_idx] = (scene_name, local_idx, len(scene_info))

    def _set_sequence_group_flag(self):
        """Group samples into sequences; the samplers never split a group across GPUs."""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if self.seq_split_num == -2:
            return

        res = []
        curr_sequence = -1
        for idx in range(len(self.samples)):
            if self.samples[idx].get('prev') == -1:
                curr_sequence += 1
            res.append(curr_sequence)
        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            bin_counts = np.bincount(self.flag)
            new_flags = []
            curr_new_flag = 0
            for curr_flag in range(len(bin_counts)):
                seq_length = int(round(bin_counts[curr_flag] / self.seq_split_num))
                bounds = list(range(0, bin_counts[curr_flag], seq_length)) + [bin_counts[curr_flag]]
                # a single leftover sample joins the last sub-sequence
                if bounds[-1] - bounds[-2] <= 1:
                    bounds = bounds[:-2] + [bounds[-1]]
                bounds = np.array(bounds)
                for sub_seq_len in (bounds[1:] - bounds[:-1]):
                    new_flags.extend([curr_new_flag] * sub_seq_len)
                    curr_new_flag += 1
            assert len(new_flags) == len(self.flag)
            self.flag = np.array(new_flags, dtype=np.int64)

    def get_sample(self, idx):
        """Sample info with its map geometries in the (lidar-centered) ego frame."""
        sample = self.samples[idx]
        location = sample['location']

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = sample['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        ego2global[:3, 3] = sample['e2g_translation']

        # The map is queried at the lidar position, as in MapTR.
        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = [float(x) for x in lidar2global[:3, 3]]
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        map_geoms = self.map_extractor.get_map_geom(
            location, lidar2global_translation, lidar2global_rotation)

        lidar_shifted_e2g_translation = np.array(sample['e2g_translation'])
        lidar_shifted_e2g_translation[0] = lidar2global_translation[0]
        lidar_shifted_e2g_translation[1] = lidar2global_translation[1]
        lidar_shifted_e2g_translation = lidar_shifted_e2g_translation.tolist()
        e2g_rotation = sample['e2g_rotation']

        lidar2global = np.eye(4)
        lidar2global[:3, :3] = Quaternion(e2g_rotation).rotation_matrix
        lidar2global[:3, 3] = lidar_shifted_e2g_translation
        ego2lidar = np.linalg.inv(lidar2global) @ ego2global

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v

        ego2img_rts = []
        ego2cam_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(c['extrinsics']), np.array(c['intrinsics'])
            cam2lidar_rt = ego2lidar @ np.linalg.inv(extrinsic)
            ego2cam_rt = np.linalg.inv(cam2lidar_rt)
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rts.append(ego2cam_rt)
            ego2img_rts.append(viewpad @ ego2cam_rt)

        return {
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'ego2cam': ego2cam_rts,
            'map_geoms': map_label2geom,  # {label: [LineString]}
            'ego2global_translation': lidar_shifted_e2g_translation,
            'ego2global_rotation': Quaternion(e2g_rotation).rotation_matrix.tolist(),
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar2ego_translation': sample['lidar2ego_translation'],
            'lidar2ego_rotation': sample['lidar2ego_rotation'],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_dict = self.get_sample(idx)
        data = self.pipeline(input_dict)
        data['seq_info'] = DC(self.idx2scene[idx], cpu_only=True)

        if self.multi_frame:
            # Training clip: the current frame plus multi_frame - 1 earlier
            # frames drawn from the preceding sampling_span samples.
            scene_seq_info = self.scene_name2idx[input_dict['scene_name']]
            local_idx_curr = input_dict['sample_idx'] - scene_seq_info[0]

            span = max(self.sampling_span, self.multi_frame)
            min_idx = local_idx_curr - span
            sampled_indices = sorted(
                np.random.choice(span, self.multi_frame - 1, replace=False).tolist())
            local_indices_prev = [max(min_idx + x, 0) for x in sampled_indices]
            data['img_metas'].data['local_idx'] = local_idx_curr

            all_prev_data = []
            for local_idx_prev in local_indices_prev:
                data_prev = self.pipeline(self.get_sample(local_idx_prev + scene_seq_info[0]))
                data_prev['img_metas'].data['local_idx'] = local_idx_prev
                all_prev_data.append(data_prev)
            data['all_prev_data'] = all_prev_data

        return data

    def format_results(self, results, prefix, save_semantic=False):
        """Write predictions (normalized -> meters) to <prefix>/submission_vector.json."""
        submissions = {'meta': self.meta, 'results': {}}
        roi_size = np.array(self.roi_size)
        origin = -np.array([self.roi_size[0] / 2, self.roi_size[1] / 2])
        for pred in results:
            single_case = {'vectors': [], 'scores': [], 'labels': []}
            if save_semantic:
                single_case['semantic_mask'] = pred['semantic_mask'].tolist()
            for vector, score, label in zip(pred['vectors'], pred['scores'], pred['labels']):
                if len(vector) < 2:
                    continue
                single_case['vectors'].append(vector * (roi_size + 1e-5) + origin)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            submissions['results'][pred['token']] = single_case

        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(prefix, exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def evaluate(self, results, logger=None, **kwargs):
        save_semantic = self.eval_semantic or kwargs.get('save_semantic', False)
        result_path = self.format_results(results, self.work_dir, save_semantic=save_semantic)
        return self.evaluate_file(result_path, logger=logger)

    def evaluate_file(self, result_path, logger=None):
        if self.eval_semantic:
            evaluator = RasterEvaluate(self.eval_config)
        else:
            evaluator = VectorEvaluate(self.eval_config)
        return evaluator.evaluate(result_path, logger=logger)
