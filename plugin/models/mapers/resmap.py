"""ReSMap: camera-satellite fusion for online HD map construction.

The camera stream is a temporal BEVFormer encoder with a short BEV memory
(adapted from MapTracker / StreamMapNet). The satellite stream encodes an
ego-aligned aerial tile into the same BEV grid. The two streams are combined
by Predict-then-Fuse (PTF) and decoded into vectors by the OCMQ head.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head

from ..heads import OCMQHead


def predict_then_fuse(cam_bev, sat_bev, cam_rel, sat_rel, eps=1e-6):
    """Reliability-weighted average of the two BEV maps (Eq. 2-3)."""
    return (cam_rel * cam_bev + sat_rel * sat_bev) / (cam_rel + sat_rel + eps)


@DETECTORS.register_module()
class ReSMap(nn.Module):
    """
    Args:
        bev_h, bev_w (int): BEV grid size.
        roi_size (tuple): BEV range in meters (x, y).
        backbone_cfg (dict): temporal camera BEV encoder.
        seg_cfg (dict): BEV segmentation head, instantiated three times
            (camera, satellite, fused).
        sat_encoder_cfg (dict): satellite encoder.
        mm_head_cfg (dict, optional): OCMQ vector head. None in stage 1.
        history_steps (int): BEV memory length during training.
        test_time_history_steps (int): BEV memory bank size at test time.
        mem_select_dist_ranges (list): distance bins for picking memory
            entries from the test-time bank.
        freeze_bev (bool): freeze the camera encoder and camera seg head.
        freeze_sat_encoder (bool): freeze the satellite encoder.
        use_modality_masking (bool): randomly zero / perturb the BEV inputs
            of the fused branch during training.
        mask_*_prob (float): probabilities of each masking type.
        scene_drop_probs (tuple): per-scene probability of dropping the
            (camera, satellite, fused) evidence in the OCMQ gate.
    """

    def __init__(self,
                 bev_h,
                 bev_w,
                 roi_size,
                 backbone_cfg,
                 seg_cfg,
                 sat_encoder_cfg,
                 mm_head_cfg=None,
                 history_steps=4,
                 test_time_history_steps=20,
                 mem_select_dist_ranges=(1, 5, 10, 15),
                 freeze_bev=False,
                 freeze_sat_encoder=False,
                 use_modality_masking=False,
                 mask_cam_zero_prob=0.10,
                 mask_cam_noise_prob=0.10,
                 mask_cam_block_prob=0.05,
                 mask_sat_zero_prob=0.05,
                 mask_noise_scale=1.0,
                 scene_drop_probs=(0.3, 0.3, 0.1),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.seg_decoder = build_head(seg_cfg)

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size
        self.history_steps = history_steps
        self.test_time_history_steps = test_time_history_steps
        self.mem_select_dist_ranges = mem_select_dist_ranges
        self.freeze_bev = freeze_bev

        xmin, xmax = -roi_size[0] / 2, roi_size[0] / 2
        ymin, ymax = -roi_size[1] / 2, roi_size[1] / 2
        x = torch.linspace(xmin, xmax, bev_w)
        y = torch.linspace(ymax, ymin, bev_h)
        y, x = torch.meshgrid(y, x)
        z = torch.zeros_like(x)
        ones = torch.ones_like(x)
        plane = torch.stack([x, y, z, ones], dim=-1)
        self.register_buffer('plane', plane.double())

        self.sat_encoder = build_backbone(sat_encoder_cfg)
        if freeze_sat_encoder:
            self.sat_encoder.freeze()

        # seg_decoder: camera branch, seg_decoder_sat: satellite branch,
        # seg_decoder_fused: PTF-fused BEV.
        self.seg_decoder_fused = build_head(seg_cfg)
        self.seg_decoder_sat = build_head(seg_cfg)

        self.use_modality_masking = use_modality_masking
        self.mask_cam_zero_prob = mask_cam_zero_prob
        self.mask_cam_noise_prob = mask_cam_noise_prob
        self.mask_cam_block_prob = mask_cam_block_prob
        self.mask_sat_zero_prob = mask_sat_zero_prob
        self.mask_noise_scale = mask_noise_scale

        self.mm_head = None
        if mm_head_cfg is not None:
            self.mm_head = OCMQHead(**mm_head_cfg)
        self.scene_drop_probs = scene_drop_probs

    def init_weights(self):
        pass

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        kwargs.pop('rescale', None)
        return self.forward_test(*args, **kwargs)

    def train_step(self, data_dict, optimizer):
        loss, log_vars, num_samples = self(**data_dict)
        return dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

    def train(self, mode=True):
        super().train(mode)
        for param in (list(self.backbone.parameters())
                      + list(self.seg_decoder.parameters())):
            param.requires_grad = not self.freeze_bev
        return self

    # ------------------------------------------------------------------
    # Camera BEV with temporal memory
    # ------------------------------------------------------------------

    def history_coords(self, img_metas, history_img_metas):
        """Sampling grids that warp each memory entry into the current frame."""
        all_history_coord = []
        if len(history_img_metas) == 0:
            return all_history_coord

        for b_i in range(len(img_metas)):
            history_e2g_trans = torch.stack([
                self.plane.new_tensor(prev[b_i]['ego2global_translation'], dtype=torch.float64)
                for prev in history_img_metas], dim=0)
            history_e2g_rot = torch.stack([
                self.plane.new_tensor(prev[b_i]['ego2global_rotation'], dtype=torch.float64)
                for prev in history_img_metas], dim=0)
            curr_e2g_trans = self.plane.new_tensor(
                img_metas[b_i]['ego2global_translation'], dtype=torch.float64)
            curr_e2g_rot = self.plane.new_tensor(
                img_metas[b_i]['ego2global_rotation'], dtype=torch.float64)

            eye = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
            history_g2e_matrix = torch.stack([eye] * len(history_e2g_trans), dim=0)
            history_g2e_matrix[:, :3, :3] = torch.transpose(history_e2g_rot, 1, 2)
            history_g2e_matrix[:, :3, 3] = -torch.bmm(
                torch.transpose(history_e2g_rot, 1, 2), history_e2g_trans[..., None]).squeeze(-1)

            curr_e2g_matrix = eye.clone()
            curr_e2g_matrix[:3, :3] = curr_e2g_rot
            curr_e2g_matrix[:3, 3] = curr_e2g_trans

            curr2prev = torch.bmm(
                history_g2e_matrix,
                repeat(curr_e2g_matrix, 'n1 n2 -> r n1 n2', r=len(history_g2e_matrix)))
            coord = torch.einsum('nlk,ijk->nijl', curr2prev, self.plane).float()[..., :2]

            # metric -> [-1, 1]
            coord[..., 0] = coord[..., 0] / (self.roi_size[0] / 2)
            coord[..., 1] = -coord[..., 1] / (self.roi_size[1] / 2)
            all_history_coord.append(coord)

        return all_history_coord

    def select_memory_entries(self, history_metas, curr_meta):
        """Pick `history_steps` entries from the test-time memory bank by distance."""
        if len(history_metas) <= self.history_steps:
            return np.arange(len(history_metas))

        history_e2g_trans = np.array(
            [item[0]['ego2global_translation'] for item in history_metas])[:, :2]
        curr_e2g_trans = np.array(curr_meta[0]['ego2global_translation'])[:2]
        dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)

        sorted_indices = np.argsort(dists)
        sorted_dists = dists[sorted_indices]
        covered = np.zeros_like(sorted_indices).astype(bool)
        selected_ids = []
        for dist_range in self.mem_select_dist_ranges[::-1]:
            outer_valid = (sorted_dists >= dist_range) & ~covered
            if outer_valid.any():
                pick_id = np.where(outer_valid)[0][0]
                covered[pick_id:] = True
            else:
                inner_valid = (sorted_dists < dist_range) & ~covered
                if inner_valid.any():
                    pick_id = np.where(inner_valid)[0][-1]
                    covered[pick_id] = True
                else:
                    return np.arange(len(history_metas))[-4:]
            selected_ids.append(pick_id)

        return sorted_indices[np.array(selected_ids)]

    # ------------------------------------------------------------------
    # Satellite stream and fusion
    # ------------------------------------------------------------------

    def encode_satellite(self, sat_img):
        # H-flip to match the BEVFormer y-axis convention.
        return torch.flip(self.sat_encoder(sat_img), [2])

    def reliability(self, seg_decoder, bev):
        """Per-pixel max-sigmoid confidence of a branch seg head, at BEV size."""
        with torch.no_grad():
            logits, _ = seg_decoder(bev_features=bev, return_loss=False)
            conf = logits.sigmoid().max(dim=1, keepdim=True)[0]
            return F.interpolate(conf, size=bev.shape[2:], mode='bilinear',
                                 align_corners=False)

    def fuse(self, cam_bev, sat_bev, cam_bev_for_rel=None, sat_bev_for_rel=None):
        """PTF. Reliability can be taken from the unmasked BEV during training."""
        if cam_bev_for_rel is None:
            cam_bev_for_rel = cam_bev
        if sat_bev_for_rel is None:
            sat_bev_for_rel = sat_bev
        cam_rel = self.reliability(self.seg_decoder, cam_bev_for_rel)
        sat_rel = self.reliability(self.seg_decoder_sat, sat_bev_for_rel)
        return predict_then_fuse(cam_bev, sat_bev, cam_rel, sat_rel)

    def apply_modality_masking(self, cam_bev, sat_bev):
        """Simulate sensor failure on the fused-branch inputs (training only)."""
        if not self.training or not self.use_modality_masking:
            return cam_bev, sat_bev

        p = random.random()
        cumulative = self.mask_cam_zero_prob
        if p < cumulative:
            return torch.zeros_like(cam_bev), sat_bev

        cumulative += self.mask_cam_noise_prob
        if p < cumulative:
            noise = torch.randn_like(cam_bev) * self.mask_noise_scale
            return cam_bev + noise, sat_bev

        cumulative += self.mask_cam_block_prob
        if p < cumulative:
            B, C, H, W = cam_bev.shape
            mask = torch.ones_like(cam_bev)
            y0 = random.randint(0, H // 2)
            x0 = random.randint(0, W // 2)
            mask[:, :, y0:y0 + H // 2, x0:x0 + W // 2] = 0
            return cam_bev * mask, sat_bev

        cumulative += self.mask_sat_zero_prob
        if p < cumulative:
            return cam_bev, torch.zeros_like(sat_bev)

        return cam_bev, sat_bev

    def sample_scene_drops(self, img_metas):
        """One evidence-drop decision per scene in the batch, or None."""
        p_cam, p_sat, p_fuse = self.scene_drop_probs
        decisions = {}
        drops = []
        for meta in img_metas:
            scene = meta['scene_name']
            if scene not in decisions:
                r = random.random()
                if r < p_cam:
                    decisions[scene] = 'cam'
                elif r < p_cam + p_sat:
                    decisions[scene] = 'sat'
                elif r < p_cam + p_sat + p_fuse:
                    decisions[scene] = 'fuse'
                else:
                    decisions[scene] = None
            drops.append(decisions[scene])
        return drops if any(d is not None for d in drops) else None

    # ------------------------------------------------------------------
    # Train / test
    # ------------------------------------------------------------------

    @staticmethod
    def vector_targets(vectors, device):
        """Collate per-sample GT polylines into label / line tensors."""
        labels_list, lines_list = [], []
        for sample in vectors:
            labels, lines = [], []
            for label, sample_lines in sample.items():
                for line in sample_lines:
                    labels.append(label)
                    # (num_permute, num_points, 2) -> (num_permute, num_points * 2)
                    lines.append(torch.tensor(line).flatten(-2))
            assert len(labels) > 0, 'every training sample must have GT vectors'
            labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            lines_list.append(torch.stack(lines).float().to(device))
        return dict(labels=labels_list, lines=lines_list)

    def forward_train(self, img, vectors, semantic_mask, img_metas, sat_img,
                      all_prev_data=None, **kwargs):
        """
        Args:
            img: (B, N, 3, H, W) current-frame camera images.
            vectors: per-sample GT polylines, {label: [array]}.
            semantic_mask: (B, K, H, W) BEV semantic GT of the current frame.
            img_metas: per-sample meta dicts.
            sat_img: (B, 3, H_s, W_s) current-frame satellite tile.
            all_prev_data: earlier frames of the clip, each a dict with
                'img', 'img_metas' and 'semantic_mask'.
        """
        num_prev_frames = len(all_prev_data) if all_prev_data is not None else 0
        # Only the first and the last frame backprop into the image backbone.
        backprop_backbone_ids = [0, num_prev_frames]

        history_bev_feats = []
        history_img_metas = []
        frame_losses = []
        frames = list(all_prev_data or []) + [
            dict(img=img, img_metas=img_metas, semantic_mask=semantic_mask)]

        # Camera stream over the clip, with a dense seg loss on every frame.
        for t, frame in enumerate(frames):
            history_coord = self.history_coords(frame['img_metas'], history_img_metas)
            cam_bev, _ = self.backbone(
                frame['img'], frame['img_metas'], t, history_bev_feats,
                history_img_metas, history_coord,
                img_backbone_gradient=(t in backprop_backbone_ids))
            gt_semantic = torch.flip(frame['semantic_mask'], [2])
            _, _, seg_loss, seg_dice_loss = self.seg_decoder(
                cam_bev, gt_semantic, return_loss=True)
            frame_losses.append(seg_loss + seg_dice_loss)

            history_bev_feats.append(cam_bev)
            history_img_metas.append(frame['img_metas'])
            if len(history_bev_feats) > self.history_steps:
                history_bev_feats.pop(0)
                history_img_metas.pop(0)

        loss = 0
        for frame_loss in frame_losses:
            loss += frame_loss
        log_vars = {'seg': seg_loss.item(), 'seg_dice': seg_dice_loss.item()}

        # Satellite and fused branches, current frame only.
        sat_bev = self.encode_satellite(sat_img)
        _, _, seg_loss_sat, seg_dice_sat = self.seg_decoder_sat(
            sat_bev, gt_semantic, return_loss=True)
        loss = loss + seg_loss_sat + seg_dice_sat
        log_vars['seg_sat'] = seg_loss_sat.item()
        log_vars['seg_dice_sat'] = seg_dice_sat.item()

        cam_bev_masked, sat_bev_masked = self.apply_modality_masking(cam_bev, sat_bev)
        fused_bev = self.fuse(cam_bev_masked, sat_bev_masked,
                              cam_bev_for_rel=cam_bev, sat_bev_for_rel=sat_bev)
        _, _, seg_loss_fused, seg_dice_fused = self.seg_decoder_fused(
            fused_bev, gt_semantic, return_loss=True)
        loss = loss + seg_loss_fused + seg_dice_fused
        log_vars['seg_fused'] = seg_loss_fused.item()
        log_vars['seg_dice_fused'] = seg_dice_fused.item()

        if self.mm_head is not None:
            gts = self.vector_targets(vectors, img.device)
            gated_bev = self.fuse(cam_bev, sat_bev)
            mm_losses, _ = self.mm_head(
                cam_bev, sat_bev, gated_bev, return_loss=True, gts=gts,
                drop_groups=self.sample_scene_drops(img_metas))
            for k, v in mm_losses.items():
                loss = loss + v
                log_vars[f'mm_{k}'] = v.item()

        log_vars['total'] = loss.item()
        return loss, log_vars, img.size(0)

    @staticmethod
    def seg_to_mask(seg_preds, thr=0.4):
        scores, labels = seg_preds[0].max(0)
        scores = scores.sigmoid()
        mask = torch.zeros(labels.shape, dtype=torch.uint8, device=labels.device)
        pos = scores >= thr
        mask[pos] = labels[pos].type(torch.uint8) + 1
        return mask.cpu().numpy()

    @torch.no_grad()
    def forward_test(self, img, img_metas, sat_img, seq_info, **kwargs):
        assert img.shape[0] == 1, 'only batch size 1 per GPU is supported at test time'

        scene_name, local_idx, seq_length = seq_info[0]
        img_metas[0]['local_idx'] = local_idx
        if local_idx == 0:
            self.history_bev_feats_all = []
            self.history_img_metas_all = []

        selected = self.select_memory_entries(self.history_img_metas_all, img_metas)
        history_img_metas = [self.history_img_metas_all[i] for i in selected]
        history_bev_feats = [self.history_bev_feats_all[i] for i in selected]
        history_coord = self.history_coords(img_metas, history_img_metas)

        cam_bev, _ = self.backbone(img, img_metas, local_idx, history_bev_feats,
                                   history_img_metas, history_coord)

        self.history_bev_feats_all.append(cam_bev)
        self.history_img_metas_all.append(img_metas)
        if len(self.history_bev_feats_all) > self.test_time_history_steps:
            self.history_bev_feats_all.pop(0)
            self.history_img_metas_all.pop(0)

        sat_bev = self.encode_satellite(sat_img)
        fused_bev = self.fuse(cam_bev, sat_bev)
        seg_preds, _ = self.seg_decoder_fused(bev_features=fused_bev, return_loss=False)

        result = dict(token=img_metas[0]['token'], semantic_mask=self.seg_to_mask(seg_preds),
                      vectors=[], scores=[], labels=[])

        if self.mm_head is not None:
            outputs = self.mm_head(cam_bev, sat_bev, fused_bev, return_loss=False)
            lines = outputs['lines'][0]
            scores, labels = outputs['scores'][0].sigmoid().max(dim=-1)
            for i in range(lines.shape[0]):
                result['vectors'].append(lines[i].view(-1, 2).cpu().numpy())
                result['scores'].append(scores[i].item())
                result['labels'].append(labels[i].item())

        return [result]
