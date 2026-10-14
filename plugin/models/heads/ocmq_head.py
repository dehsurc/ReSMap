"""OCMQ vector head: object queries decoded against camera, satellite and fused BEV."""
import copy

import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, Linear, bias_init_with_prob, xavier_init
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.core import build_assigner, reduce_mean
from mmdet.models import build_loss

from ..transformer_utils.ocmq_decoder import OCMQDecoder


class OCMQHead(nn.Module):
    """
    Args:
        num_classes (int): number of map element classes.
        in_channels (int): BEV feature channels.
        embed_dims (int): query dimension.
        num_points (int): points per polyline.
        num_queries (int): number of object queries.
        roi_size (tuple): BEV range in meters.
        decoder_cfg (dict): OCMQDecoder arguments.
        loss_cls, loss_reg, assigner (dict): loss and matching configs.
        sync_cls_avg_factor (bool): average the cls normalizer across GPUs.
        bg_cls_weight (float): weight of background queries in the normalizer.
        different_heads (bool): separate prediction branches per layer.
    """

    def __init__(self,
                 num_classes=3,
                 in_channels=256,
                 embed_dims=512,
                 num_points=20,
                 num_queries=100,
                 roi_size=(60, 30),
                 decoder_cfg=dict(),
                 loss_cls=dict(),
                 loss_reg=dict(),
                 assigner=dict(),
                 sync_cls_avg_factor=True,
                 bg_cls_weight=0.,
                 different_heads=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.num_queries = num_queries
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight

        # Kept as buffers so that released checkpoints load without missing keys.
        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0] / 2, -roi_size[1] / 2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))

        self.cls_out_channels = num_classes if loss_cls.get('use_sigmoid', False) \
            else num_classes + 1

        self.obj_query_embed = nn.Embedding(num_queries, embed_dims)
        self.obj_ref_pts_embed = nn.Linear(embed_dims, num_points * 2)

        self.cam_input_proj = Conv2d(in_channels, embed_dims, kernel_size=1)
        self.sat_input_proj = Conv2d(in_channels, embed_dims, kernel_size=1)
        self.fuse_input_proj = Conv2d(in_channels, embed_dims, kernel_size=1)
        self.bev_pos_embed = build_positional_encoding(
            dict(type='SinePositionalEncoding', num_feats=embed_dims // 2, normalize=True))

        self.decoder = OCMQDecoder(embed_dims=embed_dims, num_points=num_points, **decoder_cfg)

        cls_branch = Linear(embed_dims, self.cls_out_channels)
        reg_branch = nn.Sequential(
            Linear(embed_dims, 2 * embed_dims),
            nn.LayerNorm(2 * embed_dims),
            nn.ReLU(),
            Linear(2 * embed_dims, 2 * embed_dims),
            nn.LayerNorm(2 * embed_dims),
            nn.ReLU(),
            Linear(2 * embed_dims, num_points * 2),
        )
        num_layers = self.decoder.num_layers
        if different_heads:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(cls_branch) for _ in range(num_layers)])
            self.reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_layers)])
        else:
            self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_layers)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_layers)])

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.assigner = build_assigner(assigner)

        self.init_weights()

    def init_weights(self):
        for proj in [self.cam_input_proj, self.sat_input_proj, self.fuse_input_proj]:
            for p in proj.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        xavier_init(self.obj_ref_pts_embed, distribution='uniform', bias=0.)
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)

    def _bev_tokens(self, bev, proj):
        B, _, H, W = bev.shape
        pos = self.bev_pos_embed(bev.new_zeros(B, H, W))
        tokens = (proj(bev) + pos).flatten(2).transpose(1, 2)
        spatial_shapes = torch.tensor([[H, W]], device=bev.device)
        level_start = torch.tensor([0], device=bev.device)
        return tokens, spatial_shapes, level_start

    def forward(self, cam_bev, sat_bev, fused_bev, return_loss=True, gts=None,
                drop_groups=None):
        """
        Args:
            cam_bev, sat_bev, fused_bev: (B, C, H, W) BEV features.
            gts (dict): 'labels' list[(M,)] and 'lines' list[(M, K, P*2)].
            drop_groups (list, optional): per-sample evidence to drop in the
                gate ('cam' / 'sat' / 'fuse' / None), training only.
        """
        B = cam_bev.shape[0]
        query = self.obj_query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        ref_pts = self.obj_ref_pts_embed(query).sigmoid().view(
            B, self.num_queries, self.num_points, 2)

        inter_queries, inter_ref_pts = self.decoder(
            query, ref_pts, self.reg_branches,
            self._bev_tokens(cam_bev, self.cam_input_proj),
            self._bev_tokens(sat_bev, self.sat_input_proj),
            self._bev_tokens(fused_bev, self.fuse_input_proj),
            drop_groups=drop_groups)

        all_cls_scores = []
        all_lines_preds = []
        for layer_idx, (q, ref) in enumerate(zip(inter_queries, inter_ref_pts)):
            all_cls_scores.append(list(self.cls_branches[layer_idx](q)))
            all_lines_preds.append(list(ref.flatten(2)))

        outputs = dict(lines=all_lines_preds[-1], scores=all_cls_scores[-1])
        if not return_loss:
            return outputs

        loss_dict = {}
        num_layers = len(all_cls_scores)
        for layer_idx in range(num_layers):
            layer_loss = self.loss_single(
                all_lines_preds[layer_idx], all_cls_scores[layer_idx], gts)
            prefix = '' if layer_idx == num_layers - 1 else f'd{layer_idx}.'
            for k, v in layer_loss.items():
                loss_dict[prefix + k] = v
        return loss_dict, outputs

    def loss_single(self, lines_preds, cls_scores, gts):
        """Hungarian-matched cls + line regression loss for one decoder layer."""
        all_labels, all_label_weights = [], []
        all_lines_targets, all_lines_weights = [], []
        num_total_pos = 0
        num_total_neg = 0

        for pred_lines, pred_scores, gt_labels, gt_lines in zip(
                lines_preds, cls_scores, gts['labels'], gts['lines']):
            num_queries = pred_lines.shape[0]
            labels = gt_labels.new_full((num_queries,), self.num_classes)
            label_weights = pred_lines.new_ones((num_queries,))
            lines_targets = pred_lines.new_zeros(pred_lines.shape)
            lines_weights = pred_lines.new_zeros(pred_lines.shape)

            if gt_labels.shape[0] == 0:
                num_total_neg += num_queries
            else:
                assigned_gt_inds, gt_permute_idx = self.assigner.assign(
                    dict(lines=pred_lines, scores=pred_scores),
                    dict(lines=gt_lines, labels=gt_labels))
                pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
                neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1)
                pos_gt_inds = assigned_gt_inds[pos_inds] - 1

                labels[pos_inds] = gt_labels[pos_gt_inds]
                if len(pos_inds) > 0:
                    # Regress to the best-matching permutation of each GT line.
                    lines_targets[pos_inds] = gt_lines[
                        pos_gt_inds, gt_permute_idx[pos_inds, pos_gt_inds]]
                    lines_weights[pos_inds] = 1.0
                num_total_pos += len(pos_inds)
                num_total_neg += len(neg_inds)

            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_lines_targets.append(lines_targets)
            all_lines_weights.append(lines_weights)

        pred_scores = torch.cat(cls_scores, dim=0)
        pred_lines = torch.cat(lines_preds, dim=0)

        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(pred_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_scores, torch.cat(all_labels, dim=0), torch.cat(all_label_weights, dim=0),
            avg_factor=cls_avg_factor)

        num_pos = pred_lines.new_tensor([num_total_pos])
        num_pos = torch.clamp(reduce_mean(num_pos), min=1).item()
        loss_reg = self.loss_reg(
            pred_lines, torch.cat(all_lines_targets, dim=0), torch.cat(all_lines_weights, dim=0),
            avg_factor=num_pos)

        return dict(cls=loss_cls, reg=loss_reg)
