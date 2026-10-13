from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models import BACKBONES
from mmdet.models.utils import build_transformer
from mmdet3d.models import builder

from .bevformer.grid_mask import GridMask


@BACKBONES.register_module()
class BEVFormerBackbone(nn.Module):
    """Multi-view images -> BEV features, with warped BEV memory from earlier frames.

    Args:
        roi_size (tuple): BEV range in meters (x, y).
        bev_h, bev_w (int): BEV grid size.
        img_backbone, img_neck (dict): image feature extractor.
        transformer (dict): BEVFormer encoder.
        positional_encoding (dict): BEV positional encoding.
        use_grid_mask (bool): GridMask augmentation during training.
        history_steps (int): length of the BEV memory.
    """

    def __init__(self,
                 roi_size,
                 bev_h,
                 bev_w,
                 img_backbone=None, 
                 img_neck=None,               
                 transformer=None,
                 positional_encoding=None,
                 use_grid_mask=True,
                 history_steps=None):
        super(BEVFormerBackbone, self).__init__()

        # image feature
        self.default_ratio = 0.5
        self.default_prob = 0.7
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=self.default_ratio, mode=1, 
                prob=self.default_prob)
        self.use_grid_mask = use_grid_mask

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
            self.with_img_neck = True
        else:
            self.with_img_neck = False

        self.bev_h = bev_h
        self.bev_w = bev_w

        self.real_w = roi_size[0]
        self.real_h = roi_size[1]
    
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.history_steps = history_steps

        self._init_layers()
        self.init_weights()


    def _init_layers(self):
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)


    def init_weights(self):
        self.transformer.init_weights()
        self.img_backbone.init_weights()
        self.img_neck.init_weights()

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def forward(self, img, img_metas, timestep, history_bev_feats, history_img_metas,
                all_history_coord, img_backbone_gradient=True):
        """
        Args:
            img: (B, N, 3, H, W) multi-view images.
            history_bev_feats: list of (B, C, H, W) BEV features from earlier frames.
            all_history_coord: per-sample grids that warp the memory into this frame.
            img_backbone_gradient: if False, the image backbone runs without grad.
        Returns:
            (B, C, bev_h, bev_w) BEV features and the multi-level image features.
        """
        backprop_context = torch.no_grad if img_backbone_gradient is False else nullcontext
        with backprop_context():
            mlvl_feats = self.extract_img_feat(img=img, img_metas=img_metas)

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        # Warp the BEV memory into the current frame.
        if len(history_bev_feats) > 0:
            all_warped_history_feat = []
            for b_i in range(bs):
                history_coord = all_history_coord[b_i]
                history_bev_feats_i = torch.stack([feats[b_i] for feats in history_bev_feats], 0)
                warped_history_feat_i = F.grid_sample(history_bev_feats_i, 
                            history_coord, padding_mode='zeros', align_corners=False)
                all_warped_history_feat.append(warped_history_feat_i)
            all_warped_history_feat = torch.stack(all_warped_history_feat, dim=0) # BTCHW
            prop_bev_feat = all_warped_history_feat[:, -1]
        else:
            all_warped_history_feat = None
            prop_bev_feat = None

        # Zero-pad the memory to a fixed length.
        if len(history_bev_feats) < self.history_steps:
            num_repeat = self.history_steps - len(history_bev_feats)
            zero_bev_feats = torch.zeros([bs, bev_queries.shape[1], self.bev_h, self.bev_w]).to(bev_queries.device)
            padding_history_bev_feats = torch.stack([zero_bev_feats,] * num_repeat, dim=1)
            if all_warped_history_feat is not None:
                all_warped_history_feat = torch.cat([padding_history_bev_feats, all_warped_history_feat], dim=1)
            else:
                all_warped_history_feat = padding_history_bev_feats
        
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outs =  self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
                bev_pos=bev_pos,
                prop_bev=prop_bev_feat,
                img_metas=img_metas,
                warped_history_bev=all_warped_history_feat,
            )
        
        outs = outs.unflatten(1, (self.bev_h, self.bev_w)).permute(0, 3, 1, 2).contiguous()
        return outs, mlvl_feats
