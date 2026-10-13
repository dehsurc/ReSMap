import numpy as np
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmdet.models import HEADS, build_loss


@HEADS.register_module(force=True)
class MapSegHead(nn.Module):
    """BEV semantic segmentation head, upsampling from the BEV grid to the canvas."""

    def __init__(self,
                 num_classes=3,
                 in_channels=256,
                 embed_dims=256,
                 bev_size=(100, 50),
                 canvas_size=(200, 100),
                 loss_seg=dict(),
                 loss_dice=dict()):
        super().__init__()
        self.loss_seg = build_loss(loss_seg)
        self.loss_dice = build_loss(loss_dice)
        cls_out_channels = num_classes if self.loss_seg.use_sigmoid else num_classes + 1

        assert canvas_size[0] % bev_size[0] == 0, 'canvas size must be a multiple of the bev size'
        num_up_blocks = int(np.log2(canvas_size[0] // bev_size[0]))

        self.conv_in = nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_mid_layers = nn.ModuleList([])
        self.downsample_layers = nn.ModuleList([])
        for _ in range(num_up_blocks):
            self.conv_mid_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ))
            self.downsample_layers.append(nn.Upsample(scale_factor=0.5, mode='bilinear'))
        self.conv_out = nn.Conv2d(embed_dims, cls_out_channels, kernel_size=1, padding=0)

        if self.loss_seg.use_sigmoid:
            nn.init.constant_(self.conv_out.bias, bias_init_with_prob(0.01))

    def forward(self, bev_features, gts=None, return_loss=True):
        """Returns (preds, seg_feats) and, with return_loss, the focal and dice losses."""
        x = self.relu(self.conv_in(bev_features))
        for conv_mid in self.conv_mid_layers:
            x = conv_mid(x)
        preds = self.conv_out(x)

        seg_feats = x
        for downsample in self.downsample_layers:
            seg_feats = downsample(seg_feats)

        if not return_loss:
            return preds, seg_feats
        return preds, seg_feats, self.loss_seg(preds, gts), self.loss_dice(preds, gts)
