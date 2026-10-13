import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES, build_backbone


@BACKBONES.register_module()
class SatelliteEncoder(nn.Module):
    """Satellite tile -> BEV feature map on the camera BEV grid.

    A ResNet whose stages are summed FPN-style at the finest scale, then
    resized to the BEV size.

    Args:
        backbone_cfg (dict): image backbone, e.g. ResNet-50.
        neck_in_channels (list): channels of each backbone stage.
        out_channels (int): output channels (= camera BEV channels).
        bev_size (tuple): output size (H, W).
        frozen (bool): freeze all parameters after init.
    """

    def __init__(self,
                 backbone_cfg,
                 neck_in_channels=None,
                 out_channels=256,
                 bev_size=(50, 100),
                 frozen=False):
        super().__init__()
        self.bev_size = bev_size
        self.out_channels = out_channels

        if neck_in_channels is None:
            depth = backbone_cfg.get('depth', 50)
            if depth in (18, 34):
                neck_in_channels = [64, 128, 256, 512]
            else:
                neck_in_channels = [256, 512, 1024, 2048]

        self.backbone = build_backbone(backbone_cfg)

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in neck_in_channels
        ])
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if frozen:
            self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True
        self.eval()

    def train(self, mode=True):
        # a frozen encoder stays in eval mode (BN uses running stats)
        if getattr(self, '_frozen', False):
            return super().train(False)
        return super().train(mode)

    def forward(self, sat_img):
        """(B, 3, H, W) normalized tile -> (B, C, bev_h, bev_w)."""
        feats = self.backbone(sat_img)

        target_size = feats[0].shape[2:]
        out = self.lateral_convs[0](feats[0])
        for i in range(1, len(feats)):
            lateral = self.lateral_convs[i](feats[i])
            lateral = F.interpolate(lateral, size=target_size,
                                    mode='bilinear', align_corners=False)
            out = out + lateral

        out = self.output_conv(out)
        if out.shape[2:] != (self.bev_size[0], self.bev_size[1]):
            out = F.interpolate(out, size=self.bev_size,
                                mode='bilinear', align_corners=False)
        return out
