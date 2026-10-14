from mmdet.models.builder import LOSSES
from mmdet.models.losses import l1_loss, smooth_l1_loss
from torch import nn


@LOSSES.register_module()
class LinesL1Loss(nn.Module):
    """Smooth-L1 (beta > 0) or L1 loss on flattened polylines, averaged over points."""

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        else:
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)

        num_points = pred.shape[-1] // 2
        return loss / num_points * self.loss_weight
