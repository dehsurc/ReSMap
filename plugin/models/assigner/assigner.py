import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import smooth_l1_loss


@MATCH_COST.register_module()
class LinesL1Cost:
    """Permutation-invariant (smooth) L1 cost between predicted and GT polylines.

    Args:
        weight (float): cost weight.
        beta (float): smooth-L1 beta; 0 gives plain L1.
    """

    def __init__(self, weight=1.0, beta=0.0, permute=True):
        assert permute, 'GT lines are always given with all their permutations'
        self.weight = weight
        self.beta = beta

    def __call__(self, lines_pred, gt_lines):
        """
        Args:
            lines_pred: (num_pred, P * 2) normalized polylines.
            gt_lines: (num_gt, num_permute, P * 2).
        Returns:
            cost (num_pred, num_gt) and the best permutation index (num_pred, num_gt).
        """
        num_pred, num_gt = len(lines_pred), len(gt_lines)
        gt_lines = gt_lines.flatten(0, 1)
        num_pts = lines_pred.shape[-1] // 2

        if self.beta > 0:
            lines_pred = lines_pred.unsqueeze(1).repeat(1, len(gt_lines), 1)
            gt_lines = gt_lines.unsqueeze(0).repeat(num_pred, 1, 1)
            dist_mat = smooth_l1_loss(
                lines_pred, gt_lines, reduction='none', beta=self.beta).sum(-1)
        else:
            dist_mat = torch.cdist(lines_pred, gt_lines, p=1)
        dist_mat = dist_mat / num_pts

        dist_mat = dist_mat.view(num_pred, num_gt, -1)
        dist_mat, gt_permute_idx = torch.min(dist_mat, 2)
        return dist_mat * self.weight, gt_permute_idx


@BBOX_ASSIGNERS.register_module()
class HungarianLinesAssigner:
    """One-to-one Hungarian matching between query predictions and GT lines.

    Args:
        cls_cost (dict): classification cost config.
        reg_cost (dict): line regression cost config.
    """

    def __init__(self, cls_cost, reg_cost):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)

    @torch.no_grad()
    def assign(self, preds, gts):
        """
        Args:
            preds (dict): 'lines' (num_pred, P * 2), 'scores' (num_pred, C) logits.
            gts (dict): 'lines' (num_gt, num_permute, P * 2), 'labels' (num_gt,).
        Returns:
            assigned_gt_inds (num_pred,): 0 for background, otherwise the
                1-based index of the matched GT.
            gt_permute_idx (num_pred, num_gt): best permutation of each GT.
        """
        num_preds = preds['lines'].size(0)
        assigned_gt_inds = preds['lines'].new_zeros((num_preds,), dtype=torch.long)

        reg_cost, gt_permute_idx = self.reg_cost(preds['lines'], gts['lines'])
        cost = self.cls_cost(preds['scores'], gts['labels']) + reg_cost

        row_inds, col_inds = linear_sum_assignment(cost.cpu().numpy())
        row_inds = torch.from_numpy(row_inds).to(preds['lines'].device)
        col_inds = torch.from_numpy(col_inds).to(preds['lines'].device)
        assigned_gt_inds[row_inds] = col_inds + 1
        return assigned_gt_inds, gt_permute_idx
