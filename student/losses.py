"""
Distillation losses for the compact student.

Three components:
  L_gt   : binary cross-entropy with GT mask (always used)
  L_pred : MSE between student and teacher heatmaps (when teacher available)
  L_resp : response-consistency under label-preserving augmentation
            ||student(x_aug) - student(x)||_2  (encourages causal invariance)

Total loss:
  L = λ_gt * L_gt + λ_pred * L_pred + λ_resp * L_resp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LAMBDA_GT, LAMBDA_PRED, LAMBDA_RESP


# weighted BCE to handle extreme class imbalance (target pixels << background)
class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: float = 100.0):
        super().__init__()
        self.pw = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor(self.pw, device=pred.device)
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pw)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        inter  = (pred * target).sum(dim=1)
        dice   = (2. * inter + self.smooth) / \
                 (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return 1. - dice.mean()


class DistillationLoss(nn.Module):
    def __init__(self, pos_weight: float = 100.0):
        super().__init__()
        self.bce  = WeightedBCE(pos_weight)
        self.dice = DiceLoss()

    def forward(self,
                student_logit: torch.Tensor,   # (B,1,H,W) pre-sigmoid
                mask: torch.Tensor,            # (B,1,H,W) binary GT
                student_heatmap: torch.Tensor, # (B,1,H,W) sigmoid output
                student_aug: torch.Tensor,     # (B,1,H,W) student on aug clip
                teacher_heatmap=None,          # (B,1,H,W) or None
                ) -> dict:

        # GT supervision
                # GT supervision: BCE + Dice
        l_bce  = self.bce(student_logit, mask)
        l_dice = self.dice(student_heatmap, mask)
        l_gt   = l_bce + l_dice

        # Prediction matching (teacher soft labels)
        if teacher_heatmap is not None:
            l_pred = F.mse_loss(student_heatmap, teacher_heatmap.detach())
        else:
            l_pred = torch.tensor(0.0, device=student_logit.device)

        # Response consistency (augmentation invariance)
        l_resp = F.mse_loss(student_aug, student_heatmap.detach())

        total = (LAMBDA_GT   * l_gt
               + LAMBDA_PRED * l_pred
               + LAMBDA_RESP * l_resp)

        return {
            "total" : total,
            "l_gt"  : l_gt.item(),
            "l_pred": l_pred.item(),
            "l_resp": l_resp.item(),
        }
