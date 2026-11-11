"""
Distillation losses for the compact student. The model outputs LOGITS.

  L_gt   : BCE-with-logits + SoftIoU vs. the GT mask
           (SoftIoU handles the extreme target/background imbalance; the old
            pos_weight=100 BCE pushed the net to over-predict blobs, which is
            what kept precision below 0.5)
  L_pred : MSE between the student heatmap (max-pooled to CHAL's stride-8
           grid) and CHAL's confidence map (only with a trained teacher)
  L_resp : response consistency, MSE(sigmoid(student(x_aug)), sigmoid(student(x)))

  L = λ_gt * L_gt + λ_pred * L_pred + λ_resp * L_resp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LAMBDA_GT, LAMBDA_PRED, LAMBDA_RESP


class SoftIoULoss(nn.Module):
    """Batch-level soft IoU (stable when a crop contains no target)."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob, target):
        inter = (prob * target).sum()
        union = prob.sum() + target.sum() - inter
        return 1.0 - (inter + self.smooth) / (union + self.smooth)


class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.iou = SoftIoULoss()

    def forward(self,
                student_logit: torch.Tensor,        # (B,1,H,W)
                mask: torch.Tensor,                 # (B,1,H,W) binary GT
                student_aug_logit: torch.Tensor,    # (B,1,H,W) on augmented clip
                teacher_heatmap=None,               # (B,1,H,W) in [0,1] or None
                ) -> dict:
        prob = torch.sigmoid(student_logit)

        l_bce = F.binary_cross_entropy_with_logits(student_logit, mask)
        l_iou = self.iou(prob, mask)
        l_gt  = l_bce + l_iou

        if teacher_heatmap is not None:
            t = teacher_heatmap.detach()
            # CHAL predicts on a stride-8 grid: compare on that grid, taking the
            # max student response inside each cell
            s = (prob if prob.shape[-2:] == t.shape[-2:] else
                 F.adaptive_max_pool2d(prob, t.shape[-2:]))
            l_pred = F.mse_loss(s, t)
        else:
            l_pred = torch.zeros((), device=student_logit.device)

        l_resp = F.mse_loss(torch.sigmoid(student_aug_logit), prob.detach())

        total = LAMBDA_GT * l_gt + LAMBDA_PRED * l_pred + LAMBDA_RESP * l_resp
        return {
            "total" : total,
            "l_gt"  : l_gt.item(),
            "l_pred": l_pred.item(),
            "l_resp": l_resp.item(),
        }
