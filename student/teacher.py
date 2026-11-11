"""
Teacher wrapper – loads a TRAINED CHAL model (UESTC-nnLab/CHAL) and exposes
`predict(clip) -> heatmap` for distillation.

Notes on CHAL (checked against the official repo):
  • input  : (B, 3, T, H, W), RGB, normalised with ImageNet mean/std
             (utils/utils.py::preprocess_input), H, W divisible by 32
  • output : list with ONE tensor (B, 4 + 1 + num_classes, H/8, W/8)
             channels = [reg(4), objectness(1), class(num_classes)]
  • The teacher heatmap is sigmoid(obj) * max_c sigmoid(cls) at stride 8.
    It is returned at that native resolution; the loss compares it with the
    student's heatmap max-pooled to the same grid (CHAL is a box detector, so
    its stride-8 confidence must not be upsampled into a fake pixel mask).
  • CHAL/model_data/pretrain.pth is the COCO YOLOX-S initialisation used to
    START training CHAL – it is NOT a teacher and is rejected below.

If no trained CHAL checkpoint exists, DummyTeacher is used (GT-only training).
"""

import os
import sys
from typing import Optional

import torch

from config import CHAL_DIR, TEACHER_CKPT, NUM_FRAMES

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


class DummyTeacher:
    """Stand-in when no CHAL checkpoint is available (pure GT supervision)."""
    def predict(self, clip: torch.Tensor) -> None:
        return None


class CHALTeacher:
    def __init__(self, ckpt_path: str = TEACHER_CKPT, num_classes: int = 1,
                 device: Optional[str] = None):
        if CHAL_DIR not in sys.path:
            sys.path.insert(0, CHAL_DIR)
        try:
            from nets.slowfastnet import slowfastnet  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Could not import CHAL ({e}). The public CHAL repo imports "
                "nets/module/CBAM.py, which it does not ship; add that file "
                f"under {CHAL_DIR}/nets/module/.") from e

        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

        cls_w = state.get("head.cls_preds.0.weight")
        if (not any(k.startswith("neck.") for k in state)
                or cls_w is None or cls_w.shape[0] != num_classes):
            raise RuntimeError(
                f"{ckpt_path} is not a trained CHAL checkpoint (no CHAL neck "
                "weights and/or a non-1-class head). model_data/pretrain.pth is "
                "only the COCO YOLOX initialisation – train CHAL first and point "
                "TEACHER_CKPT at its best weights.")

        self.model = slowfastnet(num_classes=num_classes, num_frame=NUM_FRAMES)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Teacher checkpoint is missing {len(missing)} "
                               f"weights, e.g. {missing[:3]}")
        if unexpected:
            print(f"[teacher] ignoring {len(unexpected)} unexpected keys")
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, clip: torch.Tensor) -> torch.Tensor:
        """clip (B,T,1,H,W) in [0,1] -> heatmap (B,1,H/8,W/8) in [0,1]."""
        x = clip.to(self.device).permute(0, 2, 1, 3, 4).repeat(1, 3, 1, 1, 1)
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)          # (B,3,T,H,W)
        out = self.model(x)
        out = out[0] if isinstance(out, (list, tuple)) else out    # (B,5+C,h,w)
        obj = torch.sigmoid(out[:, 4:5])
        cls = torch.sigmoid(out[:, 5:]).amax(dim=1, keepdim=True)
        return (obj * cls).clamp(0, 1)
