"""
Teacher wrapper – loads a trained CHAL model and exposes a unified
`predict(clip) -> heatmap` interface that the distillation trainer uses.

If CHAL weights are unavailable (e.g. during development), a DummyTeacher
that returns the ground-truth mask can be used instead.

Usage:
    teacher = CHALTeacher(ckpt_path)        # real teacher
    teacher = DummyTeacher()               # GT-based stand-in
    heatmap = teacher.predict(clip)         # (B, 1, H, W) in [0, 1]
"""

import sys, os
import torch
import torch.nn.functional as F

from config import CHAL_DIR, TEACHER_CKPT, NUM_FRAMES


class DummyTeacher:
    """Stand-in when no CHAL checkpoint is available.
    Distillation collapses to pure GT supervision in this mode."""
    def predict(self, clip: torch.Tensor) -> None:
        return None   # signals trainer to skip pred-matching loss


class CHALTeacher:
    """
    Wraps a trained CHAL model for inference only.
    CHAL must be on the Python path (we add CHAL_DIR here).
    """
    def __init__(self, ckpt_path: str = TEACHER_CKPT,
                 num_classes: int = 1,
                 device: str | None = None):
        if CHAL_DIR not in sys.path:
            sys.path.insert(0, CHAL_DIR)

        from nets.slowfastnet import slowfastnet  # type: ignore

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = slowfastnet(num_classes=num_classes, num_frame=NUM_FRAMES)
        state = torch.load(ckpt_path, map_location=self.device)
        # CHAL checkpoint may be wrapped; handle both cases
        if "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip : (B, T, 1, H, W) in [0, 1]
        Returns teacher heatmap (B, 1, H, W) in [0, 1].

        CHAL's slowfastnet expects input as (B, C, T, H, W) with C channels
        per frame.  We adapt here.
        """
        B, T, C, H, W = clip.shape
        # CHAL uses RGB-like 3-ch per frame; replicate grayscale to 3 channels
        x = clip.permute(0, 2, 1, 3, 4)   # (B, 1, T, H, W)
        x = x.repeat(1, 3, 1, 1, 1)       # (B, 3, T, H, W)
        x = x.to(self.device)

        raw = self.model(x)                # list of prediction tensors
        # CHAL returns YOLOv-style outputs; we convert the first head's
        # confidence map to a spatial heatmap via sigmoid + max-pool
        if isinstance(raw, (list, tuple)):
            conf = raw[0]                  # (B, anchors*(5+cls), H', W')
        else:
            conf = raw
        # Take every 5th channel (objectness) across anchors and max-reduce
        # Adjust if CHAL uses a different stride convention
        obj = conf[:, 4::6, :, :]         # objectness channels
        heatmap = torch.sigmoid(obj).max(dim=1, keepdim=True)[0]
        heatmap = F.interpolate(heatmap, size=(H, W),
                                mode='bilinear', align_corners=False)
        return heatmap.clamp(0, 1)
