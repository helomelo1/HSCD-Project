"""
Label-preserving background augmentations.

These transforms change *only* background intensity statistics.
Target pixel locations are unchanged, so ground-truth masks remain valid.

Usage:
    aug_clip = label_preserving_aug(clip)   # clip: (B, T, 1, H, W) in [0,1]
"""

import torch
import torch.nn.functional as F
from config import AUG_BRIGHTNESS, AUG_CONTRAST, AUG_NOISE_STD


def label_preserving_aug(clip: torch.Tensor) -> torch.Tensor:
    """
    clip : (B, T, 1, H, W), float32 in [0, 1]
    Returns an augmented version with the same shape.
    Augmentations:
        1. Random brightness shift  (additive, per-batch-item)
        2. Random contrast scale    (multiplicative, per-batch-item)
        3. Additive Gaussian noise  (per-pixel)
    All are label-preserving: they do not move or remove targets.
    """
    B = clip.shape[0]
    out = clip.clone()

    # 1. brightness
    delta = (torch.rand(B, 1, 1, 1, 1, device=clip.device) * 2 - 1) \
            * AUG_BRIGHTNESS
    out = out + delta

    # 2. contrast (scale around per-frame mean)
    mean = out.mean(dim=[2, 3, 4], keepdim=True)
    alpha = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=clip.device) * 2 - 1) \
            * AUG_CONTRAST
    out = (out - mean) * alpha + mean

    # 3. Gaussian noise
    noise = torch.randn_like(out) * AUG_NOISE_STD
    out = out + noise

    return out.clamp(0.0, 1.0)
