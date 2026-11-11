"""
Label-preserving background augmentations (for the response-consistency loss).

The SAME brightness / contrast change is applied to every frame of a clip, so
frame differences (motion cues) are preserved; only the mild per-pixel noise
differs between frames. Strengths are kept small (see config.py) so dim,
low-SNR targets are not erased - otherwise the transform would no longer be
label-preserving.

Usage:
    aug_clip = label_preserving_aug(clip)   # clip: (B, T, 1, H, W) in [0,1]
"""

import torch
from config import AUG_BRIGHTNESS, AUG_CONTRAST, AUG_NOISE_STD


def label_preserving_aug(clip: torch.Tensor) -> torch.Tensor:
    B = clip.shape[0]
    dev = clip.device
    out = clip

    # 1. brightness (per clip)
    delta = (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * AUG_BRIGHTNESS
    out = out + delta

    # 2. contrast around the clip mean (per clip, identical for all frames)
    mean = out.mean(dim=[1, 2, 3, 4], keepdim=True)
    alpha = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * AUG_CONTRAST
    out = (out - mean) * alpha + mean

    # 3. mild Gaussian noise
    out = out + torch.randn_like(out) * AUG_NOISE_STD

    # no clamping: clamping would saturate bright targets into the background
    return out
