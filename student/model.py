"""
Compact Student Model – CausalMARD (Motion-Anomaly Response Distillation).

Input : clip (B, T, 1, H, W) in [0, 1]; frame T-1 is the current (key) frame.
Output: LOGITS (B, 1, H, W). Apply torch.sigmoid() to get the heatmap.

Why this design (vs. the previous version):
  The previous temporal block mixed frames only through *globally pooled*
  channel statistics, so a 2-3 px moving target had no path from the past
  frames to the output (the model was effectively single-frame). Here time is
  fused PER PIXEL, and parameter-free background/motion anomaly priors are
  provided explicitly (the lightweight analogue of CHAL's background-centred
  anomaly modelling):

  1. Anomaly priors (no parameters, causal: only frames <= t are used)
       cur, cur - mean(past frames)  [temporal background residual],
       cur - previous frame          [motion],
       cur - local mean              [spatial local contrast]
  2. Shared per-frame stem -> frames stacked on channels -> 3x3 conv
     (pixel-wise spatio-temporal fusion; no Conv3d, so it runs on MPS/CPU/CUDA)
  3. 4-level residual U-Net with full-resolution skips (small targets need them)
  4. 1x1 head -> logits (bias initialised to a 1 % prior for stable training)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

def conv_bn_relu(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    """3x3 conv - BN - ReLU - 3x3 conv - BN, plus (projected) identity."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.skip = (nn.Identity() if in_ch == out_ch else
                     nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                   nn.BatchNorm2d(out_ch)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


def anomaly_priors(clip: torch.Tensor) -> torch.Tensor:
    """clip (B,T,1,H,W) -> (B,4,H,W) parameter-free causal anomaly cues."""
    cur  = clip[:, -1]                                   # (B,1,H,W)
    past = clip[:, :-1]
    bg   = past.mean(dim=1) if past.shape[1] > 0 else cur
    prev = clip[:, -2] if clip.shape[1] > 1 else cur
    local_mean = F.avg_pool2d(cur, 9, stride=1, padding=4,
                              count_include_pad=False)
    return torch.cat([cur, cur - bg, cur - prev, cur - local_mean], dim=1)


# ── Full Student ──────────────────────────────────────────────────────────────

class CompactStudent(nn.Module):
    """
    Input : clip  (B, T, 1, H, W)
    Output: logits (B, 1, H, W)   (H, W must be divisible by 8)
    """
    def __init__(self, num_frames=5, in_channels=1, base_ch=16):
        super().__init__()
        self.T = num_frames
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # per-frame shared stem, then pixel-wise temporal fusion
        self.frame_stem = conv_bn_relu(in_channels, c1)
        self.temporal_fuse = conv_bn_relu(c1 * num_frames, c1)
        # explicit anomaly-prior branch
        self.prior_stem = conv_bn_relu(4, c1)

        # encoder
        self.enc1 = ResBlock(2 * c1, c1)                 # H
        self.enc2 = ResBlock(c1, c2)                     # H/2
        self.enc3 = ResBlock(c2, c3)                     # H/4
        self.enc4 = ResBlock(c3, c4)                     # H/8 (bottleneck)
        self.pool = nn.MaxPool2d(2)

        # decoder
        self.dec3 = ResBlock(c4 + c3, c3)
        self.dec2 = ResBlock(c3 + c2, c2)
        self.dec1 = ResBlock(c2 + c1, c1)

        self.head = nn.Conv2d(c1, 1, 1)
        nn.init.constant_(self.head.bias, -math.log((1 - 0.01) / 0.01))

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear',
                             align_corners=False)

    def forward(self, clip):
        B, T, C, H, W = clip.shape
        if T != self.T:
            raise ValueError(f"model built for {self.T} frames, got {T}")

        # pixel-wise spatio-temporal fusion (all frames are <= t: causal)
        f = self.frame_stem(clip.reshape(B * T, C, H, W))        # (B*T,c1,H,W)
        f = f.reshape(B, T * f.shape[1], H, W)                   # (B,T*c1,H,W)
        f = self.temporal_fuse(f)                                # (B,c1,H,W)
        p = self.prior_stem(anomaly_priors(clip))                # (B,c1,H,W)

        e1 = self.enc1(torch.cat([f, p], dim=1))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self._up(e4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return self.head(d1)                                     # logits


# ── Parameter count utility ───────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CompactStudent(num_frames=5, base_ch=16)
    x = torch.rand(2, 5, 1, 256, 256)
    y = m(x)
    print(f"Output shape : {y.shape} (logits)")
    print(f"Parameters   : {count_params(m):,}")
