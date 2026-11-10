"""
Compact Student Model – CausalMARD (Motion-Anomaly Response Distillation).

Design goals:
  • Much fewer parameters than CHAL (SlowFast + neural background field)
  • Explicit temporal modelling with causal (past→present) convolutions
  • Outputs a per-pixel anomaly heatmap in [0, 1]

Architecture (encoder-decoder with temporal causal convs):
  Stem  → T×DepthwiseTempConv blocks (causal, no future leakage)
        → Spatial encoder (3 scales, depthwise-separable)
        → Lightweight decoder with skip connections
        → 1×1 conv → sigmoid heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class DepthwiseSep(nn.Module):
    """Depthwise-separable 2-D convolution."""
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=stride,
                            padding=k // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class CausalTempConv(nn.Module):
    """
    1-D causal convolution along the temporal axis (kernel_size=3).
    Applies *per-channel* (groups=C) so it is very cheap.
    Padding = 2 on the left (past), 0 on the right (no future).
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              groups=channels, bias=False)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.act(self.bn(self.conv(x)))


class TemporalBlock(nn.Module):
    """Spatial DW-Sep conv + causal temporal mixing."""
    def __init__(self, ch):
        super().__init__()
        self.spatial = DepthwiseSep(ch, ch)
        self.temporal = CausalTempConv(ch)

    def forward(self, frames):
        """
        frames: list of T tensors each (B, C, H, W)
        returns: list of T tensors (B, C, H, W)
        """
        T = len(frames)
        B, C, H, W = frames[0].shape
        # spatial
        sp = [self.spatial(f) for f in frames]       # list of (B,C,H,W)
        # temporal along spatial average (global avg pool → 1d conv → scale)
        pooled = torch.stack([f.mean(dim=[2, 3]) for f in sp], dim=2)  # (B,C,T)
        mixed  = self.temporal(pooled)                                   # (B,C,T)
        # multiply back as a channel-wise scale (SE-style)
        out = []
        for t in range(T):
            gate = torch.sigmoid(mixed[:, :, t]).unsqueeze(-1).unsqueeze(-1)
            out.append(sp[t] * gate)
        return out


# ── Spatial Encoder ───────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch):
        super().__init__()
        # 3 downsampling stages, each halves spatial resolution
        self.s1 = nn.Sequential(
            DepthwiseSep(in_ch,       base_ch,   k=3),
            DepthwiseSep(base_ch,     base_ch,   k=3))
        self.s2 = nn.Sequential(
            nn.MaxPool2d(2),
            DepthwiseSep(base_ch,     base_ch*2, k=3),
            DepthwiseSep(base_ch*2,   base_ch*2, k=3))
        self.s3 = nn.Sequential(
            nn.MaxPool2d(2),
            DepthwiseSep(base_ch*2,   base_ch*4, k=3),
            DepthwiseSep(base_ch*4,   base_ch*4, k=3))

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        return s1, s2, s3   # skips + deepest


class Decoder(nn.Module):
    def __init__(self, base_ch):
        super().__init__()
        self.up2 = DepthwiseSep(base_ch*4 + base_ch*2, base_ch*2)
        self.up1 = DepthwiseSep(base_ch*2 + base_ch,   base_ch)

    def forward(self, s1, s2, s3):
        # upsample s3 to s2 resolution, concat, refine
        x = torch.cat([F.interpolate(s3, size=s2.shape[-2:],
                        mode='bilinear', align_corners=False), s2], dim=1)
        x = self.up2(x)   # (B, base*2, H/2, W/2)
        # upsample to s1 resolution, concat, refine
        x = torch.cat([F.interpolate(x, size=s1.shape[-2:],
                        mode='bilinear', align_corners=False), s1], dim=1)
        x = self.up1(x)   # (B, base, H, W)
        return x


# ── Full Student ──────────────────────────────────────────────────────────────

class CompactStudent(nn.Module):
    """
    Input : clip  (B, T, 1, H, W)
    Output: heatmap (B, 1, H, W) in [0, 1]
    """
    def __init__(self, num_frames=5, in_channels=1, base_ch=16):
        super().__init__()
        self.T = num_frames
        # Per-frame stem
        self.stem = DepthwiseSep(in_channels, base_ch, k=3)
        # Temporal blocks
        self.t_block1 = TemporalBlock(base_ch)
        self.t_block2 = TemporalBlock(base_ch)
        # Encode the last (most-recent, causal) frame representation
        self.encoder = Encoder(base_ch, base_ch)
        self.decoder = Decoder(base_ch)
        self.head    = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Sigmoid())

    def forward(self, clip):
        """clip: (B, T, 1, H, W)"""
        B, T, C, H, W = clip.shape
        # Stem on each frame
        frames = [self.stem(clip[:, t]) for t in range(T)]   # T×(B,C,H,W)
        # Temporal blocks (causal)
        frames = self.t_block1(frames)
        frames = self.t_block2(frames)
        # Use only the last frame (most-recent, full temporal context)
        feat = frames[-1]                                      # (B, C, H, W)
        s1, s2, s3 = self.encoder(feat)
        out        = self.decoder(s1, s2, s3)                 # (B, base, H, W)
        heatmap    = self.head(out)                           # (B, 1, H, W)
        return heatmap


# ── Parameter count utility ───────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CompactStudent(num_frames=5, base_ch=16)
    x = torch.randn(2, 5, 1, 256, 256)
    y = m(x)
    print(f"Output shape : {y.shape}")
    print(f"Parameters   : {count_params(m):,}")
