"""
Dataset for NUDT-MIRSDT (same file layout as the official DTUM release).

Each line in train/test.txt looks like:
    Sequence1/Mix/00001.mat
Each .mat has key 'Mix' with shape (T, H, W); the LAST frame is the current
frame and its mask is:
    Sequence1/masks/00001.png  (binary 0/255)

Frames are NOT resized (NUDT-MIRSDT is 512x512, targets are 1-3 px; resizing
to 256 with a nearest-neighbour mask deletes or shifts many of them).
  • train : random TRAIN_CROP x TRAIN_CROP crops (mostly containing a target)
            + flips / transpose applied identically to every frame
  • eval  : full frame, zero-padded by `pad_collate`; the true (h, w) is
            returned so metrics are computed on the un-padded region only.

Returns:
    clip : FloatTensor (T, 1, h, w) in [0, 1]
    mask : FloatTensor (1, h, w)    in {0, 1}
    hw   : (h, w) original size
    path : str
"""

import os
import random

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import (DATASET_DIR, NUM_FRAMES, TRAIN_CROP, TARGET_CROP_PROB,
                    PAD_MULTIPLE)


def read_list(txt_path):
    with open(txt_path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def split_train_val(lines, every_nth):
    """Hold out whole sequences (every n-th, sorted) as validation."""
    seqs = sorted({ln.replace("\\", "/").split("/")[0] for ln in lines},
                  key=lambda s: (len(s), s))
    val_seqs = set(seqs[every_nth - 1::every_nth])
    train = [ln for ln in lines if ln.replace("\\", "/").split("/")[0] not in val_seqs]
    val   = [ln for ln in lines if ln.replace("\\", "/").split("/")[0] in val_seqs]
    return train, val, sorted(val_seqs)


def to_unit_range(x: np.ndarray) -> np.ndarray:
    """Map raw frames to [0, 1] regardless of how the .mat stores them."""
    x = x.astype(np.float32)
    mx = float(x.max()) if x.size else 0.0
    if mx <= 1.0:
        return x
    if mx <= 255.0:
        return x / 255.0
    return x / 65535.0


class MIRSDTDataset(Dataset):
    def __init__(self, lines, train: bool = False, num_frames: int = NUM_FRAMES,
                 crop: int = TRAIN_CROP):
        """lines: list of 'SequenceX/Mix/xxxxx.mat' or a path to a .txt list."""
        if isinstance(lines, str):
            lines = read_list(lines)
        self.train = train
        self.T = num_frames
        self.crop = crop
        self.samples = []
        missing = 0
        for rel in lines:
            rel = rel.replace("\\", "/")
            mat_path = os.path.join(DATASET_DIR, rel)
            parts = rel.split("/")
            mask_path = os.path.join(DATASET_DIR, parts[0], "masks",
                                     os.path.splitext(parts[-1])[0] + ".png")
            if os.path.exists(mat_path) and os.path.exists(mask_path):
                self.samples.append((mat_path, mask_path))
            else:
                missing += 1
        if missing:
            print(f"[dataset] warning: {missing} listed samples not found on disk")
        if not self.samples:
            raise RuntimeError(f"No samples found under {DATASET_DIR}")

    # ── loading ───────────────────────────────────────────────────────────────
    def _load(self, mat_path, mask_path):
        mix = sio.loadmat(mat_path)["Mix"]
        if mix.ndim == 2:
            mix = mix[None]
        clip = to_unit_range(mix)[-self.T:]                # last T frames
        if clip.shape[0] < self.T:                         # repeat oldest frame
            pad = np.repeat(clip[:1], self.T - clip.shape[0], axis=0)
            clip = np.concatenate([pad, clip], axis=0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        if mask.shape != clip.shape[1:]:
            raise ValueError(f"mask/frame size mismatch: {mask_path}")
        return clip, mask

    # ── training augmentation (geometry identical for every frame) ────────────
    def _random_crop(self, clip, mask):
        c = self.crop
        T, H, W = clip.shape
        if H < c or W < c:                                 # pad small frames
            ph, pw = max(0, c - H), max(0, c - W)
            clip = np.pad(clip, ((0, 0), (0, ph), (0, pw)))
            mask = np.pad(mask, ((0, ph), (0, pw)))
            T, H, W = clip.shape
        ys, xs = np.nonzero(mask)
        if len(ys) and random.random() < TARGET_CROP_PROB:
            k = random.randrange(len(ys))
            y0_lo, y0_hi = max(0, ys[k] - c + 1), min(H - c, ys[k])
            x0_lo, x0_hi = max(0, xs[k] - c + 1), min(W - c, xs[k])
            y0 = random.randint(y0_lo, y0_hi)
            x0 = random.randint(x0_lo, x0_hi)
        else:
            y0 = random.randint(0, H - c)
            x0 = random.randint(0, W - c)
        return clip[:, y0:y0 + c, x0:x0 + c], mask[y0:y0 + c, x0:x0 + c]

    @staticmethod
    def _random_geom(clip, mask):
        if random.random() < 0.5:
            clip, mask = clip[:, :, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            clip, mask = clip[:, ::-1, :], mask[::-1, :]
        if random.random() < 0.5:                          # square crops only
            clip, mask = clip.transpose(0, 2, 1), mask.T
        return np.ascontiguousarray(clip), np.ascontiguousarray(mask)

    # ── public ────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path, mask_path = self.samples[idx]
        clip, mask = self._load(mat_path, mask_path)
        if self.train:
            clip, mask = self._random_crop(clip, mask)
            clip, mask = self._random_geom(clip, mask)
        h, w = mask.shape
        clip_t = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(1)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return clip_t, mask_t, (h, w), mat_path


def pad_collate(batch, multiple: int = PAD_MULTIPLE):
    """Zero-pad every sample (bottom/right) to a common size divisible by
    `multiple`. Returns clips (B,T,1,H,W), masks (B,1,H,W), hw list, paths."""
    Hm = max(b[0].shape[-2] for b in batch)
    Wm = max(b[0].shape[-1] for b in batch)
    Hp = -(-Hm // multiple) * multiple
    Wp = -(-Wm // multiple) * multiple
    clips, masks, hws, paths = [], [], [], []
    for clip, mask, hw, path in batch:
        dh, dw = Hp - clip.shape[-2], Wp - clip.shape[-1]
        clips.append(F.pad(clip, (0, dw, 0, dh)))
        masks.append(F.pad(mask, (0, dw, 0, dh)))
        hws.append(hw)
        paths.append(path)
    return torch.stack(clips), torch.stack(masks), hws, paths
