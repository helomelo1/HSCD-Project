"""
Dataset for NUDT-MIRSDT.

Each line in train/test.txt looks like:
    Sequence1/Mix/00001.mat

Each .mat has key 'Mix' with shape (5, H, W) – five consecutive grayscale frames.
The corresponding mask for the *last* (most-recent) frame is loaded from:
    <seq_root>/masks/<frame_id>.png  (binary 0/255)

We return:
    clip  : FloatTensor (5, 1, H, W)  – normalised to [0, 1]
    mask  : FloatTensor (1, H, W)     – binary {0, 1}
    path  : str (for debugging)
"""

import os
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from config import DATASET_DIR, NUM_FRAMES, IMG_SIZE


class MIRSDTDataset(Dataset):
    def __init__(self, txt_path: str, img_size: int = IMG_SIZE,
                 augment: bool = False):
        self.img_size  = img_size
        self.augment   = augment
        self.samples   = self._parse_txt(txt_path)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _parse_txt(self, txt_path: str):
        samples = []
        with open(txt_path) as f:
            for line in f:
                rel = line.strip()           # e.g. "Sequence1/Mix/00001.mat"
                if not rel:
                    continue
                mat_path = os.path.join(DATASET_DIR, rel)
                # Derive mask path: Sequence1/masks/00001.png
                parts    = rel.split(os.sep)  # ['Sequence1', 'Mix', '00001.mat']
                seq      = parts[0]
                frame_id = os.path.splitext(parts[-1])[0]  # '00001'
                mask_path = os.path.join(DATASET_DIR, seq, "masks",
                                         frame_id + ".png")
                if os.path.exists(mat_path) and os.path.exists(mask_path):
                    samples.append((mat_path, mask_path))
        return samples

    def _load_clip(self, mat_path: str):
        """Load (5, H, W) uint8 from .mat, return float32 normalised."""
        d    = sio.loadmat(mat_path)
        clip = d["Mix"].astype(np.float32) / 255.0   # (5, H, W)
        return clip

    def _load_mask(self, mask_path: str):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)        # binary 0/1
        return mask

    def _resize(self, clip, mask):
        s = self.img_size
        resized_frames = []
        for t in range(clip.shape[0]):
            fr = cv2.resize(clip[t], (s, s), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(fr)
        clip_r = np.stack(resized_frames, axis=0)             # (T, s, s)
        mask_r = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)
        return clip_r, mask_r

    def _random_flip(self, clip, mask):
        if np.random.rand() < 0.5:
            clip = clip[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if np.random.rand() < 0.5:
            clip = clip[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        return clip, mask

    # ── public ────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path, mask_path = self.samples[idx]
        clip  = self._load_clip(mat_path)    # (T, H, W)
        mask  = self._load_mask(mask_path)   # (H, W)
        clip, mask = self._resize(clip, mask)

        if self.augment:
            clip, mask = self._random_flip(clip, mask)

        # (T, 1, H, W)  and  (1, H, W)
        clip_t = torch.from_numpy(clip).unsqueeze(1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return clip_t, mask_t, mat_path
