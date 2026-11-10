"""
Main training script for the compact student model.

Usage (no teacher / GT-only mode):
    python train_student.py

Usage (with CHAL teacher for distillation):
    python train_student.py --teacher

The script saves:
  logs/best.pth        – best checkpoint by validation F1
  logs/last.pth        – most recent checkpoint
  logs/train_log.csv   – per-epoch loss & metrics
"""

import argparse
import csv
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── add project root to path so imports work from anywhere ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BATCH_SIZE, BASE_CH, LR, LOG_DIR, NUM_EPOCHS,
                    NUM_FRAMES, TEACHER_CKPT, TRAIN_TXT, TEST_TXT,
                    WEIGHT_DECAY, IMG_SIZE)
from dataset import MIRSDTDataset
from model import CompactStudent, count_params
from losses import DistillationLoss
from augment import label_preserving_aug
from evaluate import evaluate


# ── helpers ───────────────────────────────────────────────────────────────────

def build_teacher(use_teacher: bool):
    from teacher import CHALTeacher, DummyTeacher
    if not use_teacher:
        return DummyTeacher()
    if not os.path.exists(TEACHER_CKPT):
        print(f"[warn] CHAL checkpoint not found at {TEACHER_CKPT}."
              " Falling back to DummyTeacher (GT-only mode).")
        return DummyTeacher()
    return CHALTeacher(TEACHER_CKPT)


def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Datasets
    train_ds = MIRSDTDataset(TRAIN_TXT, img_size=IMG_SIZE, augment=True)
    val_ds   = MIRSDTDataset(TEST_TXT,  img_size=IMG_SIZE, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # Model
    model     = CompactStudent(num_frames=NUM_FRAMES, base_ch=BASE_CH).to(device)
    print(f"Student parameters: {count_params(model):,}")

    # Teacher
    teacher = build_teacher(args.teacher)

    # Optimiser & loss
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)
    criterion = DistillationLoss(pos_weight=10.0)

    best_f1   = 0.0
    log_path  = os.path.join(LOG_DIR, "train_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "l_gt", "l_pred", "l_resp",
                         "Pd", "Fa", "F1"])

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ep_loss = ep_gt = ep_pred = ep_resp = 0.0

        for clip, mask, _ in tqdm(train_dl,
                                  desc=f"Epoch {epoch}/{NUM_EPOCHS}",
                                  leave=False):
            clip = clip.to(device)   # (B, T, 1, H, W)
            mask = mask.to(device)   # (B, 1, H, W)

            # ── forward pass on clean clip ──────────────────────────────────
            heatmap = model(clip)    # (B, 1, H, W) in [0,1]

            # logit for BCE (we expose the last sigmoid separately below)
            # For clean logit we recompute: logit = log(h/(1-h))
            eps = 1e-6
            logit = torch.log((heatmap + eps) / (1 - heatmap + eps))

            # ── teacher prediction ──────────────────────────────────────────
            teacher_hm = teacher.predict(clip)   # None or (B,1,H,W)

            # ── augmented forward pass ──────────────────────────────────────
            clip_aug  = label_preserving_aug(clip)
            hm_aug    = model(clip_aug)           # (B, 1, H, W)

            # ── loss ────────────────────────────────────────────────────────
            losses = criterion(logit, mask, heatmap, hm_aug, teacher_hm)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            ep_loss  += losses["total"].item()
            ep_gt    += losses["l_gt"]
            ep_pred  += losses["l_pred"]
            ep_resp  += losses["l_resp"]

        scheduler.step()
        n = len(train_dl)
        print(f"[Epoch {epoch:3d}] loss={ep_loss/n:.4f}  "
              f"gt={ep_gt/n:.4f}  pred={ep_pred/n:.4f}  resp={ep_resp/n:.4f}")

        # ── validation ───────────────────────────────────────────────────────
        metrics = evaluate(model, TEST_TXT, device=device)
        print(f"           Pd={metrics['Pd']:.4f}  Fa={metrics['Fa']:.6f}  "
              f"F1={metrics['F1']:.4f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, ep_loss/n, ep_gt/n, ep_pred/n, ep_resp/n,
                             metrics["Pd"], metrics["Fa"], metrics["F1"]])

        save_ckpt(model, os.path.join(LOG_DIR, "last.pth"))
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            save_ckpt(model, os.path.join(LOG_DIR, "best.pth"))
            print(f"           ✓ New best F1={best_f1:.4f} – saved.")

    print(f"\nTraining complete. Best F1={best_f1:.4f}")
    print(f"Checkpoints in: {LOG_DIR}")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train compact student on NUDT-MIRSDT")
    parser.add_argument("--teacher", action="store_true",
                        help="Enable CHAL teacher distillation "
                             "(requires CHAL/logs/best_model.pth)")
    args = parser.parse_args()
    train(args)
