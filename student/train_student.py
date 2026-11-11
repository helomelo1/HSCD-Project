"""
Main training script for the compact student model.

Usage (no teacher / GT-only mode):
    python train_student.py
Usage (with CHAL teacher for distillation):
    python train_student.py --teacher

Model selection is done on a VALIDATION split (whole sequences held out from
train.txt). The test set is never used during training. The detection
threshold is tuned on validation and stored inside best.pth.

The script saves:
  logs/best.pth        – best EMA weights by validation F1 (+ tuned threshold)
  logs/last.pth        – most recent EMA weights
  logs/train_log.csv   – per-epoch loss & validation metrics
"""

import argparse
import copy
import csv
import math
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BATCH_SIZE, BASE_CH, LR, LOG_DIR, NUM_EPOCHS, NUM_FRAMES,
                    TEACHER_CKPT, TRAIN_TXT, WEIGHT_DECAY, WARMUP_EPOCHS,
                    GRAD_CLIP, EMA_DECAY, NUM_WORKERS, VAL_EVERY_NTH_SEQ,
                    THRESH_GRID)
from dataset import MIRSDTDataset, pad_collate, read_list, split_train_val
from model import CompactStudent, count_params
from losses import DistillationLoss
from augment import label_preserving_aug
from evaluate import evaluate_sweep, best_threshold, get_device


# ── helpers ───────────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def build_teacher(use_teacher: bool):
    from teacher import CHALTeacher, DummyTeacher
    if not use_teacher:
        return DummyTeacher()
    if not os.path.exists(TEACHER_CKPT):
        print(f"[warn] CHAL checkpoint not found at {TEACHER_CKPT}."
              " Falling back to DummyTeacher (GT-only mode).")
        return DummyTeacher()
    return CHALTeacher(TEACHER_CKPT)


class EMA:
    """Exponential moving average of weights (BN buffers copied)."""
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))     # warm start
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def save_ckpt(model, path, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "num_frames": NUM_FRAMES,
                "base_ch": BASE_CH, **extra}, path)


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    seed_everything(args.seed)
    os.makedirs(LOG_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Data: hold out whole sequences of the TRAIN list for validation
    train_lines, val_lines, val_seqs = split_train_val(read_list(TRAIN_TXT),
                                                       VAL_EVERY_NTH_SEQ)
    train_ds = MIRSDTDataset(train_lines, train=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True,
                          collate_fn=pad_collate, worker_init_fn=worker_init,
                          persistent_workers=NUM_WORKERS > 0,
                          pin_memory=(device == "cuda"))
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_lines)} "
          f"(held-out sequences: {', '.join(val_seqs)})")

    model = CompactStudent(num_frames=NUM_FRAMES, base_ch=BASE_CH).to(device)
    ema = EMA(model, EMA_DECAY)
    print(f"Student parameters: {count_params(model):,}")

    teacher = build_teacher(args.teacher)
    criterion = DistillationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_dl)
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup = WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, step = -1.0, 0
    log_path = os.path.join(LOG_DIR, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "l_gt", "l_pred", "l_resp",
                                "val_thresh", "val_Pd", "val_Fa",
                                "val_Precision", "val_F1", "val_pix_F1"])

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ep = np.zeros(4)
        for clip, mask, _, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS}",
                                     leave=False):
            clip, mask = clip.to(device), mask.to(device)

            logit = model(clip)
            teacher_hm = teacher.predict(clip)
            if teacher_hm is not None:
                teacher_hm = teacher_hm.to(device)
            logit_aug = model(label_preserving_aug(clip))

            losses = criterion(logit, mask, logit_aug, teacher_hm)
            if not torch.isfinite(losses["total"]):
                print("[warn] non-finite loss, batch skipped")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            ema.update(model, step)
            step += 1

            ep += [losses["total"].item(), losses["l_gt"],
                   losses["l_pred"], losses["l_resp"]]

        ep /= steps_per_epoch
        print(f"[Epoch {epoch:3d}] loss={ep[0]:.4f}  gt={ep[1]:.4f}  "
              f"pred={ep[2]:.4f}  resp={ep[3]:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # ── validation on held-out sequences (EMA weights) ───────────────────
        sweep = evaluate_sweep(ema.model, val_lines, device=device,
                               thresholds=THRESH_GRID)
        thr, m = best_threshold(sweep)
        print(f"           val@{thr:.2f}: Pd={m['Pd']:.4f}  Fa={m['Fa']:.2e}  "
              f"Prec={m['Precision']:.4f}  F1={m['F1']:.4f}  pixF1={m['pix_F1']:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, *ep, thr, m["Pd"], m["Fa"],
                                    m["Precision"], m["F1"], m["pix_F1"]])

        save_ckpt(ema.model, os.path.join(LOG_DIR, "last.pth"),
                  epoch=epoch, threshold=thr, val=m)
        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            save_ckpt(ema.model, os.path.join(LOG_DIR, "best.pth"),
                      epoch=epoch, threshold=thr, val=m)
            print(f"           ✓ New best val F1={best_f1:.4f} (thr {thr}) – saved.")

    print(f"\nTraining complete. Best val F1={best_f1:.4f}")
    print(f"Now run:  python evaluate.py --ckpt {os.path.join(LOG_DIR, 'best.pth')}")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train compact student on NUDT-MIRSDT")
    parser.add_argument("--teacher", action="store_true",
                        help="Enable CHAL teacher distillation "
                             "(requires CHAL/logs/best_model.pth)")
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())
