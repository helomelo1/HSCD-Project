"""
Evaluation metrics for infrared small target detection:
  - Pd  (probability of detection)  = TP / (TP + FN)
  - Fa  (false-alarm rate)          = FP / total_pixels
  - F1  = 2 * Pd * Precision / (Pd + Precision)
  - IoU (bonus)

A detection is counted as a "true positive" if the predicted binary mask
overlaps with the GT mask (IoU > 0 suffices for small targets).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DETECT_THRESH, IMG_SIZE, NUM_FRAMES, BATCH_SIZE
from dataset import MIRSDTDataset
from model import CompactStudent


# ── per-sample metrics ────────────────────────────────────────────────────────

def pixel_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    pred_bin, gt_bin : (H, W) boolean arrays
    Returns tp, fp, fn, total_pixels
    """
    tp = int((pred_bin & gt_bin).sum())
    fp = int((pred_bin & ~gt_bin).sum())
    fn = int((~pred_bin & gt_bin).sum())
    total = pred_bin.size
    return tp, fp, fn, total


# ── full dataset evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: CompactStudent, txt_path: str,
             device: str = "cpu", threshold: float = DETECT_THRESH):
    """
    Returns dict with Pd, Fa, Precision, F1 (macro-averaged over frames).
    """
    dataset = MIRSDTDataset(txt_path, img_size=IMG_SIZE, augment=False)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    model.eval()
    total_tp = total_fp = total_fn = total_px = 0

    for clip, mask, _ in tqdm(loader, desc="Eval"):
        clip = clip.to(device)                       # (B,T,1,H,W)
        pred = model(clip).squeeze(1).cpu().numpy()  # (B,H,W)
        gt   = mask.squeeze(1).numpy().astype(bool)  # (B,H,W)

        for b in range(pred.shape[0]):
            pred_bin = pred[b] > threshold
            tp, fp, fn, px = pixel_metrics(pred_bin, gt[b])
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_px += px

    eps = 1e-8
    pd        = total_tp / (total_tp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    fa        = total_fp / (total_px + eps)
    f1        = 2 * pd * precision / (pd + precision + eps)

    return {"Pd": pd, "Fa": fa, "Precision": precision, "F1": f1}


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os
    from config import TEST_TXT, BASE_CH, LOG_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  default=os.path.join(LOG_DIR, "best.pth"))
    parser.add_argument("--txt",   default=TEST_TXT)
    parser.add_argument("--thresh", type=float, default=DETECT_THRESH)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CompactStudent(num_frames=NUM_FRAMES, base_ch=BASE_CH).to(device)
    state  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    metrics = evaluate(model, args.txt, device=device, threshold=args.thresh)
    print("\n── Evaluation Results ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
