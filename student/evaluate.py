"""
Evaluation for NUDT-MIRSDT, following the official protocol of the dataset
authors (DTUM, `ShootingRules.py`):

  • Pd  (target-level detection probability)
        a GT target (8-connected component) is detected if ANY predicted pixel
        lies within DET_RADIUS (=1) px of any of its pixels.  Pd = det / #targets
  • Fa  (pixel false-alarm rate)
        predicted pixels farther than FA_RADIUS (=4) px from every GT target,
        divided by the number of image pixels (padding excluded).
  • Precision / F1 (object level)
        a predicted connected component with no pixel inside the 4 px target
        zone is a false-alarm object.  Precision = det / (det + FA objects),
        F1 = 2·Pd·Precision / (Pd + Precision).

The previous script's numbers were PIXEL-level (recall/precision of
individual pixels), which is not detection probability; they are still
reported below as pix_* for comparison. Metrics are computed at native
resolution on the un-padded image.
"""

import os
import time

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (DETECT_THRESH, NUM_FRAMES, BATCH_SIZE, THRESH_GRID,
                    DET_RADIUS, FA_RADIUS, BASE_CH, NUM_WORKERS)
from dataset import MIRSDTDataset, pad_collate
from model import CompactStudent, count_params

_S8 = np.ones((3, 3), dtype=bool)                  # 8-connectivity
COUNT_KEYS = ["targets", "det", "fa_px", "fa_obj", "px",
              "tp_px", "fp_px", "fn_px"]


def image_counts(prob: np.ndarray, gt: np.ndarray, thresholds):
    """prob (h,w) float, gt (h,w) bool -> {thr: counts dict}."""
    lab, n_t = ndimage.label(gt, structure=_S8)
    zone = ndimage.maximum_filter(gt, size=2 * FA_RADIUS + 1)
    idx = np.arange(1, n_t + 1)
    out = {}
    for thr in thresholds:
        pred = prob >= thr
        c = dict.fromkeys(COUNT_KEYS, 0)
        c["targets"], c["px"] = n_t, gt.size
        if n_t:
            near = ndimage.maximum_filter(pred, size=2 * DET_RADIUS + 1)
            c["det"] = int(np.count_nonzero(ndimage.maximum(near, lab, idx)))
        if pred.any():
            c["fa_px"] = int(np.count_nonzero(pred & ~zone))
            plab, n_p = ndimage.label(pred, structure=_S8)
            hit = ndimage.maximum(zone, plab, np.arange(1, n_p + 1))
            c["fa_obj"] = int(n_p - np.count_nonzero(hit))
        c["tp_px"] = int(np.count_nonzero(pred & gt))
        c["fp_px"] = int(np.count_nonzero(pred & ~gt))
        c["fn_px"] = int(np.count_nonzero(~pred & gt))
        out[thr] = c
    return out


def summarize(c):
    eps = 1e-12
    pd = c["det"] / (c["targets"] + eps)
    prec = c["det"] / (c["det"] + c["fa_obj"] + eps)
    pix_pd = c["tp_px"] / (c["tp_px"] + c["fn_px"] + eps)
    pix_pr = c["tp_px"] / (c["tp_px"] + c["fp_px"] + eps)
    return {
        "Pd": pd,
        "Fa": c["fa_px"] / (c["px"] + eps),
        "Precision": prec,
        "F1": 2 * pd * prec / (pd + prec + eps),
        "FA_objects": c["fa_obj"],
        "pix_Pd": pix_pd,
        "pix_Prec": pix_pr,
        "pix_F1": 2 * pix_pd * pix_pr / (pix_pd + pix_pr + eps),
        "IoU": c["tp_px"] / (c["tp_px"] + c["fp_px"] + c["fn_px"] + eps),
    }


@torch.no_grad()
def evaluate_sweep(model, lines, device="cpu", thresholds=THRESH_GRID,
                   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Returns {threshold: metrics dict}. `lines` = list or .txt path."""
    ds = MIRSDTDataset(lines, train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=pad_collate)
    model.eval()
    totals = {t: dict.fromkeys(COUNT_KEYS, 0) for t in thresholds}
    for clip, mask, hws, _ in tqdm(dl, desc="Eval", leave=False):
        prob = torch.sigmoid(model(clip.to(device))).squeeze(1).cpu().numpy()
        gt = mask.squeeze(1).numpy() > 0.5
        for b, (h, w) in enumerate(hws):
            per = image_counts(prob[b, :h, :w], gt[b, :h, :w], thresholds)
            for t in thresholds:
                for k in COUNT_KEYS:
                    totals[t][k] += per[t][k]
    return {t: summarize(totals[t]) for t in thresholds}


def evaluate(model, lines, device="cpu", threshold=DETECT_THRESH, **kw):
    return evaluate_sweep(model, lines, device, [threshold], **kw)[threshold]


def best_threshold(sweep: dict):
    t = max(sweep, key=lambda k: (sweep[k]["F1"], -k))
    return t, sweep[t]


# ── efficiency profiling (params, MACs, latency) ──────────────────────────────

@torch.no_grad()
def profile(model, device, size=512, runs=50):
    macs = [0]

    def hook(m, inp, out):
        if isinstance(m, torch.nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            macs[0] += out.numel() * k
    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, torch.nn.Conv2d)]
    x = torch.rand(1, NUM_FRAMES, 1, size, size, device=device)
    model.eval()
    model(x)
    for h in hs:
        h.remove()
    for _ in range(5):
        model(x)
    sync = (torch.cuda.synchronize if device == "cuda" else
            torch.mps.synchronize if device == "mps" else (lambda: None))
    sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    sync()
    ms = (time.perf_counter() - t0) / runs * 1000
    return {"params": count_params(model), "GMACs": macs[0] / 1e9,
            "latency_ms": ms, "input": f"1x{NUM_FRAMES}x1x{size}x{size}"}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_student(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if not (isinstance(state, dict) and "model" in state):
        raise RuntimeError(
            f"{ckpt_path} is a checkpoint from the OLD architecture. "
            "Retrain with the updated train_student.py.")
    model = CompactStudent(num_frames=state.get("num_frames", NUM_FRAMES),
                           base_ch=state.get("base_ch", BASE_CH)).to(device)
    model.load_state_dict(state["model"])
    return model, state


def print_metrics(m, title):
    print(f"\n── {title} " + "─" * max(0, 40 - len(title)))
    print(f"  Pd          : {m['Pd']:.4f}   (target-level, official protocol)")
    print(f"  Fa          : {m['Fa']:.3e}   (false-alarm pixels / pixels)")
    print(f"  Precision   : {m['Precision']:.4f}   (object-level)")
    print(f"  F1          : {m['F1']:.4f}   (object-level)")
    print(f"  FA objects  : {m['FA_objects']}")
    print(f"  pixel Pd / Prec / F1 / IoU : {m['pix_Pd']:.4f} / "
          f"{m['pix_Prec']:.4f} / {m['pix_F1']:.4f} / {m['IoU']:.4f}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from config import TEST_TXT, LOG_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(LOG_DIR, "best.pth"))
    parser.add_argument("--txt", default=TEST_TXT)
    parser.add_argument("--thresh", type=float, default=None,
                        help="override the validation-tuned threshold")
    parser.add_argument("--sweep", action="store_true",
                        help="also print all thresholds (for PR/ROC curves only;"
                             " do NOT pick the threshold on the test set)")
    parser.add_argument("--profile", action="store_true",
                        help="report params, MACs and latency")
    args = parser.parse_args()

    device = get_device()
    model, state = load_student(args.ckpt, device)
    thr = args.thresh if args.thresh is not None else state.get("threshold", DETECT_THRESH)
    print(f"Checkpoint: {args.ckpt} (epoch {state.get('epoch')}, "
          f"threshold {thr} {'(given)' if args.thresh is not None else '(tuned on val)'})")

    grid = sorted(set(THRESH_GRID) | {thr}) if args.sweep else [thr]
    sweep = evaluate_sweep(model, args.txt, device=device, thresholds=grid)

    if args.sweep:
        print(f"\n{'Thresh':>7} {'Pd':>8} {'Fa':>10} {'Prec':>8} {'F1':>8} {'pixF1':>8}")
        for t in grid:
            m = sweep[t]
            print(f"{t:7.2f} {m['Pd']:8.4f} {m['Fa']:10.3e} {m['Precision']:8.4f}"
                  f" {m['F1']:8.4f} {m['pix_F1']:8.4f}")
    print_metrics(sweep[thr], f"Test results @ {thr}")

    if args.profile:
        p = profile(model, device)
        print(f"\n── Efficiency ({device}) ──")
        print(f"  Params  : {p['params']:,}")
        print(f"  GMACs   : {p['GMACs']:.2f}  (input {p['input']})")
        print(f"  Latency : {p['latency_ms']:.1f} ms / clip")
