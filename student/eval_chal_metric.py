"""
Evaluate the student with CHAL's metric (box-level, COCO / pycocotools),
replicating CHAL's test.py exactly:

  mAP50     = COCO AP at IoU 0.50 (cocoEval.stats[1])
  Recall    = recall at IoU 0.50, maxDets=100, area=all
  Precision = mean interpolated precision over recall points up to Recall
  F1        = 2 * P * R / (P + R)

Boxes from the student:
  predicted heatmap -> binarised at the val-tuned threshold (stored in the
  checkpoint, or --box_thresh) -> 8-connected components -> one box per
  component (tight pixel extent), score = max probability in the component.
GT boxes: tight boxes around each 8-connected component of the GT masks.

NOTE: CHAL itself is scored against its own box annotations (released with
the CHAL datasets). This script uses boxes derived from the NUDT-MIRSDT masks,
so the metric is identical but GT box sizes may differ slightly from CHAL's
annotation files.

Usage:
    python eval_chal_metric.py --ckpt logs/best.pth
"""

import argparse
import contextlib
import io
import os

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import TEST_TXT, LOG_DIR, BATCH_SIZE, NUM_WORKERS, DETECT_THRESH
from dataset import MIRSDTDataset, pad_collate
from evaluate import load_student, get_device

_S8 = np.ones((3, 3), dtype=bool)


def boxes_from_binary(binary):
    """-> list of (x, y, w, h, slice) for each 8-connected component."""
    lab, n = ndimage.label(binary, structure=_S8)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        out.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0),
                    lab[sl] == k, sl))
    return out


def chal_metrics(gt_dict, dets):
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    if not dets:
        return {"mAP50": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision_50 = ev.eval["precision"][0, :, 0, 0, -1]     # same as CHAL
    recall_50 = float(ev.eval["recall"][0, 0, 0, -1])
    n = int(recall_50 * 100)
    p = float(np.mean(precision_50[:n])) if n > 0 else 0.0
    f1 = 2 * recall_50 * p / (recall_50 + p) if (recall_50 + p) > 0 else 0.0
    return {"mAP50": float(ev.stats[1]), "Precision": p,
            "Recall": recall_50, "F1": f1}


@torch.no_grad()
def run(model, lines, device, box_thresh):
    ds = MIRSDTDataset(lines, train=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, collate_fn=pad_collate)
    model.eval()
    images, anns, dets = [], [], []
    img_id = ann_id = 0
    for clip, mask, hws, paths in tqdm(dl, desc="Eval (CHAL metric)"):
        prob = torch.sigmoid(model(clip.to(device))).squeeze(1).cpu().numpy()
        gt = mask.squeeze(1).numpy() > 0.5
        for b, (h, w) in enumerate(hws):
            img_id += 1
            images.append({"id": img_id, "width": int(w), "height": int(h),
                           "file_name": paths[b]})
            for x, y, bw, bh, _, _ in boxes_from_binary(gt[b, :h, :w]):
                ann_id += 1
                anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                             "bbox": [x, y, bw, bh], "area": bw * bh,
                             "iscrowd": 0})
            p = prob[b, :h, :w]
            for x, y, bw, bh, comp, sl in boxes_from_binary(p >= box_thresh):
                dets.append({"image_id": img_id, "category_id": 1,
                             "bbox": [x, y, bw, bh],
                             "score": float(p[sl][comp].max())})
    gt_dict = {"images": images, "annotations": anns,
               "categories": [{"id": 1, "name": "target"}]}
    return chal_metrics(gt_dict, dets), len(anns), len(dets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(LOG_DIR, "best.pth"))
    ap.add_argument("--txt", default=TEST_TXT)
    ap.add_argument("--box_thresh", type=float, default=None,
                    help="binarisation threshold for box extraction "
                         "(default: val-tuned threshold in the checkpoint)")
    args = ap.parse_args()

    device = get_device()
    model, state = load_student(args.ckpt, device)
    thr = args.box_thresh if args.box_thresh is not None else state.get("threshold", DETECT_THRESH)
    m, n_gt, n_det = run(model, args.txt, device, thr)
    print(f"\n── CHAL metric (COCO box, IoU 0.5) @ box_thresh {thr} ──")
    print(f"  GT boxes: {n_gt}   predicted boxes: {n_det}")
    print(f"  mAP50     : {m['mAP50']*100:.2f}")
    print(f"  Precision : {m['Precision']*100:.2f}")
    print(f"  Recall    : {m['Recall']*100:.2f}")
    print(f"  F1        : {m['F1']*100:.2f}")