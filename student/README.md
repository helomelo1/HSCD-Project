# Compact Motion-Anomaly Response Distillation (CausalMARD)

**Project P3 — Thermal Target Detection**

Distils CHAL's hierarchical anomaly-aware predictions into a compact temporal student model, while enforcing response consistency under label-preserving background augmentations.

---

## Directory layout

```
student/
  config.py          – all paths & hyperparameters
  dataset.py         – NUDT-MIRSDT dataloader (5-frame .mat clips + PNG masks)
  model.py           – CompactStudent (causal temporal conv + DW-Sep UNet)
  augment.py         – label-preserving background augmentations
  losses.py          – distillation loss (GT-BCE + pred-match + resp-consistency)
  teacher.py         – CHAL teacher wrapper (or DummyTeacher for GT-only mode)
  evaluate.py        – Pd, Fa, Precision, F1 metrics
  train_student.py   – main training entry-point
  logs/              – checkpoints & CSV log (created on first run)
```

---

## Step-by-step guide

### Step 0 – Install dependencies

```bash
pip install torch torchvision scipy opencv-python tqdm numpy
```

### Step 1 – Check config

Open `config.py` and verify:
- `DATASET_DIR` points to `NUDT-MIRSDT/`
- `TRAIN_TXT` / `TEST_TXT` point to `train.txt` / `test.txt`
- `TEACHER_CKPT` points to a trained CHAL `best_model.pth` *(if you want distillation)*

### Step 2 – (Optional) Train CHAL teacher first

Follow the CHAL `README.md` to train on NUDT-MIRSDT, then copy the best checkpoint:

```bash
# in CHAL/
cp logs/<run>/best_model.pth logs/best_model.pth
```

### Step 3 – Train the student

**GT-only mode** (no teacher needed – good for initial experiments):
```bash
cd student
python train_student.py
```

**Distillation mode** (requires CHAL checkpoint):
```bash
python train_student.py --teacher
```

### Step 4 – Evaluate

```bash
python evaluate.py --ckpt logs/best.pth
```

Outputs:
| Metric     | Meaning                          |
|------------|----------------------------------|
| **Pd**     | Detection probability (recall)   |
| **Fa**     | False-alarm rate (FP/total px)   |
| **Precision** | TP / (TP + FP)              |
| **F1**     | Harmonic mean of Pd & Precision  |

---

## Model design

```
CompactStudent
  ├── Stem (DW-Sep, 1→16 ch)
  ├── TemporalBlock × 2
  │     ├── Spatial DW-Sep (per frame)
  │     └── CausalTempConv (past→present, no future leakage)
  ├── Spatial Encoder (3-scale DW-Sep, max-pool)
  ├── Decoder (skip connections, bilinear upsample)
  └── Head (1×1 conv → sigmoid)
```

**Parameters: ~16 k** (vs CHAL's SlowFast + background field: ~tens of millions)

---

## Distillation losses

| Loss | Weight | Description |
|------|--------|-------------|
| `L_gt`   | 1.0 | Weighted BCE vs GT mask (pos_weight=10 for class imbalance) |
| `L_pred` | 1.0 | MSE between student and CHAL teacher heatmaps |
| `L_resp` | 0.5 | MSE between student(x) and student(x_aug) – response invariance |

Weights are in `config.py`.

---

## Augmentation (label-preserving)

Applied only to background intensity — target pixel locations are unchanged:
1. Random brightness shift (±30%)
2. Random contrast scale (±30%)
3. Additive Gaussian noise (σ=0.05)
