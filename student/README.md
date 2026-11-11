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

`CHAL/model_data/pretrain.pth` is only the COCO YOLOX-S initialisation used to
*start* CHAL training – it is **not** a teacher (teacher.py rejects it).
Train CHAL on NUDT-MIRSDT following its README (note: the public CHAL repo
imports `nets/module/CBAM.py`, which it does not ship), then:

```bash
cp CHAL/logs/<run>/best_epoch_weights.pth CHAL/logs/best_model.pth
```

### Step 3 – Train the student

**GT-only mode** (no teacher needed):
```bash
cd student
python train_student.py
```

**Distillation mode** (requires a trained CHAL checkpoint):
```bash
python train_student.py --teacher
```

Training uses random 256×256 crops at native resolution; 10 % of the training
sequences are held out for validation (model selection + threshold tuning).
The test set is never used during training.

### Step 4 – Evaluate

```bash
python evaluate.py --ckpt logs/best.pth --profile
```

Metrics follow the official NUDT-MIRSDT protocol (DTUM `ShootingRules.py`):

| Metric     | Meaning |
|------------|---------|
| **Pd**     | target-level detection probability (pred pixel within 1 px of target) |
| **Fa**     | false-alarm pixels (> 4 px from every target) / total pixels |
| **Precision** | object-level: detected targets / (detected + false-alarm components) |
| **F1**     | harmonic mean of Pd and Precision |
| pix_*      | pixel-level metrics (what the previous version reported) |

The threshold tuned on validation is stored in `best.pth`. `--sweep` prints all
thresholds for PR/ROC curves – do not pick the threshold on the test set.

---

## Model design

```
CompactStudent  (~0.53 M params, ~16 GMACs at 5×512×512)
  ├── Anomaly priors (parameter-free, causal):
  │     current frame, current − mean(past frames), current − previous,
  │     current − local mean
  ├── Shared per-frame stem → frames stacked on channels → 3×3 conv
  │     (pixel-wise spatio-temporal fusion)
  ├── 4-level residual U-Net encoder/decoder with full-resolution skips
  └── 1×1 head → logits
```

---

## Distillation losses

| Loss | Weight | Description |
|------|--------|-------------|
| `L_gt`   | 1.0 | BCE-with-logits + SoftIoU vs GT mask |
| `L_pred` | 1.0 | MSE between CHAL's stride-8 confidence map and the student heatmap max-pooled to that grid |
| `L_resp` | 0.5 | MSE between student(x) and student(x_aug) – response invariance |

Weights are in `config.py`.

---

## Augmentation (label-preserving)

Identical for every frame of a clip (motion cues preserved), kept mild so
low-SNR targets survive; no clamping:
1. Random brightness shift (±0.1)
2. Random contrast scale (±20%)
3. Additive Gaussian noise (σ=0.01)
