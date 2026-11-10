"""
Central config for the compact student distillation project.
Edit paths and hyper-parameters here before running any script.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(ROOT, "NUDT-MIRSDT")          # dataset root
TRAIN_TXT    = os.path.join(DATASET_DIR, "train.txt")     # list of .mat clips
TEST_TXT     = os.path.join(DATASET_DIR, "test.txt")
CHAL_DIR     = os.path.join(ROOT, "CHAL")                 # CHAL repo root
TEACHER_CKPT = os.path.join(CHAL_DIR, "logs", "best_model.pth")  # fill in after training CHAL
STUDENT_DIR  = os.path.join(ROOT, "student")
LOG_DIR      = os.path.join(STUDENT_DIR, "logs")

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_FRAMES   = 5          # temporal clip length (matches CHAL Mix .mat shape[0])
IMG_SIZE     = 256        # spatial resize for student (smaller than CHAL's 512)
IN_CHANNELS  = 1          # grayscale
BASE_CH      = 16         # student backbone base channels (keep light)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 4
NUM_EPOCHS   = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4

# Distillation loss weights
LAMBDA_GT    = 1.0    # ground-truth binary segmentation loss
LAMBDA_PRED  = 1.0    # prediction matching loss  (teacher soft output)
LAMBDA_RESP  = 0.5    # response-consistency loss (augmented views)

# ── Augmentation (label-preserving) ──────────────────────────────────────────
# These only perturb background intensity – target location is unchanged.
AUG_BRIGHTNESS = 0.3   # ±fraction
AUG_CONTRAST   = 0.3
AUG_NOISE_STD  = 0.05  # Gaussian noise std (after [0,1] normalisation)

# ── Evaluation ────────────────────────────────────────────────────────────────
DETECT_THRESH = 0.5    # binarise student heatmap at this threshold
