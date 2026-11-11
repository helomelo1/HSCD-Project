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

# ── Data ─────────────────────────────────────────────────────────────────────
NUM_FRAMES   = 5          # last NUM_FRAMES frames of each 'Mix' clip are used
IN_CHANNELS  = 1          # grayscale
# NUDT-MIRSDT frames are 512x512 (a few are smaller). Images are NOT resized:
# downscaling destroys 1-3 px targets. Evaluation runs at native resolution
# (zero-padded to a multiple of PAD_MULTIPLE, padding is cropped off again).
PAD_MULTIPLE = 32
TRAIN_CROP   = 256        # random crops for training (model is fully conv.)
TARGET_CROP_PROB = 0.8    # prob. that a training crop is forced to contain a target
# Validation = whole sequences held out from train.txt (never select on test).
VAL_EVERY_NTH_SEQ = 10    # every 10th training sequence -> validation (~10%)

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_CH      = 16         # student base channels (widths 16/32/64/128)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 8
NUM_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
GRAD_CLIP    = 5.0
EMA_DECAY    = 0.999      # evaluate/save an exponential moving average of weights
NUM_WORKERS  = 4

# Distillation loss weights
LAMBDA_GT    = 1.0    # ground-truth segmentation loss (BCE + SoftIoU)
LAMBDA_PRED  = 1.0    # prediction matching loss  (teacher soft output)
LAMBDA_RESP  = 0.5    # response-consistency loss (augmented views)

# ── Augmentation (label-preserving) ──────────────────────────────────────────
# Applied identically to all frames of a clip, so frame differences / motion
# are preserved. Kept mild: strong noise/contrast loss erases low-SNR targets,
# which would no longer be label-preserving.
AUG_BRIGHTNESS = 0.1   # ±additive shift
AUG_CONTRAST   = 0.2   # ±fraction, contrast is only ever scaled in [0.8,1.2]
AUG_NOISE_STD  = 0.01  # Gaussian noise std (inputs in [0,1])

# ── Evaluation ────────────────────────────────────────────────────────────────
# Default threshold; train_student.py tunes it on the validation split and
# stores it inside best.pth, which evaluate.py uses automatically.
DETECT_THRESH = 0.5
THRESH_GRID   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Official NUDT-MIRSDT protocol (DTUM ShootingRules): a target is detected if
# any predicted pixel lies within DET_RADIUS px of it; predicted pixels farther
# than FA_RADIUS px from every target are false alarms.
DET_RADIUS   = 1
FA_RADIUS    = 4
