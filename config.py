"""config.py — all constants for know-when-you-dont-know."""
from __future__ import annotations
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
ACTS_DIR = REPO_ROOT / "acts"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
ACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Model IDs ─────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SMOKE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Answer with a short factual answer only (a few words). "
    "If unsure, still give your best guess."
)

# ── Generation ─────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 24

# ── Dataset sizes ─────────────────────────────────────────────────────────────
N_TRAIN = 3000
N_VAL = 500
N_TEST = 1000
N_OOD = 1000

# ── Probe ─────────────────────────────────────────────────────────────────────
C_VALUES = [0.01, 0.1, 1.0]
TARGET_COVERAGE = 0.70
COVERAGES = [0.9, 0.8, 0.7, 0.6]

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED = 42

# ── Hub ───────────────────────────────────────────────────────────────────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
