"""Single source of truth for every constant in the abstention project.

RULE 5: no other file hardcodes a model id, size, seed, path or dtype -- they all
import from here.  Run `python config.py` to print the resolved environment facts
(device, dtype, layer count / hidden size read live from the model's config.json).

Label convention used throughout (never inverted): y = 1 means the model's greedy
answer is WRONG (hallucination), y = 0 means correct; probe risk score = P(y=1).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- models
MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"       # the only real target model
SMOKE_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"  # CPU smoke target

# Verified 2026-11-12 with huggingface_hub==0.33.4 `model_info()` (network call):
#   Qwen/Qwen2.5-3B-Instruct    exists, 3,085,938,688 params (BF16), 12 files
#   Qwen/Qwen2.5-0.5B-Instruct  exists,   494,032,768 params (BF16), 10 files

# --------------------------------------------------------------------------- prompt
SYSTEM_PROMPT: str = (
    "Answer with a short factual answer only (a few words). "
    "If unsure, still give your best guess."
)
MAX_NEW_TOKENS: int = 24          # greedy decoding budget
GREEDY_NOTE: str = "do_sample=False; temperature/top_p are never passed"

# --------------------------------------------------------------------------- sizes
N_TRAIN: int = 3000
N_VAL: int = 500
N_TEST: int = 1000
N_OOD: int = 1000
SMOKE_N_TRAIN: int = 24           # --smoke overrides the four sizes above
SMOKE_N_VAL: int = 8
SMOKE_N_TEST: int = 8
SMOKE_N_OOD: int = 8

# --------------------------------------------------------------------------- probe
C_VALUES: list[float] = [0.01, 0.1, 1.0]   # logistic-regression C grid
TARGET_COVERAGE: float = 0.70              # headline operating point
COVERAGES: list[float] = [0.9, 0.8, 0.7, 0.6]
SEED: int = 42
BOOTSTRAP_N: int = 2000

LABEL_CONVENTION: str = (
    "y = 1 means the greedy answer matches NO gold alias (the model is WRONG); "
    "y = 0 means at least one gold alias matched (the model is CORRECT). "
    "The probe's risk score is P(y=1); higher risk => abstain."
)

# --------------------------------------------------------------------------- data ids
TRIVIAQA_NAME: str = "mandarjoshi/trivia_qa"
TRIVIAQA_CONFIG: str = "rc.nocontext"
NQOPEN_NAME: str = "google-research-datasets/nq_open"
SPLITS: tuple[str, ...] = ("train", "val", "test", "ood")

# --------------------------------------------------------------------------- hub
HF_REPO_ID: str | None = os.environ.get("HF_REPO_ID")  # set in the env, never hardcoded

# --------------------------------------------------------------------------- paths
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
ACTS_DIR: Path = ROOT_DIR / "acts"
MODELS_DIR: Path = ROOT_DIR / "models"
RESULTS_DIR: Path = ROOT_DIR / "results"


class Paths:
    """All output paths for one run.  `smoke=True` redirects everything into
    *_smoke/ directories so a 0.5B smoke run can never clobber a real run."""

    def __init__(self, smoke: bool = False, suffix: str = "_smoke") -> None:
        self.smoke = smoke
        s = suffix if smoke else ""
        self.root: Path = ROOT_DIR
        self.data_dir: Path = Path(f"{DATA_DIR}{s}")
        self.acts_dir: Path = Path(f"{ACTS_DIR}{s}")
        self.models_dir: Path = Path(f"{MODELS_DIR}{s}")
        self.results_dir: Path = Path(f"{RESULTS_DIR}{s}")

    # ---- individual files
    def questions(self, split: str) -> Path:
        return self.data_dir / f"{split}.jsonl"

    def labeled(self, split: str) -> Path:
        return self.data_dir / f"{split}_labeled.jsonl"

    def acts(self, split: str) -> Path:
        return self.acts_dir / f"{split}.npy"

    @property
    def probe(self) -> Path:
        return self.models_dir / "probe.joblib"

    @property
    def probe_meta(self) -> Path:
        return self.models_dir / "probe_meta.json"

    @property
    def metrics(self) -> Path:
        return self.results_dir / "metrics.json"

    def mkdirs(self) -> None:
        for d in (self.data_dir, self.acts_dir, self.models_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {
            "smoke": self.smoke,
            "data_dir": str(self.data_dir),
            "acts_dir": str(self.acts_dir),
            "models_dir": str(self.models_dir),
            "results_dir": str(self.results_dir),
        }


# --------------------------------------------------------------------------- dtype
def pick_dtype(torch_module=None) -> tuple[object, str]:
    """Return (torch.dtype, name): bfloat16 when the device supports it, else
    float16 on CUDA, float32 on CPU.  Callers MUST log the returned name
    (free Colab T4s have no bf16 -- the README states this deviation)."""
    import torch as _torch

    t = torch_module or _torch
    if not t.cuda.is_available():
        return t.float32, "float32(cpu)"
    if t.cuda.is_bf16_supported():
        return t.bfloat16, "bfloat16"
    return t.float16, "float16"


def resolve_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- seeds
def set_seed(seed: int = SEED) -> None:
    """Fixed seeds everywhere (rule 5): python, numpy, torch (+CUDA), and every
    sklearn estimator that takes random_state gets `seed` explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch always present in the pipeline
        pass


# --------------------------------------------------------------------------- CLI glue
def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags shared by every script: --model_id override, --smoke, --seed, sizes."""
    parser.add_argument("--model_id", default=None,
                        help=f"override MODEL_ID ({MODEL_ID}); e.g. {SMOKE_MODEL_ID}")
    parser.add_argument("--smoke", action="store_true",
                        help=f"use SMOKE_MODEL_ID={SMOKE_MODEL_ID}, tiny sizes and "
                             "*_smoke/ output dirs")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def apply_run_flags(args: argparse.Namespace,
                    n_train: int | None = None,
                    n_val: int | None = None,
                    n_test: int | None = None,
                    n_ood: int | None = None) -> dict:
    """Resolve --model_id / --smoke into a plain dict every stage passes around.

    Precedence: explicit flag > smoke defaults > config defaults.  Sizes given as
    None fall back to N_TRAIN/N_VAL/N_TEST/N_OOD (or the SMOKE_* values).
    """
    smoke = bool(getattr(args, "smoke", False))
    model_id = getattr(args, "model_id", None) or (SMOKE_MODEL_ID if smoke else MODEL_ID)
    if smoke:
        n_train = SMOKE_N_TRAIN if n_train is None else n_train
        n_val = SMOKE_N_VAL if n_val is None else n_val
        n_test = SMOKE_N_TEST if n_test is None else n_test
        n_ood = SMOKE_N_OOD if n_ood is None else n_ood
    else:
        n_train = N_TRAIN if n_train is None else n_train
        n_val = N_VAL if n_val is None else n_val
        n_test = N_TEST if n_test is None else n_test
        n_ood = N_OOD if n_ood is None else n_ood
    seed = int(getattr(args, "seed", SEED))
    set_seed(seed)
    paths = Paths(smoke=smoke)
    paths.mkdirs()
    print(f"[config] model_id={model_id} smoke={smoke} seed={seed} "
          f"n=(train={n_train}, val={n_val}, test={n_test}, ood={n_ood}) "
          f"dirs={paths.as_dict()}")
    return {"model_id": model_id, "smoke": smoke, "seed": seed, "paths": paths,
            "n_train": n_train, "n_val": n_val, "n_test": n_test, "n_ood": n_ood}


def describe_environment(model_id: str = MODEL_ID) -> dict:
    """Layer count / hidden size read from the model's config.json (never assumed)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    return {
        "model_id": model_id,
        "n_hidden_layers": cfg.num_hidden_layers,
        "n_hidden_states_rows": cfg.num_hidden_layers + 1,
        "d_model": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
    }


if __name__ == "__main__":
    import torch

    dev = resolve_device()
    dt, dt_name = pick_dtype()
    out = {
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": dev,
        "dtype": dt_name,
        "label_convention": LABEL_CONVENTION,
        "paths": Paths(smoke=False).as_dict(),
        "smoke_paths": Paths(smoke=True).as_dict(),
        "env_3B": describe_environment(MODEL_ID),
        "env_smoke": describe_environment(SMOKE_MODEL_ID),
    }
    print(json.dumps(out, indent=2))
