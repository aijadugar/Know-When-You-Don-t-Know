"""Central configuration for the abstention project.

Everything that later stages (scripts 02-08, app.py) need is defined here so no
script hardcodes a model id, a path, or a dtype.  Run `python config.py` to print
the resolved environment facts (device, dtype, layer count read from config.json).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- models
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"      # real run
MODEL_500M = "Qwen/Qwen2.5-0.5B-Instruct"  # fast smoke-test model

# ----------------------------------------------------------------------------- datasets
TRIVIAQA_NAME = "mandarjoshi/trivia_qa"
TRIVIAQA_CONFIG = "rc.nocontext"
NQOPEN_NAME = "google-research-datasets/nq_open"

# Verified on 2026-11-12 against datasets==4.0.0:
#   * TriviaQA rc.nocontext TEST split answers are WITHHELD (all aliases empty,
#     value == '<unk>').  We therefore never use the test split; train+validation
#     are sampled into disjoint pools (see scripts/02_prepare_data.py).
#   * NQ-Open rows are {'question': str, 'answer': list[str]} -- NOT answer.aliases.
TRIVIAQA_TEST_SPLIT_HASHELD = True

# ----------------------------------------------------------------------------- paths
DATA_DIR = os.environ.get("ABSTAIN_DATA_DIR", "data")
ACTS_DIR = os.environ.get("ABSTAIN_ACTS_DIR", "acts")
MODELS_DIR = os.environ.get("ABSTAIN_MODELS_DIR", "models")
RESULTS_DIR = os.environ.get("ABSTAIN_RESULTS_DIR", "results")

SPLIT_FILES = {
    "train": os.path.join(DATA_DIR, "train_labeled.jsonl"),
    "val": os.path.join(DATA_DIR, "val_labeled.jsonl"),
    "test": os.path.join(DATA_DIR, "test_labeled.jsonl"),
    "ood": os.path.join(DATA_DIR, "ood_labeled.jsonl"),
}
SPLIT_FILES_UNLABELED = {k: v.replace("_labeled", "") for k, v in SPLIT_FILES.items()}

def acts_path(split: str) -> str:
    return os.path.join(ACTS_DIR, f"{split}.npy")

def acts_logp_path(split: str) -> str:
    return os.path.join(ACTS_DIR, f"{split}_logprob.npy")

PROBE_PATH = os.path.join(MODELS_DIR, "probe.joblib")
PROBE_META_PATH = os.path.join(MODELS_DIR, "probe_meta.json")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# ----------------------------------------------------------------------------- method
TARGET_COVERAGE = 0.70          # headline operating point
C_GRID = [0.01, 0.1, 1.0, 10.0]  # logistic-regression regularization grid
MAX_NEW_TOKENS = 24             # greedy decoding budget
GEN_TEMPERATURE = 0.0           # greedy
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 1234
LABEL_CONVENTION = (
    "y = 1 means the greedy answer matches NO gold alias (model is WRONG); "
    "y = 0 means at least one gold alias matched (model is CORRECT)."
)

PROMPT_TEMPLATE = "{question}"  # wrapped by the chat template in common.build_prompt


@dataclass
class RunFlags:
    """Every script takes --model_id and --n_* flags so the same code runs a 0.5B
    smoke test on a tiny sample and then the real run."""
    model_id: str = MODEL_3B
    n_train: int = 3000
    n_val: int = 1000
    n_test: int = 1000
    n_ood: int = 1000
    batch_size: int = 8
    seed: int = 0
    max_new_tokens: int = MAX_NEW_TOKENS
    device: str | None = None      # auto-detected when None
    dtype: str | None = None       # auto-selected when None (see pick_dtype)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--model_id", default=RunFlags.model_id)
        parser.add_argument("--n_train", type=int, default=RunFlags.n_train)
        parser.add_argument("--n_val", type=int, default=RunFlags.n_val)
        parser.add_argument("--n_test", type=int, default=RunFlags.n_test)
        parser.add_argument("--n_ood", type=int, default=RunFlags.n_ood)
        parser.add_argument("--batch_size", type=int, default=RunFlags.batch_size)
        parser.add_argument("--seed", type=int, default=RunFlags.seed)
        parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
        parser.add_argument("--device", default=None)
        parser.add_argument("--dtype", default=None, choices=[None, "bf16", "fp16", "fp32"])
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunFlags":
        return cls(**{f.name: getattr(args, f.name) for f in cls.__dataclass_fields__.values()})


# ----------------------------------------------------------------------------- dtype
def pick_dtype(device: str, forced: str | None = None):
    """Return (torch_dtype, name).

    The free Colab T4 has no native bf16, so: bf16 where the hardware supports it,
    fp16 otherwise on CUDA, fp32 on CPU.  Callers must LOG which one ran; the
    README states this deviation from the original 'bf16' constraint.
    """
    import torch

    if forced == "fp32":
        return torch.float32, "fp32"
    if device == "cpu":
        return torch.float32, "fp32"
    if forced == "fp16":
        return torch.float16, "fp16"
    if forced == "bf16" or forced is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        if forced == "bf16":
            raise RuntimeError("bf16 requested but not supported on this device")
        return torch.float16, "fp16"
    raise ValueError(forced)


def resolve_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_environment(model_id: str) -> dict:
    """Read layer count / hidden size from the model's config.json -- never trusted
    from prose.  Cheap: loads only the config, not the weights."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    return {
        "model_id": model_id,
        "n_hidden_layers": cfg.num_hidden_layers,
        "n_hidden_states_rows": cfg.num_hidden_layers + 1,  # + embedding row
        "d_model": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
    }


if __name__ == "__main__":
    import json

    import torch

    dev = resolve_device()
    dt, dt_name = pick_dtype(dev)
    out = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": dev,
        "dtype": dt_name,
        **describe_environment(MODEL_3B),
    }
    print(json.dumps(out, indent=2))
