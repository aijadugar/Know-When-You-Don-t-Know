"""Shared, GPU-free-at-import helpers for the abstention pipeline.

Importing this module must stay cheap (no torch/transformers at module top level)
so `pytest` runs in seconds; anything that needs a model lives in a function-local
import inside the stage scripts.

Conventions baked in here (they are the contract every stage relies on):
  * LABEL: y = 1 means the model's greedy answer is WRONG, y = 0 correct.
    The probe's risk score is P(y=1).  See config.LABEL_CONVENTION -- never invert.
  * PROMPT: build_prompt() renders [system, user] through
    apply_chat_template(messages, tokenize=False, add_generation_prompt=True).
    The prompt contains the QUESTION ONLY, never the answer or the gold aliases.
  * PADDING: tokenizer padding_side="left", so position -1 of input_ids is the last
    REAL prompt token for every row and generated text starts at
    input_ids.shape[1] for every row of the batch.
  * MATCHING: whole-word span matching after normalize_answer(); see its docstring
    for the two known failure modes (valid paraphrases -> labelled wrong; very
    short gold answers -> matched strictly).
"""
from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

MIN_ALIAS_LEN = 2          # normalized aliases shorter than this are ignored
_EXTRA_PUNCT = "\u201c\u201d\u2018\u2019\u2013\u2014\u2026"   # “ ” ‘ ’ – — …
_PUNCT_TABLE = str.maketrans(string.punctuation + _EXTRA_PUNCT,
                             " " * (len(string.punctuation) + len(_EXTRA_PUNCT)))
_ARTICLES = {"a", "an", "the"}


# --------------------------------------------------------------------------- seeds
def set_seed(seed: int) -> None:
    """Delegate to config.set_seed (python/numpy/torch/dataset shuffles all use it)."""
    from config import set_seed as _set_seed

    _set_seed(seed)


# --------------------------------------------------------------------------- jsonl
def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run the earlier pipeline stage first "
            f"(see Makefile targets), e.g. `make data`."
        )
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:  # loud, with the offending line
                raise ValueError(f"{p}:{i} is not valid JSON: {e}\n  {line[:200]}") from e
    return rows


def save_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def count_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# --------------------------------------------------------------------------- prompt
def build_prompt(tokenizer: Any, question: str) -> str:
    """Render the chat prompt (system + question) as TEXT.

    Verified API (transformers==4.53.2):
      tokenizer.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True) -> str
    The system prompt comes from config.SYSTEM_PROMPT; the answer/golds are NOT
    part of the prompt, so activations at the last prompt token cannot leak them.
    """
    from config import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------- model I/O
def load_model_and_tokenizer(model_id: str, device: str | None = None):
    """Return (model, tokenizer, dtype_name).

    * dtype: bfloat16 if torch.cuda.is_bf16_supported() else float16 on CUDA,
      float32 on CPU (config.pick_dtype); the chosen name is printed (rule 5).
    * model: eval(), no grad needed by callers, moved to `device` when CUDA exists.
    * tokenizer: padding_side="left"; pad token verified before touching it --
      Qwen2.5 ships pad_token=<|padding|> (id 151643), so pad=eos is a fallback
      that only fires if a future checkpoint drops it.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from config import pick_dtype, resolve_device

    dev = resolve_device(device)
    dtype, dtype_name = pick_dtype()
    if dev == "cpu":
        dtype, dtype_name = torch.float32, "float32(cpu)"

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        print(f"[common] WARNING: {model_id} has no pad token ({tok.pad_token!r}); "
              f"falling back to eos={tok.eos_token!r}")
        tok.pad_token = tok.eos_token
    else:
        print(f"[common] pad token OK: {tok.pad_token!r} (id={tok.pad_token_id}), "
              f"eos={tok.eos_token!r} (id={tok.eos_token_id})")

    kwargs: dict[str, Any] = {}
    if dev == "cuda":
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    if dev != "cuda":
        model = model.to(dev)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"[common] model={model_id} device={dev} dtype={dtype_name} "
          f"torch={torch.__version__}")
    return model, tok, dtype_name


# --------------------------------------------------------------------------- normalization
def normalize_answer(text: str) -> str:
    """lowercase -> punctuation to SPACES -> drop articles -> collapse whitespace.

    Punctuation becomes a space rather than being deleted on purpose: deleting it
    would splice "19:69" into "1969" and create false matches.
    """
    s = str(text).lower().translate(_PUNCT_TABLE)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def _whole_word_span(haystack: str, needle: str) -> bool:
    """True iff `needle` occurs in `haystack` delimited by non-word chars."""
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def is_match(answer: str, gold_aliases: Sequence[str]) -> bool:
    """Whole-word match between the model answer and ANY gold alias, both sides
    normalized.  Match counts when the answer contains an alias as a whole-word
    span OR an alias contains the answer as a whole-word span.

    Ignored on purpose: empty/whitespace answers (never match -> labelled wrong)
    and aliases shorter than MIN_ALIAS_LEN=2 chars after normalization (a 1-char
    alias such as "1" would otherwise match "1990").

    KNOWN LIMITS (accepted, they define the label noise we report):
      1. Valid paraphrases are labelled WRONG: gold "Mark Twain" vs answer
         "Samuel Clemens" -> y=1 even though the model is right.
      2. Very short gold answers are matched strictly: gold "7" vs answer
         "Seven" -> y=1; conversely any 1-char alias is dropped entirely, so
         gold ["A"] can never be matched and always yields y=1.
    """
    a = normalize_answer(answer)
    if not a:
        return False
    for g in gold_aliases:
        gn = normalize_answer(g)
        if len(gn) < MIN_ALIAS_LEN:
            continue
        if gn == a or _whole_word_span(a, gn) or _whole_word_span(gn, a):
            return True
    return False


def label_from_match(matched: bool) -> int:
    """THE label convention: matched (correct) -> y=0; not matched (WRONG) -> y=1."""
    return 0 if matched else 1


def make_label(answer: str, gold_aliases: Sequence[str]) -> int:
    """Convenience: is_match + label_from_match in one call."""
    return label_from_match(is_match(answer, gold_aliases))


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --------------------------------------------------------------------------- metrics
def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC of `score` against y (y=1 wrong).  A risk score (higher => more likely
    wrong) gives AUROC > 0.5; pass -confidence for the confidence baselines."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64)))


def calibrate_threshold_for_coverage(p_wrong: np.ndarray, target: float) -> float:
    """Threshold on P(y=1) answering ~`target` of the rows (ties break conservatively)."""
    p = np.sort(np.asarray(p_wrong, dtype=np.float64))
    k = max(1, int(round(target * len(p))))
    return float(p[k - 1])


def accuracy_at_threshold(y: np.ndarray, p_wrong: np.ndarray, thr: float) -> dict:
    y = np.asarray(y)
    p = np.asarray(p_wrong, dtype=np.float64)
    answered = p <= thr
    n = int(answered.sum())
    return {
        "threshold_p_wrong": float(thr),
        "realised_coverage": n / len(y) if len(y) else float("nan"),
        "n_answered": n,
        "n_abstained": int(len(y) - n),
        "accuracy_answered": float((y[answered] == 0).mean()) if n else float("nan"),
        "selective_error_rate": float((y[answered] == 1).mean()) if n else float("nan"),
    }


def coverage_accuracy_curve(y: np.ndarray, p_wrong: np.ndarray,
                            coverages: Sequence[float]) -> dict:
    """Answer the `coverage` fraction of rows with the LOWEST predicted risk."""
    y = np.asarray(y)
    order = np.argsort(np.asarray(p_wrong, dtype=np.float64), kind="stable")
    ys = y[order]
    out: dict[str, dict] = {}
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        out[f"{c:.2f}"] = {
            "requested_coverage": float(c),
            "realised_coverage": k / len(y),
            "n_answered": k,
            "accuracy_answered": float((ys[:k] == 0).mean()),
        }
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def sample_rows(rows: Sequence[dict], k: int, seed: int) -> list[dict]:
    """Deterministic random sample without replacement (seeded, rule 5)."""
    if k >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(list(rows), k)
