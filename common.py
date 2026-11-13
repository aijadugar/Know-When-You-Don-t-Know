"""common.py — shared utilities. No GPU imports at module level."""
from __future__ import annotations
import json
import re
import string
from typing import Any
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Seeds
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from sklearn.utils import check_random_state  # noqa: E402

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # sklearn's check_random_state also respects this
    np.random.default_rng(seed)

# ─────────────────────────────────────────────────────────────────────────────
# JSONL helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """Save a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(tokenizer: Any, question: str, system_prompt: str) -> str:
    """Build a chat-formatted prompt string.
    Uses tokenizer.apply_chat_template with tokenize=False so the caller
    can tokenize with left-padding in a batch.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(
    model_id: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[Any, Any]:
    """Load a causal model and tokenizer.
    - dtype is chosen at runtime based on hardware support (bfloat16 or float16).
    - Tokenizer: padding_side="left", pad token set to EOS if missing.
    - Model: eval mode, on CUDA if available.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Determine dtype for the model (bf16 if supported, else fp16)
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    print(f"[load_model_and_tokenizer] model={model_id}, dtype={dtype}")
    print(
        f"[load_model_and_tokenizer] CUDA available: {torch.cuda.is_available()}, "
        f"bf16 supported: {torch.cuda.is_bf16_supported()}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        trust_remote_code=True,
    )

    # Ensure pad token exists; set to EOS if missing
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("[load_model_and_tokenizer] Set pad_token = eos_token")
        else:
            raise ValueError("Neither pad_token nor eos_token found in tokenizer")

    return model, tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Answer normalization
# ─────────────────────────────────────────────────────────────────────────────
def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison.
    Lowercase, strip, remove articles a/an/the, collapse whitespace.
    """
    text = text.lower().strip()
    # Remove articles
    for article in ("a", "an", "the"):
        text = re.sub(r"\b" + article + r"\b", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────
def _whole_word_match(text: str, pattern: str) -> bool:
    """Return True if pattern appears as a whole word (or number) in text."""
    # Escape special regex chars except word boundaries
    escaped = re.escape(pattern)
    return bool(re.search(rf"\b{escaped}\b", text))

def is_match(answer: str, gold_aliases: list[str]) -> bool:
    """Check if answer matches any gold alias.
    Strategy:
    - Normalize both answer and each alias.
    - Ignore aliases shorter than 2 characters after normalization.
    - Ignore empty answers.
    - Match if:
      (a) normalized answer equals a normalized alias, OR
      (b) the alias contains the answer as a whole-word span, OR
      (c) the answer contains the alias as a whole-word span.
    Known limits:
    - Valid paraphrases are labelled wrong (e.g. "Paris, France" vs "Paris").
    - Short gold answers are matched strictly.
    """
    if not answer or not answer.strip():
        return False

    norm_answer = normalize_answer(answer)

    for alias in gold_aliases:
        norm_alias = normalize_answer(alias)
        # Ignore very short aliases
        if len(norm_alias) < 2:
            continue

        # (a) Exact equality
        if norm_answer == norm_alias:
            return True

        # (b) Alias contains answer as whole word
        if _whole_word_match(norm_alias, norm_answer):
            return True

        # (c) Answer contains alias as whole word
        if _whole_word_match(norm_answer, norm_alias):
            return True

    return False

def label_from_match(matched: bool) -> int:
    """Convert a match result to a label.
    y=1 means the model's answer is WRONG (not matched).
    y=0 means the model's answer is CORRECT (matched).
    """
    return 0 if matched else 1
