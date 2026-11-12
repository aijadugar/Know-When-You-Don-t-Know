"""Shared utilities: normalization/matching, labeling, prompting, generation,
logit-based baselines, activation extraction, AUROC and bootstrap CIs.

Design rules baked in (each corresponds to a defect found in the first draft):
  * y = 1 means WRONG (matches no gold alias).  See LABEL_CONVENTION.
  * Whole-word matching with a minimum alias length -- plain substring would let
    "1" match "1990" and short answers match everything.
  * TriviaQA golds come from answer.normalized_aliases; NQ-Open golds are a plain
    list of strings (NOT answer.aliases).
  * Prompts go through apply_chat_template(..., tokenize=False,
    add_generation_prompt=True) and are THEN tokenized as strings -- chat-template
    dicts passed straight to the tokenizer crash or build the wrong prompt.
  * Left padding; generated text is sliced at input_ids.shape[1] (attention_mask
    sums break under left padding).
  * Baseline scores are computed over REAL tokens only: positions after the first
    EOS are masked out so padding can't bias them by answer length.
  * Activations come from output_hidden_states=True, index -1 on the sequence axis
    (the last prompt token), never from per-layer forward hooks.
"""
from __future__ import annotations

import json
import os
import re
import string
from typing import Iterable, Sequence

import numpy as np

MIN_ALIAS_LEN = 2  # normalized aliases shorter than this are dropped from matching

# --------------------------------------------------------------------------- jsonl
def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- matching
def normalize(s: str) -> str:
    """TriviaQA-style normalization: lowercase, punctuation becomes a SPACE (not
    deleted -- deleting would splice '19:69' into '1969' and create false hits),
    whitespace collapsed."""
    s = s.lower()
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _wordwise_regex(alias_norm: str) -> re.Pattern:
    # whole-word match: \b boundaries so "1" does not match "1990"
    esc = re.escape(alias_norm)
    return re.compile(rf"(?<!\w){esc}(?!\w)")


def answer_matches_gold(answer: str, golds_norm: Sequence[str]) -> bool:
    """True iff `answer` contains any gold alias as a whole word.

    golds_norm must already be normalized(); aliases shorter than MIN_ALIAS_LEN
    are ignored (a 1-char alias matches too much to be trusted).
    """
    a = normalize(answer)
    if not a:
        return False
    for g in golds_norm:
        if len(g) < MIN_ALIAS_LEN:
            continue
        if _wordwise_regex(g).search(a):
            return True
    return False


def make_label(answer: str, golds_norm: Sequence[str]) -> int:
    """LABEL CONVENTION: y = 1 when the answer matches NO gold alias (WRONG)."""
    return 0 if answer_matches_gold(answer, golds_norm) else 1


def golds_for(row_kind: str, answer_field) -> list[str]:
    """Extract normalized gold aliases.

    row_kind == 'triviaqa': answer_field is the HF dict with aliases /
        normalized_aliases (use normalized_aliases; fall back to aliases).
    row_kind == 'nq_open': answer_field is a LIST OF STRINGS (verified field
        layout); there is no .aliases attribute.
    """
    if row_kind == "triviaqa":
        gl = answer_field.get("normalized_aliases") or []
        if not gl:
            gl = answer_field.get("aliases") or []
    elif row_kind == "nq_open":
        assert isinstance(answer_field, list), type(answer_field)
        gl = answer_field
    else:
        raise ValueError(row_kind)
    return sorted({normalize(g) for g in gl if normalize(g)})


# --------------------------------------------------------------------------- model I/O
def load_model_and_tokenizer(model_id: str, device: str, dtype_name: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from config import pick_dtype

    torch_dtype, used = (None, "fp32") if device == "cpu" else pick_dtype(device, dtype_name)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    print(f"[common] model={model_id} device={device} dtype={used}")
    return model, tok, used


def build_prompt(question: str, tokenizer) -> str:
    """Chat template applied to a message DICT but rendered to TEXT
    (tokenize=False, add_generation_prompt=True); we tokenize the string later."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def first_eos_position(gen_ids: np.ndarray, eos_id: int) -> int:
    """Index of the first EOS in generated ids, or len(gen_ids)."""
    hits = np.flatnonzero(np.asarray(gen_ids) == eos_id)
    return int(hits[0]) if hits.size else int(len(gen_ids))


def decode_answer(tokenizer, gen_ids: Sequence[int], eos_id: int) -> str:
    """Decode generated ids up to (excluding) the FIRST EOS.

    Caller passes only the continuation ids, i.e. out_ids[:, input_ids.shape[1]:]
    -- under left padding attention_mask.sum() would cut the wrong position.
    """
    real = list(gen_ids[: first_eos_position(np.asarray(gen_ids), eos_id)])
    if not real:
        return ""
    return tokenizer.decode(real, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- generation + baselines
def generate_and_score(model, tokenizer, prompts: Sequence[str], device: str,
                       max_new_tokens: int, batch_size: int):
    """Greedy-generate answers for `prompts` and compute, per row:

      answers        decoded text (stops at first EOS)
      mean_logprob   baseline (a): MEAN LOG-PROB of the CHOSEN token over real
                     generated tokens only (positions after the first EOS masked;
                     padding never averaged in)
      first_maxprob  baseline (b): MAX SOFTMAX PROB of the first generated token
      gen_ids        raw continuation ids [N, T]

    Returns dict of lists/arrays.  Scores are oriented so that HIGHER = more
    confident; AUROC is computed against y with sign flip handled by the caller.
    """
    import torch
    import torch.nn.functional as F

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    answers, mean_logs, first_maxs, all_ids = [], [], [], []

    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        in_len = enc["input_ids"].shape[1]  # slice point under LEFT padding
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen = out.sequences[:, in_len:]                      # [B, T]
        T = gen.shape[1]
        # stack per-step logits -> log-softmax -> log-prob of chosen token
        step_logits = torch.stack(out.scores, dim=1).float()  # [B, T, V]
        step_logp = F.log_softmax(step_logits, dim=-1)
        chosen = step_logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1)  # [B, T]
        probs = step_logp.exp()
        maxp_first = probs[:, 0].max(dim=-1).values                 # [B]

        for b in range(gen.shape[0]):
            ids_b = gen[b].tolist()
            n_real = first_eos_position(ids_b, eos_id)
            n_real = max(n_real, 1)  # keep at least the first token
            answers.append(decode_answer(tokenizer, ids_b, eos_id))
            mean_logs.append(float(chosen[b, :n_real].mean()))
            first_maxs.append(float(maxp_first[b]))
            all_ids.append(ids_b)
        del out, step_logits, step_logp, chosen, probs
        if device == "cuda":
            torch.cuda.empty_cache()

    maxlen = max(len(x) for x in all_ids) if all_ids else 0
    ids_arr = np.full((len(all_ids), maxlen), pad_id, dtype=np.int64)
    for b, x in enumerate(all_ids):
        ids_arr[b, : len(x)] = x
    return {
        "answers": answers,
        "mean_logprob": np.asarray(mean_logs, dtype=np.float64),
        "first_maxprob": np.asarray(first_maxs, dtype=np.float64),
        "gen_ids": ids_arr,
    }


# --------------------------------------------------------------------------- activations
def extract_last_token_activations(model, tokenizer, prompts: Sequence[str],
                                   device: str, batch_size: int) -> np.ndarray:
    """Hidden state at the LAST PROMPT TOKEN for every layer, float16.

    output_hidden_states=True; hidden_states[l] is [B, L, d]; we take [:, -1, :]
    which is the last prompt token because padding is on the LEFT.  No forward
    hooks (fragile, easy to read the wrong tensor).  Result: [N, n_layers+1, d].
    """
    import torch

    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)  # [B, L+1, d]
        outs.append(hs.to(torch.float16).cpu().numpy())
        del out, hs
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------- metrics
def auroc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64)))


def coverage_accuracy_curve(y: np.ndarray, p_wrong: np.ndarray, coverages=(0.5, 0.6, 0.7, 0.8)):
    """Answer the top-(coverage fraction) most-confident-correct rows.

    p_wrong = probe probability of y=1 (wrong).  Answer when p_wrong <= threshold.
    Returns {coverage: {"realised_coverage", "accuracy_answered", "n_answered"}}.
    """
    y, p = np.asarray(y), np.asarray(p_wrong)
    order = np.argsort(p)                    # least-likely-wrong first
    ys = y[order]
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        acc = float((ys[:k] == 0).mean())
        out[f"{c:.2f}"] = {
            "requested_coverage": c,
            "realised_coverage": k / len(y),
            "accuracy_answered": acc,
            "n_answered": k,
        }
    return out


def bootstrap_ci(values: np.ndarray, n: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def accuracy_at_threshold(y: np.ndarray, p_wrong: np.ndarray, thr: float):
    answer = p_wrong <= thr
    na = int(answer.sum())
    return {
        "threshold": float(thr),
        "realised_coverage": na / len(y) if len(y) else float("nan"),
        "n_answered": na,
        "accuracy_answered": float((np.asarray(y)[answer] == 0).mean()) if na else float("nan"),
    }


def calibrate_threshold_for_coverage(y: np.ndarray, p_wrong: np.ndarray, target: float) -> float:
    """Threshold on p_wrong giving ~`target` coverage (ties broken conservatively)."""
    p = np.sort(np.asarray(p_wrong))
    k = max(1, int(round(target * len(p))))
    return float(p[k - 1])
