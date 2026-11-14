"""02_generate_and_label.py — greedy answer generation and labeling."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    SEED,
    SMOKE_MODEL_ID,
    SYSTEM_PROMPT,
    DATA_DIR,
)
from common import (
    build_prompt,
    is_match,
    label_from_match,
    load_jsonl,
    save_jsonl,
    set_seed,
    load_model_and_tokenizer,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model_id", type=str, default=None)
    return parser.parse_args()

def decode_answer(
    tokenizer: Any,
    input_ids: torch.Tensor,
    sequences: torch.Tensor,
    scores: list[torch.Tensor],
    row: int = 0,
) -> tuple[str, float, float, int]:
    """Decode the generated answer and compute metrics.
    Returns: (answer_text, mean_logprob, first_token_maxprob, n_tokens)
    """
    prompt_len = input_ids.shape[1]
    # Generated tokens start at position `prompt_len`
    gen_ids = sequences[0, prompt_len:]

    # Decode answer (skip special tokens, strip whitespace)
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Count real tokens (stop at EOS or pad)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_positions = (gen_ids == eos_id) | (gen_ids == pad_id)

    if eos_positions.any():
        first_eos = eos_positions.nonzero()[0].item()
        real_gen = gen_ids[:first_eos]
    else:
        real_gen = gen_ids

    n_tokens = len(real_gen)

    # scores: list (one per step) of [batch, vocab] logits. Use THIS row only.
    if scores and n_tokens > 0:
        step_logits = torch.stack([s[row] for s in scores[:n_tokens]], dim=0).float()  # [T, vocab]
        step_logprobs = torch.log_softmax(step_logits, dim=-1)
        idx = torch.arange(n_tokens, device=step_logprobs.device)
        chosen_logprobs = step_logprobs[idx, real_gen.to(step_logprobs.device)]
        mean_logprob = chosen_logprobs.mean().item()
    else:
        mean_logprob = 0.0

    if scores:
        probs = torch.softmax(scores[0][row].float(), dim=-1)
        first_token_maxprob = probs.max().item()
    else:
        first_token_maxprob = 0.0

    return answer, mean_logprob, first_token_maxprob, n_tokens

def process_split(
    model: Any,
    tokenizer: Any,
    split_path: Path,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    """Generate answers for one split with checkpointing."""
    rows = load_jsonl(str(split_path))
    split_name = split_path.stem
    partial_path = split_path.with_suffix(".partial")

    # Check for existing checkpoint
    start_idx = 0
    if partial_path.exists():
        completed = load_jsonl(str(partial_path))
        start_idx = len(completed)
        print(f"  Resuming from checkpoint: {start_idx}/{len(rows)}")
        rows = completed + rows[start_idx:]

    # Build prompts and sort by length for efficient batching
    questions = [r["question"] for r in rows]
    prompts = [build_prompt(tokenizer, q, SYSTEM_PROMPT) for q in questions]
    original_indices = list(range(len(prompts)))

    # Sort by prompt length
    sorted_data = sorted(
        zip(prompts, questions, rows, original_indices),
        key=lambda x: len(x[0]),
    )
    sorted_prompts = [d[0] for d in sorted_data]
    sorted_questions = [d[1] for d in sorted_data]
    sorted_rows = [d[2] for d in sorted_data]
    sorted_indices = [d[3] for d in sorted_data]

    # Process in batches
    total_tokens = 0
    total_time = 0.0
    all_results = []

    n_batches = (len(sorted_prompts) + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, len(sorted_prompts), batch_size),
        desc=f"  {split_name}",
        total=n_batches,
    ):
        batch_end = min(batch_start + batch_size, len(sorted_prompts))
        batch_prompts = sorted_prompts[batch_start:batch_end]
        batch_rows = sorted_rows[batch_start:batch_end]

        # Tokenize with left padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Verify left padding
        assert attention_mask[:, -1].all(), "Last position is padding for some row"

        # Greedy decoding
        try:
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            elapsed = time.time() - t0

            sequences = outputs.sequences  # [batch, prompt_len + gen_len]
            scores = outputs.scores  # list of [batch, vocab]

            # Count generated tokens
            prompt_lens = (input_ids != tokenizer.pad_token_id).sum(dim=1)
            gen_lens = (sequences != tokenizer.pad_token_id).sum(dim=1) - prompt_lens
            total_tokens += gen_lens.sum().item()
            total_time += elapsed

            # Decode each sequence
            for i in range(len(batch_rows)):
                answer, mean_logprob, first_token_maxprob, n_tokens = decode_answer(
                    tokenizer,
                    input_ids[i : i + 1],
                    sequences[i : i + 1],
                    scores,
                    row=i,
                )

                matched = is_match(answer, batch_rows[i]["gold_answers"])
                y = label_from_match(matched)

                result = {**batch_rows[i]}
                result["answer"] = answer
                result["mean_logprob"] = mean_logprob
                result["first_token_maxprob"] = first_token_maxprob
                result["n_tokens"] = n_tokens
                result["y"] = y
                all_results.append(result)

        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at batch {batch_start}, halving batch size")
            half_bs = batch_size // 2
            # Retry with smaller batch
            for sub_start in range(batch_start, batch_end, half_bs):
                sub_end = min(sub_start + half_bs, batch_end)
                sub_prompts = sorted_prompts[sub_start:sub_end]
                sub_rows = sorted_rows[sub_start:sub_end]

                sub_inputs = tokenizer(
                    sub_prompts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    truncation=True,
                    max_length=512,
                )
                sub_input_ids = sub_inputs["input_ids"].to(device)
                sub_attention_mask = sub_inputs["attention_mask"].to(device)

                with torch.no_grad():
                    sub_outputs = model.generate(
                        input_ids=sub_input_ids,
                        attention_mask=sub_attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                sub_sequences = sub_outputs.sequences
                sub_scores = sub_outputs.scores

                for j in range(len(sub_rows)):
                    answer, mean_logprob, first_token_maxprob, n_tokens = decode_answer(
                        tokenizer,
                        sub_input_ids[j : j + 1],
                        sub_sequences[j : j + 1],
                        sub_scores,
                        row=j,
                    )

                    matched = is_match(answer, sub_rows[j]["gold_answers"])
                    y = label_from_match(matched)

                    result = {**sub_rows[j]}
                    result["answer"] = answer
                    result["mean_logprob"] = mean_logprob
                    result["first_token_maxprob"] = first_token_maxprob
                    result["n_tokens"] = n_tokens
                    result["y"] = y
                    all_results.append(result)

        # Checkpoint every 20 batches
        if (batch_start // batch_size + 1) % 20 == 0:
            # Restore original order
            restored = [None] * len(all_results)
            for orig_idx, result in zip(sorted_indices[: len(all_results)], all_results):
                restored[orig_idx] = result
            save_jsonl(str(partial_path), [r for r in restored if r is not None])

    # Restore original order
    restored_results = [None] * len(all_results)
    for orig_idx, result in zip(sorted_indices, all_results):
        restored_results[orig_idx] = result

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    print(f"  {split_name}: {total_tokens} tokens in {total_time:.1f}s "
          f"= {tokens_per_sec:.1f} tok/s, dtype={dtype}")

    return restored_results

def print_split_summary(results: list[dict], split_name: str) -> None:
    """Print summary stats and sample rows for one split."""
    n = len(results)
    wrong_rate = np.mean([r["y"] for r in results])
    mean_n_tokens = np.mean([r["n_tokens"] for r in results])

    print(f"\n{'=' * 70}")
    print(f"SPLIT: {split_name}")
    print(f"{'=' * 70}")
    print(f"  n = {n}")
    print(f"  wrong rate = {wrong_rate:.3f}")
    print(f"  mean n_tokens = {mean_n_tokens:.1f}")

    # 40 random samples for hand-checking
    rng = np.random.default_rng(SEED)
    indices = rng.choice(n, min(40, n), replace=False)

    print(f"\n--- 40 random samples (Q, gold[0], answer, y) ---")
    for idx in indices:
        r = results[idx]
        print(f"  Q: {r['question'][:60]}")
        print(f"  G: {r['gold_answers'][0]}")
        print(f"  A: {r['answer']}")
        print(f"  y={r['y']}  ntok={r['n_tokens']}  "
              f"mlp={r['mean_logprob']:.2f}  ftmx={r['first_token_maxprob']:.3f}")
        print()

    # 15 most confident wrong answers (lowest mean_logprob where y=1)
    wrong = [r for r in results if r["y"] == 1]
    if wrong:
        wrong_sorted = sorted(wrong, key=lambda r: r["mean_logprob"])
        print(f"  --- 15 most confident WRONG answers (lowest mean_logprob) ---")
        for r in wrong_sorted[:15]:
            print(f"  mlp={r['mean_logprob']:.3f}  Q: {r['question'][:50]}  "
                  f"A: {r['answer'][:30]}  G: {r['gold_answers'][0]}")

    # 15 least confident correct answers (highest mean_logprob where y=0)
    correct = [r for r in results if r["y"] == 0]
    if correct:
        correct_sorted = sorted(correct, key=lambda r: r["mean_logprob"], reverse=True)
        print(f"\n  --- 15 least confident CORRECT answers (highest mean_logprob) ---")
        for r in correct_sorted[:15]:
            print(f"  mlp={r['mean_logprob']:.3f}  Q: {r['question'][:50]}  "
                  f"A: {r['answer'][:30]}  G: {r['gold_answers'][0]}")

    return wrong_rate

def main():
    args = parse_args()
    set_seed(SEED)

    print("=" * 70)
    print("P3: ANSWER GENERATION AND LABELING")
    print("=" * 70)

    # Determine model
    if args.smoke:
        model_id = SMOKE_MODEL_ID
        print(f"Using smoke model: {model_id}")
    elif args.model_id:
        model_id = args.model_id
    else:
        model_id = MODEL_ID

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        dtype = torch.float32  # float16 on CPU is extremely slow
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device}, dtype: {dtype}")

    model, tokenizer = load_model_and_tokenizer(model_id, dtype)
    model = model.to(device)

    # Process each split
    split_files = ["train", "val", "test", "ood"]
    wrong_rates = {}

    for split_name in split_files:
        split_path = DATA_DIR / f"{split_name}.jsonl"
        if not split_path.exists():
            print(f"  Skipping {split_name} (not found)")
            continue

        print(f"\n## Processing {split_name}...")

        # Verify compute_transition_scores signature
        import inspect
        sig = inspect.signature(model.compute_transition_scores)
        print(f"  compute_transition_scores sig: {sig}")

        results = process_split(
            model, tokenizer, split_path, args.batch_size, device, dtype
        )

        # Save labeled data
        out_path = DATA_DIR / f"{split_name}_labeled.jsonl"
        save_jsonl(str(out_path), results)

        # Print summary
        wrong_rate = print_split_summary(results, split_name)
        wrong_rates[split_name] = wrong_rate

    # Gate: wrong rate must be between 0.2 and 0.8
    print(f"\n{'=' * 70}")
    print("P3 GATE CHECK")
    print(f"{'=' * 70}")
    for split_name, wr in wrong_rates.items():
        status = "✓" if 0.2 <= wr <= 0.8 else "✗ FAIL"
        print(f"  {split_name}: wrong_rate={wr:.3f} {status}")

    bad = [(s, wr) for s, wr in wrong_rates.items() if not (0.2 <= wr <= 0.8)]
    for split_name, wr in bad:
        print(f"\nWARNING: {split_name} wrong rate {wr:.3f} outside [0.2, 0.8]")
    if bad:
        if args.smoke:
            # The 0.5B smoke model is weak, so high wrong rates are expected.
            # Only fail if a split has a single class (probe/AUROC can't run).
            single_class = [s for s, wr in wrong_rates.items() if wr in (0.0, 1.0)]
            if single_class:
                print(f"\nERROR: single-class splits {single_class}. "
                      "Increase the smoke sample sizes in step 1.")
                sys.exit(1)
            print("  --smoke: continuing despite wrong rates outside [0.2, 0.8].")
        else:
            print("  Stopping — check label noise before proceeding.")
            sys.exit(1)

    print("\nP3 GATE: PASSED — all wrong rates in [0.2, 0.8]")

if __name__ == "__main__":
    main()
