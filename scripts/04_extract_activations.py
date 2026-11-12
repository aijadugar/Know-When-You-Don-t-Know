"""Stage 04 (gate P4): last-prompt-token activations for every layer.

Reads data/*_labeled.jsonl, writes acts/<split>.npy (float16, shape
[N, n_layers+1, d_model]) plus acts/<split>_logprob.npy and acts/<split>_y.npy.

The probe input is the hidden state at the LAST PROMPT TOKEN -- i.e. index -1 on
the sequence axis under LEFT padding -- taken from model(..., output_hidden_states=True).
No per-layer forward hooks (fragile; easy to read the wrong tensor).  This is the
same input recipe Semantic Entropy Probes (Kossen et al.) used for their linear
probes, letting us abstain BEFORE generating.

Gate P4: batched activations must equal single-prompt activations within float16
tolerance (a real padding test over a length-mixed mini-batch).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import config  # noqa: E402
from common import build_prompt, extract_last_token_activations, load_model_and_tokenizer, read_jsonl  # noqa: E402


def padding_test(model, tok, device, n=8):
    """Batch of mixed-length prompts vs one-at-a-time; assert equality."""
    questions = [
        "Capital of France?",
        "What is the name of the largest ocean on Earth and where is it located?",
        "Who wrote Hamlet?",
        f"In what year did {'NASA ' * 3}land humans on the Moon for the first time?",
        "Speed of light?",
        "Which planet in our solar system has the most confirmed natural satellites orbiting it?",
        "Author of Dune?",
        "Currency of Japan?",
    ][:n]
    prompts = [build_prompt(q, tok) for q in questions]
    big = extract_last_token_activations(model, tok, prompts, device, batch_size=len(prompts))
    small = extract_last_token_activations(model, tok, prompts, device, batch_size=1)
    assert big.shape == small.shape, (big.shape, small.shape)
    atol = 5e-2 if big.dtype == np.float16 else 1e-5
    bad = np.abs(big.astype(np.float32) - small.astype(np.float32)).max()
    print(f"[04] padding test max|batch-single| = {bad:.4g} (atol={atol})")
    assert bad <= atol, "gate P4 failed: batching changed activations"
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    ap.add_argument("--splits", default="train,val,test,ood")
    args = ap.parse_args(argv)
    fl = config.RunFlags.from_args(args)
    device = config.resolve_device(fl.device)

    model, tok, dtype_used = load_model_and_tokenizer(fl.model_id, device, fl.dtype)
    os.makedirs(config.ACTS_DIR, exist_ok=True)

    padding_test(model, tok, device)

    env = config.describe_environment(fl.model_id)
    for split in args.splits.split(","):
        rows = read_jsonl(config.SPLIT_FILES[split])
        prompts = [build_prompt(r["question"], tok) for r in rows]
        acts = extract_last_token_activations(model, tok, prompts, device, fl.batch_size)
        assert acts.dtype == np.float16
        assert acts.shape == (len(rows), env["n_hidden_states_rows"], env["d_model"]), acts.shape
        np.save(config.acts_path(split), acts)
        y = np.asarray([r["y"] for r in rows], dtype=np.int8)
        np.save(os.path.join(config.ACTS_DIR, f"{split}_y.npy"), y)
        lp = np.asarray([[r["mean_logprob"], r["first_maxprob"]] for r in rows], dtype=np.float64)
        np.save(config.acts_logp_path(split), lp)
        print(f"[04] {config.acts_path(split)}: shape={acts.shape} dtype={acts.dtype}")
    print("[gate P4] shapes/dtype/padding test: OK")


if __name__ == "__main__":
    main()
