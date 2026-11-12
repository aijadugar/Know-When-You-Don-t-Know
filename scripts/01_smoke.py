"""P1 smoke harness: 0.5B end-to-end dry run on 20 questions (no dataset downloads).

Hardcoded question/gold pairs, real tokenizer+chat-template path, greedy answers,
labels, last-prompt-token activations for every layer, per-layer probe on the tiny
sample, AUROC printed.  This is the thing that must reach an AUROC number without
crashing before any big stage runs.  Runtime here is also the reference point for
the 'agent must time it' estimates in the README.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import config  # noqa: E402
from common import (auroc, build_prompt, extract_last_token_activations,  # noqa: E402
                    generate_and_score, load_model_and_tokenizer, make_label)

PAIRS = [
    ("Capital of France?", ["paris"]),
    ("Capital of Japan?", ["tokyo"]),
    ("Who wrote Hamlet?", ["william shakespeare", "shakespeare"]),
    ("Largest planet in the Solar System?", ["jupiter"]),
    ("Chemical symbol for gold?", ["au"]),
    ("What year did World War II end?", ["1945"]),
    ("Tallest mountain above sea level?", ["everest", "mount everest"]),
    ("How many continents are there on Earth?", ["seven", "7"]),
    ("What gas do plants absorb from the atmosphere?", ["carbon dioxide", "co2"]),
    ("Who painted the Mona Lisa?", ["leonardo da vinci", "da vinci"]),
    ("What is the chemical formula for water?", ["h2o"]),
    ("Planet known as the Red Planet?", ["mars"]),
    ("Who developed the theory of general relativity?", ["einstein", "albert einstein"]),
    ("Currency of the United Kingdom?", ["pound sterling", "pound"]),
    ("What is the longest river in Africa?", ["nile", "the nile"]),
    ("Name of our galaxy?", ["milky way"]),
    ("Who wrote 'Romeo and Juliet'?", ["shakespeare", "william shakespeare"]),
    ("What is the largest ocean on Earth?", ["pacific", "pacific ocean"]),
    ("In what year did humans first land on the Moon?", ["1969"]),
    ("What is H2O commonly known as?", ["water"]),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    ap.set_defaults(model_id=config.MODEL_500M)  # smoke test defaults to the small model
    args = ap.parse_args(argv)
    fl = config.RunFlags.from_args(args)
    device = config.resolve_device(fl.device)

    t0 = time.time()
    model, tok, dtype_used = load_model_and_tokenizer(fl.model_id, device, fl.dtype)
    env = config.describe_environment(fl.model_id)
    print(f"[smoke] env={json.dumps(env)} dtype={dtype_used} load_s={time.time()-t0:.1f}")

    questions = [q for q, _ in PAIRS]
    golds = [g for _, g in PAIRS]
    prompts = [build_prompt(q, tok) for q in questions]

    t0 = time.time()
    gen = generate_and_score(model, tok, prompts, device, fl.max_new_tokens, fl.batch_size)
    y = np.array([make_label(a, g) for a, g in zip(gen["answers"], golds)], dtype=np.int8)
    print(f"[smoke] generate {len(prompts)} q in {time.time()-t0:.1f}s; "
          f"wrong_rate={y.mean():.2f}")
    for q, a, yy in list(zip(questions, gen["answers"], y))[:5]:
        print(f"   y={yy} | {q} -> {a!r}")

    t0 = time.time()
    X = extract_last_token_activations(model, tok, prompts, device, fl.batch_size)
    print(f"[smoke] acts {X.shape} {X.dtype} in {time.time()-t0:.1f}s")
    assert X.shape == (len(prompts), env["n_hidden_states_rows"], env["d_model"])

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    best = None
    for l in range(X.shape[1]):
        Xi = X[:, l, :].astype(np.float32)
        if len(set(y.tolist())) < 2:
            break
        # LOO predictions so the AUROC is not trivially 1.0 on 20 points
        loo = np.zeros(len(y))
        for tr_idx, te_idx in LeaveOneOut().split(Xi):
            try:
                clf = LogisticRegression(C=1.0, max_iter=200).fit(Xi[tr_idx], y[tr_idx])
                loo[te_idx] = clf.predict_proba(Xi[te_idx])[:, 1]
            except Exception:
                loo[te_idx] = 0.5
        auc = auroc(y, loo)
        if best is None or (auc == auc and auc > best[1]):  # nan-safe
            best = (l, auc, loo)
    bl_mean = auroc(y, -gen["mean_logprob"])
    bl_first = auroc(y, -gen["first_maxprob"])
    if best is None:
        print("[smoke] labels degenerate (all correct or all wrong) — no AUROC computable")
        raise SystemExit(1)
    print(f"[smoke] PROBE AUROC (LOO, layer {best[0]}) = {best[1]:.3f} | "
          f"baselines mean_logprob={bl_mean:.3f} first_maxprob={bl_first:.3f}")
    print("[gate P1] dry run reached a probe AUROC number without crashing: OK")


if __name__ == "__main__":
    main()
