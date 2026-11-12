"""Stage 03 (gate P3): greedy answers + error labels.

Reads data/{train,val,test,ood}.jsonl, writes data/*_labeled.jsonl with rows:
  {"id","question","golds","source","answer","y","mean_logprob","first_maxprob"}

LABEL CONVENTION (unit-tested in tests/test_common.py):
  y = 1  <=>  the greedy answer matches NO gold alias  <=>  the model is WRONG.

Also stores the two logit baselines computed HERE, once, from the same greedy
rollout used for the label:
  mean_logprob  : mean log-prob of the CHOSEN token over real generated tokens
                  only (positions after the first EOS are masked out, so padding
                  cannot bias the score by answer length).
  first_maxprob : max softmax probability of the FIRST generated token.

Gate P3: prints 40 random labeled rows for a hand-check and asserts the overall
wrong-rate lies between 20% and 80%.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

sys.path.insert(0, ".")
import config  # noqa: E402
from common import (build_prompt, generate_and_score, load_model_and_tokenizer,  # noqa: E402
                    make_label, read_jsonl, write_jsonl)


def label_rows(rows, gen):
    out = []
    for row, ans, ml, fm in zip(rows, gen["answers"], gen["mean_logprob"], gen["first_maxprob"]):
        y = make_label(ans, row["golds"])          # y=1 when WRONG
        out.append({**{k: row[k] for k in ("id", "question", "golds", "source")},
                    "answer": ans, "y": int(y),
                    "mean_logprob": float(ml), "first_maxprob": float(fm)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    ap.add_argument("--splits", default="train,val,test,ood")
    args = ap.parse_args(argv)
    fl = config.RunFlags.from_args(args)
    device = config.resolve_device(fl.device)

    model, tok, dtype_used = load_model_and_tokenizer(fl.model_id, device, fl.dtype)
    eos_id = tok.eos_token_id

    summary = {}
    rng = random.Random(fl.seed)
    all_labeled = {}
    for split in args.splits.split(","):
        rows = read_jsonl(config.SPLIT_FILES_UNLABELED[split])
        prompts = [build_prompt(r["question"], tok) for r in rows]
        gen = generate_and_score(model, tok, prompts, device, fl.max_new_tokens, fl.batch_size)
        labeled = label_rows(rows, gen)
        write_jsonl(config.SPLIT_FILES[split], labeled)
        all_labeled[split] = labeled
        wr = sum(r["y"] for r in labeled) / len(labeled)
        summary[split] = {"n": len(labeled), "wrong_rate": wr}
        print(f"[03] {split}: n={len(labeled)} wrong_rate={wr:.3f}")

    # ---- gate P3: print 40 rows for hand-checking
    pool = [r for lst in all_labeled.values() for r in lst]
    sample = rng.sample(pool, min(40, len(pool)))
    print("\n[03] hand-check sample (question | answer | y | golds):")
    for r in sample:
        print(json.dumps({"q": r["question"], "a": r["answer"], "y": r["y"],
                          "golds": r["golds"][:3]}, ensure_ascii=False))

    for split, s in summary.items():
        assert 0.20 <= s["wrong_rate"] <= 0.80 or s["n"] < 50, \
            f"gate P3 failed for {split}: wrong_rate={s['wrong_rate']:.3f} outside [0.2, 0.8]"
    print("[gate P3] wrong-rates within [0.2, 0.8] per split: OK")
    meta = {"model_id": fl.model_id, "device": device, "dtype": dtype_used,
            "max_new_tokens": fl.max_new_tokens, "summary": summary}
    with open(config.DATA_DIR + "/label_summary.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
