"""Stage 02 (gate P2): build disjoint question pools.

Verified dataset facts this script depends on (checked live, datasets==4.0.0):
  * TriviaQA rc.nocontext TEST split answers are WITHHELD -> never used.
    train+validation are sampled into three DISJOINT pools: train/val/test.
  * NQ-Open has no test set with golds; its validation split (3,610 rows, all
    non-empty) is the OOD pool.  answer is a LIST OF STRINGS.

Outputs data/{train,val,test,ood}.jsonl rows:
  {"id", "question", "golds": [normalized aliases], "source"}
Gate checks (asserted here and in tests): counts match --n_* flags, no duplicate
questions across splits, every gold list non-empty.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys

sys.path.insert(0, ".")
import config  # noqa: E402
from common import golds_for, write_jsonl  # noqa: E402


def qid(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def load_triviaqa_pools(n_train: int, n_val: int, n_test: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset(config.TRIVIAQA_NAME, config.TRIVIAQA_CONFIG)
    rows = []
    for split in ("train", "validation"):  # NEVER "test": answers withheld
        for r in ds[split]:
            golds = golds_for("triviaqa", r["answer"])
            if not golds:
                continue
            rows.append({"id": qid(r["question"]), "question": r["question"],
                         "golds": golds, "source": f"trivia_qa/{split}"})
    # de-duplicate identical questions across train/validation
    seen, uniq = set(), []
    for r in rows:
        key = r["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    need = n_train + n_val + n_test
    if len(uniq) < need:
        raise RuntimeError(f"only {len(uniq)} usable TriviaQA rows for {need} requested")
    return uniq[:n_train], uniq[n_train:n_train + n_val], uniq[n_train + n_val:need]


def load_nqopen_pool(n_ood: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset(config.NQOPEN_NAME)
    rows = []
    for split in ("validation", "train"):  # validation first: fully labelled
        for r in ds[split]:
            golds = golds_for("nq_open", r["answer"])  # plain list[str]
            if not golds:
                continue
            rows.append({"id": qid(r["question"]), "question": r["question"],
                         "golds": golds, "source": f"nq_open/{split}"})
    seen, uniq = set(), []
    for r in rows:
        key = r["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    # keep validation-sourced rows at the front so tiny smoke runs use val golds
    uniq.sort(key=lambda r: 0 if r["source"] == "nq_open/validation" else 1)
    if len(uniq) < n_ood:
        raise RuntimeError(f"only {len(uniq)} usable NQ-Open rows for {n_ood} requested")
    return uniq[:n_ood]


def check_gate(pools: dict[str, list[dict]]) -> None:
    all_ids: set[str] = set()
    for name, rows in pools.items():
        ids = {r["id"] for r in rows}
        assert len(ids) == len(rows), f"duplicate questions inside {name}"
        assert not (ids & all_ids), f"{name} overlaps an earlier split"
        all_ids |= ids
        assert all(r["golds"] for r in rows), f"empty gold list in {name}"
    print("[gate P2] disjoint pools, non-empty golds: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    args = ap.parse_args(argv)
    fl = config.RunFlags.from_args(args)

    tr, va, te = load_triviaqa_pools(fl.n_train, fl.n_val, fl.n_test, fl.seed)
    od = load_nqopen_pool(fl.n_ood, fl.seed + 1)
    pools = {"train": tr, "val": va, "test": te, "ood": od}

    expected = {"train": fl.n_train, "val": fl.n_val, "test": fl.n_test, "ood": fl.n_ood}
    for name, rows in pools.items():
        path = config.SPLIT_FILES_UNLABELED[name]
        write_jsonl(path, rows)
        assert len(rows) == expected[name], (name, len(rows), expected[name])
        print(f"[02] {path}: {len(rows)} rows")
    check_gate(pools)


if __name__ == "__main__":
    main()
