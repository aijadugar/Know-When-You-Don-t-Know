"""01_build_data.py — build train/val/test/ood splits."""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    N_OOD,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED,
    DATA_DIR,
    SMOKE_MODEL_ID,
    MODEL_ID,
)
from common import normalize_answer, set_seed, save_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=N_TRAIN)
    parser.add_argument("--n_val", type=int, default=N_VAL)
    parser.add_argument("--n_test", type=int, default=N_TEST)
    parser.add_argument("--n_ood", type=int, default=N_OOD)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)

    print("=" * 70)
    print("P2: DATA BUILDING")
    print("=" * 70)

    # ── 1. Load TriviaQA ────────────────────────────────────────────────────
    print("\n## Loading TriviaQA (rc.nocontext)...")
    tqa = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    print(f"  Splits: {list(tqa.keys())}")

    for split in tqa:
        print(f"\nSplit: {split}")
        print(f"  Features: {tqa[split].features}")
        print(f"  Rows: {len(tqa[split])}")

    # Count non-empty aliases per split
    print("\nNon-empty aliases per split:")
    for split in tqa:
        non_empty = sum(
            1 for row in tqa[split]
            if row.get("answer", {}).get("aliases")
        )
        print(f"    {split}: {non_empty}/{len(tqa[split])}")

    # Check if test answers are withheld
    test_has_answers = False
    for row in tqa["test"].take(min(10, len(tqa["test"]))):
        aliases = row.get("answer", {}).get("aliases", [])
        if aliases:
            test_has_answers = True
            break

    print(f"\nTest split has populated answers: {test_has_answers}")

    # ── 2. Decide TriviaQA route ───────────────────────────────────────────
    if test_has_answers:
        print("\n## Route A: test split has answers → use standard splits")
        tqa_splits = {
            "train": list(tqa["train"]),
            "val": list(tqa["validation"]),
            "test": list(tqa["test"]),
        }
    else:
        print("\n## Route B: test split answers withheld → pool train+val, re-split")
        pool = list(tqa["train"]) + list(tqa["validation"])
        print(f"  Pool size: {len(pool)}")

        # Dedupe by normalized question text
        seen_q: dict[str, Any] = {}
        for row in pool:
            norm_q = normalize_answer(row["question"])
            if norm_q not in seen_q:
                seen_q[norm_q] = row
        pool = list(seen_q.values())
        print(f"  After dedup by normalized question: {len(pool)}")

        np.random.seed(SEED)
        indices = np.random.permutation(len(pool))
        pool = [pool[i] for i in indices]

        n_test = int(len(pool) * 0.15)
        n_val = int(len(pool) * 0.10)
        tqa_splits = {
            "test": pool[:n_test],
            "val": pool[n_test:n_test + n_val],
            "train": pool[n_test + n_val:],
        }
        print(f"  Re-split: train={len(tqa_splits['train'])}, "
              f"val={len(tqa_splits['val'])}, test={len(tqa_splits['test'])}")

    # ── 3. Load NQ-Open ────────────────────────────────────────────────────
    print("\n## Loading NQ-Open...")
    nq = load_dataset("google-research-datasets/nq_open")
    print(f"  Splits: {list(nq.keys())}")

    for split in nq:
        print(f"  Split: {split}, Rows: {len(nq[split])}, Features: {nq[split].features}")

    # Verify answer field format
    print("\nSample NQ-Open rows (validation split):")
    for i, row in enumerate(nq["validation"].take(3)):
        print(f"\nRow {i}:")
        print(f"    question: {row['question'][:80]}...")
        print(f"    answer: {row['answer']} (type: {type(row['answer']).__name__})")
        assert isinstance(row["answer"], list), (
            f"NQ-Open answer is not a list: {type(row['answer'])}"
        )

    # Use validation split for OOD
    nq_rows = list(nq["validation"])

    # Dedupe by normalized question text
    seen_nq: dict[str, Any] = {}
    for row in nq_rows:
        norm_q = normalize_answer(row["question"])
        if norm_q not in seen_nq:
            seen_nq[norm_q] = row
    nq_rows = list(seen_nq.values())
    print(f"\nAfter dedup by normalized question: {len(nq_rows)}")

    # Drop NQ questions that appear in any TriviaQA split
    tqa_questions: set[str] = set()
    for split_rows in tqa_splits.values():
        for row in split_rows:
            tqa_questions.add(normalize_answer(row["question"]))

    before = len(nq_rows)
    nq_rows = [r for r in nq_rows if normalize_answer(r["question"]) not in tqa_questions]
    print(f"  After dropping overlap with TriviaQA: {len(nq_rows)} (removed {before - len(nq_rows)})")

    # ── 4. Build gold_answers ─────────────────────────────────────────────
    def get_tqa_gold(row: dict) -> list[str]:
        ans = row.get("answer", {})
        aliases = ans.get("aliases", [])
        norm_aliases = ans.get("normalized_aliases", [])
        value = ans.get("value", "")
        gold = set(aliases)
        gold.update(norm_aliases)
        if value:
            gold.add(value)
        return list(gold)

    def get_nq_gold(row: dict) -> list[str]:
        # answer is already a list of strings
        return list(row["answer"])

    # ── 5. Sample ──────────────────────────────────────────────────────────
    def sample_split(rows: list, n: int, source: str) -> list[dict]:
        if len(rows) < n:
            print(f"  WARNING: {source} has only {len(rows)} rows, need {n}")
            n = len(rows)
        indices = np.random.choice(len(rows), n, replace=False)
        return [rows[i] for i in indices]

    train_rows = sample_split(tqa_splits["train"], args.n_train, "train")
    val_rows = sample_split(tqa_splits["val"], args.n_val, "val")
    test_rows = sample_split(tqa_splits["test"], args.n_test, "test")
    ood_rows = sample_split(nq_rows, args.n_ood, "ood")

    # ── 6. Ensure disjointness ─────────────────────────────────────────────
    def build_index(rows: list[dict], prefix: str) -> tuple[list[dict], set, set]:
        ids: set = set()
        norm_qs: set = set()
        indexed = []
        for i, row in enumerate(rows):
            qid = f"{prefix}:{row.get('question_id', i)}"
            nq = normalize_answer(row["question"])
            if qid in ids or nq in norm_qs:
                print(f"  DUPLICATE in {prefix}: {qid} / {nq[:50]}")
            ids.add(qid)
            norm_qs.add(nq)
            indexed.append((qid, row, nq))
        return indexed, ids, norm_qs

    train_idx, train_ids, train_nqs = build_index(train_rows, "tqa_train")
    val_idx, val_ids, val_nqs = build_index(val_rows, "tqa_val")
    test_idx, test_ids, test_nqs = build_index(test_rows, "tqa_test")
    ood_idx, ood_ids, ood_nqs = build_index(ood_rows, "nq_ood")

    # Check cross-split overlaps
    overlaps = []
    for name_a, ids_a, nqs_a in [
        ("train", train_ids, train_nqs),
        ("val", val_ids, val_nqs),
        ("test", test_ids, test_nqs),
        ("ood", ood_ids, ood_nqs),
    ]:
        for name_b, ids_b, nqs_b in [
            ("train", train_ids, train_nqs),
            ("val", val_ids, val_nqs),
            ("test", test_ids, test_nqs),
            ("ood", ood_ids, ood_nqs),
        ]:
            if name_a >= name_b:
                continue
            id_overlap = ids_a & ids_b
            q_overlap = nqs_a & nqs_b
            if id_overlap or q_overlap:
                overlaps.append(
                    f"  {name_a} ∩ {name_b}: {len(id_overlap)} id overlaps, "
                    f"{len(q_overlap)} question overlaps"
                )

    if overlaps:
        print("\nOVERLAP ERRORS:")
        for o in overlaps:
            print(o)
        raise AssertionError("Splits are not disjoint")

    # ── 7. Build output rows ───────────────────────────────────────────────
    def make_row(qid: str, row: dict, gold_fn) -> dict:
        gold = gold_fn(row)
        assert gold, f"Empty gold_answers for {qid}"
        return {
            "id": qid,
            "question": row["question"],
            "gold_answers": gold,
        }

    train_out = [make_row(r[0], r[1], get_tqa_gold) for r in train_idx]
    val_out = [make_row(r[0], r[1], get_tqa_gold) for r in val_idx]
    test_out = [make_row(r[0], r[1], get_tqa_gold) for r in test_idx]
    ood_out = [make_row(r[0], r[1], get_nq_gold) for r in ood_idx]

    # ── 8. Save ───────────────────────────────────────────────────────────
    save_jsonl(str(DATA_DIR / "train.jsonl"), train_out)
    save_jsonl(str(DATA_DIR / "val.jsonl"), val_out)
    save_jsonl(str(DATA_DIR / "test.jsonl"), test_out)
    save_jsonl(str(DATA_DIR / "ood.jsonl"), ood_out)

    print(f"\nSaved: train={len(train_out)}, val={len(val_out)}, "
          f"test={len(test_out)}, ood={len(ood_out)}")

    # ── 9. Stats table ─────────────────────────────────────────────────────
    def stats(rows: list[dict]) -> dict:
        q_lens = [len(r["question"].split()) for r in rows]
        n_aliases = [len(r["gold_answers"]) for r in rows]
        return {
            "n": len(rows),
            "mean_q_len": np.mean(q_lens),
            "mean_aliases": np.mean(n_aliases),
        }

    print("\n## Stats table")
    print(f"{'Split':<8} {'N':>6} {'Mean Q len (words)':>20} {'Mean aliases':>14}")
    print("-" * 52)
    for name, rows in [("train", train_out), ("val", val_out),
                       ("test", test_out), ("ood", ood_out)]:
        s = stats(rows)
        print(f"{name:<8} {s['n']:>6} {s['mean_q_len']:>20.1f} {s['mean_aliases']:>14.1f}")

    # ── 10. Sample rows ─────────────────────────────────────────────────────
    print("\n## Sample rows")
    print("\nTrain:")
    for r in train_out[:5]:
        print(f"  [{r['id']}] Q: {r['question'][:60]}...")
        print(f"       Gold: {r['gold_answers'][:3]}")

    print("\nOOD (NQ-Open):")
    for r in ood_out[:5]:
        print(f"  [{r['id']}] Q: {r['question'][:60]}...")
        print(f"       Gold: {r['gold_answers'][:3]}")

    # ── Gate assertions ────────────────────────────────────────────────────
    print("\n## GATE assertions")
    assert len(train_out) == args.n_train, f"train: {len(train_out)} != {args.n_train}"
    assert len(val_out) == args.n_val, f"val: {len(val_out)} != {args.n_val}"
    assert len(test_out) == args.n_test, f"test: {len(test_out)} != {args.n_test}"
    assert len(ood_out) == args.n_ood, f"ood: {len(ood_out)} != {args.n_ood}"

    assert all(r["gold_answers"] for r in train_out), "train has empty gold_answers"
    assert all(r["gold_answers"] for r in val_out), "val has empty gold_answers"
    assert all(r["gold_answers"] for r in test_out), "test has empty gold_answers"
    assert all(r["gold_answers"] for r in ood_out), "ood has empty gold_answers"

    print("  ✓ Row counts match flags")
    print("  ✓ No question overlaps across splits")
    print("  ✓ All gold_answers non-empty")
    print("\nP2 GATE: PASSED")

if __name__ == "__main__":
    main()
