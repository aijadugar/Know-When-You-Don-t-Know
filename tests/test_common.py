"""P1 gate tests: normalization, whole-word matching, the label convention, and a
left-padding check against the smoke tokenizer.

Run: python -m pytest -q            (fast: only one test touches the HF hub)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402


# --------------------------------------------------------------------- normalization
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Paris", "paris"),
        ("  New  York, City ", "new york city"),
        ("The City of London", "city of london"),           # articles removed
        ("a/an/the", ""),                                   # only articles -> empty
        ("Trump's  win!!", "trump s win"),                  # punctuation -> space
        ("ratio of 19:69", "ratio of 19 69"),               # never spliced into 1969
        ("THE THE", ""),
        ("", ""),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert common.normalize_answer(raw) == expected


def test_normalize_punctuation_becomes_space_not_deleted() -> None:
    """Deleting punctuation would splice '19:69' -> '1969' and create false hits."""
    assert common.normalize_answer("19:69") == "19 69"
    assert common.is_match("ratio of 19:69", ["1969"]) is False


# --------------------------------------------------------------------- whole-word span
def test_one_does_not_match_1990() -> None:
    """"1" must not match "1990": 1-char aliases are ignored AND boundaries are strict."""
    assert common.is_match("1990", ["1"]) is False           # alias < 2 chars ignored
    assert common.is_match("1", ["1990"]) is False           # answer too short to span
    assert common.is_match("the year 1", ["1990"]) is False
    assert common.is_match("1990", ["1990"]) is True
    assert common.is_match("Apollo 11", ["11"]) is True      # 2-char alias, whole word
    assert common.is_match("Apollo 11", ["1"]) is False


def test_paris_matches_longer_answer_and_vice_versa() -> None:
    assert common.is_match("The city of Paris", ["Paris"]) is True
    assert common.is_match("Paris", ["The city of Paris"]) is True   # alias contains answer
    assert common.is_match("Paris, France", ["France"]) is True
    assert common.is_match("Palermo", ["Paris"]) is False
    assert common.is_match("The Seine", ["see"]) is False            # no partial words


def test_case_insensitive_matching() -> None:
    assert common.is_match("PARIS", ["paris"]) is True
    assert common.is_match("sinclair LEWIS", ["Sinclair Lewis"]) is True


def test_short_aliases_and_empty_answers_are_ignored() -> None:
    assert common.is_match("anything at all", ["a"]) is False
    assert common.is_match("anything", ["", "  "]) is False
    assert common.is_match("", ["paris"]) is False
    assert common.is_match("   ", ["paris"]) is False
    assert common.is_match("?", ["paris"]) is False     # normalizes to empty


# --------------------------------------------------------------------- label convention
def test_label_convention_matched_is_zero_wrong_is_one() -> None:
    """y = 1 means WRONG (hallucination), y = 0 means correct. Never invert."""
    assert common.label_from_match(True) == 0
    assert common.label_from_match(False) == 1
    assert common.make_label("Paris", ["Paris"]) == 0                # correct -> 0
    assert common.make_label("Berlin", ["Paris"]) == 1               # wrong    -> 1
    assert common.make_label("", ["Paris"]) == 1                     # empty    -> wrong
    assert common.make_label("Samuel Clemens", ["Mark Twain"]) == 1  # paraphrase -> wrong


def test_label_convention_string_in_config() -> None:
    import config

    assert "y = 1" in config.LABEL_CONVENTION and "WRONG" in config.LABEL_CONVENTION
    assert "P(y=1)" in config.LABEL_CONVENTION


def test_build_prompt_contains_question_not_answer() -> None:
    class FakeTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False and add_generation_prompt is True
            return "\n".join(m["content"] for m in messages)

    p = common.build_prompt(FakeTok(), "Capital of France?")
    assert "Capital of France?" in p
    assert "short factual answer" in p          # system prompt is included
    assert "Paris" not in p                     # answers never enter the prompt


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [{"id": "a/1", "question": "Q?", "gold_answers": ["x", "y"]}]
    p = tmp_path / "r.jsonl"
    assert common.save_jsonl(rows, p) == 1
    assert common.load_jsonl(p) == rows
    with pytest.raises(FileNotFoundError):
        common.load_jsonl(tmp_path / "missing.jsonl")


def test_metrics_helpers_orientation() -> None:
    y = np.array([0, 0, 1, 1])
    risk = np.array([0.1, 0.2, 0.8, 0.9])       # risk = P(y=1): higher => worse
    assert common.auroc(y, risk) == pytest.approx(1.0)
    thr = common.calibrate_threshold_for_coverage(risk, 0.5)
    at = common.accuracy_at_threshold(y, risk, thr)
    assert at["realised_coverage"] == pytest.approx(0.5)
    assert at["accuracy_answered"] == pytest.approx(1.0)
    curve = common.coverage_accuracy_curve(y, risk, coverages=[0.5, 1.0])
    assert curve["0.50"]["accuracy_answered"] >= curve["1.00"]["accuracy_answered"]
    mean, lo, hi = common.bootstrap_ci(np.r_[np.ones(40), np.zeros(10)], 200, 42)
    assert lo <= mean <= hi and mean == pytest.approx(0.8)


# --------------------------------------------------------------------- left padding
SMOKE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # verified to exist on the Hub


def test_left_padding_last_position_is_real_token() -> None:
    """Batch two prompts of very different length; position -1 must be the final REAL
    token for both rows (attention_mask[:, -1] all ones) -- that is what makes
    hidden_states[l][:, -1, :] the last-prompt-token activation for every row."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SMOKE_MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:                    # verified: Qwen2.5 HAS a pad token
        tok.pad_token = tok.eos_token
    assert tok.pad_token is not None

    short = common.build_prompt(tok, "Capital of France?")
    long = common.build_prompt(
        tok,
        "In the 19th century, which composer from Bohemia wrote a symphony that is "
        "commonly associated with the discovery of a new continent, and what was its "
        "nickname in English-speaking countries?",
    )
    enc = tok([short, long], return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]

    assert ids.shape[0] == 2 and int(am.sum(-1).min()) < int(am.sum(-1).max())
    assert bool(am[:, -1].all()), "position -1 must be unmasked under left padding"
    # the shorter row really was padded on the LEFT
    assert int(am[0].argmax()) > 0
    # and each row's last token equals its unpadded encoding's last token
    solo_l = tok(long, return_tensors="pt")["input_ids"][0]
    solo_s = tok(short, return_tensors="pt")["input_ids"][0]
    assert int(ids[1, -1]) == int(solo_l[-1])
    assert int(ids[0, -1]) == int(solo_s[-1])
