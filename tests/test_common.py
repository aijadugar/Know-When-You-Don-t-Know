"""tests/test_common.py — unit tests for common.py."""
from __future__ import annotations
import pytest
from transformers import AutoTokenizer
from common import (
    normalize_answer,
    is_match,
    label_from_match,
)

class TestNormalize:
    def test_lowercase(self):
        assert normalize_answer("Paris") == "paris"

    def test_strip(self):
        assert normalize_answer("  hello  ") == "hello"

    def test_remove_articles(self):
        assert normalize_answer("the city of Paris") == "city of paris"
        assert normalize_answer("a quick brown fox") == "quick brown fox"

    def test_collapse_whitespace(self):
        assert normalize_answer("hello    world") == "hello world"

    def test_remove_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"

class TestIsMatch:
    def test_exact_match(self):
        assert is_match("Paris", ["Paris", "City of Light"])

    def test_normalized_match(self):
        assert is_match("  THE   city  of  paris  ", ["Paris"])

    def test_whole_word_in_alias(self):
        assert is_match("Paris", ["The city of Paris"])
        assert is_match("Paris", ["city of Paris"])

    def test_alias_contains_answer(self):
        assert is_match("Paris", ["Paris, France"])

    def test_short_alias_ignored(self):
        """Aliases shorter than 2 chars after normalization are ignored."""
        assert not is_match("Paris", ["x"])
        assert not is_match("Paris", [""])

    def test_empty_answer_returns_false(self):
        assert not is_match("", ["Paris"])
        assert not is_match("   ", ["Paris"])

    def test_no_match(self):
        assert not is_match("London", ["Paris", "France"])

    def test_number_does_not_match_year(self):
        """'1' must not match '1990'."""
        assert not is_match("1", ["1990", "1991"])
        assert not is_match("1", ["one thousand"])

    def test_paris_matches_the_city_of_paris(self):
        """Short answer matching a longer alias is valid."""
        assert is_match("Paris", ["The city of Paris"])

class TestLabelConvention:
    def test_matched_is_y0(self):
        """Matched answer -> y=0 (correct)."""
        assert label_from_match(matched=True) == 0

    def test_not_matched_is_y1(self):
        """Unmatched answer -> y=1 (wrong)."""
        assert label_from_match(matched=False) == 1

class TestLeftPadding:
    def test_last_token_is_real_token(self):
        """In a left-padded batch, position -1 is the last real token for all rows."""
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompts = [
            "What is the capital of France?",
            "What is the capital of France? Who was the first president of the United States?",
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # Under left padding, the last position is the last real token for all rows
        assert inputs["attention_mask"][:, -1].all(), (
            "Last position must be a real (non-padding) token for all rows"
        )

        # The last token in input_ids should NOT be pad for either row
        for i, seq in enumerate(inputs["input_ids"]):
            last_token = seq[-1].item()
            pad_token = tokenizer.pad_token_id
            assert last_token != pad_token, (
                f"Row {i}: last token is pad token {last_token}, "
                f"pad_id={pad_token}"
            )
