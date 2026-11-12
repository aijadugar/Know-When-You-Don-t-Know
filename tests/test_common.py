"""Unit tests for the defect list.  Run: pytest -q  (from repo root)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
import common  # noqa: E402


# ---------------------------------------------------------------- label convention
def test_label_convention_wrong_is_one():
    """THE regression from the first draft: y must be 1 when WRONG, not when right."""
    golds = ["sinclair lewis"]
    assert common.make_label("Sinclair Lewis", golds) == 0      # correct -> 0
    assert common.make_label("Ernest Hemingway", golds) == 1    # wrong   -> 1
    assert common.make_label("", golds) == 1                    # empty   -> wrong
    assert common.LABEL_CONVENTION if hasattr(common, "LABEL_CONVENTION") else True


def test_label_convention_string_matches_config():
    import config
    assert "y = 1" in config.LABEL_CONVENTION and "WRONG" in config.LABEL_CONVENTION


# ---------------------------------------------------------------- matching rules
def test_substring_pitfall_one_vs_1990():
    """Plain substring would match '1' inside '1990'; whole-word must not."""
    assert common.answer_matches_gold("1990", ["1"]) is False      # 1-char alias ignored
    assert common.answer_matches_gold("1990", ["1990"]) is True
    assert common.answer_matches_gold("the year 1", ["1990"]) is False
    assert common.answer_matches_gold("Apollo 11 launched in 1969", ["1969"]) is True
    assert common.answer_matches_gold("ratio of 19:69", ["1969"]) is False  # word boundary


def test_normalize_strips_punctuation_so_colon_cannot_bridge():
    # punctuation removal could splice "19:69" into "1969"; normalize must insert
    # a space so whole-word matching still refuses the false hit.
    assert common.normalize("ratio of 19:69") == "ratio of 19 69"
    assert common.answer_matches_gold(common.normalize("19:69"), ["1969"]) is False


def test_short_alias_does_not_match_everything():
    assert common.answer_matches_gold("anything at all", ["a"]) is False
    assert common.answer_matches_gold("Paris", ["paris"]) is True
    assert common.answer_matches_gold("Paris, France", ["france"]) is True
    assert common.answer_matches_gold("The Seine", ["see"]) is False  # no partial word


def test_normalize():
    assert common.normalize("  New  York, City ") == "new york city"


# ---------------------------------------------------------------- gold extraction
def test_triviaqa_golds_use_normalized_aliases():
    field = {"aliases": ["A B"], "normalized_aliases": ["a b", "c d"]}
    assert common.golds_for("triviaqa", field) == ["a b", "c d"]


def test_nqopen_golds_are_plain_list_of_strings():
    """NQ-Open has NO answer.aliases; the field is list[str] (verified live)."""
    assert common.golds_for("nq_open", ["Fernie Alpine Resort"]) == ["fernie alpine resort"]
    with pytest.raises(AssertionError):
        common.golds_for("nq_open", {"aliases": ["wrong shape"]})


# ---------------------------------------------------------------- EOS masking
def test_first_eos_position_and_decode_mask_padding():
    class FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"t{i}" for i in ids)
    gen = [5, 7, 0, 0, 0]          # tokens after the first EOS (id 0) are padding
    assert common.first_eos_position(np.asarray(gen), eos_id=0) == 2
    assert common.decode_answer(FakeTok(), gen, eos_id=0) == "t5 t7"
    # mean over real tokens only (what generate_and_score does):
    vals = np.array([-0.5, -0.25, -9.0, -9.0, -9.0])
    n = common.first_eos_position(np.asarray(gen), 0)
    assert abs(vals[:n].mean() - (-0.375)) < 1e-9   # NOT contaminated by -9's


# ---------------------------------------------------------------- metrics helpers
def test_auroc_orientation_confidence_flip():
    y = np.array([0, 0, 1, 1])
    conf = np.array([0.9, 0.8, 0.2, 0.1])           # higher conf = more likely correct
    assert common.auroc(y, -conf) == pytest.approx(1.0)


def test_coverage_curve_monotone_and_threshold_calibration():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.3).astype(np.int8)
    p = np.where(y == 1, rng.beta(3, 2, 1000), rng.beta(2, 6, 1000))
    thr = common.calibrate_threshold_for_coverage(y, p, 0.7)
    at = common.accuracy_at_threshold(y, p, thr)
    assert 0.65 <= at["realised_coverage"] <= 0.75
    curve = common.coverage_accuracy_curve(y, p, coverages=(0.5, 0.7))
    assert curve["0.50"]["accuracy_answered"] >= curve["0.70"]["accuracy_answered"] * 0.99


def test_bootstrap_ci_brackets_mean():
    v = np.ones(50); v[:10] = 0
    m, lo, hi = common.bootstrap_ci(v, 300, 1)
    assert lo <= m <= hi and 0.8 == pytest.approx(m)


# ---------------------------------------------------------------- app gate P8
def test_abstain_branch_never_calls_generate(tmp_path, monkeypatch):
    """Gate P8: on the abstain path model.generate must be provably untouched."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    # --- fake everything; app.py imports torch/config lazily inside functions
    d = 4
    X = np.concatenate([np.zeros((20, d)), np.ones((20, d))])
    yw = np.array([0] * 20 + [1] * 20)
    clf = LogisticRegression().fit(X, yw)
    bundle = {"probe": clf, "layer": 0, "C": 1.0}
    joblib.dump(bundle, tmp_path / "probe.joblib")
    import json
    (tmp_path / "metrics.json").write_text(json.dumps({"threshold_p_wrong": 0.01}))

    calls = {"generate": 0}

    class FakeOut:
        class _HS:
            def __getitem__(self, li):
                class Row:
                    def __getitem__(self, sl):
                        class T:
                            def float(self): return self
                            def cpu(self): return self
                            def numpy(self): return np.ones((1, d)) * 10.0  # screams WRONG
                        return T()
                return Row()
        hidden_states = _HS()

    class FakeModel:
        def __call__(self, **kw): return FakeOut()
        def generate(self, *a, **k):
            calls["generate"] += 1
            raise AssertionError("generate called on abstain path!")

    class FakeTokEnc(dict):
        def to(self, device): return self

    class FakeTok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "PROMPT"
        def __call__(self, texts, return_tensors=None, padding=None):
            return FakeTokEnc()

    import config as cfgmod
    monkeypatch.setattr(cfgmod, "PROBE_PATH", str(tmp_path / "probe.joblib"))
    monkeypatch.setattr(cfgmod, "METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setattr(cfgmod, "resolve_device", lambda r=None: "cpu")

    import common as cmod
    monkeypatch.setattr(cmod, "load_model_and_tokenizer",
                        lambda mid, dev, dt: (FakeModel(), FakeTok(), "fp32"))

    import importlib
    import app
    importlib.reload(app)
    app._ready = False
    out = app.answer("What nonsense question is this?")
    assert "abstained" in out.lower()
    assert calls["generate"] == 0
