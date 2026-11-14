"""tests/test_app.py — verify abstention logic without calling generate."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
import torch

# Set APP_MOCK before importing app module
os.environ["APP_MOCK"] = "1"
os.environ["HF_REPO_ID"] = ""  # Force local fallback

# Create mock files before importing app
MOCK_DIR = Path(__file__).parent.parent / "models"
MOCK_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

import json as _json

# Mock probe
MOCK_PROBE = MagicMock()
MOCK_PROBE.predict_proba.return_value = [[0.5, 0.5]]  # risk = 0.5

# Mock meta
MOCK_META = {
    "hidden_state_index": 0,
    "threshold": 0.4,
    "val_risk_quantiles": [float(i) / 100 for i in range(101)],
    "C": 0.1,
    "n_layers": 2,
    "d_model": 64,
    "target_coverage": 0.7,
    "realised_val_coverage": 0.7,
}

# Mock metrics
MOCK_METRICS = {
    "results": [
        {
            "split": "test", "scorer": "probe",
            "auroc": 0.71, "auroc_ci": [0.68, 0.74],
            "acc_at_70": 0.62, "base_accuracy": 0.55,
        },
        {
            "split": "ood", "scorer": "probe",
            "auroc": 0.68, "auroc_ci": [0.64, 0.72],
            "acc_at_70": 0.59, "base_accuracy": 0.52,
        },
    ],
    "verdict": {
        "ood_degradation": -0.03,
        "probe_beats_both_baselines_auroc_test": True,
        "probe_beats_both_baselines_auroc_ood": False,
    },
}

# Write mock files
with open(MOCK_DIR / "probe_meta.json", "w") as f:
    _json.dump(MOCK_META, f)

with open(RESULTS_DIR / "metrics.json", "w") as f:
    _json.dump(MOCK_METRICS, f)

# Placeholder so app.py's existence check passes (joblib.load is mocked below)
(MOCK_DIR / "probe.joblib").touch()

# Mock joblib to return our mock probe
import joblib
original_load = joblib.load

def mock_load(path):
    if "probe.joblib" in str(path):
        return MOCK_PROBE
    return original_load(path)

joblib.load = mock_load

# Now import app
sys.path.insert(0, str(Path(__file__).parent.parent))
import app


class TestAppAbstention:
    """Test that the app abstains correctly based on probe risk."""

    def test_abstain_when_risk_high(self):
        """When risk > threshold, should abstain and NOT call generate."""
        # Set risk above threshold
        MOCK_PROBE.predict_proba.return_value = [[0.5, 0.8]]  # risk = 0.8 > threshold 0.4
        
        # Spy on model.generate
        generate_spy = MagicMock()
        app.model.generate = generate_spy
        
        risk, decision, answer = app.answer_question(
            question="What is the capital of France?",
            strictness=70,
            probe_on=True,
        )
        
        # Should abstain
        assert "Abstained" in decision or "not confident" in decision.lower()
        assert risk == 0.8
        
        # Generate should NOT be called
        generate_spy.assert_not_called()

    def test_answer_when_risk_low(self):
        """When risk <= threshold, should answer and CALL generate."""
        # Set risk below threshold
        MOCK_PROBE.predict_proba.return_value = [[0.5, 0.3]]  # risk = 0.3 < threshold 0.4
        
        # Spy on model.generate
        generate_spy = MagicMock()
        generate_spy.return_value = MagicMock(
            sequences=torch.tensor([[0, 1, 2, 3, 4]])
        )
        app.model.generate = generate_spy
        
        risk, decision, answer = app.answer_question(
            question="What is the capital of France?",
            strictness=70,
            probe_on=True,
        )
        
        # Should answer
        assert "Answered" in decision or "✅" in decision
        assert risk == 0.3
        
        # Generate SHOULD be called
        generate_spy.assert_called_once()

    def test_probe_off_always_answers(self):
        """When probe_on=False, should always answer regardless of risk."""
        # Set risk high
        MOCK_PROBE.predict_proba.return_value = [[0.5, 0.9]]  # risk = 0.9
        
        # Spy on model.generate
        generate_spy = MagicMock()
        generate_spy.return_value = MagicMock(
            sequences=torch.tensor([[0, 1, 2, 3, 4]])
        )
        app.model.generate = generate_spy
        
        risk, decision, answer = app.answer_question(
            question="What is the capital of France?",
            strictness=70,
            probe_on=False,  # Probe disabled
        )
        
        # Should answer even though risk is high
        assert "Answered" in decision or "✅" in decision
        assert risk == 0.9
        
        # Generate SHOULD be called
        generate_spy.assert_called_once()

    def test_strictness_slider_affects_threshold(self):
        """Different strictness values should map to different thresholds."""
        # Set risk at 0.5
        MOCK_PROBE.predict_proba.return_value = [[0.5, 0.5]]
        
        # Test with strictness=50 (target coverage 50%, threshold at 50th percentile = 0.5)
        # Risk 0.5 == threshold 0.5, so should answer (not > threshold)
        generate_spy = MagicMock()
        generate_spy.return_value = MagicMock(sequences=torch.tensor([[0, 1, 2, 3, 4]]))
        app.model.generate = generate_spy
        
        risk, decision, answer = app.answer_question(
            question="Test question",
            strictness=50,
            probe_on=True,
        )
        
        # With strictness=50, threshold=0.5, risk=0.5, should answer
        assert "Answered" in decision or "✅" in decision
        generate_spy.assert_called()
        
        # Test with strictness=90 (target coverage 90%, threshold at 10th percentile = 0.1)
        # Risk 0.5 > threshold 0.1, so should abstain
        generate_spy.reset_mock()
        
        risk, decision, answer = app.answer_question(
            question="Test question",
            strictness=90,
            probe_on=True,
        )
        
        # With strictness=90, threshold=0.1, risk=0.5, should abstain
        assert "Abstained" in decision or "not confident" in decision.lower()
        generate_spy.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])