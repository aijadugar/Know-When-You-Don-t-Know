"""tests/test_app.py — verify abstention logic without calling generate."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set APP_MOCK before importing app module
os.environ["APP_MOCK"] = "1"
os.environ["HF_REPO_ID"] = ""  # Force local fallback

# Mock hf_hub_download to return local files
import json as _json

MOCK_PROBE = MagicMock()
MOCK_PROBE.predict_proba.return_value = [[0.5, 0.5]]  # risk = 0.5

MOCK_META = {
    "hidden_state_index": 0,
    "threshold": 0.4,
    "val_risk_quantiles": [float(i) / 100 for i in range(101)],
}

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
    },
}

# This test file is incomplete in the conversation
# The conversation cuts off before completing the test implementation
