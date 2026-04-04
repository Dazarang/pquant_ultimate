"""Regression tests for data/scripts/6_build_features.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "data" / "scripts" / "6_build_features.py"
SPEC = importlib.util.spec_from_file_location("build_features_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestValidateFinal:
    def test_rejects_label_columns_in_feature_set(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=3),
            "stock_id": ["AAA"] * 3,
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.0, 11.0, 12.0],
            "volume": [100, 110, 120],
            "feat_a": [0.1, 0.2, 0.3],
            "PivotLow": [0, 1, 0],
            "PivotLow_base": [0, 1, 0],
            "PivotLow_event_id": [0, 1, 0],
            "PivotLow_event_offset": [np.nan, 0.0, np.nan],
        })

        assert MODULE.validate_final(df, ["feat_a", "PivotLow_event_offset"]) is False

    def test_rejects_all_positive_labels(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=3),
            "stock_id": ["AAA"] * 3,
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.0, 11.0, 12.0],
            "volume": [100, 110, 120],
            "feat_a": [0.1, 0.2, 0.3],
            "PivotLow": [1, 1, 1],
            "PivotLow_base": [1, 1, 1],
            "PivotLow_event_id": [1, 2, 3],
            "PivotLow_event_offset": [0.0, 0.0, 0.0],
        })

        assert MODULE.validate_final(df, ["feat_a"]) is False
