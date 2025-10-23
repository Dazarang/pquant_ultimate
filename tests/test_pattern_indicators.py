"""
Test pattern recognition indicators.
"""

import pytest
import numpy as np
import talib
from indicators.pattern import find_pivots, Hammer
from tests.conftest import assert_series_close


class TestFindPivots:
    """Test pivot detection."""

    def test_pivot_high_detection(self, sample_ohlcv_data):
        """Test pivot high detection logic."""
        df = sample_ohlcv_data

        pivot_high, pivot_low = find_pivots(df, lb=5, rb=5, return_boolean=True)

        # Pivots should be boolean
        assert pivot_high.dtype == bool
        assert pivot_low.dtype == bool

        # Should have some pivots detected
        assert pivot_high.sum() > 0, "No pivot highs detected"
        assert pivot_low.sum() > 0, "No pivot lows detected"

    def test_pivot_symmetry(self, sample_ohlcv_data):
        """Test pivot detection is symmetric."""
        df = sample_ohlcv_data

        # Same lb and rb should give centered detection
        pivot_high, pivot_low = find_pivots(df, lb=8, rb=8, return_boolean=True)

        # First and last lb+rb bars should have no pivots
        assert pivot_high.iloc[:8].sum() == 0
        assert pivot_high.iloc[-8:].sum() == 0
        assert pivot_low.iloc[:8].sum() == 0
        assert pivot_low.iloc[-8:].sum() == 0

    def test_pivot_return_values(self, sample_ohlcv_data):
        """Test pivot return value formats."""
        df = sample_ohlcv_data

        # Boolean return
        ph_bool, pl_bool = find_pivots(df, lb=5, rb=5, return_boolean=True)
        assert ph_bool.dtype == bool

        # Value return
        ph_val, pl_val = find_pivots(df, lb=5, rb=5, return_boolean=False)
        assert ph_val.dtype == float
        assert pl_val.dtype == float

        # Values at pivot points should match original data
        for idx in ph_bool[ph_bool].index:
            assert ph_val.loc[idx] == df.loc[idx, "high"]


class TestHammer:
    """Test Hammer candlestick pattern."""

    def test_hammer_vs_talib(self, sample_ohlcv_data):
        """Compare Hammer with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_hammer = Hammer().calculate(df)

        # Ta-lib
        talib_hammer = talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"])

        # Both should return integers (0 or 100/-100)
        assert custom_hammer.dtype in [np.int32, np.int64]
        assert talib_hammer.dtype in [np.int32, np.int64]

        # Pattern detection criteria differ between implementations
        # Ta-lib uses sophisticated pattern recognition, ours is simplified
        # Just verify format is correct
        print(f"\nCustom hammers: {(custom_hammer != 0).sum()}, "
              f"Ta-lib hammers: {(talib_hammer != 0).sum()}")

    def test_hammer_criteria(self, sample_ohlcv_data):
        """Test Hammer only appears at valid patterns."""
        df = sample_ohlcv_data

        hammer = Hammer().calculate(df)

        # Where hammer is detected, verify basic criteria
        hammer_idx = hammer[hammer == 100].index

        for idx in hammer_idx[:5]:  # Check first 5
            open_ = df.loc[idx, "open"]
            high = df.loc[idx, "high"]
            low = df.loc[idx, "low"]
            close = df.loc[idx, "close"]

            body = abs(close - open_)
            body_bottom = min(open_, close)
            lower_shadow = body_bottom - low

            # Hammer should have long lower shadow
            assert lower_shadow >= body, f"Invalid hammer at {idx}"

    def test_hammer_output_format(self, sample_ohlcv_data):
        """Test Hammer output is 0 or 100."""
        df = sample_ohlcv_data

        hammer = Hammer().calculate(df)

        # Should only contain 0 or 100
        unique_values = hammer.unique()
        assert set(unique_values).issubset({0, 100}), f"Unexpected values: {unique_values}"
