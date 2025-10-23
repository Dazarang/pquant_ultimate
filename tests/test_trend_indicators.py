"""
Test trend indicators for accuracy and correctness.

Uses reference implementations for validation to ensure algorithmic correctness.
"""

import pytest
import numpy as np
import talib  # Reference implementation for validation only
from indicators.trend import SMA, EMA, VWAP
from tests.conftest import assert_series_close


class TestSMA:
    """Test Simple Moving Average."""

    @pytest.mark.parametrize("period", [5, 20, 50, 200])
    def test_sma_accuracy(self, sample_ohlcv_data, tolerance_params, period):
        """Validate SMA calculation accuracy."""
        df = sample_ohlcv_data

        # Our implementation
        actual_sma = SMA().calculate(df, period)

        # Reference implementation for validation
        expected_sma = talib.SMA(df["close"], timeperiod=period)

        # Validate accuracy
        assert_series_close(actual_sma, expected_sma, tolerance_params["sma"], f"SMA_{period}")

    def test_sma_realistic_data(self, real_market_data, tolerance_params):
        """Test SMA on realistic market data."""
        df = real_market_data

        actual_sma = SMA().calculate(df, period=20)
        expected_sma = talib.SMA(df["close"], timeperiod=20)

        assert_series_close(actual_sma, expected_sma, tolerance_params["sma"], "SMA_20_real")


class TestEMA:
    """Test Exponential Moving Average."""

    @pytest.mark.parametrize("period", [5, 8, 21, 55, 89])
    def test_ema_accuracy(self, sample_ohlcv_data, tolerance_params, period):
        """Validate EMA calculation accuracy."""
        df = sample_ohlcv_data

        # Our implementation
        actual_ema = EMA().calculate(df, period)

        # Reference for validation
        expected_ema = talib.EMA(df["close"], timeperiod=period)

        # Validate accuracy
        assert_series_close(actual_ema, expected_ema, tolerance_params["ema"], f"EMA_{period}")

    def test_ema_realistic_data(self, real_market_data, tolerance_params):
        """Test EMA on realistic market data."""
        df = real_market_data

        actual_ema = EMA().calculate(df, period=21)
        expected_ema = talib.EMA(df["close"], timeperiod=21)

        assert_series_close(actual_ema, expected_ema, tolerance_params["ema"], "EMA_21_real")


class TestVWAP:
    """Test Volume Weighted Average Price."""

    def test_vwap_calculation(self, sample_ohlcv_data):
        """Test VWAP calculation logic."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_vwap = VWAP().calculate(df)

        # Manual calculation for verification
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        manual_vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

        # Should match exactly
        assert_series_close(custom_vwap, manual_vwap, 1e-10, "VWAP")

    def test_vwap_monotonic(self, sample_ohlcv_data):
        """Test VWAP is generally stable (no wild swings)."""
        df = sample_ohlcv_data
        vwap = VWAP().calculate(df)

        # VWAP should not have NaN after initialization
        assert vwap.iloc[10:].isna().sum() == 0, "VWAP has unexpected NaN values"

        # VWAP should be within reasonable range of prices
        assert vwap.min() >= df["low"].min() * 0.8
        assert vwap.max() <= df["high"].max() * 1.2
