"""
Test momentum indicators for accuracy and correctness.

Uses reference implementations for validation to ensure algorithmic correctness.
"""

import pytest
import numpy as np
import talib  # Reference implementation for validation only
from indicators.momentum import RSI, MACD, ADX, ROC, MOM
from indicators.volatility import ATR
from tests.conftest import assert_series_close


class TestRSI:
    """Test Relative Strength Index."""

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_rsi_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare RSI with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_rsi = RSI().calculate(df, period)

        # Ta-lib
        talib_rsi = talib.RSI(df["close"], timeperiod=period)

        # Compare
        assert_series_close(custom_rsi, talib_rsi, tolerance_params["rsi"], f"RSI_{period}")

    def test_rsi_range(self, sample_ohlcv_data):
        """Test RSI stays in 0-100 range."""
        df = sample_ohlcv_data
        rsi = RSI().calculate(df, period=14)

        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0, "RSI below 0"
        assert valid_rsi.max() <= 100, "RSI above 100"

    def test_rsi_real_data(self, real_market_data, tolerance_params):
        """Test RSI on real market data."""
        df = real_market_data

        custom_rsi = RSI().calculate(df, period=14)
        talib_rsi = talib.RSI(df["close"], timeperiod=14)

        assert_series_close(custom_rsi, talib_rsi, tolerance_params["rsi"], "RSI_14_real")


class TestMACD:
    """Test MACD (custom ATR-normalized version)."""

    def test_macd_calculation(self, sample_ohlcv_data):
        """Test MACD calculation logic."""
        df = sample_ohlcv_data

        # Custom implementation
        macd, signal, hist = MACD().calculate(df, 12, 26, 9)

        # Test relationships
        # Histogram should be macd - signal
        assert_series_close(hist, macd - signal, 1e-6, "MACD_histogram")

    def test_macd_components(self, sample_ohlcv_data):
        """Test MACD components are reasonable."""
        df = sample_ohlcv_data

        macd, signal, hist = MACD().calculate(df, 12, 26, 9)

        # Should have valid values (not all NaN)
        assert macd.dropna().shape[0] > 100
        assert signal.dropna().shape[0] > 100
        assert hist.dropna().shape[0] > 100


class TestADX:
    """Test Average Directional Index."""

    @pytest.mark.parametrize("period", [14])
    def test_adx_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare ADX with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_adx = ADX().calculate(df, period)

        # Ta-lib
        talib_adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)

        # Compare (ADX has more complex calculation, use looser tolerance)
        assert_series_close(custom_adx, talib_adx, tolerance_params["adx"], f"ADX_{period}")

    def test_adx_range(self, sample_ohlcv_data):
        """Test ADX stays in valid range."""
        df = sample_ohlcv_data
        adx = ADX().calculate(df, period=14)

        valid_adx = adx.dropna()
        assert valid_adx.min() >= 0, "ADX below 0"
        assert valid_adx.max() <= 100, "ADX above 100"


class TestROC:
    """Test Rate of Change."""

    @pytest.mark.parametrize("period", [5, 10, 20])
    def test_roc_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare ROC with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_roc = ROC().calculate(df, period)

        # Ta-lib
        talib_roc = talib.ROC(df["close"], timeperiod=period)

        # Compare
        assert_series_close(custom_roc, talib_roc, tolerance_params["roc"], f"ROC_{period}")


class TestMOM:
    """Test Momentum."""

    @pytest.mark.parametrize("period", [5, 10, 20])
    def test_mom_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare MOM with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_mom = MOM().calculate(df, period)

        # Ta-lib
        talib_mom = talib.MOM(df["close"], timeperiod=period)

        # Compare
        assert_series_close(custom_mom, talib_mom, tolerance_params["mom"], f"MOM_{period}")
