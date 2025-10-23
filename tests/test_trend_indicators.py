"""
Test trend indicators against ta-lib.
"""

import pytest
import numpy as np
import talib
from indicators.trend import SMA, EMA, WMA, VWAP
from tests.conftest import assert_series_close


class TestSMA:
    """Test Simple Moving Average."""

    @pytest.mark.parametrize("period", [5, 20, 50, 200])
    def test_sma_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare SMA with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_sma = SMA().calculate(df, period)

        # Ta-lib
        talib_sma = talib.SMA(df["close"], timeperiod=period)

        # Compare
        assert_series_close(custom_sma, talib_sma, tolerance_params["sma"], f"SMA_{period}")

    def test_sma_real_data(self, real_market_data, tolerance_params):
        """Test SMA on real market data."""
        df = real_market_data

        custom_sma = SMA().calculate(df, period=20)
        talib_sma = talib.SMA(df["close"], timeperiod=20)

        assert_series_close(custom_sma, talib_sma, tolerance_params["sma"], "SMA_20_real")


class TestEMA:
    """Test Exponential Moving Average."""

    @pytest.mark.parametrize("period", [5, 8, 21, 55, 89])
    def test_ema_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare EMA with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_ema = EMA().calculate(df, period)

        # Ta-lib
        talib_ema = talib.EMA(df["close"], timeperiod=period)

        # Compare
        assert_series_close(custom_ema, talib_ema, tolerance_params["ema"], f"EMA_{period}")

    def test_ema_real_data(self, real_market_data, tolerance_params):
        """Test EMA on real market data."""
        df = real_market_data

        custom_ema = EMA().calculate(df, period=21)
        talib_ema = talib.EMA(df["close"], timeperiod=21)

        assert_series_close(custom_ema, talib_ema, tolerance_params["ema"], "EMA_21_real")


class TestWMA:
    """Test Weighted Moving Average."""

    @pytest.mark.parametrize("period", [5, 10, 21])
    def test_wma_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare WMA with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_wma = WMA().calculate(df, period)

        # Ta-lib
        talib_wma = talib.WMA(df["close"], timeperiod=period)

        # Compare
        assert_series_close(custom_wma, talib_wma, tolerance_params["wma"], f"WMA_{period}")


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
