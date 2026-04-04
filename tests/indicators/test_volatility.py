"""
Test volatility indicators for accuracy and correctness.

Uses reference implementations for validation to ensure algorithmic correctness.
"""

import pytest
import talib  # Reference implementation for validation only

from indicators.volatility import ADR, ATR, SAR, BBands
from tests.conftest import assert_series_close


class TestBBands:
    """Test Bollinger Bands."""

    @pytest.mark.parametrize("period", [20])
    @pytest.mark.parametrize("nbdev", [2.0])
    def test_bbands_vs_talib(self, sample_ohlcv_data, tolerance_params, period, nbdev):
        """Compare Bollinger Bands with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_upper, custom_middle, custom_lower = BBands().calculate(df, period, nbdev, nbdev)

        # Ta-lib
        talib_upper, talib_middle, talib_lower = talib.BBANDS(
            df["close"], timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev
        )

        # Compare all three bands
        assert_series_close(custom_upper, talib_upper, tolerance_params["bbands"], "BBands_upper")
        assert_series_close(custom_middle, talib_middle, tolerance_params["bbands"], "BBands_middle")
        assert_series_close(custom_lower, talib_lower, tolerance_params["bbands"], "BBands_lower")

    def test_bbands_relationships(self, sample_ohlcv_data):
        """Test Bollinger Bands relationships."""
        df = sample_ohlcv_data

        upper, middle, lower = BBands().calculate(df, 20, 2.0, 2.0)

        # Upper should be > middle > lower
        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_mask] >= middle[valid_mask]).all(), "Upper < middle"
        assert (middle[valid_mask] >= lower[valid_mask]).all(), "Middle < lower"


class TestATR:
    """Test Average True Range."""

    @pytest.mark.parametrize("period", [14])
    def test_atr_vs_talib(self, sample_ohlcv_data, tolerance_params, period):
        """Compare ATR with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_atr = ATR().calculate(df, period)

        # Ta-lib
        talib_atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)

        # Compare
        assert_series_close(custom_atr, talib_atr, tolerance_params["atr"], f"ATR_{period}")

    def test_atr_positive(self, sample_ohlcv_data):
        """Test ATR is always positive."""
        df = sample_ohlcv_data
        atr = ATR().calculate(df, period=14)

        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all(), "ATR has negative values"


class TestADR:
    """Test Average Daily Range."""

    def test_adr_calculation(self, sample_ohlcv_data):
        """Test ADR calculation logic."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_adr = ADR().calculate(df, length=20)

        # Manual verification
        hl_ratio = df["high"] / df["low"]
        manual_adr = 100 * (hl_ratio.rolling(window=20).mean() - 1)
        manual_adr = manual_adr.round(2)

        # Should match exactly
        assert_series_close(custom_adr, manual_adr, 1e-10, "ADR")

    def test_adr_positive(self, sample_ohlcv_data):
        """Test ADR is always positive."""
        df = sample_ohlcv_data
        adr = ADR().calculate(df, length=20)

        valid_adr = adr.dropna()
        assert (valid_adr >= 0).all(), "ADR has negative values"


class TestSAR:
    """Test Parabolic SAR."""

    def test_sar_vs_talib(self, sample_ohlcv_data, tolerance_params):
        """Compare SAR with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_sar = SAR().calculate(df, acceleration=0.02, maximum=0.2)

        # Ta-lib
        talib_sar = talib.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)

        # Compare (SAR can have differences in reversal points)
        assert_series_close(custom_sar, talib_sar, tolerance_params["sar"], "SAR")

    def test_sar_in_price_range(self, sample_ohlcv_data):
        """Test SAR stays within reasonable price range."""
        df = sample_ohlcv_data
        sar = SAR().calculate(df)

        valid_sar = sar.dropna()

        # SAR should be within or near price range
        min_price = df["low"].min()
        max_price = df["high"].max()

        assert valid_sar.min() >= min_price * 0.8
        assert valid_sar.max() <= max_price * 1.2
