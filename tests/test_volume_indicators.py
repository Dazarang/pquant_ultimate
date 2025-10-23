"""
Test volume indicators for accuracy and correctness.

Uses reference implementations for validation to ensure algorithmic correctness.
"""

import pytest
import numpy as np
import talib  # Reference implementation for validation only
from indicators.volume import OBV, ADOSC
from tests.conftest import assert_series_close


class TestOBV:
    """Test On-Balance Volume."""

    def test_obv_vs_talib(self, sample_ohlcv_data, tolerance_params):
        """Compare OBV with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_obv, custom_obv_ema = OBV().calculate(df, ema_period=55)

        # Ta-lib
        talib_obv = talib.OBV(df["close"], df["volume"])

        # Compare OBV (not EMA, as smoothing may differ slightly)
        assert_series_close(custom_obv, talib_obv, tolerance_params["obv"], "OBV")

    def test_obv_logic(self, sample_ohlcv_data):
        """Test OBV calculation logic."""
        df = sample_ohlcv_data[:10]  # Use small sample

        obv, _ = OBV().calculate(df, ema_period=5)

        # Manual verification for first few points
        manual_obv = [df["volume"].iloc[0]]

        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                manual_obv.append(manual_obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                manual_obv.append(manual_obv[-1] - df["volume"].iloc[i])
            else:
                manual_obv.append(manual_obv[-1])

        # Compare first few values
        assert np.allclose(obv.iloc[:5].values, manual_obv[:5], atol=1)


class TestADOSC:
    """Test Chaikin A/D Oscillator."""

    @pytest.mark.parametrize("fast,slow", [(3, 10)])
    def test_adosc_vs_talib(self, sample_ohlcv_data, tolerance_params, fast, slow):
        """Compare ADOSC with ta-lib."""
        df = sample_ohlcv_data

        # Custom implementation
        custom_adosc = ADOSC().calculate(df, fastperiod=fast, slowperiod=slow)

        # Ta-lib
        talib_adosc = talib.ADOSC(
            df["high"], df["low"], df["close"], df["volume"],
            fastperiod=fast, slowperiod=slow
        )

        # Compare
        assert_series_close(
            custom_adosc, talib_adosc, tolerance_params["adosc"], f"ADOSC_{fast}_{slow}"
        )

    def test_adosc_real_data(self, real_market_data, tolerance_params):
        """Test ADOSC on real market data."""
        df = real_market_data

        custom_adosc = ADOSC().calculate(df, fastperiod=3, slowperiod=10)
        talib_adosc = talib.ADOSC(
            df["high"], df["low"], df["close"], df["volume"],
            fastperiod=3, slowperiod=10
        )

        assert_series_close(custom_adosc, talib_adosc, tolerance_params["adosc"], "ADOSC_real")
