"""
Volume indicators: OBV, ADOSC.
Vectorized implementations for performance.
"""

import pandas as pd
import numpy as np
from numba import njit
from indicators.base import BaseIndicator, ensure_numpy_array


@njit
def _calculate_obv_numba(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Numba-optimized OBV calculation.

    Args:
        close: Close prices
        volume: Volume

    Returns:
        OBV array
    """
    n = len(close)
    result = np.empty(n)
    result[0] = volume[0]

    for i in range(1, n):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]

    return result


class OBV(BaseIndicator):
    """On-Balance Volume - Numba optimized."""

    def __init__(self):
        super().__init__("OBV")

    def calculate(
        self, df: pd.DataFrame, ema_period: int = 55
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate OBV and its EMA.

        Args:
            df: DataFrame with 'close', 'volume' columns
            ema_period: EMA smoothing period

        Returns:
            Tuple of (OBV, OBV_EMA)
        """
        required_cols = ["close", "volume"]
        self.validate_dataframe(df, required_cols)
        self.validate_period(ema_period)

        close = ensure_numpy_array(df["close"])
        volume = ensure_numpy_array(df["volume"])

        # Calculate OBV
        obv = _calculate_obv_numba(close, volume)
        obv_series = pd.Series(obv, index=df.index, name="OBV")

        # Calculate OBV EMA
        obv_ema = obv_series.ewm(span=ema_period, adjust=False).mean()

        return obv_series, obv_ema


@njit
def _calculate_ad_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized Accumulation/Distribution calculation.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume

    Returns:
        A/D array
    """
    n = len(close)
    result = np.empty(n)
    ad = 0.0

    for i in range(n):
        hl_diff = high[i] - low[i]

        if hl_diff == 0:
            mfm = 0.0
        else:
            mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl_diff

        mfv = mfm * volume[i]
        ad += mfv
        result[i] = ad

    return result


class ADOSC(BaseIndicator):
    """Chaikin A/D Oscillator - Numba optimized."""

    def __init__(self):
        super().__init__("ADOSC")

    def calculate(
        self, df: pd.DataFrame, fastperiod: int = 3, slowperiod: int = 10
    ) -> pd.Series:
        """
        Calculate Chaikin A/D Oscillator.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns
            fastperiod: Fast EMA period
            slowperiod: Slow EMA period

        Returns:
            ADOSC series
        """
        required_cols = ["high", "low", "close", "volume"]
        self.validate_dataframe(df, required_cols)
        self.validate_period(fastperiod)
        self.validate_period(slowperiod)

        high = ensure_numpy_array(df["high"])
        low = ensure_numpy_array(df["low"])
        close = ensure_numpy_array(df["close"])
        volume = ensure_numpy_array(df["volume"])

        # Calculate A/D line
        ad = _calculate_ad_numba(high, low, close, volume)
        ad_series = pd.Series(ad, index=df.index, name="AD")

        # Calculate fast and slow EMAs
        ad_fast = ad_series.ewm(span=fastperiod, adjust=False).mean()
        ad_slow = ad_series.ewm(span=slowperiod, adjust=False).mean()

        # ADOSC = fast EMA - slow EMA
        adosc = ad_fast - ad_slow

        return adosc


# Convenience functions
def calculate_obv(
    df: pd.DataFrame, ema_period: int = 55
) -> tuple[pd.Series, pd.Series]:
    """Calculate OBV - convenience function."""
    return OBV().calculate(df, ema_period)


def calculate_adosc(
    df: pd.DataFrame, fastperiod: int = 3, slowperiod: int = 10
) -> pd.Series:
    """Calculate ADOSC - convenience function."""
    return ADOSC().calculate(df, fastperiod, slowperiod)
