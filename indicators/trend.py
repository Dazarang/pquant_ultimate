"""
Trend indicators: SMA, EMA, VWAP.
Pure vectorized implementations for maximum performance.
"""

import pandas as pd
import numpy as np
from numba import njit
from indicators.base import BaseIndicator, ensure_numpy_array


class SMA(BaseIndicator):
    """Simple Moving Average - vectorized implementation."""

    def __init__(self):
        super().__init__("SMA")

    def calculate(self, df: pd.DataFrame, period: int = 50) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            df: DataFrame with 'close' column
            period: Lookback period

        Returns:
            SMA series
        """
        self.validate_dataframe(df, ["close"])
        self.validate_period(period)
        self.validate_data_length(df, period)

        return df["close"].rolling(window=period, min_periods=1).mean()


@njit
def _calculate_ema_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    Numba-optimized EMA calculation using industry-standard algorithm.

    Initializes with SMA at period-1, then applies exponential smoothing
    with alpha = 2/(period+1) for subsequent values.

    Args:
        data: Price array
        period: Lookback period

    Returns:
        EMA array
    """
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    if n < period:
        return result

    # Alpha factor for exponential smoothing
    alpha = 2.0 / (period + 1.0)

    # Initialize with SMA at position period-1
    sma = np.mean(data[:period])
    result[period - 1] = sma

    # Apply exponential smoothing for rest
    for i in range(period, n):
        result[i] = result[i - 1] + alpha * (data[i] - result[i - 1])

    return result


class EMA(BaseIndicator):
    """Exponential Moving Average using standard industry algorithm."""

    def __init__(self):
        super().__init__("EMA")

    def calculate(self, df: pd.DataFrame, period: int = 50) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with 'close' column
            period: Lookback period

        Returns:
            EMA series
        """
        self.validate_dataframe(df, ["close"])
        self.validate_period(period)

        data = ensure_numpy_array(df["close"])
        result = _calculate_ema_numba(data, period)

        return pd.Series(result, index=df.index, name=f"EMA_{period}")


class VWAP(BaseIndicator):
    """Volume Weighted Average Price - vectorized implementation."""

    def __init__(self):
        super().__init__("VWAP")

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            VWAP series
        """
        required_cols = ["high", "low", "close", "volume"]
        self.validate_dataframe(df, required_cols)

        # Typical price = (high + low + close) / 3
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # VWAP = cumsum(typical_price * volume) / cumsum(volume)
        cumulative_tpv = (typical_price * df["volume"]).cumsum()
        cumulative_volume = df["volume"].cumsum()

        # Avoid division by zero
        vwap = cumulative_tpv / cumulative_volume.replace(0, np.nan)

        return vwap.ffill()


# Convenience functions for backward compatibility
def calculate_sma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Calculate SMA - convenience function."""
    return SMA().calculate(df, period)


def calculate_ema(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Calculate EMA - convenience function."""
    return EMA().calculate(df, period)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP - convenience function."""
    return VWAP().calculate(df)
