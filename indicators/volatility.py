"""
Volatility indicators: BBands, ATR, ADR, APZ, SAR.
Optimized with vectorization and Numba.
"""

import pandas as pd
import numpy as np
from numba import njit
from indicators.base import BaseIndicator, ensure_numpy_array


class BBands(BaseIndicator):
    """Bollinger Bands - vectorized."""

    def __init__(self):
        super().__init__("BBands")

    def calculate(
        self,
        df: pd.DataFrame,
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with 'close' column
            period: Lookback period
            nbdevup: Standard deviations for upper band
            nbdevdn: Standard deviations for lower band
            matype: MA type (0=SMA, 1=EMA) - for compatibility

        Returns:
            Tuple of (upper, middle, lower) bands
        """
        self.validate_dataframe(df, ["close"])
        self.validate_period(period)

        # Middle band (SMA)
        if matype == 1:  # EMA
            middle = df["close"].ewm(span=period, adjust=False).mean()
        else:  # SMA (default)
            middle = df["close"].rolling(window=period).mean()

        # Standard deviation
        std = df["close"].rolling(window=period).std()

        # Upper and lower bands
        upper = middle + (std * nbdevup)
        lower = middle - (std * nbdevdn)

        return upper, middle, lower


@njit
def _calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    Numba-optimized ATR calculation.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period

    Returns:
        ATR array
    """
    n = len(close)
    result = np.empty(n)
    result[:] = np.nan

    if n < 2:
        return result

    # Calculate True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Wilder's smoothing
    if n >= period:
        result[period - 1] = np.mean(tr[:period])

        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


class ATR(BaseIndicator):
    """Average True Range - Numba optimized."""

    def __init__(self):
        super().__init__("ATR")

    def calculate(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Lookback period

        Returns:
            ATR series
        """
        required_cols = ["high", "low", "close"]
        self.validate_dataframe(df, required_cols)
        self.validate_period(period)

        high = ensure_numpy_array(df["high"])
        low = ensure_numpy_array(df["low"])
        close = ensure_numpy_array(df["close"])

        result = _calculate_atr_numba(high, low, close, period)

        return pd.Series(result, index=df.index, name=f"ATR_{period}")


class ADR(BaseIndicator):
    """Average Daily Range - vectorized."""

    def __init__(self):
        super().__init__("ADR")

    def calculate(self, df: pd.DataFrame, length: int = 20) -> pd.Series:
        """
        Calculate Average Daily Range.

        Args:
            df: DataFrame with 'high', 'low' columns
            length: Lookback period

        Returns:
            ADR series (percentage)
        """
        required_cols = ["high", "low"]
        self.validate_dataframe(df, required_cols)
        self.validate_period(length)

        # Calculate high/low ratio
        hl_ratio = df["high"] / df["low"]

        # ADR = 100 * (SMA(high/low) - 1)
        adr = 100 * (hl_ratio.rolling(window=length).mean() - 1)

        return adr.round(2)


class APZ(BaseIndicator):
    """Adaptive Price Zone - custom double smooth EMA."""

    def __init__(self):
        super().__init__("APZ")

    def _double_smooth_ema(self, price: pd.Series, length: int) -> pd.Series:
        """
        Calculate Double Smooth EMA.

        Args:
            price: Price series
            length: Period

        Returns:
            Double smoothed EMA
        """
        period = int(np.sqrt(length))
        ema1 = price.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return ema2

    def calculate(
        self, df: pd.DataFrame, period: int = 21, band_pct: float = 2.0
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Adaptive Price Zone.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Lookback period
            band_pct: Band width percentage

        Returns:
            Tuple of (upper_band, lower_band)
        """
        required_cols = ["high", "low", "close"]
        self.validate_dataframe(df, required_cols)
        self.validate_period(period)

        # Double smooth EMA of close
        ds_ema = self._double_smooth_ema(df["close"], period)

        # Double smooth EMA of range
        range_hl = df["high"] - df["low"]
        range_ds_ema = self._double_smooth_ema(range_hl, period)

        # Calculate bands
        upper_band = ds_ema + band_pct * range_ds_ema
        lower_band = ds_ema - band_pct * range_ds_ema

        return upper_band, lower_band


@njit
def _calculate_sar_numba(
    high: np.ndarray,
    low: np.ndarray,
    acceleration: float,
    maximum: float,
) -> np.ndarray:
    """
    Numba-optimized Parabolic SAR calculation.

    Args:
        high: High prices
        low: Low prices
        acceleration: Acceleration factor
        maximum: Maximum acceleration factor

    Returns:
        SAR array
    """
    n = len(high)
    result = np.empty(n)
    result[:] = np.nan

    if n < 2:
        return result

    # Initialize
    sar = low[0]
    ep = high[0]
    af = acceleration
    is_long = True

    result[0] = sar

    for i in range(1, n):
        # Update SAR
        sar = sar + af * (ep - sar)

        # Check for reversal
        reverse = False

        if is_long:
            if low[i] < sar:
                reverse = True
                sar = ep
                ep = low[i]
                af = acceleration
                is_long = False
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
                # SAR cannot be above prior two lows
                if i >= 2:
                    sar = min(sar, low[i - 1], low[i - 2])
        else:
            if high[i] > sar:
                reverse = True
                sar = ep
                ep = high[i]
                af = acceleration
                is_long = True
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)
                # SAR cannot be below prior two highs
                if i >= 2:
                    sar = max(sar, high[i - 1], high[i - 2])

        result[i] = sar

    return result


class SAR(BaseIndicator):
    """Parabolic SAR - Numba optimized."""

    def __init__(self):
        super().__init__("SAR")

    def calculate(
        self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2
    ) -> pd.Series:
        """
        Calculate Parabolic SAR.

        Args:
            df: DataFrame with 'high', 'low' columns
            acceleration: Acceleration factor
            maximum: Maximum acceleration factor

        Returns:
            SAR series
        """
        required_cols = ["high", "low"]
        self.validate_dataframe(df, required_cols)

        high = ensure_numpy_array(df["high"])
        low = ensure_numpy_array(df["low"])

        result = _calculate_sar_numba(high, low, acceleration, maximum)

        return pd.Series(result, index=df.index, name="SAR")


# Convenience functions
def calculate_bbands(
    df: pd.DataFrame,
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands - convenience function."""
    return BBands().calculate(df, period, nbdevup, nbdevdn, matype)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR - convenience function."""
    return ATR().calculate(df, period)


def calculate_adr(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Calculate ADR - convenience function."""
    return ADR().calculate(df, length)


def calculate_apz(
    df: pd.DataFrame, period: int = 21, band_pct: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Calculate APZ - convenience function."""
    return APZ().calculate(df, period, band_pct)


def calculate_sar(
    df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2
) -> pd.Series:
    """Calculate SAR - convenience function."""
    return SAR().calculate(df, acceleration, maximum)


def double_smooth_ema(price: pd.Series, length: int) -> pd.Series:
    """Calculate double smooth EMA - convenience function."""
    return APZ()._double_smooth_ema(price, length)
