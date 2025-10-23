"""
Momentum indicators: RSI, MACD, ADX, ROC, MOM.
Optimized with Numba JIT compilation for hot loops.
"""

import pandas as pd
import numpy as np
from numba import njit
from indicators.base import BaseIndicator, ensure_numpy_array


@njit
def _calculate_rsi_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    Numba-optimized RSI calculation using Wilder's smoothing.

    Args:
        data: Price array
        period: Lookback period

    Returns:
        RSI array
    """
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    if n < period + 1:
        return result

    # Calculate price changes
    deltas = np.diff(data)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial average using SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # First RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's smoothing for subsequent values
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


class RSI(BaseIndicator):
    """Relative Strength Index - Numba optimized."""

    def __init__(self):
        super().__init__("RSI")

    def calculate(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing method.

        Args:
            df: DataFrame with 'close' column
            period: Lookback period (default 14)

        Returns:
            RSI series (0-100)
        """
        self.validate_dataframe(df, ["close"])
        self.validate_period(period, min_period=2)

        data = ensure_numpy_array(df["close"])
        result = _calculate_rsi_numba(data, period)

        return pd.Series(result, index=df.index, name=f"RSI_{period}")


class MACD(BaseIndicator):
    """MACD with ATR normalization - custom implementation from original code."""

    def __init__(self):
        super().__init__("MACD")

    def calculate(
        self,
        df: pd.DataFrame,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD normalized by ATR (custom implementation).

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            fastperiod: Fast EMA period
            slowperiod: Slow EMA period
            signalperiod: Signal line EMA period

        Returns:
            Tuple of (MACD, signal_line, histogram)
        """
        required_cols = ["high", "low", "close"]
        self.validate_dataframe(df, required_cols)

        # Calculate EMAs
        ema_fast = df["close"].ewm(span=fastperiod, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slowperiod, adjust=False).mean()

        # Calculate ATR for normalization
        from indicators.volatility import calculate_atr
        atr = calculate_atr(df, period=slowperiod)

        # MACD normalized by ATR
        macd_v = ((ema_fast - ema_slow) / atr.replace(0, np.nan)) * 100
        signal_line = macd_v.ewm(span=signalperiod, adjust=False).mean()
        histogram = macd_v - signal_line

        return macd_v, signal_line, histogram


@njit
def _calculate_adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    Numba-optimized ADX calculation.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period

    Returns:
        ADX array
    """
    n = len(close)
    result = np.empty(n)
    result[:] = np.nan

    if n < period + 1:
        return result

    # Calculate True Range and Directional Movement
    tr = np.empty(n)
    plus_dm = np.empty(n)
    minus_dm = np.empty(n)

    tr[0] = high[0] - low[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # Smooth TR and DM
    atr = np.empty(n)
    plus_di = np.empty(n)
    minus_di = np.empty(n)
    dx = np.empty(n)

    atr[:period] = np.nan
    plus_di[:period] = np.nan
    minus_di[:period] = np.nan
    dx[:period] = np.nan

    # Initial values
    atr[period] = np.mean(tr[:period + 1])
    smoothed_plus_dm = np.mean(plus_dm[:period + 1])
    smoothed_minus_dm = np.mean(minus_dm[:period + 1])

    for i in range(period, n):
        if i > period:
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            smoothed_plus_dm = (smoothed_plus_dm * (period - 1) + plus_dm[i]) / period
            smoothed_minus_dm = (smoothed_minus_dm * (period - 1) + minus_dm[i]) / period

        if atr[i] > 0:
            plus_di[i] = 100 * smoothed_plus_dm / atr[i]
            minus_di[i] = 100 * smoothed_minus_dm / atr[i]

            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # Smooth DX to get ADX
    adx = np.empty(n)
    adx[:] = np.nan

    start_idx = period * 2
    if start_idx < n:
        adx[start_idx] = np.mean(dx[period:start_idx + 1])

        for i in range(start_idx + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


class ADX(BaseIndicator):
    """Average Directional Index - Numba optimized."""

    def __init__(self):
        super().__init__("ADX")

    def calculate(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ADX.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Lookback period

        Returns:
            ADX series
        """
        required_cols = ["high", "low", "close"]
        self.validate_dataframe(df, required_cols)
        self.validate_period(period)

        high = ensure_numpy_array(df["high"])
        low = ensure_numpy_array(df["low"])
        close = ensure_numpy_array(df["close"])

        result = _calculate_adx_numba(high, low, close, period)

        return pd.Series(result, index=df.index, name=f"ADX_{period}")


class ROC(BaseIndicator):
    """Rate of Change - vectorized."""

    def __init__(self):
        super().__init__("ROC")

    def calculate(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change.

        Args:
            df: DataFrame with 'close' column
            period: Lookback period

        Returns:
            ROC series (percentage)
        """
        self.validate_dataframe(df, ["close"])
        self.validate_period(period)

        # ROC = ((close - close_n_periods_ago) / close_n_periods_ago) * 100
        close_shifted = df["close"].shift(period)
        roc = ((df["close"] - close_shifted) / close_shifted) * 100

        return roc


class MOM(BaseIndicator):
    """Momentum - vectorized."""

    def __init__(self):
        super().__init__("MOM")

    def calculate(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Calculate Momentum.

        Args:
            df: DataFrame with 'close' column
            period: Lookback period

        Returns:
            MOM series
        """
        self.validate_dataframe(df, ["close"])
        self.validate_period(period)

        # MOM = close - close_n_periods_ago
        return df["close"] - df["close"].shift(period)


# Convenience functions
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI - convenience function."""
    return RSI().calculate(df, period)


def calculate_macd(
    df: pd.DataFrame,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD - convenience function."""
    return MACD().calculate(df, fastperiod, slowperiod, signalperiod)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX - convenience function."""
    return ADX().calculate(df, period)


def calculate_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate ROC - convenience function."""
    return ROC().calculate(df, period)


def calculate_mom(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate MOM - convenience function."""
    return MOM().calculate(df, period)
