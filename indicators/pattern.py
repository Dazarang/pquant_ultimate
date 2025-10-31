"""
Pattern recognition: find_pivots, Hammer candlestick patterns.
Core labeling mechanism for ML model.
"""

import numpy as np
import pandas as pd
from numba import njit

from indicators.base import BaseIndicator, ensure_numpy_array


@njit
def _find_pivot_high_numba(data: np.ndarray, lb: int, rb: int) -> np.ndarray:
    """
    Numba-optimized pivot high detection.

    Args:
        data: Price array (highs)
        lb: Left bars
        rb: Right bars

    Returns:
        Boolean array indicating pivot highs
    """
    n = len(data)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(lb, n - rb):
        center_value = data[i]
        is_pivot = True

        # Check left side
        for j in range(i - lb, i):
            if data[j] >= center_value:
                is_pivot = False
                break

        # Check right side
        if is_pivot:
            for j in range(i + 1, i + rb + 1):
                if data[j] >= center_value:
                    is_pivot = False
                    break

        result[i] = is_pivot

    return result


@njit
def _find_pivot_low_numba(data: np.ndarray, lb: int, rb: int) -> np.ndarray:
    """
    Numba-optimized pivot low detection.

    Args:
        data: Price array (lows)
        lb: Left bars
        rb: Right bars

    Returns:
        Boolean array indicating pivot lows
    """
    n = len(data)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(lb, n - rb):
        center_value = data[i]
        is_pivot = True

        # Check left side
        for j in range(i - lb, i):
            if data[j] <= center_value:
                is_pivot = False
                break

        # Check right side
        if is_pivot:
            for j in range(i + 1, i + rb + 1):
                if data[j] <= center_value:
                    is_pivot = False
                    break

        result[i] = is_pivot

    return result


def find_pivots(
    df: pd.DataFrame,
    lb: int = 8,
    rb: int = 13,
    return_boolean: bool = True,
    window_variations: list[int] | None = None,
    use_close: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """
    Identifies pivot highs and lows in DataFrame.
    CRITICAL: Core labeling mechanism for ML model.

    Args:
        df: DataFrame with 'high', 'low', and 'close' columns
        lb: Number of bars to the left of pivot
        rb: Number of bars to the right of pivot
        return_boolean: If True, return boolean indicators;
                       otherwise return original values with NaNs
        window_variations: Optional list of day offsets to expand base pivots.
                          E.g., [-2, -1, 1, 2] will mark days at base_pivot±1
                          and base_pivot±2 as pivots.
                          If day X is a base pivot, then days X-2, X-1, X, X+1,
                          X+2 will all be marked as pivots.
                          Out-of-bounds indices are safely ignored.
        use_close: If True, detect pivots using close price.
                  If False, use high/low prices (legacy behavior).

    Returns:
        Tuple of (PivotHigh, PivotLow) series
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    if use_close:
        required_cols = ["close"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        close = ensure_numpy_array(df["close"])
        high_data = close
        low_data = close
    else:
        required_cols = ["high", "low"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        high_data = ensure_numpy_array(df["high"])
        low_data = ensure_numpy_array(df["low"])

    # Handle window variations
    if window_variations is None:
        # Default: single window
        pivot_high_bool = _find_pivot_high_numba(high_data, lb, rb)
        pivot_low_bool = _find_pivot_low_numba(low_data, lb, rb)
    else:
        # Find base pivots, then expand to adjacent days
        n = len(high_data)
        pivot_high_base = _find_pivot_high_numba(high_data, lb, rb)
        pivot_low_base = _find_pivot_low_numba(low_data, lb, rb)

        # Start with base pivots
        pivot_high_bool = pivot_high_base.copy()
        pivot_low_bool = pivot_low_base.copy()

        # For each base pivot, mark adjacent days
        high_indices = np.where(pivot_high_base)[0]
        low_indices = np.where(pivot_low_base)[0]

        for idx in high_indices:
            for offset in window_variations:
                adj_idx = idx + offset
                if 0 <= adj_idx < n:
                    pivot_high_bool[adj_idx] = True

        for idx in low_indices:
            for offset in window_variations:
                adj_idx = idx + offset
                if 0 <= adj_idx < n:
                    pivot_low_bool[adj_idx] = True

    if return_boolean:
        pivot_high = pd.Series(pivot_high_bool, index=df.index, name="PivotHigh")
        pivot_low = pd.Series(pivot_low_bool, index=df.index, name="PivotLow")
    else:
        # Return actual values at pivot points, NaN elsewhere
        values_col = df["close"] if use_close else df["high"]
        pivot_high = pd.Series(np.where(pivot_high_bool, values_col, np.nan), index=df.index, name="PivotHigh")
        values_col = df["close"] if use_close else df["low"]
        pivot_low = pd.Series(np.where(pivot_low_bool, values_col, np.nan), index=df.index, name="PivotLow")

    return pivot_high, pivot_low


@njit
def _detect_hammer_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Numba-optimized Hammer pattern detection.

    Hammer criteria:
    - Small body at upper end of range
    - Long lower shadow (>= 2x body)
    - Little to no upper shadow
    - Bullish reversal pattern

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Integer array (100 = hammer, 0 = no pattern)
    """
    n = len(close)
    result = np.zeros(n, dtype=np.int32)

    for i in range(n):
        body = abs(close[i] - open_[i])
        total_range = high[i] - low[i]

        if total_range == 0:
            continue

        # Body position in range
        body_top = max(open_[i], close[i])
        body_bottom = min(open_[i], close[i])

        lower_shadow = body_bottom - low[i]
        upper_shadow = high[i] - body_top

        # Hammer criteria
        is_hammer = (
            lower_shadow >= 2 * body  # Long lower shadow
            and upper_shadow <= 0.1 * total_range  # Small upper shadow
            and body <= 0.3 * total_range  # Small body
        )

        if is_hammer:
            result[i] = 100

    return result


class Hammer(BaseIndicator):
    """Hammer candlestick pattern detector."""

    def __init__(self):
        super().__init__("Hammer")

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Hammer candlestick patterns.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
            Integer series (100 = hammer, 0 = no pattern)
        """
        required_cols = ["open", "high", "low", "close"]
        self.validate_dataframe(df, required_cols)

        open_ = ensure_numpy_array(df["open"])
        high = ensure_numpy_array(df["high"])
        low = ensure_numpy_array(df["low"])
        close = ensure_numpy_array(df["close"])

        result = _detect_hammer_numba(open_, high, low, close)

        return pd.Series(result, index=df.index, name="Hammer")


def detect_rsi_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect RSI divergence at pivot lows.
    From original code - preserved for ML features.

    Args:
        df: DataFrame with 'PivotLow', 'close', 'RSI' columns

    Returns:
        DataFrame with 'RSI_Divergence' column added
    """
    required_cols = ["PivotLow", "close", "RSI"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Initialize divergence column
    df["RSI_Divergence"] = 0

    # Find pivot low indices
    pivot_indices = df[df["PivotLow"] == 1].index.tolist()

    # Compare consecutive pivots
    for i in range(len(pivot_indices) - 1):
        first_idx = pivot_indices[i]
        second_idx = pivot_indices[i + 1]

        first_price = df.loc[first_idx, "close"]
        second_price = df.loc[second_idx, "close"]

        first_rsi = df.loc[first_idx, "RSI"]
        second_rsi = df.loc[second_idx, "RSI"]

        # Bullish divergence: lower price, higher RSI
        if second_price < first_price and second_rsi > first_rsi:
            df.loc[second_idx, "RSI_Divergence"] = 1

    return df


def calculate_hammer(df: pd.DataFrame) -> pd.Series:
    """Calculate Hammer pattern - convenience function."""
    return Hammer().calculate(df)
