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
    price_tolerance: float = 0.01,
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
                          and base_pivot±2 as pivots, but ONLY if the adjacent
                          day's close price is within price_tolerance of the
                          base pivot's close price.
                          Out-of-bounds indices are safely ignored.
        use_close: If True, detect pivots using close price.
                  If False, use high/low prices (legacy behavior).
        price_tolerance: Percentage tolerance for window variations (default 0.05 = 5%).
                        Adjacent days only marked as pivots if their close price
                        is within this percentage of base pivot's close.
                        Only applies when window_variations is not None.

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

    # Window variations require close prices for tolerance check
    if window_variations is not None and "close" not in df.columns:
        raise ValueError("window_variations requires 'close' column for price tolerance check")

    # Handle window variations
    if window_variations is None:
        # Default: single window
        pivot_high_bool = _find_pivot_high_numba(high_data, lb, rb)
        pivot_low_bool = _find_pivot_low_numba(low_data, lb, rb)
    else:
        # Find base pivots, then expand to adjacent days with price tolerance
        n = len(high_data)
        pivot_high_base = _find_pivot_high_numba(high_data, lb, rb)
        pivot_low_base = _find_pivot_low_numba(low_data, lb, rb)

        # Start with base pivots
        pivot_high_bool = pivot_high_base.copy()
        pivot_low_bool = pivot_low_base.copy()

        # Get close prices for tolerance check
        close_prices = ensure_numpy_array(df["close"])

        # For each base pivot, mark adjacent days within price tolerance
        high_indices = np.where(pivot_high_base)[0]
        low_indices = np.where(pivot_low_base)[0]

        for idx in high_indices:
            base_price = close_prices[idx]
            for offset in window_variations:
                adj_idx = idx + offset
                if 0 <= adj_idx < n:
                    adj_price = close_prices[adj_idx]
                    # Check if adjacent price is within tolerance
                    price_diff = abs(adj_price - base_price) / base_price
                    if price_diff <= price_tolerance:
                        pivot_high_bool[adj_idx] = True

        for idx in low_indices:
            base_price = close_prices[idx]
            for offset in window_variations:
                adj_idx = idx + offset
                if 0 <= adj_idx < n:
                    adj_price = close_prices[adj_idx]
                    # Check if adjacent price is within tolerance
                    price_diff = abs(adj_price - base_price) / base_price
                    if price_diff <= price_tolerance:
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


@njit
def _find_local_extrema_rolling(data: np.ndarray, window: int, find_max: bool) -> np.ndarray:
    """
    Find local extrema using rolling window (backward-looking only).
    At position i, checks if data[i] is max/min within window [i-window+1, i].

    Args:
        data: Price or indicator array
        window: Lookback window size
        find_max: True for local maxima, False for local minima

    Returns:
        Boolean array marking local extrema
    """
    n = len(data)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(window - 1, n):
        start_idx = i - window + 1
        is_extreme = True

        if find_max:
            # Check if current value is maximum in window
            for j in range(start_idx, i + 1):
                if data[j] > data[i]:
                    is_extreme = False
                    break
        else:
            # Check if current value is minimum in window
            for j in range(start_idx, i + 1):
                if data[j] < data[i]:
                    is_extreme = False
                    break

        result[i] = is_extreme

    return result


def find_local_extrema(
    df: pd.DataFrame,
    price_col: str = "close",
    lookback_window: int = 8,
    find_lows: bool = True,
    find_highs: bool = True,
) -> pd.DataFrame:
    """
    Find local extrema (lows/highs) using backward-looking method.
    NO LOOKAHEAD BIAS - safe for ML features.

    Similar to find_pivots() but uses only historical data.
    At position i, checks if value is min/max within [i-lookback_window+1, i].

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        lookback_window: Window size for extrema detection (similar to lb in pivots)
        find_lows: If True, detect local lows
        find_highs: If True, detect local highs

    Returns:
        DataFrame with added columns:
        - LocalLow: 1 if local low detected, 0 otherwise
        - LocalHigh: 1 if local high detected, 0 otherwise
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")

    df = df.copy()
    price_data = ensure_numpy_array(df[price_col])

    if find_lows:
        local_lows = _find_local_extrema_rolling(price_data, lookback_window, find_max=False)
        df["LocalLow"] = local_lows.astype(np.int32)

    if find_highs:
        local_highs = _find_local_extrema_rolling(price_data, lookback_window, find_max=True)
        df["LocalHigh"] = local_highs.astype(np.int32)

    return df


def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_col: str = "RSI",
    price_col: str = "close",
    lookback_window: int = 5,
    max_lookback: int = 60,
    min_distance: int = 5,
) -> pd.DataFrame:
    """
    Detect RSI divergence using backward-looking method (NO LOOKAHEAD BIAS).
    Suitable for real-time trading and ML features.

    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high

    Args:
        df: DataFrame with price and RSI columns
        rsi_col: Name of RSI column
        price_col: Name of price column
        lookback_window: Window to detect local extrema (bars back for peak detection)
        max_lookback: Maximum bars back to search for previous extreme
        min_distance: Minimum bars between extrema to avoid noise

    Returns:
        DataFrame with added columns:
        - Bullish_Divergence: 1 if bullish divergence detected, 0 otherwise
        - Bearish_Divergence: 1 if bearish divergence detected, 0 otherwise
        - Divergence_Strength: Normalized strength of divergence
    """
    required_cols = [price_col, rsi_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Find local lows and highs using backward-looking rolling window
    price_data = ensure_numpy_array(df[price_col])
    rsi_data = ensure_numpy_array(df[rsi_col])

    local_lows = _find_local_extrema_rolling(price_data, lookback_window, find_max=False)
    local_highs = _find_local_extrema_rolling(price_data, lookback_window, find_max=True)

    # Initialize divergence columns
    n = len(df)
    bullish_div = np.zeros(n, dtype=np.int32)
    bearish_div = np.zeros(n, dtype=np.int32)
    div_strength = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Check for bullish divergence at local lows
        if local_lows[i]:
            # Find previous low within max_lookback and min_distance
            search_start = max(0, i - max_lookback)
            prev_low_idx = -1

            for j in range(i - min_distance, search_start - 1, -1):
                if local_lows[j]:
                    prev_low_idx = j
                    break

            if prev_low_idx >= 0:
                curr_price = price_data[i]
                prev_price = price_data[prev_low_idx]
                curr_rsi = rsi_data[i]
                prev_rsi = rsi_data[prev_low_idx]

                # Bullish divergence: lower price, higher RSI
                if curr_price < prev_price and curr_rsi > prev_rsi:
                    bullish_div[i] = 1
                    # Strength: normalized by price/RSI differences
                    price_drop = (prev_price - curr_price) / prev_price
                    rsi_rise = (curr_rsi - prev_rsi) / 100.0
                    div_strength[i] = (price_drop + rsi_rise) / 2.0

        # Check for bearish divergence at local highs
        if local_highs[i]:
            # Find previous high within max_lookback and min_distance
            search_start = max(0, i - max_lookback)
            prev_high_idx = -1

            for j in range(i - min_distance, search_start - 1, -1):
                if local_highs[j]:
                    prev_high_idx = j
                    break

            if prev_high_idx >= 0:
                curr_price = price_data[i]
                prev_price = price_data[prev_high_idx]
                curr_rsi = rsi_data[i]
                prev_rsi = rsi_data[prev_high_idx]

                # Bearish divergence: higher price, lower RSI
                if curr_price > prev_price and curr_rsi < prev_rsi:
                    bearish_div[i] = 1
                    # Strength: normalized by price/RSI differences
                    price_rise = (curr_price - prev_price) / prev_price
                    rsi_drop = (prev_rsi - curr_rsi) / 100.0
                    div_strength[i] = (price_rise + rsi_drop) / 2.0

    df["Bullish_Divergence"] = bullish_div
    df["Bearish_Divergence"] = bearish_div
    df["Divergence_Strength"] = div_strength

    return df


def calculate_hammer(df: pd.DataFrame) -> pd.Series:
    """Calculate Hammer pattern - convenience function."""
    return Hammer().calculate(df)
