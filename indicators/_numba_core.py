"""Numba-optimized core functions for indicators.

These are low-level functions JIT-compiled for performance.
Used internally by higher-level indicator functions.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def count_support_tests_numba(
    prices: np.ndarray,
    local_low_mask: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Count prior local lows within tolerance of current price.

    For each row, counts how many prior local lows have prices
    within `tolerance` percentage of the current price.

    Args:
        prices: Array of close prices
        local_low_mask: Boolean array (1=local low, 0=not)
        tolerance: Price similarity tolerance (e.g., 0.02 for 2%)

    Returns:
        Array of support test counts for each row
    """
    n = len(prices)
    result = np.zeros(n, dtype=np.int32)

    # Collect local low indices and prices
    local_low_indices = []
    local_low_prices = []
    for i in range(n):
        if local_low_mask[i] == 1:
            local_low_indices.append(i)
            local_low_prices.append(prices[i])

    # For each row, count matching prior local lows
    for i in range(n):
        current_price = prices[i]
        count = 0
        for j in range(len(local_low_indices)):
            if local_low_indices[j] < i:  # Prior only
                low_price = local_low_prices[j]
                price_diff_pct = abs(low_price - current_price) / current_price
                if price_diff_pct < tolerance:
                    count += 1
        result[i] = count

    return result


@jit(nopython=True, cache=True)
def count_consecutive_down_numba(returns: np.ndarray) -> np.ndarray:
    """Count consecutive negative return days.

    Args:
        returns: Array of daily returns

    Returns:
        Array of consecutive down day counts
    """
    n = len(returns)
    result = np.zeros(n, dtype=np.int32)
    count = 0

    for i in range(n):
        if not np.isnan(returns[i]) and returns[i] < 0:
            count += 1
        else:
            count = 0
        result[i] = count

    return result


@jit(nopython=True, cache=True)
def days_since_last_low_numba(local_low_mask: np.ndarray, day_offsets: np.ndarray) -> np.ndarray:
    """Calculate calendar days since last local low.

    Args:
        local_low_mask: Boolean array (1=local low, 0=not)
        day_offsets: Array of day offsets from reference date (calendar days)

    Returns:
        Array of calendar days since last low (NaN if no prior low)
    """
    n = len(local_low_mask)
    result = np.full(n, np.nan, dtype=np.float64)
    last_low_day = -1

    for i in range(n):
        if local_low_mask[i] == 1:
            last_low_day = day_offsets[i]
            result[i] = 0
        elif last_low_day >= 0:
            result[i] = day_offsets[i] - last_low_day

    return result


@jit(nopython=True, cache=True)
def detect_divergence_pairs_numba(
    prices: np.ndarray,
    indicator1: np.ndarray,
    indicator2: np.ndarray,
    indicator3: np.ndarray,
    local_low_mask: np.ndarray,
) -> np.ndarray:
    """Detect multi-indicator divergence at local lows.

    Bullish divergence: Lower low in price + higher low in indicator.
    Counts across 3 indicators (RSI, MACD, Stochastic).

    Args:
        prices: Close prices
        indicator1: First indicator (e.g., RSI)
        indicator2: Second indicator (e.g., MACD)
        indicator3: Third indicator (e.g., Stochastic)
        local_low_mask: Boolean array (1=local low, 0=not)

    Returns:
        Array of divergence scores (0-3)
    """
    n = len(prices)
    result = np.zeros(n, dtype=np.int32)

    # Collect local low indices
    local_low_indices = []
    for i in range(n):
        if local_low_mask[i] == 1:
            local_low_indices.append(i)

    # Check consecutive pairs of local lows
    for k in range(len(local_low_indices) - 1):
        first_idx = local_low_indices[k]
        second_idx = local_low_indices[k + 1]

        first_price = prices[first_idx]
        second_price = prices[second_idx]

        # Only check if lower low in price
        if second_price < first_price:
            div_count = 0

            # Check indicator 1
            if indicator1[second_idx] > indicator1[first_idx]:
                div_count += 1

            # Check indicator 2
            if indicator2[second_idx] > indicator2[first_idx]:
                div_count += 1

            # Check indicator 3
            if indicator3[second_idx] > indicator3[first_idx]:
                div_count += 1

            result[second_idx] = div_count

    return result


@jit(nopython=True, cache=True)
def detect_hidden_divergence_numba(
    prices: np.ndarray,
    rsi: np.ndarray,
    local_low_mask: np.ndarray,
) -> np.ndarray:
    """Detect hidden bullish divergence.

    Hidden bullish: Higher low in price + lower low in RSI.

    Args:
        prices: Close prices
        rsi: RSI values
        local_low_mask: Boolean array (1=local low, 0=not)

    Returns:
        Array of hidden divergence flags (0 or 1)
    """
    n = len(prices)
    result = np.zeros(n, dtype=np.int32)

    # Collect local low indices
    local_low_indices = []
    for i in range(n):
        if local_low_mask[i] == 1:
            local_low_indices.append(i)

    # Check consecutive pairs of local lows
    for k in range(len(local_low_indices) - 1):
        first_idx = local_low_indices[k]
        second_idx = local_low_indices[k + 1]

        first_price = prices[first_idx]
        second_price = prices[second_idx]
        first_rsi = rsi[first_idx]
        second_rsi = rsi[second_idx]

        # Hidden bullish: higher low in price, lower low in RSI
        if second_price > first_price and second_rsi < first_rsi:
            result[second_idx] = 1

    return result
