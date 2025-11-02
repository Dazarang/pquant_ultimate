"""
Advanced ML features for stock bottom detection.

Feature engineering functions that combine multiple indicators
to detect patterns associated with market bottoms:
- Multi-indicator divergence
- Volume exhaustion and panic selling
- Support level testing
- Statistical mean reversion
- Temporal patterns

These features are designed to work with both single-stock
and multi-stock DataFrames (auto-detects stock_id column).
"""

import numpy as np
import pandas as pd

from indicators.momentum import calculate_macd, calculate_rsi, calculate_stochastic
from indicators.pattern import find_local_extrema
from indicators.volatility import calculate_bbands


def _has_stock_id(df: pd.DataFrame) -> bool:
    """Check if DataFrame has stock_id column for multi-stock processing."""
    return "stock_id" in df.columns


def _get_groupby_or_single(df: pd.DataFrame):
    """
    Return groupby object if multi-stock, else return dict with single key.

    This allows writing code once that works for both cases:
    for stock_id, group in _get_groupby_or_single(df):
        # process group
    """
    if _has_stock_id(df):
        return df.groupby("stock_id")
    else:
        return [(None, df)]


def _get_date_column(df: pd.DataFrame) -> pd.Series:
    """
    Get date column from DataFrame.

    Uses 'date' column if exists, otherwise uses index.
    Ensures result is datetime type.
    """
    if "date" in df.columns:
        return pd.to_datetime(df["date"])
    else:
        return pd.to_datetime(df.index)


def _detect_local_extrema_safe(
    df: pd.DataFrame, price_col: str = "close", lookback_window: int = 8, find_lows: bool = True, find_highs: bool = False
) -> pd.DataFrame:
    """
    Detect local extrema, handling multi-stock DataFrames properly.

    If df has stock_id, processes each stock separately to avoid
    cross-contamination in interleaved data.

    Args:
        df: DataFrame with price data
        price_col: Column to detect extrema in
        lookback_window: Window for extrema detection
        find_lows: Whether to detect lows
        find_highs: Whether to detect highs

    Returns:
        DataFrame with LocalLow and/or LocalHigh columns added
    """
    if _has_stock_id(df):
        # Process per stock to avoid cross-contamination
        result_dfs = []
        for stock_id in df["stock_id"].unique():
            stock_mask = df["stock_id"] == stock_id
            stock_df = df[stock_mask].copy()
            stock_df = find_local_extrema(
                stock_df, price_col=price_col, lookback_window=lookback_window, find_lows=find_lows, find_highs=find_highs
            )
            result_dfs.append(stock_df)
        return pd.concat(result_dfs, ignore_index=False).sort_index()
    else:
        return find_local_extrema(df, price_col=price_col, lookback_window=lookback_window, find_lows=find_lows, find_highs=find_highs)


def detect_multi_indicator_divergence(
    df: pd.DataFrame, lookback_window: int = 8
) -> pd.DataFrame:
    """
    Detect bullish divergence across multiple indicators.
    NO LOOKAHEAD BIAS - uses backward-looking local extrema.

    Bullish divergence = Lower low in PRICE but Higher low in INDICATOR.
    Checks across RSI, MACD, and Stochastic simultaneously.

    Score: 0=no divergence, 1=single, 2=double, 3=triple

    Args:
        df: DataFrame with price data
        lookback_window: Window for local extrema detection (default 8, similar to pivot lb)

    Returns:
        DataFrame with new column: multi_divergence_score (0-3)
    """
    df = df.copy()

    # Ensure indicators exist
    if "rsi" not in df.columns:
        df["rsi"] = calculate_rsi(df, period=14)

    if "macd" not in df.columns:
        macd, _, _ = calculate_macd(df)
        df["macd"] = macd

    if "stoch" not in df.columns:
        stoch_k, _ = calculate_stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3)
        df["stoch"] = stoch_k

    # Detect local lows using backward-looking method
    df = _detect_local_extrema_safe(df, price_col="close", lookback_window=lookback_window, find_lows=True, find_highs=False)

    df["multi_divergence_score"] = 0

    # Process each stock separately (or single stock if no stock_id)
    for stock_id, _group in _get_groupby_or_single(df):
        if _has_stock_id(df):
            stock_mask = df["stock_id"] == stock_id
            stock_data = df[stock_mask].copy()
        else:
            stock_data = df.copy()

        local_low_indices = stock_data[stock_data["LocalLow"] == 1].index.tolist()

        for i in range(len(local_low_indices) - 1):
            first_idx = local_low_indices[i]
            second_idx = local_low_indices[i + 1]

            first_price = df.loc[first_idx, "close"]
            second_price = df.loc[second_idx, "close"]

            # Only check if lower low in price
            if second_price < first_price:
                divergence_count = 0

                # Check RSI: higher low?
                if df.loc[second_idx, "rsi"] > df.loc[first_idx, "rsi"]:
                    divergence_count += 1

                # Check MACD: higher low?
                if df.loc[second_idx, "macd"] > df.loc[first_idx, "macd"]:
                    divergence_count += 1

                # Check Stochastic: higher low?
                if df.loc[second_idx, "stoch"] > df.loc[first_idx, "stoch"]:
                    divergence_count += 1

                df.loc[second_idx, "multi_divergence_score"] = divergence_count

    return df


def detect_volume_exhaustion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect volume exhaustion patterns.

    Price declining + Volume declining = Selling exhaustion.
    Sellers giving up despite price still falling.

    Normal selling has price down + volume up.
    Exhaustion is price down + volume also down.

    Returns:
        DataFrame with new columns:
        - volume_exhaustion (binary): price down >2% AND volume down >10%
        - exhaustion_strength (continuous): magnitude of exhaustion
    """
    df = df.copy()

    if _has_stock_id(df):
        grouped = df.groupby("stock_id")
    else:
        grouped = df.groupby(lambda x: 0)  # Single group

    # Calculate 5-day price change
    df["price_change_5d"] = grouped["close"].transform(lambda x: x.pct_change(5))

    # Calculate 5-day volume change
    df["volume_change_5d"] = grouped["volume"].transform(lambda x: x.pct_change(5))

    # Handle inf from pct_change when volume was 0 (holidays)
    df["volume_change_5d"] = df["volume_change_5d"].replace([np.inf, -np.inf], 0)

    # Exhaustion: price down but volume also down
    df["volume_exhaustion"] = (
        (df["price_change_5d"] < -0.02)  # Price down >2%
        & (df["volume_change_5d"] < -0.10)  # Volume down >10%
    ).astype(int)

    # Continuous version: magnitude of exhaustion
    df["exhaustion_strength"] = np.where(
        (df["price_change_5d"] < 0) & (df["volume_change_5d"] < 0),
        abs(df["price_change_5d"]) * abs(df["volume_change_5d"]) * 100,
        0,
    )

    return df


def detect_panic_selling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect panic selling / capitulation events.

    Extreme volume spike + Extreme price drop = Panic/Capitulation.
    Often marks exact bottom.

    Returns:
        DataFrame with new columns:
        - panic_selling (binary): volume >2x average AND drop >2 std devs
        - panic_severity (0-10): how extreme is the event
    """
    df = df.copy()

    if _has_stock_id(df):
        grouped = df.groupby("stock_id")
    else:
        grouped = df.groupby(lambda x: 0)

    # Volume spike (per stock)
    df["volume_ma20"] = grouped["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_spike_ratio"] = df["volume"] / df["volume_ma20"].replace(0, 1)

    # Daily return
    df["ret_1d"] = grouped["close"].transform(lambda x: x.pct_change(fill_method=None))

    # Historical volatility (per stock)
    df["volatility_20d"] = grouped["ret_1d"].transform(lambda x: x.rolling(20).std())

    # Panic conditions
    df["panic_selling"] = ((df["volume_spike_ratio"] > 2.0) & (df["ret_1d"] < -2 * df["volatility_20d"])).astype(int)

    # Severity: how extreme is the event?
    df["panic_severity"] = np.where(
        df["panic_selling"] == 1, df["volume_spike_ratio"] * abs(df["ret_1d"] / df["volatility_20d"].replace(0, 1)), 0
    ).clip(0, 10)

    return df


def detect_support_tests(
    df: pd.DataFrame, tolerance: float = 0.02, lookback_window: int = 8
) -> pd.DataFrame:
    """
    Count how many times price has tested similar support levels.
    NO LOOKAHEAD BIAS - uses backward-looking local extrema.

    More tests = stronger support = higher probability bounce.

    Args:
        df: DataFrame with price data
        tolerance: Price similarity tolerance (default 2%)
        lookback_window: Window for local extrema detection (default 8)

    Returns:
        DataFrame with new column: support_test_count
    """
    df = df.copy()

    # Detect local lows using backward-looking method
    if "LocalLow" not in df.columns:
        df = _detect_local_extrema_safe(df, price_col="close", lookback_window=lookback_window, find_lows=True, find_highs=False)

    df["support_test_count"] = 0

    # Process each stock separately
    for stock_id, _group in _get_groupby_or_single(df):
        if _has_stock_id(df):
            stock_mask = df["stock_id"] == stock_id
            stock_data = df[stock_mask].copy().reset_index(drop=True)
        else:
            stock_data = df.copy().reset_index(drop=True)

        # Get all local lows with their original indices
        local_low_data = []
        for idx in stock_data.index:
            if stock_data.loc[idx, "LocalLow"] == 1:
                local_low_data.append(
                    {
                        "idx": idx,
                        "price": stock_data.loc[idx, "close"],
                    }
                )

        # For each row, count similar prior local lows
        for current_idx in stock_data.index:
            current_price = stock_data.loc[current_idx, "close"]

            # Count prior local lows within tolerance
            test_count = 0
            for local_low in local_low_data:
                if local_low["idx"] < current_idx:  # Only prior lows
                    price_diff_pct = abs(local_low["price"] - current_price) / current_price
                    if price_diff_pct < tolerance:  # Within tolerance
                        test_count += 1

            # Update original dataframe
            if _has_stock_id(df):
                original_idx = df[stock_mask].index[current_idx]
            else:
                original_idx = stock_data.index[current_idx]
            df.loc[original_idx, "support_test_count"] = test_count

    return df


def detect_exhaustion_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect exhaustion in consecutive down days.

    Many down days + slowing drops = exhaustion = bottom near.

    Tracks:
    - How many consecutive down days
    - Whether selling is accelerating or decelerating

    Returns:
        DataFrame with new columns:
        - consecutive_down_days: count of consecutive negative days
        - selling_acceleration: change in daily return (negative=accelerating)
        - exhaustion_signal: many down days BUT decelerating
    """
    df = df.copy()

    if _has_stock_id(df):
        grouped = df.groupby("stock_id")
    else:
        grouped = df.groupby(lambda x: 0)

    df["ret_1d"] = grouped["close"].transform(lambda x: x.pct_change(fill_method=None))

    # Count consecutive down days (per stock)
    def count_consecutive_down(series):
        result = pd.Series(0, index=series.index)
        count = 0
        for idx in series.index:
            if pd.notna(series[idx]) and series[idx] < 0:
                count += 1
            else:
                count = 0
            result[idx] = count
        return result

    df["consecutive_down_days"] = grouped["ret_1d"].transform(count_consecutive_down)

    # Selling acceleration/deceleration
    df["ret_1d_prev"] = grouped["ret_1d"].shift(1)
    df["selling_acceleration"] = df["ret_1d"] - df["ret_1d_prev"]

    # Exhaustion: many down days BUT today's drop smaller than yesterday's
    # acceleration > 0 means LESS negative today = decelerating
    df["exhaustion_signal"] = (
        (df["consecutive_down_days"] >= 4)
        & (df["selling_acceleration"] > 0)
        & (df["ret_1d"] < 0)  # Still negative, just less so
    ).astype(int)

    return df


def detect_hidden_divergence(df: pd.DataFrame, lookback_window: int = 8) -> pd.DataFrame:
    """
    Detect hidden bullish divergence.
    NO LOOKAHEAD BIAS - uses backward-looking local extrema.

    Hidden bullish divergence:
    - HIGHER low in price (making progress)
    - But LOWER low in RSI (underlying weakness)

    Different from regular divergence. Less useful for bottoms
    but can indicate false breakdowns.

    Args:
        df: DataFrame with price data
        lookback_window: Window for local extrema detection (default 8)

    Returns:
        DataFrame with new column: hidden_bullish_divergence
    """
    df = df.copy()

    if "rsi" not in df.columns:
        df["rsi"] = calculate_rsi(df, period=14)

    # Detect local lows using backward-looking method
    if "LocalLow" not in df.columns:
        df = _detect_local_extrema_safe(df, price_col="close", lookback_window=lookback_window, find_lows=True, find_highs=False)

    df["hidden_bullish_divergence"] = 0

    for stock_id, _group in _get_groupby_or_single(df):
        if _has_stock_id(df):
            stock_mask = df["stock_id"] == stock_id
            stock_data = df[stock_mask].copy()
        else:
            stock_data = df.copy()

        local_low_indices = stock_data[stock_data["LocalLow"] == 1].index.tolist()

        for i in range(len(local_low_indices) - 1):
            first_idx = local_low_indices[i]
            second_idx = local_low_indices[i + 1]

            first_price = df.loc[first_idx, "close"]
            second_price = df.loc[second_idx, "close"]
            first_rsi = df.loc[first_idx, "rsi"]
            second_rsi = df.loc[second_idx, "rsi"]

            # Hidden bullish: Higher low in price + Lower low in RSI
            if (second_price > first_price) and (second_rsi < first_rsi):
                df.loc[second_idx, "hidden_bullish_divergence"] = 1

    return df


def calculate_mean_reversion_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean reversion signals using statistical methods.

    Statistical bottom: How far is price from long-term mean?
    Z-score < -2 means 2+ standard deviations below mean (97.5th percentile).

    Returns:
        DataFrame with new columns:
        - price_zscore: (current - mean) / std
        - statistical_bottom: price >2 std devs below mean
        - at_zscore_extreme: at or near the lowest zscore in recent history
    """
    df = df.copy()

    if _has_stock_id(df):
        grouped = df.groupby("stock_id")
    else:
        grouped = df.groupby(lambda x: 0)

    # Long-term (1 year) mean and std
    df["price_ma252"] = grouped["close"].transform(lambda x: x.rolling(252, min_periods=100).mean())
    df["price_std252"] = grouped["close"].transform(lambda x: x.rolling(252, min_periods=100).std())

    # Z-score: (current - mean) / std
    df["price_zscore"] = (df["close"] - df["price_ma252"]) / df["price_std252"].replace(0, 1)

    # Extreme oversold (2 std devs below mean)
    df["statistical_bottom"] = (df["price_zscore"] < -2.0).astype(int)

    # Is this the most extreme we've been recently?
    df["min_zscore_252d"] = grouped["price_zscore"].transform(lambda x: x.rolling(252, min_periods=100).min())

    # At or near the extreme
    df["at_zscore_extreme"] = (df["price_zscore"] <= df["min_zscore_252d"] + 0.2).astype(int)

    return df


def detect_bb_squeeze_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Bollinger Band squeeze followed by breakdown.

    Bollinger Band squeeze (low volatility) followed by breakdown
    below lower band. Often precedes sharp reversal.

    Returns:
        DataFrame with new columns:
        - bb_squeeze: band width at 20-day low
        - below_lower_band: price below lower band
        - squeeze_breakdown: both conditions met
    """
    df = df.copy()

    if _has_stock_id(df):
        grouped = df.groupby("stock_id")
    else:
        grouped = df.groupby(lambda x: 0)

    # Calculate Bollinger Bands
    upper, middle, lower = calculate_bbands(df, period=20)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower

    # Band width (normalized)
    df["bb_width"] = (upper - lower) / middle

    # Is width at 20-day low? (squeeze)
    df["bb_width_min20"] = grouped["bb_width"].transform(lambda x: x.rolling(20, min_periods=10).min())

    # Squeeze = current width within 5% of minimum
    df["bb_squeeze"] = (df["bb_width"] <= df["bb_width_min20"] * 1.05).astype(int)

    # Price below lower band
    df["below_lower_band"] = (df["close"] < df["bb_lower"]).astype(int)

    # Squeeze + breakdown = potential reversal
    df["squeeze_breakdown"] = ((df["bb_squeeze"] == 1) & (df["below_lower_band"] == 1)).astype(int)

    return df


def add_time_features(df: pd.DataFrame, lookback_window: int = 8) -> pd.DataFrame:
    """
    Add time-based features for temporal patterns.
    NO LOOKAHEAD BIAS - uses backward-looking local extrema.

    Temporal patterns in bottoms:
    - Day of week effects
    - Days since last bottom (cyclicality)
    - Month-end and quarter-end effects

    Args:
        df: DataFrame with price data
        lookback_window: Window for local extrema detection (default 8)

    Returns:
        DataFrame with new columns:
        - day_of_week (0-6)
        - is_monday, is_friday
        - days_since_last_low
        - is_month_end, is_quarter_end
    """
    df = df.copy()

    # Get date column (from 'date' column or index)
    date_series = _get_date_column(df)
    if "date" not in df.columns:
        df["date"] = date_series
    else:
        df["date"] = date_series

    # Day of week (0=Monday, 4=Friday)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # Detect local lows using backward-looking method
    if "LocalLow" not in df.columns:
        df = _detect_local_extrema_safe(df, price_col="close", lookback_window=lookback_window, find_lows=True, find_highs=False)

    # Days since last local low (per stock)
    def calculate_days_since_low(group):
        result = pd.Series(np.nan, index=group.index)
        last_low_date = None

        for idx in group.index:
            if group.loc[idx, "LocalLow"] == 1:
                last_low_date = group.loc[idx, "date"]
                result[idx] = 0
            elif last_low_date is not None:
                result[idx] = (group.loc[idx, "date"] - last_low_date).days

        return result

    if _has_stock_id(df):
        df["days_since_last_low"] = (
            df.groupby("stock_id", group_keys=False).apply(calculate_days_since_low)
        )
    else:
        df["days_since_last_low"] = calculate_days_since_low(df)

    # Month-end indicator
    df["is_month_end"] = (df["date"].dt.is_month_end | (df["date"].dt.day >= 28)).astype(int)

    # Quarter-end
    df["is_quarter_end"] = (
        df["date"].dt.is_quarter_end | ((df["date"].dt.month.isin([3, 6, 9, 12])) & (df["date"].dt.day >= 28))
    ).astype(int)

    return df


def create_all_advanced_features(
    df: pd.DataFrame,
    support_tolerance: float = 0.02,
) -> pd.DataFrame:
    """
    Create all advanced ML features for bottom detection.

    Applies all feature engineering functions in sequence.
    Handles missing dependencies gracefully.

    Args:
        df: DataFrame with OHLCV data and base indicators
        support_tolerance: Price tolerance for support testing (default 2%)

    Returns:
        DataFrame with all advanced features added

    Features created:
        - multi_divergence_score (0-3)
        - volume_exhaustion, exhaustion_strength
        - panic_selling, panic_severity
        - support_test_count
        - consecutive_down_days, exhaustion_signal, selling_acceleration
        - hidden_bullish_divergence
        - price_zscore, statistical_bottom, at_zscore_extreme
        - bb_squeeze, below_lower_band, squeeze_breakdown
        - day_of_week, is_monday, is_friday, days_since_last_pivot
        - is_month_end, is_quarter_end
    """
    df = df.copy()

    # 1. Multi-indicator divergence (strongest signal)
    df = detect_multi_indicator_divergence(df)

    # 2. Volume exhaustion
    df = detect_volume_exhaustion(df)

    # 3. Panic selling detection
    df = detect_panic_selling(df)

    # 4. Support level testing
    df = detect_support_tests(df, tolerance=support_tolerance)

    # 5. Exhaustion sequence
    df = detect_exhaustion_sequence(df)

    # 6. Hidden divergence
    df = detect_hidden_divergence(df)

    # 7. Mean reversion
    df = calculate_mean_reversion_signal(df)

    # 9. Bollinger Band squeeze
    df = detect_bb_squeeze_breakdown(df)

    # 10. Time-based features
    df = add_time_features(df)

    return df


# List of all advanced feature columns for easy reference
ADVANCED_FEATURE_COLUMNS = [
    # Multi-indicator divergence
    "multi_divergence_score",  # 0-3 (most powerful)
    # Volume patterns
    "volume_exhaustion",  # 0 or 1
    "exhaustion_strength",  # continuous
    "panic_selling",  # 0 or 1
    "panic_severity",  # 0-10
    # Support levels
    "support_test_count",  # 0, 1, 2, 3+
    # Momentum exhaustion
    "consecutive_down_days",  # 0, 1, 2, 3+
    "exhaustion_signal",  # 0 or 1
    "selling_acceleration",  # negative=accelerating, positive=decelerating
    # Divergence patterns
    "hidden_bullish_divergence",  # 0 or 1
    # Statistical
    "price_zscore",  # typically -3 to +3
    "statistical_bottom",  # 0 or 1
    "at_zscore_extreme",  # 0 or 1
    # Volatility patterns
    "bb_squeeze",  # 0 or 1
    "below_lower_band",  # 0 or 1
    "squeeze_breakdown",  # 0 or 1
    # Temporal
    "day_of_week",  # 0-6
    "is_monday",  # 0 or 1
    "is_friday",  # 0 or 1
    "days_since_last_pivot",  # days
    "is_month_end",  # 0 or 1
    "is_quarter_end",  # 0 or 1
]
