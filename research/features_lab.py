"""Feature laboratory -- experimental features created by the researcher.

Winning features accumulate here across iterations.
Battle-tested features get promoted to lib/features.py later.

Rules:
  - All features MUST be backward-looking (no future data).
  - Compute per stock (groupby stock_id) to avoid cross-stock contamination.
  - Return the same DataFrame with new columns added.
"""

import pandas as pd


def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add experimental features. Returns (df, new_feature_names).

    The researcher edits this function to add new features.
    Must return the df with new columns AND a list of the new column names.
    """
    new_features = []

    # --- RESEARCHER: add features below ---

    g = df.groupby("stock_id")
    daily_range = df["high"] - df["low"]

    # Range contraction: today's range vs peak range in last 10 days
    # Values < 1 and declining = volatility narrowing = stabilization after selloff
    max_range_10d = daily_range.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(10, min_periods=1).max()
    )
    df["range_contraction_10d"] = daily_range / max_range_10d.replace(0, float("nan"))
    new_features.append("range_contraction_10d")

    # Price position in 10-day range (more responsive than bb_position / percentile_252)
    # Near 0 = at 10-day lows; near 1 = at 10-day highs
    high_10d = g["high"].transform(lambda x: x.rolling(10, min_periods=1).max())
    low_10d = g["low"].transform(lambda x: x.rolling(10, min_periods=1).min())
    range_10d = high_10d - low_10d
    df["price_pos_10d"] = (df["close"] - low_10d) / range_10d.replace(0, float("nan"))
    new_features.append("price_pos_10d")

    # Buying pressure ratio: 5-day avg of (close - low) / (high - low)
    # > 0.5 = consistently closing in upper half of range = accumulation
    close_in_range = (df["close"] - df["low"]) / daily_range.replace(0, float("nan"))
    df["buying_pressure_5d"] = close_in_range.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    new_features.append("buying_pressure_5d")

    # Overnight gap: (open - prev_close) / ATR
    # Captures pre-market/after-hours institutional order flow
    # At bottoms: gaps stop widening (exhaustion) or start filling (accumulation)
    prev_close = g["close"].shift(1)
    df["overnight_gap_atr"] = (df["open"] - prev_close) / df["atr_14"].replace(
        0, float("nan")
    )
    new_features.append("overnight_gap_atr")

    # 5-day mean of overnight gaps: persistent overnight selling trend
    # Shift from negative toward zero = overnight selling exhausting
    df["overnight_gap_trend_5d"] = df["overnight_gap_atr"].groupby(
        df["stock_id"]
    ).transform(lambda x: x.rolling(5, min_periods=1).mean())
    new_features.append("overnight_gap_trend_5d")

    # Selling exhaustion: recent (5d) down-day volume as fraction of 20d total
    # Declining ratio = selling volume drying up = genuine bottom forming
    # Targets knife_rate: falling knives have sustained/increasing sell volume
    down_vol = df["volume"] * (df["ret_1d"] < 0).astype(float)
    recent_down_vol = down_vol.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    total_down_vol_20d = down_vol.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    )
    df["sell_vol_exhaustion"] = recent_down_vol / total_down_vol_20d.replace(
        0, float("nan")
    )
    new_features.append("sell_vol_exhaustion")

    # Volume-weighted momentum (10d): returns scaled by relative volume
    # Heavy-volume drops dominate; as volume thins on declines, this rises toward zero
    # Complements sell_vol_exhaustion with directional momentum context
    vol_wt_ret = df["ret_1d"] * df["volume_ratio"]
    df["vol_wtd_momentum_10d"] = vol_wt_ret.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    new_features.append("vol_wtd_momentum_10d")

    # Variance ratio (5d/1d): mean-reversion vs trending regime
    # VR < 1 = mean-reverting (stabilizing, safer bottom entry)
    # VR > 1 = trending (directional, falling knife risk)
    ret_5d = g["close"].transform(lambda x: x.pct_change(5))
    var_1d = g["ret_1d"].transform(lambda x: x.rolling(20, min_periods=5).var())
    var_5d = ret_5d.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(20, min_periods=5).var()
    )
    df["variance_ratio"] = var_5d / (5 * var_1d).replace(0, float("nan"))
    new_features.append("variance_ratio")

    # Sign-flip frequency: fraction of consecutive return sign changes in 10 days
    # High = choppy/mean-reverting (bottoming pattern)
    # Low = persistent direction (trending, knife risk)
    ret_lag1 = g["ret_1d"].shift(1)
    sign_flip = ((df["ret_1d"] * ret_lag1) < 0).astype(float)
    df["sign_flip_ratio_10d"] = sign_flip.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    new_features.append("sign_flip_ratio_10d")

    # Consecutive down days: running count of consecutive negative-return days
    # Captures discrete decline structure that rolling windows smooth away
    # Longer streaks = higher reversal probability (mean reversion after sustained decline)
    is_down = (df["ret_1d"] < 0).astype(int)
    streak_groups = (1 - is_down).groupby(df["stock_id"]).cumsum()
    df["consec_down_days"] = is_down.groupby([df["stock_id"], streak_groups]).cumsum()
    new_features.append("consec_down_days")

    # Streak depth: cumulative return during current down streak
    # More negative = deeper capitulation = closer to exhaustion point
    df["streak_depth"] = (df["ret_1d"] * is_down).groupby(
        [df["stock_id"], streak_groups]
    ).cumsum()
    new_features.append("streak_depth")

    # --- END researcher section ---

    return df, new_features
