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

    # Consecutive down days: count of consecutive negative close-to-close returns (per stock)
    def _consec_down(ret):
        down = (ret < 0).astype(int)
        streaks = down.groupby((down != down.shift()).cumsum()).cumsum()
        return streaks * down

    df["consecutive_down_days"] = g["ret_1d"].transform(_consec_down)
    new_features.append("consecutive_down_days")

    # Intraday recovery: close near daily high = buying pressure
    df["intraday_recovery"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    new_features.append("intraday_recovery")

    # Volume climax: volume relative to 20-day rolling max (per stock)
    df["volume_climax"] = df["volume"] / g["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).max()
    )
    new_features.append("volume_climax")

    # Return acceleration: 2nd derivative of 5-day returns
    # Positive = decline decelerating (bottom), Negative = still accelerating (knife)
    def _ret_accel(close):
        r5 = close.pct_change(5)
        return r5 - r5.shift(5)

    df["ret_acceleration"] = g["close"].transform(_ret_accel)
    new_features.append("ret_acceleration")

    # Lower wick ratio: long lower wick = buying pressure defending lows
    body_bottom = df[["open", "close"]].min(axis=1)
    df["lower_wick_ratio"] = (body_bottom - df["low"]) / (df["high"] - df["low"] + 1e-8)
    new_features.append("lower_wick_ratio")

    # Close position within 5-day range: higher = already bouncing (less knife risk)
    def _close_vs_low(close):
        lo = close.rolling(5, min_periods=1).min()
        hi = close.rolling(5, min_periods=1).max()
        return (close - lo) / (hi - lo + 1e-8)

    df["close_vs_5d_low"] = g["close"].transform(_close_vs_low)
    new_features.append("close_vs_5d_low")

    # Overnight gap: (open - prev_close) / prev_close
    # Large negative gap = panic/capitulation, positive gap = gap-up recovery
    prev_close = g["close"].shift(1)
    df["overnight_gap"] = (df["open"] - prev_close) / (prev_close.abs() + 1e-8)
    new_features.append("overnight_gap")

    # Volume-price divergence: volume change when price is declining over 5d
    # Negative = volume shrinking during selloff = selling exhaustion (bullish)
    # Positive = volume rising during selloff = heavy selling (bearish/knife)
    price_chg_5d = g["close"].transform(lambda x: x.pct_change(5))
    vol_chg_5d = g["volume"].transform(lambda x: x.pct_change(5))
    df["vol_price_divergence"] = vol_chg_5d.where(price_chg_5d < 0, 0)
    new_features.append("vol_price_divergence")

    # --- END researcher section ---

    return df, new_features
