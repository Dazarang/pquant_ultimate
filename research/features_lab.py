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

    # --- END researcher section ---

    return df, new_features
