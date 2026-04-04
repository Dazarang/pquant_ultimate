"""Feature laboratory -- custom features created by the researcher.

Features accumulate here across iterations.
Stable features may be moved to lib/features.py later.

Rules:
  - All features MUST be backward-looking (no future data).
  - Compute per stock (groupby stock_id) to avoid cross-stock contamination.
  - Return the same DataFrame with new columns added.
"""

import pandas as pd


def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add custom features. Returns (df, new_feature_names).

    The researcher edits this function to add new features.
    Must return the df with new columns AND a list of the new column names.
    """
    new_features = []

    # --- RESEARCHER: add features below ---

    rng = (df["high"] - df["low"]).clip(lower=1e-10)

    df["close_position"] = (df["close"] - df["low"]) / rng

    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

    prev_close = df.groupby("stock_id")["close"].shift(1)
    df["gap_return"] = (df["open"] - prev_close) / prev_close

    df["body_ratio"] = (df["close"] - df["open"]).abs() / rng

    vol_sma20 = df.groupby("stock_id")["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    ).clip(lower=1e-10)
    df["volume_sma_ratio"] = df["volume"] / vol_sma20

    df["high_vol_reversal"] = df["close_position"] * df["volume_sma_ratio"]

    df["signed_candle"] = (df["close"] - df["open"]) / rng

    df["_range"] = df["high"] - df["low"]
    avg_range_10d = df.groupby("stock_id")["_range"].transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    ).clip(lower=1e-10)
    df["range_ratio_10d"] = df["_range"] / avg_range_10d
    df.drop(columns=["_range"], inplace=True)

    prev_high = df.groupby("stock_id")["high"].shift(1)
    df["close_vs_prev_high"] = (df["close"] - prev_high) / prev_high.clip(lower=1e-10)

    g = df.groupby("stock_id")
    lag_cols = []
    for col in ["rsi_14", "drawdown"]:
        for lag in [1, 2, 3]:
            name = f"{col}_lag{lag}"
            df[name] = g[col].shift(lag)
            lag_cols.append(name)

    new_features = [
        "close_position", "lower_wick_ratio", "gap_return",
        "body_ratio", "volume_sma_ratio", "high_vol_reversal",
        "signed_candle", "range_ratio_10d", "close_vs_prev_high",
    ] + lag_cols

    # --- END researcher section ---

    return df, new_features
