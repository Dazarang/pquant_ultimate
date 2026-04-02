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

    new_features = ["close_position", "lower_wick_ratio", "gap_return"]

    # --- END researcher section ---

    return df, new_features
