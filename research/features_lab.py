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

    # --- END researcher section ---

    return df, new_features
