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

    new_features = [
        "price_efficiency_10", "return_accel_10", "returns_skew_20",
        "returns_kurtosis_20", "volume_climax_ratio",
    ]
    g = df.groupby("stock_id")

    def _efficiency(close):
        ret = close.pct_change()
        net = ret.rolling(10, min_periods=10).sum().abs()
        total = ret.abs().rolling(10, min_periods=10).sum()
        return net / (total + 1e-10)

    def _accel(close):
        ret = close.pct_change()
        recent = ret.rolling(5, min_periods=5).sum()
        prior = ret.shift(5).rolling(5, min_periods=5).sum()
        return recent - prior

    def _skew(close):
        return close.pct_change().rolling(20, min_periods=15).skew()

    def _kurtosis(close):
        return close.pct_change().rolling(20, min_periods=15).kurt()

    def _volume_climax(volume):
        peak = volume.rolling(5, min_periods=5).max()
        avg = volume.rolling(20, min_periods=10).mean()
        return peak / (avg + 1e-10)

    df["price_efficiency_10"] = g["close"].transform(_efficiency)
    df["return_accel_10"] = g["close"].transform(_accel)
    df["returns_skew_20"] = g["close"].transform(_skew)
    df["returns_kurtosis_20"] = g["close"].transform(_kurtosis)
    df["volume_climax_ratio"] = g["volume"].transform(_volume_climax)

    # --- END researcher section ---

    return df, new_features
