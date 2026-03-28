"""Dataset loading, temporal splitting, and scaling utilities.

All splitting is strictly temporal -- no future data leaks into training.
Scaler is always fit on train only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

META_COLS = ["date", "stock_id", "open", "high", "low", "close", "volume"]
LABEL_COL = "PivotLow"


def load_dataset(
    path: str,
    stocks: str | list[str] | None = None,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a dataset.parquet and return (df, feature_cols).

    Args:
        path: Path to dataset.parquet.
        stocks: Filter to specific stocks. None = all, "AAPL" = single,
                ["AAPL", "MSFT"] = subset.
        features: Feature columns to keep. None = all. Subset = only those columns.
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)

    if stocks is not None:
        if isinstance(stocks, str):
            stocks = [stocks]
        missing = set(stocks) - set(df["stock_id"].unique())
        if missing:
            print(f"  WARNING: stocks not found in dataset: {missing}")
        df = df[df["stock_id"].isin(stocks)].reset_index(drop=True)

    if "PivotHigh" in df.columns:
        df = df.drop(columns=["PivotHigh"])

    all_features = [c for c in df.columns if c not in META_COLS + [LABEL_COL]]

    if features is not None:
        missing_f = set(features) - set(all_features)
        if missing_f:
            print(f"  WARNING: features not found in dataset: {missing_f}")
        feature_cols = [c for c in features if c in all_features]
        drop_cols = [c for c in all_features if c not in feature_cols]
        df = df.drop(columns=drop_cols)
    else:
        feature_cols = all_features

    df[feature_cols] = df[feature_cols].replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows, {df['stock_id'].nunique()} stocks, {len(feature_cols)} features")
    return df, feature_cols


def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
    embargo_sessions: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by date with embargo gap at boundaries.

    Embargo purges `embargo_sessions` trading days from the end of each earlier
    split to prevent label leakage from centered pivot windows (rb=13).
    """
    dates = sorted(df["date"].unique())

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train_end_idx = np.searchsorted(dates, train_end_ts, side="right") - 1
    val_end_idx = np.searchsorted(dates, val_end_ts, side="right") - 1

    # Purge last embargo_sessions from train; val starts after gap
    train_cutoff = dates[max(0, train_end_idx - embargo_sessions)]
    val_start = dates[min(len(dates) - 1, train_end_idx + 1)]
    val_cutoff = dates[max(0, val_end_idx - embargo_sessions)]
    test_start = dates[min(len(dates) - 1, val_end_idx + 1)]

    train = df[df["date"] <= train_cutoff].copy()
    val = df[(df["date"] >= val_start) & (df["date"] <= val_cutoff)].copy()
    test = df[df["date"] >= test_start].copy()

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        d = split["date"]
        n_stocks = split["stock_id"].nunique()
        print(f"  {name}: {len(split):>9,} rows | {n_stocks:>5} stocks | {d.min().date()} to {d.max().date()}")

    return train, val, test


def scale(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, transform all splits. Returns copies."""
    scaler = StandardScaler()

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    return train, val, test, scaler
