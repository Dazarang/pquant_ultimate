#!/usr/bin/env python3
"""Build ML-ready dataset from OHLCV data.

Usage:
    uv run python data/scripts/6_build_features.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.features import build_features  # noqa: E402


def find_latest_dataset() -> Path | None:
    """Find latest datasets folder with ohlcv.parquet."""
    datasets_dir = Path(__file__).parent.parent / "datasets"
    if not datasets_dir.exists():
        return None

    dated_dirs = sorted(
        [d for d in datasets_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: x.name,
        reverse=True,
    )

    for d in dated_dirs:
        if (d / "ohlcv.parquet").exists():
            return d

    return None


def load_ohlcv(dataset_dir: Path) -> pd.DataFrame:
    """Load OHLCV data and prepare for feature engineering."""
    ohlcv_path = dataset_dir / "ohlcv.parquet"
    print(f"Loading {ohlcv_path}...")

    df = pd.read_parquet(ohlcv_path)
    print(f"  Loaded {len(df):,} rows")

    df["date"] = pd.to_datetime(df["date"])

    # Rename ticker to stock_id for consistency
    if "ticker" in df.columns and "stock_id" not in df.columns:
        df = df.rename(columns={"ticker": "stock_id"})

    return df


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid OHLCV rows. No filling - real data only."""
    print("\nCleaning OHLCV...")
    before = len(df)

    # Drop rows with any NaN in OHLCV
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    nan_rows = df[ohlcv_cols].isnull().any(axis=1).sum()
    if nan_rows > 0:
        df = df.dropna(subset=ohlcv_cols)
        print(f"  Dropped {nan_rows:,} rows with NaN")

    # Drop zero/negative volume (no trading occurred)
    bad_volume = (df["volume"] <= 0).sum()
    if bad_volume > 0:
        df = df[df["volume"] > 0]
        print(f"  Dropped {bad_volume:,} rows with zero/negative volume")

    # Drop invalid OHLC relationships
    invalid_hl = (df["high"] < df["low"]).sum()
    if invalid_hl > 0:
        df = df[df["high"] >= df["low"]]
        print(f"  Dropped {invalid_hl:,} rows with high < low")

    invalid_hc = (df["high"] < df["close"]).sum()
    if invalid_hc > 0:
        df = df[df["high"] >= df["close"]]
        print(f"  Dropped {invalid_hc:,} rows with high < close")

    invalid_lc = (df["low"] > df["close"]).sum()
    if invalid_lc > 0:
        df = df[df["low"] <= df["close"]]
        print(f"  Dropped {invalid_lc:,} rows with low > close")

    invalid_ho = (df["high"] < df["open"]).sum()
    if invalid_ho > 0:
        df = df[df["high"] >= df["open"]]
        print(f"  Dropped {invalid_ho:,} rows with high < open")

    invalid_lo = (df["low"] > df["open"]).sum()
    if invalid_lo > 0:
        df = df[df["low"] <= df["open"]]
        print(f"  Dropped {invalid_lo:,} rows with low > open")

    # Drop negative prices
    neg_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if neg_prices > 0:
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        print(f"  Dropped {neg_prices:,} rows with non-positive prices")

    after = len(df)
    print(f"  Total: {before:,} -> {after:,} ({before - after:,} dropped)")

    return df.reset_index(drop=True)


def clean_dataset(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Remove rows with NaN in feature columns per stock."""
    print("\nCleaning NaN per stock...")

    clean_dfs = []
    total_dropped = 0
    stocks = df["stock_id"].unique()

    for stock_id in tqdm(stocks, desc="  Cleaning", ncols=80):
        stock_df = df[df["stock_id"] == stock_id].copy()
        before = len(stock_df)
        stock_df = stock_df.dropna(subset=feature_cols)
        dropped = before - len(stock_df)
        total_dropped += dropped
        clean_dfs.append(stock_df)

    result = pd.concat(clean_dfs, ignore_index=True)
    result = result.sort_values(["date", "stock_id"]).reset_index(drop=True)

    print(f"  Dropped {total_dropped:,} rows total")
    return result


def validate_final(df: pd.DataFrame, feature_cols: list[str]) -> bool:
    """Final validation."""
    print("\nFinal validation...")

    nan_count = df[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        print(f"  FAILED: {nan_count} NaN remain")
        return False

    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_df).sum().sum()
    if inf_count > 0:
        print(f"  FAILED: {inf_count} inf values")
        return False

    if "PivotLow" not in df.columns:
        print("  FAILED: Missing PivotLow")
        return False

    print("  PASSED")
    return True


def save_dataset(df: pd.DataFrame, dataset_dir: Path) -> None:
    """Save dataset.parquet and update metadata."""
    output_path = dataset_dir / "dataset.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Update metadata
    metadata_path = dataset_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    feature_cols = [
        col for col in df.columns if col not in ["date", "stock_id", "open", "high", "low", "close", "volume"]
    ]

    metadata["features"] = {
        "created_at": datetime.now().isoformat(),
        "total_features": len(feature_cols),
        "total_rows": len(df),
        "stocks": df["stock_id"].nunique(),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "labels": {
            "PivotLow": int(df["PivotLow"].sum()),
            "PivotHigh": int(df["PivotHigh"].sum()) if "PivotHigh" in df.columns else 0,
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    total_start = time.time()

    print("=" * 70)
    print("BUILD FEATURES")
    print("=" * 70)

    dataset_dir = find_latest_dataset()
    if dataset_dir is None:
        print("\nERROR: No datasets found. Run steps 1-4 first.")
        return

    print(f"\nUsing dataset: {dataset_dir.name}")

    existing = dataset_dir / "dataset.parquet"
    if existing.exists():
        print(f"\nWARNING: {existing} exists")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    df = load_ohlcv(dataset_dir)
    df = clean_ohlcv(df)

    if len(df) == 0:
        print("\nERROR: No valid data after cleaning")
        return

    start = time.time()
    df = build_features(df, verbose=True)
    print(f"  Feature building took {time.time() - start:.1f}s")

    exclude = ["date", "stock_id", "open", "high", "low", "close", "volume"]
    feature_cols = [col for col in df.columns if col not in exclude]

    before = len(df)
    df = clean_dataset(df, feature_cols)
    after = len(df)
    print(f"\nCleaned: {before:,} -> {after:,} ({(before - after) / before * 100:.1f}% dropped)")

    if not validate_final(df, feature_cols):
        return

    save_dataset(df, dataset_dir)

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Stocks: {df['stock_id'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Pivot bottoms: {df['PivotLow'].sum():,}")
    imbalance = (df["PivotLow"] == 0).sum() / max(df["PivotLow"].sum(), 1)
    print(f"  Imbalance: 1:{imbalance:.0f}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    print("=" * 70)


if __name__ == "__main__":
    main()
