#!/usr/bin/env python3
"""Build ML-ready dataset from OHLCV data.

Usage:
    uv run python data/scripts/6_build_features.py               # Full build
    uv run python data/scripts/6_build_features.py --incremental  # Update only
    uv run python data/scripts/6_build_features.py --test          # Test with 100 stocks
    uv run python data/scripts/6_build_features.py --test 50       # Test with 50 stocks
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.features import build_features  # noqa: E402

# Lookback buffer in calendar days (covers 252 trading days for indicators)
LOOKBACK_DAYS = 365


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


def load_ohlcv(dataset_dir: Path, test_stocks: int | None = None) -> pd.DataFrame:
    """Load OHLCV data and prepare for feature engineering.

    Args:
        dataset_dir: Path to dataset directory
        test_stocks: If set, limit to N random stocks for testing
    """
    ohlcv_path = dataset_dir / "ohlcv.parquet"
    print(f"Loading {ohlcv_path}...")

    df = pd.read_parquet(ohlcv_path)
    print(f"  Loaded {len(df):,} rows")

    # Keep only required columns (drop extras like 'adj close', 'is_failed')
    stock_col = "stock_id" if "stock_id" in df.columns else "ticker"
    required_cols = ["date", "open", "high", "low", "close", "volume", stock_col]
    df = df[[c for c in required_cols if c in df.columns]].copy()

    df["date"] = pd.to_datetime(df["date"])

    # Rename ticker to stock_id for consistency
    if "ticker" in df.columns and "stock_id" not in df.columns:
        df = df.rename(columns={"ticker": "stock_id"})

    # Limit to test stocks if requested
    if test_stocks:
        all_stocks = df["stock_id"].unique()
        np.random.seed(42)  # Reproducible
        selected = np.random.choice(all_stocks, min(test_stocks, len(all_stocks)), replace=False)
        df = df[df["stock_id"].isin(selected)]
        print(f"  Test mode: {len(selected)} stocks, {len(df):,} rows")

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


def save_dataset(df: pd.DataFrame, dataset_dir: Path, incremental: bool = False) -> None:
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


def find_previous_dataset() -> Path | None:
    """Find previous dataset folder that has dataset.parquet."""
    datasets_dir = Path(__file__).parent.parent / "datasets"
    if not datasets_dir.exists():
        return None

    dated_dirs = sorted(
        [d for d in datasets_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: x.name,
        reverse=True,
    )

    for d in dated_dirs:
        if (d / "dataset.parquet").exists():
            return d

    return None


def check_incremental_possible(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp] | None:
    """Check if incremental update is possible and return data splits.

    Returns:
        (safe_dataset, ohlcv_to_process, cutoff_date) or None if not possible
        - safe_dataset: rows with date < cutoff (keep existing features)
        - ohlcv_to_process: OHLCV from lookback_start onwards (for recalculation)
        - cutoff_date: date from which we replace features
    """
    ohlcv_path = dataset_dir / "ohlcv.parquet"

    if not ohlcv_path.exists():
        print("  No ohlcv.parquet found")
        return None

    # Find dataset.parquet - might be in current dir or previous dir
    dataset_path = dataset_dir / "dataset.parquet"
    if not dataset_path.exists():
        # Look for previous dataset folder
        prev_dir = find_previous_dataset()
        if prev_dir is None:
            print("  No existing dataset.parquet - use full build")
            return None
        dataset_path = prev_dir / "dataset.parquet"
        print(f"  Using previous dataset: {prev_dir.name}/dataset.parquet")

    # Load both
    dataset = pd.read_parquet(dataset_path)
    ohlcv = pd.read_parquet(ohlcv_path)

    # Filter OHLCV to required columns (same as load_ohlcv)
    stock_col = "stock_id" if "stock_id" in ohlcv.columns else "ticker"
    required_cols = ["date", "open", "high", "low", "close", "volume", stock_col]
    ohlcv = ohlcv[[c for c in required_cols if c in ohlcv.columns]].copy()

    dataset["date"] = pd.to_datetime(dataset["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Rename ticker to stock_id if needed
    if "ticker" in ohlcv.columns and "stock_id" not in ohlcv.columns:
        ohlcv = ohlcv.rename(columns={"ticker": "stock_id"})

    dataset_max = dataset["date"].max()
    ohlcv_max = ohlcv["date"].max()

    print(f"  Dataset ends: {dataset_max.date()}")
    print(f"  OHLCV ends:   {ohlcv_max.date()}")

    if ohlcv_max <= dataset_max:
        print("  OHLCV not newer than dataset - nothing to update")
        return None

    # Cutoff: LOOKBACK_DAYS back from dataset max
    # We recalculate features from cutoff onwards (replacing them)
    cutoff = dataset_max - timedelta(days=LOOKBACK_DAYS)

    # Lookback start: another LOOKBACK_DAYS for indicator warmup
    lookback_start = cutoff - timedelta(days=LOOKBACK_DAYS)

    print(f"  Cutoff: {cutoff.date()} (replacing features from here)")
    print(f"  Lookback: {lookback_start.date()} (for indicator warmup)")

    # Safe dataset: rows before cutoff (unchanged)
    safe_dataset = dataset[dataset["date"] < cutoff].copy()

    # OHLCV to process: from lookback_start onwards (includes buffer + new)
    ohlcv_to_process = ohlcv[ohlcv["date"] >= lookback_start].copy()

    print(f"  Safe rows: {len(safe_dataset):,}")
    print(f"  OHLCV to process: {len(ohlcv_to_process):,}")

    return safe_dataset, ohlcv_to_process, cutoff


def build_incremental(dataset_dir: Path) -> bool:
    """Build features incrementally.

    Returns:
        True if successful, False otherwise
    """
    total_start = time.time()

    print("=" * 70)
    print("BUILD FEATURES (INCREMENTAL)")
    print("=" * 70)

    print(f"\nUsing dataset: {dataset_dir.name}")
    print("\nChecking incremental update...")

    result = check_incremental_possible(dataset_dir)
    if result is None:
        print("\nFalling back to full build...")
        return False

    safe_dataset, ohlcv_to_process, cutoff = result

    # Clean OHLCV
    ohlcv_to_process = clean_ohlcv(ohlcv_to_process)

    if len(ohlcv_to_process) == 0:
        print("\nERROR: No valid OHLCV after cleaning")
        return False

    # Build features for the chunk (includes lookback buffer)
    start = time.time()
    new_features = build_features(ohlcv_to_process, verbose=True)
    print(f"  Feature building took {time.time() - start:.1f}s")

    # Keep only rows >= cutoff (discard lookback buffer used for warmup)
    new_features = new_features[new_features["date"] >= cutoff].copy()
    print(f"\nKept rows >= {cutoff.date()}: {len(new_features):,}")

    if len(new_features) == 0:
        print("ERROR: No rows after cutoff filter")
        return False

    # Get feature columns
    exclude = ["date", "stock_id", "open", "high", "low", "close", "volume"]
    feature_cols = [col for col in new_features.columns if col not in exclude]

    # Clean NaN from new features
    before = len(new_features)
    new_features = clean_dataset(new_features, feature_cols)
    after = len(new_features)
    print(f"\nCleaned new: {before:,} -> {after:,}")

    # Validate new features
    if not validate_final(new_features, feature_cols):
        return False

    # Ensure columns match (use intersection if different)
    safe_cols = set(safe_dataset.columns)
    new_cols = set(new_features.columns)
    if safe_cols != new_cols:
        missing = safe_cols - new_cols
        extra = new_cols - safe_cols
        if missing:
            print(f"  WARNING: Missing in new: {len(missing)} cols")
        if extra:
            print(f"  WARNING: Extra in new: {len(extra)} cols")
        common = list(safe_cols & new_cols)
        safe_dataset = safe_dataset[common]
        new_features = new_features[common]

    # Concat: safe (unchanged) + new (recalculated)
    combined = pd.concat([safe_dataset, new_features], ignore_index=True)
    combined = combined.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # Remove any duplicates (prefer new data)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["date", "stock_id"], keep="last")
    if len(combined) < before_dedup:
        print(f"  Removed {before_dedup - len(combined)} duplicates")

    save_dataset(combined, dataset_dir, incremental=True)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("INCREMENTAL UPDATE SUMMARY")
    print("=" * 70)
    print(f"  Safe rows kept: {len(safe_dataset):,}")
    print(f"  New rows added: {len(new_features):,}")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Stocks: {combined['stock_id'].nunique()}")
    print(f"  Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    print("=" * 70)

    return True


def build_full(dataset_dir: Path, test_stocks: int | None = None) -> None:
    """Build features from scratch (full build)."""
    total_start = time.time()

    print("=" * 70)
    print(f"BUILD FEATURES ({'TEST' if test_stocks else 'FULL'})")
    print("=" * 70)

    print(f"\nUsing dataset: {dataset_dir.name}")

    existing = dataset_dir / "dataset.parquet"
    if existing.exists() and not test_stocks:
        print(f"\nWARNING: {existing} exists")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    df = load_ohlcv(dataset_dir, test_stocks=test_stocks)
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

    if test_stocks:
        print("\n[TEST MODE] Not saving dataset.parquet")
    else:
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


def main():
    parser = argparse.ArgumentParser(description="Build ML features from OHLCV data")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental update: only recalculate last 365 days + new data",
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Test mode: use N random stocks (default 100 if flag used without value)",
        nargs="?",
        const=100,
    )
    args = parser.parse_args()

    dataset_dir = find_latest_dataset()
    if dataset_dir is None:
        print("\nERROR: No datasets found. Run steps 1-4 first.")
        return

    if args.incremental:
        success = build_incremental(dataset_dir)
        if not success:
            build_full(dataset_dir, test_stocks=args.test)
    else:
        build_full(dataset_dir, test_stocks=args.test)


if __name__ == "__main__":
    main()
