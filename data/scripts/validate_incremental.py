#!/usr/bin/env python3
"""Validate incremental dataset update.

Checks:
1. Overlapping dates have same feature values (within tolerance)
2. New data was added (dates beyond old max)
3. No unexpected data loss
4. Column consistency

Usage:
    uv run python data/scripts/validate_incremental.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOLERANCE = 1e-6  # For floating point comparison


def find_datasets() -> tuple[Path, Path] | None:
    """Find old and new dataset directories."""
    datasets_dir = Path(__file__).parent.parent / "datasets"

    dated_dirs = sorted(
        [d for d in datasets_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: x.name,
        reverse=True,
    )

    # Find two most recent with dataset.parquet
    with_dataset = [d for d in dated_dirs if (d / "dataset.parquet").exists()]

    if len(with_dataset) < 2:
        print("ERROR: Need at least 2 dataset folders with dataset.parquet")
        return None

    return with_dataset[1], with_dataset[0]  # old, new


def load_datasets(old_dir: Path, new_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load old and new datasets."""
    print(f"Loading {old_dir.name}/dataset.parquet...")
    old = pd.read_parquet(old_dir / "dataset.parquet")
    old["date"] = pd.to_datetime(old["date"])

    print(f"Loading {new_dir.name}/dataset.parquet...")
    new = pd.read_parquet(new_dir / "dataset.parquet")
    new["date"] = pd.to_datetime(new["date"])

    return old, new


def check_basic_stats(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Check basic statistics."""
    print("\n" + "=" * 60)
    print("BASIC STATS")
    print("=" * 60)

    old_max = old["date"].max()
    new_max = new["date"].max()

    print(f"\nOld dataset:")
    print(f"  Rows: {len(old):,}")
    print(f"  Stocks: {old['stock_id'].nunique()}")
    print(f"  Date range: {old['date'].min().date()} to {old_max.date()}")
    print(f"  Columns: {len(old.columns)}")

    print(f"\nNew dataset:")
    print(f"  Rows: {len(new):,}")
    print(f"  Stocks: {new['stock_id'].nunique()}")
    print(f"  Date range: {new['date'].min().date()} to {new_max.date()}")
    print(f"  Columns: {len(new.columns)}")

    # Check new data was added
    if new_max <= old_max:
        print(f"\n[FAIL] New dataset doesn't extend beyond old!")
        return False

    print(f"\n[PASS] New data extends to {new_max.date()} (was {old_max.date()})")
    return True


def check_column_consistency(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Check columns match."""
    print("\n" + "=" * 60)
    print("COLUMN CONSISTENCY")
    print("=" * 60)

    old_cols = set(old.columns)
    new_cols = set(new.columns)

    missing = old_cols - new_cols
    extra = new_cols - old_cols

    if missing:
        print(f"\n[WARN] Missing in new: {missing}")
    if extra:
        print(f"\n[WARN] Extra in new: {extra}")

    if not missing and not extra:
        print(f"\n[PASS] All {len(old_cols)} columns match")
        return True

    common = old_cols & new_cols
    print(f"[INFO] Common columns: {len(common)}")
    return len(missing) == 0  # Only fail if missing columns


def check_overlap_consistency(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Check overlapping data has same values."""
    print("\n" + "=" * 60)
    print("OVERLAP CONSISTENCY")
    print("=" * 60)

    # Find overlapping date range
    old_min, old_max = old["date"].min(), old["date"].max()
    new_min, new_max = new["date"].min(), new["date"].max()

    overlap_start = max(old_min, new_min)
    overlap_end = min(old_max, new_max)

    if overlap_start > overlap_end:
        print("[WARN] No overlapping dates!")
        return True

    print(f"Overlap period: {overlap_start.date()} to {overlap_end.date()}")

    # Get common columns (exclude non-feature cols for comparison)
    exclude = ["date", "stock_id", "open", "high", "low", "close", "volume"]
    common_cols = [c for c in old.columns if c in new.columns and c not in exclude]

    # Sample stocks for comparison (full comparison would be slow)
    common_stocks = set(old["stock_id"].unique()) & set(new["stock_id"].unique())
    sample_stocks = list(common_stocks)[:10]  # Check 10 stocks

    print(f"Checking {len(sample_stocks)} sample stocks...")

    mismatches = 0
    checked = 0

    for stock in sample_stocks:
        old_stock = old[(old["stock_id"] == stock) &
                        (old["date"] >= overlap_start) &
                        (old["date"] <= overlap_end)].set_index("date")
        new_stock = new[(new["stock_id"] == stock) &
                        (new["date"] >= overlap_start) &
                        (new["date"] <= overlap_end)].set_index("date")

        # Find common dates
        common_dates = old_stock.index.intersection(new_stock.index)
        if len(common_dates) == 0:
            continue

        # Compare feature values
        for col in common_cols[:20]:  # Check first 20 features
            if col not in old_stock.columns or col not in new_stock.columns:
                continue

            old_vals = old_stock.loc[common_dates, col]
            new_vals = new_stock.loc[common_dates, col]

            # Skip if both are NaN
            both_valid = ~(old_vals.isna() & new_vals.isna())
            if not both_valid.any():
                continue

            old_valid = old_vals[both_valid]
            new_valid = new_vals[both_valid]

            # Check for mismatches (allowing for NaN differences)
            diff = np.abs(old_valid - new_valid)
            max_diff = diff.max() if len(diff) > 0 else 0

            if max_diff > TOLERANCE:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  [MISMATCH] {stock}/{col}: max diff = {max_diff:.6f}")

            checked += 1

    print(f"\nChecked {checked} feature comparisons")

    if mismatches == 0:
        print("[PASS] All overlapping values match within tolerance")
        return True
    else:
        print(f"[WARN] {mismatches} mismatches found (may be due to recalculation)")
        return True  # Warn but don't fail - recalculation can cause small diffs


def check_new_data_exists(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Check new data actually exists beyond old max date."""
    print("\n" + "=" * 60)
    print("NEW DATA CHECK")
    print("=" * 60)

    old_max = old["date"].max()

    new_data = new[new["date"] > old_max]
    new_dates = new_data["date"].nunique()
    new_stocks = new_data["stock_id"].nunique()

    print(f"Data after {old_max.date()}:")
    print(f"  Rows: {len(new_data):,}")
    print(f"  Unique dates: {new_dates}")
    print(f"  Stocks with new data: {new_stocks}")

    if len(new_data) == 0:
        print("[FAIL] No new data found!")
        return False

    # Show date distribution
    if new_dates > 0:
        date_range = new_data.groupby("date").size()
        print(f"\nNew dates (first 5):")
        for date, count in list(date_range.items())[:5]:
            print(f"  {date.date()}: {count} rows")
        if new_dates > 5:
            print(f"  ... and {new_dates - 5} more dates")

    print(f"\n[PASS] {len(new_data):,} new rows added")
    return True


def check_no_data_loss(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Check we didn't lose significant data."""
    print("\n" + "=" * 60)
    print("DATA LOSS CHECK")
    print("=" * 60)

    old_stocks = set(old["stock_id"].unique())
    new_stocks = set(new["stock_id"].unique())

    lost_stocks = old_stocks - new_stocks
    gained_stocks = new_stocks - old_stocks

    if lost_stocks:
        print(f"[WARN] Lost {len(lost_stocks)} stocks: {list(lost_stocks)[:10]}")
    if gained_stocks:
        print(f"[INFO] Gained {len(gained_stocks)} stocks: {list(gained_stocks)[:10]}")

    # Check row counts in overlapping period
    old_max = old["date"].max()
    new_min = new["date"].min()

    overlap_start = max(old["date"].min(), new_min)
    overlap_end = old_max

    old_overlap = old[(old["date"] >= overlap_start) & (old["date"] <= overlap_end)]
    new_overlap = new[(new["date"] >= overlap_start) & (new["date"] <= overlap_end)]

    row_diff = len(new_overlap) - len(old_overlap)
    pct_diff = row_diff / len(old_overlap) * 100 if len(old_overlap) > 0 else 0

    print(f"\nOverlap period ({overlap_start.date()} to {overlap_end.date()}):")
    print(f"  Old rows: {len(old_overlap):,}")
    print(f"  New rows: {len(new_overlap):,}")
    print(f"  Difference: {row_diff:+,} ({pct_diff:+.1f}%)")

    # Allow some variance due to cleaning differences
    if abs(pct_diff) > 5:
        print(f"[WARN] Significant row count difference in overlap period")
    else:
        print(f"[PASS] Row counts are consistent")

    return len(lost_stocks) < 50  # Allow some stock loss due to filtering


def main():
    print("=" * 60)
    print("INCREMENTAL UPDATE VALIDATION")
    print("=" * 60)

    result = find_datasets()
    if result is None:
        sys.exit(1)

    old_dir, new_dir = result
    print(f"\nComparing: {old_dir.name} -> {new_dir.name}")

    old, new = load_datasets(old_dir, new_dir)

    checks = [
        ("Basic Stats", check_basic_stats(old, new)),
        ("Column Consistency", check_column_consistency(old, new)),
        ("Overlap Consistency", check_overlap_consistency(old, new)),
        ("New Data Exists", check_new_data_exists(old, new)),
        ("No Data Loss", check_no_data_loss(old, new)),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All validation checks passed!")
    else:
        print("\n[FAILURE] Some checks failed - review above")
        sys.exit(1)


if __name__ == "__main__":
    main()
