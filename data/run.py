#!/usr/bin/env python3
"""
Unified entry point for data pipeline.

Usage:
    uv run python data/run.py                       # Interactive menu
    uv run python data/run.py --full                # Full pipeline (1-6)
    uv run python data/run.py --update              # Update OHLCV + incremental features (5-6)
    uv run python data/run.py --features            # Build features only (6)
    uv run python data/run.py --features-inc        # Incremental features only (6)
    uv run python data/run.py --features-test       # Test features with 100 stocks (6)
    uv run python data/run.py --ohlcv               # Build OHLCV only (1-5)
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent / "scripts"


def run_script(script_name: str, extra_args: list[str] | None = None) -> bool:
    """Run a script and return True if successful."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    print(f"\n{'='*70}")
    print(f"Running: {script_name}" + (f" {' '.join(extra_args)}" if extra_args else ""))
    print("=" * 70)

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_full_pipeline() -> None:
    """Run full pipeline: tickers -> OHLCV -> features."""
    scripts = [
        "1_get_tickers.py",
        "2_filter_tickers.py",
        "3_validate_tickers.py",
        "4_build_ohlcv.py",
        "6_build_features.py",
    ]

    for script in scripts:
        if not run_script(script):
            print(f"\nPipeline stopped at {script}")
            return

    print("\n" + "=" * 70)
    print("Full pipeline complete!")
    print("=" * 70)


def run_update() -> None:
    """Update existing dataset with new data."""
    if not run_script("5_update_ohlcv.py"):
        print("\nUpdate stopped at 5_update_ohlcv.py")
        return

    if not run_script("6_build_features.py", ["--incremental"]):
        print("\nUpdate stopped at 6_build_features.py")
        return

    print("\n" + "=" * 70)
    print("Update complete!")
    print("=" * 70)


def run_features_only(incremental: bool = False, test: int | None = None) -> None:
    """Build features from existing OHLCV data."""
    extra_args = []
    if incremental:
        extra_args.append("--incremental")
    if test:
        extra_args.extend(["--test", str(test)])

    if not run_script("6_build_features.py", extra_args if extra_args else None):
        print("\nFeature building failed")
        return

    print("\n" + "=" * 70)
    print("Features built!")
    print("=" * 70)


def run_ohlcv_only() -> None:
    """Run OHLCV pipeline without features."""
    scripts = [
        "1_get_tickers.py",
        "2_filter_tickers.py",
        "3_validate_tickers.py",
        "4_build_ohlcv.py",
    ]

    for script in scripts:
        if not run_script(script):
            print(f"\nPipeline stopped at {script}")
            return

    print("\n" + "=" * 70)
    print("OHLCV pipeline complete!")
    print("=" * 70)


def interactive_menu() -> None:
    """Display interactive menu."""
    print("\n" + "=" * 70)
    print("DATA PIPELINE")
    print("=" * 70)
    print("\nWhat would you like to do?\n")
    print("  [1] Full pipeline (tickers -> OHLCV)     ~3-4 hours")
    print("  [2] Update existing data (incremental)   ~10-30 min")
    print("  [3] Build features (full)                ~30 min")
    print("  [4] Build features (incremental)         ~5 min")
    print("  [5] Build features (test 100 stocks)     ~2 min")
    print("  [6] OHLCV only (skip features)           ~3-4 hours")
    print("  [q] Quit")
    print()

    choice = input("Choice: ").strip().lower()

    if choice == "1":
        run_full_pipeline()
    elif choice == "2":
        run_update()
    elif choice == "3":
        run_features_only()
    elif choice == "4":
        run_features_only(incremental=True)
    elif choice == "5":
        run_features_only(test=100)
    elif choice == "6":
        run_ohlcv_only()
    elif choice == "q":
        print("Goodbye!")
    else:
        print(f"Unknown choice: {choice}")


def main():
    parser = argparse.ArgumentParser(description="Data pipeline entry point")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--update", action="store_true", help="Update existing data (incremental)")
    parser.add_argument("--features", action="store_true", help="Build features (full)")
    parser.add_argument("--features-inc", action="store_true", help="Build features (incremental)")
    parser.add_argument("--features-test", type=int, metavar="N", nargs="?", const=100, help="Test with N stocks")
    parser.add_argument("--ohlcv", action="store_true", help="OHLCV pipeline only")

    args = parser.parse_args()

    if args.full:
        run_full_pipeline()
    elif args.update:
        run_update()
    elif args.features:
        run_features_only()
    elif args.features_inc:
        run_features_only(incremental=True)
    elif args.features_test:
        run_features_only(test=args.features_test)
    elif args.ohlcv:
        run_ohlcv_only()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
