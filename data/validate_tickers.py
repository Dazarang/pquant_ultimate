"""
Validate filtered tickers against Yahoo Finance.
Removes delisted/invalid stocks.
REQUIRES: filtered JSON from filter_tickers.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ticker_file_utils import TickerFileFinder
from validators import TickerValidator


def load_ticker_json(json_path):
    """Load ticker JSON file."""
    with open(json_path) as f:
        return json.load(f)


def save_ticker_json(data, output_path):
    """Save validated ticker JSON."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """Main execution."""
    # Configuration
    TEST_MODE = False  # Set to True for testing with 200 tickers/category

    # Find latest filtered ticker file automatically
    finder = TickerFileFinder()
    input_file = finder.get_latest_filtered()

    if not input_file:
        print("ERROR: No filtered ticker files found matching pattern 'tickers_filtered_*.json'")
        print("Run filter_tickers.py first!")
        return

    INPUT_JSON = str(input_file)
    OUTPUT_JSON = str(input_file.parent / f"tickers_validated_{datetime.now().strftime('%Y%m%d')}.json")

    print("=" * 70)
    print("TICKER VALIDATION (API CALLS)")
    if TEST_MODE:
        print("*** TEST MODE: Validating 200 tickers per category ***")
    print("=" * 70)
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_JSON}")
    print("\nThis validates FILTERED tickers only (no ETFs, warrants, etc)")

    # Load filtered data
    print("\nLoading filtered ticker data...")
    try:
        data = load_ticker_json(INPUT_JSON)
    except FileNotFoundError:
        print(f"\nERROR: {INPUT_JSON} not found")
        print("Run filter_tickers.py first!")
        return

    # Create validator
    validator = TickerValidator(validation_days=3, rate_limit_delay=0.15)

    # Process each category
    validated_data = {}

    # Process US tickers
    if "US" in data and data["US"]:
        us_tickers = data["US"][:200] if TEST_MODE else data["US"]
        print(f"\n{'=' * 70}")
        print(f"VALIDATING US TICKERS ({len(us_tickers)} total)")
        print(f"{'=' * 70}")

        valid_us = validator.filter_ticker_list(us_tickers, verbose=True)
        validated_data["US"] = valid_us

        print(f"\nUS tickers: {len(us_tickers)} -> {len(valid_us)} (removed {len(us_tickers) - len(valid_us)})")

    # Process S&P 500
    if "SP500" in data and data["SP500"]:
        sp500_tickers = data["SP500"][:200] if TEST_MODE else data["SP500"]
        print(f"\n{'=' * 70}")
        print(f"VALIDATING S&P 500 TICKERS ({len(sp500_tickers)} total)")
        print(f"{'=' * 70}")

        valid_sp500 = validator.filter_ticker_list(sp500_tickers, verbose=True)
        validated_data["SP500"] = valid_sp500

        print(
            f"\nS&P 500: {len(sp500_tickers)} -> {len(valid_sp500)} (removed {len(sp500_tickers) - len(valid_sp500)})"
        )

    # Process Swedish stocks
    if "Sweden" in data and data["Sweden"]:
        sweden_tickers = data["Sweden"][:200] if TEST_MODE else data["Sweden"]
        print(f"\n{'=' * 70}")
        print(f"VALIDATING SWEDISH TICKERS ({len(sweden_tickers)} total)")
        print(f"{'=' * 70}")

        valid_sweden = validator.filter_ticker_list(sweden_tickers, verbose=True)
        validated_data["Sweden"] = valid_sweden

        print(
            f"\nSwedish stocks: {len(sweden_tickers)} -> {len(valid_sweden)} "
            f"(removed {len(sweden_tickers) - len(valid_sweden)})"
        )

    # Save validated data
    print(f"\n{'=' * 70}")
    print("SAVING VALIDATED DATA")
    print(f"{'=' * 70}")

    save_ticker_json(validated_data, OUTPUT_JSON)
    print(f"Saved to: {OUTPUT_JSON}")

    # Final summary
    original_total = sum(len(v) if isinstance(v, list) else 0 for v in data.values())
    validated_total = sum(len(v) if isinstance(v, list) else 0 for v in validated_data.values())

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"Filtered input: {original_total}")
    print(f"Validated output: {validated_total}")
    print(
        f"Removed (delisted): {original_total - validated_total} "
        f"({(original_total - validated_total) / original_total:.1%})"
    )

    print("\nNext step: Run build_training_set.py")


if __name__ == "__main__":
    main()
