"""
Script to validate and clean ticker JSON file.
Removes delisted and invalid tickers.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from validators import TickerValidator


def load_ticker_json(json_path: str) -> dict:
    """Load ticker JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_ticker_json(data: dict, output_path: str) -> None:
    """Save cleaned ticker JSON."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Main execution."""
    # Configuration
    TEST_MODE = False  # Set to False for full validation
    TEST_LIMIT = 200  # Tickers per category in test mode

    INPUT_JSON = '/Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_20251029.json'
    OUTPUT_JSON = f'/Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_validated_{datetime.now().strftime("%Y%m%d")}.json'

    print("=" * 70)
    print("TICKER VALIDATION AND CLEANING")
    if TEST_MODE:
        print(f"*** TEST MODE: Validating {TEST_LIMIT} tickers per category ***")
    print("=" * 70)
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_JSON}")

    # Load existing data
    print("\nLoading ticker data...")
    data = load_ticker_json(INPUT_JSON)

    # Create validator
    validator = TickerValidator(
        validation_days=3,
        rate_limit_delay=0.15  # 150ms between requests
    )

    # Process each category
    cleaned_data = {}

    # Process US tickers
    if 'US' in data:
        us_tickers = data['US'][:TEST_LIMIT] if TEST_MODE else data['US']
        print(f"\n{'=' * 70}")
        print(f"VALIDATING US TICKERS ({len(us_tickers)} total)")
        print(f"{'=' * 70}")

        valid_us = validator.filter_ticker_list(us_tickers, verbose=True)
        cleaned_data['US'] = valid_us

        print(f"\nUS tickers: {len(us_tickers)} -> {len(valid_us)} "
              f"(removed {len(us_tickers) - len(valid_us)})")

    # Process S&P 500 (usually clean, but validate anyway)
    if 'SP500' in data:
        sp500_tickers = data['SP500'][:TEST_LIMIT] if TEST_MODE else data['SP500']
        print(f"\n{'=' * 70}")
        print(f"VALIDATING S&P 500 TICKERS ({len(sp500_tickers)} total)")
        print(f"{'=' * 70}")

        valid_sp500 = validator.filter_ticker_list(sp500_tickers, verbose=True)
        cleaned_data['SP500'] = valid_sp500

        print(f"\nS&P 500: {len(sp500_tickers)} -> {len(valid_sp500)} "
              f"(removed {len(sp500_tickers) - len(valid_sp500)})")

    # Process Swedish stocks
    if 'Sweden' in data:
        sweden_tickers = data['Sweden'][:TEST_LIMIT] if TEST_MODE else data['Sweden']
        print(f"\n{'=' * 70}")
        print(f"VALIDATING SWEDISH TICKERS ({len(sweden_tickers)} total)")
        print(f"{'=' * 70}")

        valid_sweden = validator.filter_ticker_list(sweden_tickers, verbose=True)
        cleaned_data['Sweden'] = valid_sweden

        print(f"\nSwedish stocks: {len(sweden_tickers)} -> {len(valid_sweden)} "
              f"(removed {len(sweden_tickers) - len(valid_sweden)})")

    # Save cleaned data
    print(f"\n{'=' * 70}")
    print("SAVING CLEANED DATA")
    print(f"{'=' * 70}")

    save_ticker_json(cleaned_data, OUTPUT_JSON)
    print(f"Saved to: {OUTPUT_JSON}")

    # Final summary
    original_total = sum(len(v) if isinstance(v, list) else 0 for v in data.values())
    cleaned_total = sum(len(v) if isinstance(v, list) else 0 for v in cleaned_data.values())

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"Original total: {original_total}")
    print(f"Cleaned total: {cleaned_total}")
    print(f"Removed: {original_total - cleaned_total} "
          f"({(original_total - cleaned_total) / original_total:.1%})")

    # Save removal report
    report_path = OUTPUT_JSON.replace('.json', '_report.txt')
    with open(report_path, 'w') as f:
        f.write("TICKER VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input: {INPUT_JSON}\n")
        f.write(f"Output: {OUTPUT_JSON}\n\n")
        f.write(f"Original total: {original_total}\n")
        f.write(f"Cleaned total: {cleaned_total}\n")
        f.write(f"Removed: {original_total - cleaned_total}\n\n")

        for category in ['US', 'SP500', 'Sweden']:
            if category in data and category in cleaned_data:
                original = len(data[category])
                cleaned = len(cleaned_data[category])
                f.write(f"\n{category}:\n")
                f.write(f"  Original: {original}\n")
                f.write(f"  Cleaned: {cleaned}\n")
                f.write(f"  Removed: {original - cleaned}\n")

    print(f"\nReport saved to: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
