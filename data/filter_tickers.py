"""
Filter tickers BEFORE validation.
Remove ETFs, warrants, preferred shares, units, derivatives.
Pure filtering - no API calls, instant execution.
"""

import json
from datetime import datetime


def filter_junk_tickers(tickers):
    """
    Remove derivatives and junk. No API calls.
    """
    # Remove derivatives
    filtered = [
        t for t in tickers
        if not (
            t.endswith('W') or    # Warrants
            t.endswith('U') or    # Units
            t.endswith('R') or    # Rights
            '.W' in t or          # Warrants (alt format)
            '.U' in t or          # Units (alt format)
            '+' in t or           # Special shares
            '=' in t or           # When-issued
            '^' in t or           # Preferred (some formats)
            len(t) > 6 or         # Likely derivatives
            t.endswith('WW')      # Double warrants
        )
    ]

    # Additional filters for preferred shares
    filtered = [
        t for t in filtered
        if not any(t.endswith(suffix) for suffix in ['P', 'PR', 'PRA', 'PRB', 'PRC'])
    ]

    # Filter common ETF patterns
    etf_patterns = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VT', 'VO', 'AGG', 'BND',
        'XL', 'IWR', 'IWV', 'VEA', 'VWO', 'GLD', 'SLV', 'USO',
        'EFA', 'VNQ', 'TLT', 'HYG', 'LQD', 'IEMG', 'IEFA',
    ]
    filtered = [
        t for t in filtered
        if not any(t.startswith(pattern) for pattern in etf_patterns)
    ]

    return filtered


def main():
    """
    Filter tickers from raw JSON.
    Output: filtered JSON ready for validation.
    """
    # Configuration
    INPUT_JSON = '/Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_20251029.json'
    OUTPUT_JSON = f'/Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_filtered_{datetime.now().strftime("%Y%m%d")}.json'

    print("=" * 70)
    print("TICKER FILTERING (NO API CALLS)")
    print("=" * 70)
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_JSON}")

    # Load raw data
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    filtered_data = {}

    # Filter US tickers
    if 'US' in data:
        us_tickers = data['US']
        us_filtered = filter_junk_tickers(us_tickers)
        filtered_data['US'] = us_filtered

        print(f"\nUS tickers:")
        print(f"  Raw: {len(us_tickers)}")
        print(f"  Filtered: {len(us_filtered)}")
        print(f"  Removed: {len(us_tickers) - len(us_filtered)} "
              f"({(len(us_tickers) - len(us_filtered)) / len(us_tickers):.1%})")

    # S&P 500 (keep all, usually clean)
    if 'SP500' in data:
        filtered_data['SP500'] = data['SP500']
        print(f"\nS&P 500: {len(data['SP500'])} (kept all)")

    # Swedish (keep all, already clean)
    if 'Sweden' in data:
        filtered_data['Sweden'] = data['Sweden']
        print(f"Swedish: {len(data['Sweden'])} (kept all)")

    # Save filtered data
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    # Summary
    original_total = sum(len(v) if isinstance(v, list) else 0 for v in data.values())
    filtered_total = sum(len(v) if isinstance(v, list) else 0 for v in filtered_data.values())

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Original: {original_total}")
    print(f"Filtered: {filtered_total}")
    print(f"Removed: {original_total - filtered_total} "
          f"({(original_total - filtered_total) / original_total:.1%})")
    print(f"\nSaved: {OUTPUT_JSON}")
    print("\nNext step: Run validate_tickers.py on filtered list")


if __name__ == "__main__":
    main()
