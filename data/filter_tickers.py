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
    Enhanced to catch preferred shares with $ signs and letter suffixes.
    """
    # Remove derivatives and special shares
    filtered = [
        t for t in tickers
        if not (
            '$' in t or           # Preferred shares with $ (ABR$D, ACR$C)
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

    # Filter preferred shares (P, PR, PRA, PRB, PRC, PRD, PRE, etc.)
    filtered = [
        t for t in filtered
        if not any(t.endswith(suffix) for suffix in
                   ['P', 'PR', 'PRA', 'PRB', 'PRC', 'PRD', 'PRE', 'PRF', 'PRG', 'PRH', 'PRI'])
    ]

    # Filter "family" pattern: if base ticker exists + multiple single-letter variants
    # Example: OXLC exists, and OXLCG, OXLCI, OXLCL exist → remove the variants (preferred)
    base_tickers = {t[:-1] for t in filtered if len(t) >= 4}
    potential_preferred = []
    for ticker in filtered:
        if len(ticker) >= 4 and ticker[:-1] in base_tickers:
            # Check if there are multiple variants with same base
            variants = [t for t in filtered if t.startswith(ticker[:-1]) and len(t) == len(ticker)]
            if len(variants) >= 3:  # 3+ variants = likely preferred series
                if ticker[-1] in 'DEFGHIJKLMNOZ':  # Common preferred suffixes
                    potential_preferred.append(ticker)

    filtered = [t for t in filtered if t not in potential_preferred]

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


def remove_duplicates_and_class_shares(us_tickers, sp500_tickers):
    """
    Remove duplicates and handle class shares.
    1. If ticker in SP500, remove from US list
    2. If both Class A and Class B exist, keep only one (usually B is more liquid)
    """
    # Remove duplicates (priority to SP500)
    us_clean = [t for t in us_tickers if t not in sp500_tickers]

    # Handle class shares
    # Find pairs where both A and B exist
    class_a_tickers = {t for t in us_clean if len(t) <= 5 and t.endswith('A')}
    class_b_tickers = {t for t in us_clean if len(t) <= 5 and t.endswith('B')}

    pairs_to_filter = []
    for a_ticker in class_a_tickers:
        base = a_ticker[:-1]
        b_ticker = base + 'B'
        if b_ticker in class_b_tickers:
            # Both exist - keep B (usually more liquid), remove A
            pairs_to_filter.append(a_ticker)

    us_clean = [t for t in us_clean if t not in pairs_to_filter]

    return us_clean, len(pairs_to_filter)


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

    # Process S&P 500 first (priority list)
    sp500_tickers = data.get('SP500', [])
    if sp500_tickers:
        filtered_data['SP500'] = sp500_tickers
        print(f"\nS&P 500: {len(sp500_tickers)} (kept all, priority list)")

    # Filter US tickers
    us_tickers = data.get('US', [])
    if us_tickers:
        print(f"\nUS tickers:")
        print(f"  Raw: {len(us_tickers)}")

        # Step 1: Filter junk
        us_filtered = filter_junk_tickers(us_tickers)
        removed_junk = len(us_tickers) - len(us_filtered)
        print(f"  After junk filter: {len(us_filtered)} (removed {removed_junk} junk)")

        # Step 2: Remove duplicates and class shares
        us_clean, class_pairs_removed = remove_duplicates_and_class_shares(us_filtered, sp500_tickers)
        removed_dups = len(us_filtered) - len(us_clean)
        print(f"  After dedup: {len(us_clean)} (removed {removed_dups} duplicates)")
        print(f"  Class share pairs handled: {class_pairs_removed} (kept B, removed A)")

        filtered_data['US'] = us_clean

        total_removed = len(us_tickers) - len(us_clean)
        print(f"  Total removed: {total_removed} ({total_removed / len(us_tickers):.1%})")

    # Swedish (keep all, already clean)
    sweden_tickers = data.get('Sweden', [])
    if sweden_tickers:
        filtered_data['Sweden'] = sweden_tickers
        print(f"\nSwedish: {len(sweden_tickers)} (kept all, clean list)")

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
    print(f"\nRemoved categories:")
    print(f"  - Preferred shares ($, P, PR, letter series)")
    print(f"  - Warrants, units, rights")
    print(f"  - ETFs")
    print(f"  - Duplicates (US tickers already in SP500)")
    print(f"  - Class share redundancy (kept B, removed A)")
    print(f"\nSaved: {OUTPUT_JSON}")
    print("\nNext step: uv run python validate_tickers.py")


if __name__ == "__main__":
    main()
