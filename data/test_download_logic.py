"""
test download and quality check logic
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def check_data_quality(ticker, df, current_date):
    """
    Determine if stock is valid, delisted, or should be rejected
    """
    if df is None or df.empty or len(df) < 100:
        return {'status': 'rejected', 'reason': 'insufficient_data'}

    # Basic stats
    last_date = df.index[-1]
    first_date = df.index[0]
    days_of_data = len(df)

    # Handle Volume column (might be uppercase or lowercase)
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
    close_col = 'Close' if 'Close' in df.columns else 'close'

    avg_volume = df[volume_col].mean()
    avg_price = df[close_col].mean()
    last_price = df[close_col].iloc[-1]

    days_since_last = (pd.to_datetime(current_date) - last_date).days

    stats = {
        'last_date': last_date,
        'first_date': first_date,
        'days_of_data': days_of_data,
        'avg_volume': avg_volume,
        'avg_price': avg_price,
        'last_price': last_price,
        'days_since_last': days_since_last,
    }

    # === REJECTION CRITERIA ===
    # Penny stock
    if avg_price < 1.0:
        return {'status': 'rejected', 'reason': 'penny_stock', 'stats': stats}

    # Illiquid
    if avg_volume < 50_000:
        return {'status': 'rejected', 'reason': 'illiquid', 'stats': stats}

    # Too little data
    if days_of_data < 252:  # Less than 1 year
        return {'status': 'rejected', 'reason': 'insufficient_history', 'stats': stats}

    # === DELISTED DETECTION (KEEP THESE!) ===
    # Stopped trading >90 days ago
    if days_since_last > 90:
        return {
            'status': 'delisted',
            'reason': f'stopped_trading_{days_since_last}_days_ago',
            'stats': stats
        }

    # Price collapsed (bankruptcy signal)
    if last_price < 1.0 and avg_price > 5.0:
        return {
            'status': 'delisted',
            'reason': 'price_collapsed_to_penny_stock',
            'stats': stats
        }

    # === VALID ===
    return {'status': 'valid', 'stats': stats}


print("testing download and quality check logic...")

# Test with a few known good tickers
test_tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2015-01-01'
end_date = '2024-12-31'

# Download
print(f"\ndownloading {test_tickers}...")
data = yf.download(test_tickers, start=start_date, end=end_date, group_by='ticker', threads=True, progress=False)

# Test with end_date as current_date (historical data)
current_date_historical = end_date
print(f"\nusing current_date = {current_date_historical} (end_date)")

for ticker in test_tickers:
    ticker_data = data[ticker]
    quality = check_data_quality(ticker, ticker_data, current_date_historical)

    print(f"\n{ticker}:")
    print(f"  status: {quality['status']}")
    if 'stats' in quality:
        stats = quality['stats']
        print(f"  last_date: {stats['last_date']}")
        print(f"  days_since_last: {stats['days_since_last']}")
        print(f"  avg_price: ${stats['avg_price']:.2f}")
        print(f"  avg_volume: {stats['avg_volume']:,.0f}")

# Test with today as current_date (would incorrectly mark as delisted)
current_date_today = datetime.now().strftime('%Y-%m-%d')
print(f"\n\n--- WRONG APPROACH (for comparison) ---")
print(f"using current_date = {current_date_today} (today)")

for ticker in test_tickers[:1]:  # Just test one
    ticker_data = data[ticker]
    quality = check_data_quality(ticker, ticker_data, current_date_today)

    print(f"\n{ticker}:")
    print(f"  status: {quality['status']}")
    if 'stats' in quality:
        stats = quality['stats']
        print(f"  days_since_last: {stats['days_since_last']}")
        print(f"  ^ THIS IS WHY IT WAS FAILING - days_since_last > 90")

print("\n✓ test complete")
