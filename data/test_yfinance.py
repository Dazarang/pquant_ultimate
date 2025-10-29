"""
quick test of yfinance download
"""

import yfinance as yf
import pandas as pd

print("testing yfinance download...")

# test single ticker
print("\n1. testing single ticker (AAPL):")
data = yf.download('AAPL', start='2024-01-01', end='2024-12-31', progress=False)
print(f"   shape: {data.shape}")
print(f"   columns: {list(data.columns)}")
print(f"   first date: {data.index[0]}")
print(f"   last date: {data.index[-1]}")

# test multiple tickers
print("\n2. testing multiple tickers (AAPL, MSFT, GOOGL):")
data = yf.download(['AAPL', 'MSFT', 'GOOGL'], start='2024-01-01', end='2024-12-31', group_by='ticker', threads=True, progress=False)
print(f"   type: {type(data)}")
print(f"   shape: {data.shape}")
print(f"   columns: {list(data.columns[:5])}")

# test accessing individual ticker
if len(['AAPL', 'MSFT', 'GOOGL']) > 1:
    print("\n3. accessing individual ticker data:")
    aapl_data = data['AAPL']
    print(f"   AAPL shape: {aapl_data.shape}")
    print(f"   AAPL columns: {list(aapl_data.columns)}")
    print(f"   AAPL close (last 5):")
    print(aapl_data['Close'].tail())

print("\n✓ test complete")
