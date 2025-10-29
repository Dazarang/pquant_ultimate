# build_training_set.py - critical fixes applied

## issues fixed

### 1. wrong input file (line 614)
**before:**
```python
TICKER_JSON_PATH = 'data/tickers_data/tickers_cleaned_20251023.json'
```

**after:**
```python
TICKER_JSON_PATH = '/Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_validated_20251029.json'
```

**why:** must use validated JSON from clean_ticker_json.py, not original cleaned version which contains dead tickers

---

### 2. wrong date range (lines 615-616)
**before:**
```python
START_DATE = '2024-01-01'
END_DATE = '2024-12-31'
```

**after:**
```python
START_DATE = '2015-01-01'  # 10 years of data
END_DATE = '2024-12-31'    # ~2,500 trading days
```

**why:** per data_philosophy.md, need 10 years to capture:
- different market regimes (bull, bear, sideways)
- multiple recession cycles
- various sector rotations
- tech boom/bust cycles

1 year insufficient for bottom detection patterns.

---

### 3. multiindex column access bug (lines 153-175)
**problem:** batch downloads create multiindex columns like `('Close', 'AAPL')`, but code checked `'Close' in df.columns` (returns False) then tried to access `df[close_col]` (works but fragile)

**fix:**
```python
# Handle column access (yfinance returns MultiIndex or simple columns)
try:
    volume_data = df['Volume']
    close_data = df['Close']
except KeyError:
    try:
        volume_data = df['volume']
        close_data = df['close']
    except KeyError:
        return {'status': 'rejected', 'reason': 'missing_price_columns'}

# Handle both Series and DataFrame (MultiIndex case)
if isinstance(volume_data, pd.DataFrame):
    avg_volume = volume_data.iloc[:, 0].mean()
    last_price = close_data.iloc[-1, 0]
    avg_price = close_data.iloc[:, 0].mean()
else:
    avg_volume = volume_data.mean()
    last_price = close_data.iloc[-1]
    avg_price = close_data.mean()
```

---

### 4. fast_info access bug (lines 361-372)
**problem:** fast_info is object with attributes, not dict. cannot use .get()

**before:**
```python
fast_info = ticker_obj.fast_info
market_cap = fast_info.get('market_cap', 0) or fast_info.get('marketCap', 0)
```

**after:**
```python
fast_info = ticker_obj.fast_info
market_cap = getattr(fast_info, 'market_cap', None) or getattr(fast_info, 'marketCap', None)
if market_cap is None:
    market_cap = 0
```

---

### 5. missing json keys (lines 29, 73, 80)
**problem:** code assumed 'US' and 'SP500' keys exist, would crash with KeyError

**fix:**
```python
us_tickers = data.get('US', [])
sp500 = data.get('SP500', [])
sweden = data.get('Sweden', [])
```

added safe access with defaults + warning messages

---

### 6. unused imports (lines 6-11)
**removed:**
- numpy as np
- ThreadPoolExecutor
- datetime, timedelta
- Counter

**why:** never used, clutters code

---

## swedish market cap handling

### problem
swedish large-cap stocks ($5-10B) would be classified as "mid-cap" using US thresholds ($10B+), skewing stratification

### solution
separate stratification by country with adjusted thresholds:

**us stocks:**
- small: <$2B
- mid: $2-10B
- large: $10-200B
- mega: $200B+

**swedish stocks:**
- small: <$500M
- mid: $500M-2B
- large: $2-10B
- mega: $10B+

**implementation (lines 447-483):**
```python
# Separate Swedish stocks (different market cap scale)
is_swedish = df['country'] == 'Sweden'
df_us = df[~is_swedish]
df_swedish = df[is_swedish]

# Apply different market cap buckets
if len(df_us) > 0:
    df_us['cap_bucket'] = pd.cut(
        df_us['market_cap'],
        bins=[0, 2e9, 10e9, 200e9, 1e15],
        labels=['small', 'mid', 'large', 'mega']
    )

if len(df_swedish) > 0:
    # Lower thresholds for Swedish market
    df_swedish['cap_bucket'] = pd.cut(
        df_swedish['market_cap'],
        bins=[0, 500e6, 2e9, 10e9, 1e15],
        labels=['small', 'mid', 'large', 'mega']
    )

# Recombine
df = pd.concat([df_us, df_swedish], ignore_index=True)
```

**result:** swedish large-caps correctly classified as "large" in their market context, not misclassified as "mid" in us context

---

## data frequency

**answer: daily**

yfinance download without interval parameter defaults to daily data:
```python
data = yf.download(
    tickers,
    start='2015-01-01',
    end='2024-12-31',
    # no interval parameter = daily (default)
)
```

**what this means:**
- ~250 trading days per year
- 10 years = ~2,500 rows per stock
- 1,500 stocks = ~3.75 million total data points

**why daily (not intraday):**
- bottom detection is multi-day/week process, not intraday
- daily captures price action + volume patterns
- reduces noise vs 5-min/1-hour data
- easier to handle technically (smaller dataset)

**why daily (not weekly/monthly):**
- need granularity to detect reversal patterns
- weekly/monthly too coarse for precise entry points
- daily gives ~2,500 samples per stock vs ~520 (weekly) or ~120 (monthly)

---

## verification checklist

before running build_training_set.py, verify:

1. validated json exists:
```bash
ls -lh /Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_validated_20251029.json
```

2. validated json has tickers:
```bash
python -c "import json; data=json.load(open('/Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_validated_20251029.json')); print(f'US: {len(data.get(\"US\", []))}, SP500: {len(data.get(\"SP500\", []))}, Sweden: {len(data.get(\"Sweden\", []))}')"
```

3. run build_training_set.py:
```bash
cd /Users/deaz/Developer/project_quant/pQuant_ultimate/data
uv run python build_training_set.py
```

**expected runtime:** 45-90 minutes
- step 3 (download): 30-60 min (depends on rate limits)
- step 4 (metadata): 10-20 min
- other steps: <5 min

**expected output:**
- training_data.pkl (~500MB)
- training_stocks_data.parquet (~300MB)
- ~1,500 stocks
- ~3.75M rows
- date range: 2015-01-01 to 2024-12-31

---

## summary of improvements

1. uses validated tickers (no dead stocks)
2. 10 years data (proper pattern recognition)
3. handles multiindex columns correctly
4. fast_info accessed properly
5. safe json key access
6. clean imports
7. country-aware stratification (swedish market cap)
8. daily data frequency documented
9. comprehensive configuration output

all critical bugs fixed. ready to run.
