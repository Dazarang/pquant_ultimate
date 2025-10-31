# Stock Bottom Prediction Model: Complete Development Philosophy

## GUIDING PRINCIPLES

**Core Truth:** Most ML projects fail not from bad algorithms, but from data leakage, poor evaluation, and overfitting. Our philosophy prioritizes **correctness over complexity**.

**Important Note:** What is presented here is MERELY A GUIDELINE and should be treated as such.

---

# PHASE 1: DATA PREPARATION

## 1.1 Stock Universe ✅ (Already Done)

**What you did:** 13,000 tickers → 1,500 stocks via filtering

**CRITICAL VERIFICATION CHECKLIST:**

```
□ Do your 1,500 stocks include DELISTED/BANKRUPT companies?
  → If NO: You have survivorship bias. Must fix.
  → If YES: What percentage? Should be 10-20%.

□ Did you filter based on CURRENT liquidity/market cap?
  → If YES: Wrong. A stock delisted in 2020 had low liquidity in 2021 (doesn't exist).
         Use point-in-time filtering: "Was liquid DURING its existence"
  → If NO: Good.

□ Are sectors balanced?
  → If >30% of stocks in one sector (e.g., Tech): Rebalance.

□ Market cap distribution?
  → Should have: 15% mega, 25% large, 30% mid, 30% small
  → If all large-cap: Model won't generalize to small-caps.
```

**ACTION ITEMS:**

1. **Check for survivorship bias:**
```python
def audit_survivorship_bias(stock_list):
    """
    CRITICAL: Verify no survivorship bias
    """
    currently_trading = [s for s in stock_list if is_currently_trading(s)]
    delisted = [s for s in stock_list if not is_currently_trading(s)]
    
    delisted_pct = len(delisted) / len(stock_list)
    
    print(f"Currently trading: {len(currently_trading)}")
    print(f"Delisted/bankrupt: {len(delisted)}")
    print(f"Delisted percentage: {delisted_pct:.1%}")
    
    if delisted_pct < 0.10:
        print("⚠️  WARNING: Survivorship bias detected!")
        print("   Need at least 10-20% failed companies")
        return False
    else:
        print("✓ Survivorship bias check passed")
        return True
```

2. **If you lack delisted stocks, manually add them:**
```python
# Add known failures from 2015-2024
failures_to_add = [
    # Bankruptcies
    'HTGM',  # HTG Molecular (bankrupt 2023)
    'GNUS',  # Genius Brands (delisted 2023)
    'EXPR',  # Express Inc (bankrupt 2023)
    # ... add 150-250 more
    
    # Major delistings
    # Check: https://www.nasdaq.com/market-activity/stocks/delisted
]
```

---

## 1.2 Data Quality & Cleaning

### ✅ DO:

**1. Point-in-time data integrity:**
```python
def verify_data_integrity(df):
    """
    Each row must reflect information AVAILABLE at that date
    Never use future information
    """
    
    # Check 1: No forward-looking data
    # (Already handled by your data provider if using adjusted close)
    
    # Check 2: Corporate actions properly adjusted
    splits = detect_stock_splits(df)
    if len(splits) > 0:
        print(f"Found {len(splits)} splits - verify adjusted prices used")
    
    # Check 3: Dividends adjusted
    # (Most providers do this automatically)
    
    # Check 4: No impossible values
    assert (df['close'] > 0).all(), "Negative/zero prices detected"
    assert (df['volume'] >= 0).all(), "Negative volume detected"
    assert (df['high'] >= df['low']).all(), "High < Low detected"
    assert (df['high'] >= df['close']).all(), "Close > High detected"
    assert (df['low'] <= df['close']).all(), "Close < Low detected"
    
    return True
```

**2. Handle missing data correctly:**
```python
def handle_missing_data(df):
    """
    Missing data philosophy: Conservative
    """
    
    # Rule 1: If >5% of a stock's data is missing, DROP the stock
    missing_pct = df.groupby('stock_id').apply(
        lambda x: x.isnull().sum().sum() / (len(x) * len(x.columns))
    )
    stocks_to_drop = missing_pct[missing_pct > 0.05].index
    print(f"Dropping {len(stocks_to_drop)} stocks with >5% missing data")
    df = df[~df['stock_id'].isin(stocks_to_drop)]
    
    # Rule 2: For small gaps (<5 consecutive days), forward fill
    df = df.groupby('stock_id').apply(
        lambda x: x.fillna(method='ffill', limit=5)
    )
    
    # Rule 3: Remaining NaNs → drop those rows
    df = df.dropna()
    
    return df
```

**3. Detect and fix data errors:**
```python
def detect_data_errors(df):
    """
    Common data quality issues
    """
    grouped = df.groupby('stock_id')
    
    # Error 1: Unadjusted stock splits (overnight gap >50%)
    df['overnight_return'] = grouped['open'].transform(
        lambda x: x / x.shift(1)['close'] - 1
    )
    
    suspicious = df[abs(df['overnight_return']) > 0.5]
    if len(suspicious) > 0:
        print(f"⚠️  Found {len(suspicious)} suspicious overnight gaps >50%")
        print("   Likely unadjusted splits - verify with data provider")
    
    # Error 2: Days with zero volume (illiquid or data error)
    zero_vol = df[df['volume'] == 0]
    if len(zero_vol) > 0:
        print(f"⚠️  Found {len(zero_vol)} days with zero volume")
        # Drop these days
        df = df[df['volume'] > 0]
    
    # Error 3: Extreme daily returns (>75% in one day)
    df['daily_return'] = grouped['close'].pct_change()
    extreme = df[abs(df['daily_return']) > 0.75]
    if len(extreme) > 0:
        print(f"⚠️  Found {len(extreme)} extreme returns >75%")
        print("   Manual review required")
    
    return df
```

### ❌ DO NOT:

1. **DON'T drop NaNs globally:**
```python
# ❌ WRONG - drops entire stock if any NaN
df = df.dropna()

# ✅ RIGHT - handle per stock, per situation
df = handle_missing_data(df)
```

2. **DON'T use data that didn't exist at that time:**
```python
# ❌ WRONG - using future sector classification
df['sector'] = current_sector_map[df['stock_id']]  # Sector in 2024

# ✅ RIGHT - use sector at time of observation
df['sector'] = historical_sector_map[df['date']][df['stock_id']]  # Sector at date
```

3. **DON'T trust data blindly:**
- Always plot random stock samples
- Check for obvious errors
- Verify adjusted prices used

---

# PHASE 2: LABELING AND DATASET STRUCTURE

## 2.1 Labeling Philosophy - CRITICAL UNDERSTANDING

### ✅ The Pivot Labeling Approach is CORRECT

**Your pivot labeling does NOT create lookahead bias. Here's why:**

```python
# Your labeling process:
pivot_low = df['low'].rolling(window=17, center=True).apply(is_pivot_low)
df['label'] = pivot_low  # NO SHIFT NEEDED

# This is equivalent to:
# 1. Print out entire price chart on paper
# 2. Manually circle all the local bottoms
# 3. Use those circles as labels

# The KEY insight:
# - Labels CAN use future data (you're annotating history)
# - Features CANNOT use future data (must be real-time calculable)
```

### Why This is Valid

**Think of it like image classification:**
```
Dog/Cat classifier:
1. Human looks at 1000 images, labels them "dog" or "cat"
   → Human used their eyes (any method) to create labels
   
2. Model trains on pixel features to predict those labels
   → Model only sees pixels, not how human labeled

Stock bottoms:
1. Pivot algorithm looks at 17-day window, labels bottoms
   → Algorithm used future data (days +8) to create labels
   
2. Model trains on RSI/volume/etc to predict those labels
   → Model only sees RSI/volume/etc, not how pivot labeled

✓ THIS IS FINE
```

### The Real Lookahead Check

```python
def verify_no_leakage(df):
    """
    What ACTUALLY matters: Features must be backward-looking
    Labels can be created however (including using future data)
    """
    
    # ✓ FEATURES - These must ALL be backward-looking
    features_ok = {
        'rsi_14': True,  # Uses past 14 days ✓
        'volume_z': True,  # Uses past 60 days ✓
        'ret_1d': True,  # Yesterday's return ✓
        'ma_20': True,  # Past 20 days ✓
        
        # ❌ These would be leakage:
        'future_return': False,  # Uses tomorrow's price ❌
        'will_recover': False,  # Uses next week's data ❌
    }
    
    # ✓ LABELS - Can use any method
    labels_ok = {
        'pivot_low_center_window': True,  # Uses ±8 days ✓
        'manual_annotation': True,  # Human marked it ✓
        'lowest_in_month': True,  # Used full month ✓
    }
    
    print("Label creation method doesn't matter")
    print("Feature calculation is what we must verify")
```

**Key Takeaway:**
```python
# Previously incorrect advice to shift labels:
df['label'] = pivot_low.shift(8)  # ❌ UNNECESSARY

# Flawed reasoning was:
# "At day 100, label uses days 92-108, so model sees future"

# CORRECT reasoning:
# "At day 100, features use days ≤100, label is just annotation
#  Model learns: when RSI/volume look like X, it's often a bottom
#  How the bottom was identified (pivot, human, oracle) doesn't matter"
```

---

## 2.2 Raw Data Format (After Loading Tickers)

**What you should have after loading from API/CSV:**

```python
# Example: df_raw (before any processing)

      date       stock_id   open    high     low   close     volume
0   2015-01-02   AAPL      111.39  111.44  107.35  109.33   53,204,626
1   2015-01-05   AAPL      108.29  108.65  105.41  106.25   64,285,491
2   2015-01-06   AAPL      106.54  107.43  104.63  106.26   65,797,116
...
500 2015-01-02   MSFT       46.66   47.21   46.50   46.76   27,913,852
501 2015-01-05   MSFT       46.37   46.73   45.54   45.66   39,673,865
...
1000 2015-01-02  TSLA      219.31  223.00  217.25  222.41    2,968,400
...

# Key properties:
# - All stocks mixed together (not grouped)
# - Sorted by date, then stock_id
# - Standard OHLCV columns
# - Total rows: ~1,500 stocks × 2,500 days = 3,750,000 rows
```

**Verification:**
```python
def verify_raw_data_structure(df):
    """
    Check raw data is properly formatted
    """
    # 1. Required columns
    required_cols = ['date', 'stock_id', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_cols), "Missing required columns"
    
    # 2. Date format
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. No duplicates
    duplicates = df.duplicated(subset=['date', 'stock_id'])
    assert duplicates.sum() == 0, f"Found {duplicates.sum()} duplicate rows"
    
    # 4. Sorted by date (optional but recommended)
    # Don't enforce sorting by stock_id within date
    
    # 5. Data types
    assert df['close'].dtype in [float, np.float64], "Price data should be float"
    assert df['volume'].dtype in [int, np.int64, float, np.float64], "Volume should be numeric"
    
    print("✓ Raw data structure verified")
    
    # Summary stats
    print(f"\nDataset summary:")
    print(f"  Stocks: {df['stock_id'].nunique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Avg days per stock: {len(df) / df['stock_id'].nunique():.0f}")
    
    return True
```

---

## 2.3 Data Organization: INTERLEAVED BY DATE

### ✅ CORRECT Organization (Interleaved)

```python
def organize_data_for_training(df):
    """
    CORRECT: Interleave stocks by date
    Model sees diverse stocks in each batch
    """
    # Sort by date first, then stock_id (optional secondary sort)
    df = df.sort_values(['date', 'stock_id']).reset_index(drop=True)
    
    # Result: All stocks for Jan 2 together, then all for Jan 3, etc.
    return df

# What this looks like:
"""
Row 0:    2015-01-02, AAPL, ...
Row 1:    2015-01-02, GOOGL, ...
Row 2:    2015-01-02, MSFT, ...
Row 3:    2015-01-02, TSLA, ...
Row 4:    2015-01-02, ... [all other stocks for this date]
Row 1500: 2015-01-05, AAPL, ...  (next trading day)
Row 1501: 2015-01-05, GOOGL, ...
...
"""
```

**Why interleaved is better:**
```python
# During training with batch_size=1024:

# Interleaved approach:
Batch 1: [100 different stocks from Jan 2-3, 2015]
Batch 2: [100 different stocks from Jan 4-5, 2015]
→ Model learns general "bottom patterns" across diverse stocks

# Grouped approach (WRONG):
Batch 1: [1024 days of AAPL only]
Batch 2: [1024 days of AAPL only]
Batch 3: [500 days AAPL, 524 days MSFT]
→ Model overfits to AAPL quirks, then has to unlearn for MSFT
```

### ❌ WRONG Organization (Grouped by Stock)

```python
# DON'T do this:
df = df.sort_values(['stock_id', 'date'])  # ❌ Groups all AAPL, then all MSFT

# Result:
"""
Row 0:    AAPL, 2015-01-02, ...
Row 1:    AAPL, 2015-01-05, ...
...
Row 2500: AAPL, 2024-12-31, ...
Row 2501: GOOGL, 2015-01-02, ...  (new stock starts)
...
"""
# This is BAD for gradient boosting
```

---

# PHASE 3: FEATURE ENGINEERING

## 3.1 The Golden Rule: NO FUTURE INFORMATION

**Every feature must answer: "Could I calculate this in real-time trading?"**

### ✅ DO:

**1. Use rolling windows (backward-looking only):**
```python
# ✅ CORRECT - only uses past data
df['rsi_14'] = df.groupby('stock_id')['close'].transform(
    lambda x: talib.RSI(x, timeperiod=14)
)

df['ma_20'] = df.groupby('stock_id')['close'].transform(
    lambda x: x.rolling(20).mean()  # Uses last 20 days
)

df['volume_z'] = df.groupby('stock_id')['volume'].transform(
    lambda x: (x - x.rolling(60).mean()) / x.rolling(60).std()
)
```

**2. Per-stock normalization (avoid cross-stock contamination):**
```python
# ✅ CORRECT - each stock normalized using its own history
def normalize_per_stock(df, column, window=252):
    """
    Each stock normalized independently using rolling window
    """
    grouped = df.groupby('stock_id')
    
    df[f'{column}_norm'] = grouped[column].transform(
        lambda x: (x - x.rolling(window, min_periods=20).mean()) / 
                  x.rolling(window, min_periods=20).std()
    )
    
    return df
```

**3. Only use "shift" to create lags, never negative shifts:**
```python
# ✅ CORRECT - using past values
df['close_lag1'] = df.groupby('stock_id')['close'].shift(1)  # Yesterday
df['close_lag5'] = df.groupby('stock_id')['close'].shift(5)  # 5 days ago

# ❌ WRONG - looking into future
df['close_future'] = df.groupby('stock_id')['close'].shift(-1)  # Tomorrow
```

### ❌ DO NOT:

**1. NEVER normalize across entire dataset:**
```python
# ❌ WRONG - uses global statistics including future
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['price_scaled'] = scaler.fit_transform(df[['close']])
# This uses mean/std of ALL data (train + test)

# ✅ RIGHT - fit on train only, transform separately
scaler = StandardScaler()
train['price_scaled'] = scaler.fit_transform(train[['close']])
test['price_scaled'] = scaler.transform(test[['close']])  # Use train stats
```

**2. NEVER use `.min()`, `.max()`, `.mean()` without `.rolling()`:**
```python
# ❌ WRONG - sees entire series including future
df['price_normalized'] = df['close'] / df['close'].max()

# ✅ RIGHT - only uses past
df['price_normalized'] = df.groupby('stock_id')['close'].transform(
    lambda x: x / x.rolling(252, min_periods=50).max()
)
```

**3. NEVER create features that "knew the answer":**
```python
# ❌ WRONG - this is literally the target!
df['will_go_up'] = (df['close'].shift(-20) > df['close']).astype(int)

# Even subtle versions are wrong:
# ❌ WRONG - uses future returns
df['momentum_future'] = df['close'].pct_change(-10)  # 10-day forward return
```

---

## 3.2 Feature Categories

### ✅ DO: Include these feature types

**1. Price-based (returns, not absolute prices):**
```python
# Returns at multiple timeframes
df['ret_1d'] = df.groupby('stock_id')['close'].pct_change(1)
df['ret_5d'] = df.groupby('stock_id')['close'].pct_change(5)
df['ret_10d'] = df.groupby('stock_id')['close'].pct_change(10)
df['ret_20d'] = df.groupby('stock_id')['close'].pct_change(20)

# Deviation from moving averages
df['price_to_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
df['price_to_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
```

**2. Volume features:**
```python
# Volume ratios (scale-invariant)
df['volume_ratio'] = df['volume'] / df['volume_ma20']

# Volume z-score (per stock)
df['volume_z'] = df.groupby('stock_id')['volume_ratio'].transform(
    lambda x: (x - x.rolling(60).mean()) / x.rolling(60).std()
)
```

**3. Technical indicators:**
```python
# RSI, MACD, Bollinger Bands, etc.
# These are already in your indicator functions
```

**4. Pattern features (divergences, exhaustion):**
```python
# Your clever features from earlier
# Multi-indicator divergence, panic selling, etc.
```

**5. Contextual features:**
```python
# Drawdown from highs
df['drawdown'] = (df['close'] - df['high_252']) / df['high_252']

# Days since last bottom
df['days_since_pivot'] = calculate_days_since_pivot()

# Time features
df['day_of_week'] = df['date'].dt.dayofweek
```

### ❌ DO NOT:

**1. Don't use raw prices as features:**
```python
# ❌ WRONG
features = ['close', 'open', 'high', 'low']
# $5 stock vs $500 stock - incomparable

# ✅ RIGHT
features = ['ret_1d', 'price_to_sma20', 'drawdown']
# All scale-invariant
```

**2. Don't use volume without normalization:**
```python
# ❌ WRONG
features = ['volume']
# Apple: 50M shares vs penny stock: 10K shares - incomparable

# ✅ RIGHT
features = ['volume_ratio', 'volume_z']
# Both measure "how unusual is today's volume FOR THIS STOCK"
```

**3. Don't create redundant features:**
```python
# ❌ WRONG - all measure same thing
features = ['sma_20', 'sma_21', 'sma_22', 'sma_23']

# ✅ RIGHT - different timeframes
features = ['sma_20', 'sma_50', 'sma_200']
```

---

## 3.3 Feature Engineering Output

**After applying all indicators and features:**

```python
# Example: df_features (after feature engineering)

      date       stock_id   close  volume    ret_1d  ret_5d  rsi_14  macd  volume_z  drawdown  ...  label
0   2015-01-20   AAPL      110.22  51.2M    -0.021   -0.045   42.3   -1.2    0.8      -0.15    ...    0
1   2015-01-21   AAPL      109.55  48.9M    -0.006   -0.051   39.1   -1.5    0.3      -0.16    ...    0
2   2015-01-22   AAPL      112.40  53.1M     0.026   -0.032   45.2   -0.9    1.1      -0.13    ...    0
...
500 2015-01-20   MSFT       45.12  28.4M    -0.015   -0.038   38.5   -0.8    0.5      -0.09    ...    0
501 2015-01-21   MSFT       44.88  31.2M    -0.005   -0.041   36.2   -1.0    0.9      -0.10    ...    0
...
1000 2015-02-15  TSLA      210.45   5.1M    -0.085   -0.142   22.1   -3.5    2.8      -0.28    ...    1  ← Bottom!
...

# Key properties:
# - Same row structure as raw (1 row = 1 stock-day)
# - Added: All your feature columns (~30-50 features)
# - Added: 'label' column (0 or 1)
# - First ~20 days per stock may have NaN (not enough history for rolling windows)
# - Still mixed stocks, sorted by date
```

**Critical: Handle NaN from rolling windows**
```python
def handle_feature_nans(df):
    """
    NaN values appear at start of each stock's history
    (not enough data for rolling windows)
    """
    grouped = df.groupby('stock_id')
    
    # Check how many NaNs per stock
    print("NaN distribution per stock:")
    nans_per_stock = grouped.apply(lambda x: x.isnull().sum().sum())
    print(f"  Mean NaNs per stock: {nans_per_stock.mean():.0f}")
    print(f"  Typically first 20-50 rows per stock")
    
    # Strategy 1: Drop early rows with NaN
    # Lose first ~30 days per stock, but safe
    df_clean = grouped.apply(
        lambda x: x.dropna()
    ).reset_index(drop=True)
    
    print(f"\nRows before dropna: {len(df):,}")
    print(f"Rows after dropna: {len(df_clean):,}")
    print(f"Lost: {len(df) - len(df_clean):,} rows ({(len(df) - len(df_clean))/len(df):.1%})")
    
    return df_clean
```

---

# PHASE 4: TRAIN/VAL/TEST SPLIT

## 4.1 The ONLY Correct Way: Temporal Split

### ✅ DO:

```python
def temporal_split(df, train_end='2021-12-31', val_end='2022-12-31'):
    """
    CRITICAL: Split by time, not randomly
    Prevents lookahead bias
    """
    train = df[df['date'] <= train_end]
    val = df[(df['date'] > train_end) & (df['date'] <= val_end)]
    test = df[df['date'] > val_end]
    
    print(f"Train: {train['date'].min()} to {train['date'].max()}")
    print(f"Val:   {val['date'].min()} to {val['date'].max()}")
    print(f"Test:  {test['date'].min()} to {test['date'].max()}")
    
    # Verify no overlap
    assert train['date'].max() < val['date'].min()
    assert val['date'].max() < test['date'].min()
    
    return train, val, test
```

**Example split for 10 years of data (2015-2024):**
```python
# Train: 2015-01-01 to 2021-12-31 (7 years, 70%)
# Val:   2022-01-01 to 2022-12-31 (1 year, 10%)
# Test:  2023-01-01 to 2024-12-31 (2 years, 20%)
```

**Why this matters:**
```
Correct temporal split:
Train on 2015-2021 → Validate on 2022 → Test on 2023-2024
Simulates real trading: Use past to predict future

Wrong random split:
Train on [2015 days + 2020 days + 2023 days]
Test on [2016 days + 2021 days + 2024 days]
Model sees future during training!
```

### ❌ DO NOT:

**1. NEVER use random split:**
```python
# ❌ WRONG - catastrophic for time series
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Why wrong: Model trains on 2024 data, tests on 2020 data
# Learns future → predicts past
# Works in backtesting, fails in real trading
```

**2. NEVER use cross-validation (K-Fold):**
```python
# ❌ WRONG - same issue as random split
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train_idx, test_idx in kf.split(df):
    # Each fold has random mix of dates
    # Massive leakage
```

**3. NEVER split within stocks:**
```python
# ❌ WRONG - splits Apple's history randomly
apple_train, apple_test = random_split(apple_data)
msft_train, msft_test = random_split(msft_data)

# ✅ RIGHT - all stocks split at same date
train = all_stocks[all_stocks['date'] <= '2021-12-31']
test = all_stocks[all_stocks['date'] > '2021-12-31']
```

---

## 4.2 Train/Val/Test Split Structure Details

### After Temporal Split

```python
def create_train_val_test_splits(df):
    """
    Split by date, maintaining interleaved structure
    """
    # Define split dates
    train_end = '2021-12-31'
    val_end = '2022-12-31'
    
    # Split
    train_df = df[df['date'] <= train_end].copy()
    val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test_df = df[df['date'] > val_end].copy()
    
    # Verify all stocks appear in all splits
    print("Stocks in train:", train_df['stock_id'].nunique())
    print("Stocks in val:", val_df['stock_id'].nunique())
    print("Stocks in test:", test_df['stock_id'].nunique())
    
    # Check label distribution
    print(f"\nLabel distribution:")
    print(f"Train: {(train_df['label']==1).sum()} positives / {len(train_df)} total")
    print(f"Val:   {(val_df['label']==1).sum()} positives / {len(val_df)} total")
    print(f"Test:  {(test_df['label']==1).sum()} positives / {len(test_df)} total")
    
    return train_df, val_df, test_df
```

**What each split looks like:**

```python
# train_df (70% - 7 years)
      date       stock_id   ret_1d  rsi_14  volume_z  drawdown  ...  label
0   2015-01-02   AAPL      -0.021    42.3      0.8      -0.15   ...    0
1   2015-01-02   GOOGL     -0.015    45.1      0.3      -0.12   ...    0
...
2.6M rows, dates: 2015-01-02 to 2021-12-31

# val_df (10% - 1 year)
      date       stock_id   ret_1d  rsi_14  volume_z  drawdown  ...  label
0   2022-01-03   AAPL      -0.018    38.2      1.2      -0.18   ...    0
1   2022-01-03   GOOGL     -0.022    35.8      0.9      -0.21   ...    0
...
380K rows, dates: 2022-01-03 to 2022-12-30

# test_df (20% - 2 years)
      date       stock_id   ret_1d  rsi_14  volume_z  drawdown  ...  label
0   2023-01-03   AAPL      -0.012    41.5      0.5      -0.09   ...    0
1   2023-01-03   GOOGL     -0.008    43.2      0.2      -0.07   ...    0
...
760K rows, dates: 2023-01-03 to 2024-12-31
```

---

## 4.3 Walk-Forward Validation (Advanced Alternative)

### ✅ DO (if you want to be extra rigorous):

```python
def walk_forward_validation(df, train_years=5, test_years=1):
    """
    More realistic: Retrain model periodically
    
    Split 1: Train 2015-2019, Test 2020
    Split 2: Train 2016-2020, Test 2021
    Split 3: Train 2017-2021, Test 2022
    Split 4: Train 2018-2022, Test 2023
    Split 5: Train 2019-2023, Test 2024
    
    Average performance across all splits
    """
    results = []
    
    start_year = df['date'].min().year
    end_year = df['date'].max().year
    
    for test_year in range(start_year + train_years, end_year + 1):
        train_start = test_year - train_years
        train_end = test_year - 1
        
        train = df[(df['date'].dt.year >= train_start) & 
                   (df['date'].dt.year <= train_end)]
        test = df[df['date'].dt.year == test_year]
        
        # Train and evaluate
        model = train_model(train)
        metrics = evaluate_model(model, test)
        
        results.append({
            'test_year': test_year,
            'metrics': metrics
        })
    
    return results
```

**Why walk-forward is better:**
- Tests model on multiple future periods
- Accounts for regime changes (bull/bear markets)
- More conservative performance estimate

---

## 4.4 Feature Matrix Preparation

**Prepare X and y:**

```python
def prepare_feature_matrix(df):
    """
    Convert dataframe to feature matrix + labels
    """
    # Define feature columns (all except metadata and label)
    feature_cols = [
        # Returns
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
        
        # Price vs MAs
        'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
        
        # Technical indicators
        'rsi_14', 'rsi_30', 'rsi_norm',
        'macd', 'macd_signal', 'macd_hist', 'macd_cross',
        'bb_position', 'bb_width',
        'adx', 'atr_14', 'sar', 'adr',
        
        # Volume
        'volume_ratio', 'volume_z', 'obv_ratio',
        
        # Volatility
        'volatility_20d', 'vol_z',
        
        # Other
        'drawdown',
        
        # Clever features
        'multi_divergence_score',
        'RSI_Divergence',
        'volume_exhaustion',
        'panic_selling',
        'panic_severity',
        'support_test_count',
        'consecutive_down_days',
        'exhaustion_signal',
        'price_zscore',
        'statistical_bottom',
        
        # Time features
        'day_of_week',
        'is_monday',
        'days_since_last_pivot',
    ]
    
    # Verify all features exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"⚠️  Missing features: {missing}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Create feature matrix
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # Keep metadata separate (for analysis, not training)
    metadata = df[['date', 'stock_id', 'close']].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X):,}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y, metadata, feature_cols
```

**Final X and y structure:**

```python
# X (feature matrix)
       ret_1d  ret_5d  rsi_14  volume_z  drawdown  ...
0      -0.021  -0.045    42.3       0.8     -0.15  ...
1      -0.015  -0.038    45.1       0.3     -0.12  ...
2       0.012   0.005    52.8      -0.5     -0.05  ...
...
Shape: (3,750,000 rows, 45 features)

# y (labels)
0    0
1    0
2    0
3    0
4    1  ← Bottom
5    0
...
Shape: (3,750,000,)

# Distribution: ~15,000 ones (bottoms), ~3,735,000 zeros (not bottoms)
# Ratio: 1:250
```

---

## 4.5 Complete Pipeline Example

```python
def complete_data_pipeline(raw_data_path):
    """
    End-to-end: Raw data → Train/Val/Test splits
    """
    
    # Step 1: Load raw data
    print("Step 1: Loading raw data...")
    df = pd.read_parquet(raw_data_path)  # or read_csv
    verify_raw_data_structure(df)
    
    # Step 2: Feature engineering
    print("\nStep 2: Creating features...")
    df = create_ml_features(df)  # Your existing indicators
    df = create_all_clever_features(df)  # Clever features from earlier
    
    # Step 3: Create labels
    print("\nStep 3: Creating labels...")
    df = create_labels_pivot(df, lb=8, rb=8)  # Your pivot function
    
    # Step 4: Handle NaNs
    print("\nStep 4: Cleaning NaNs...")
    df = handle_feature_nans(df)
    
    # Step 5: Organize data (interleaved)
    print("\nStep 5: Organizing data...")
    df = organize_data_for_training(df)
    
    # Step 6: Temporal split
    print("\nStep 6: Creating train/val/test splits...")
    train_df, val_df, test_df = create_train_val_test_splits(df)
    
    # Step 7: Prepare feature matrices
    print("\nStep 7: Creating feature matrices...")
    X_train, y_train, meta_train, feature_cols = prepare_feature_matrix(train_df)
    X_val, y_val, meta_val, _ = prepare_feature_matrix(val_df)
    X_test, y_test, meta_test, _ = prepare_feature_matrix(test_df)
    
    # Step 8: Final verification
    print("\nStep 8: Final verification...")
    print(f"✓ Train: {X_train.shape[0]:,} samples, {(y_train==1).sum():,} bottoms")
    print(f"✓ Val:   {X_val.shape[0]:,} samples, {(y_val==1).sum():,} bottoms")
    print(f"✓ Test:  {X_test.shape[0]:,} samples, {(y_test==1).sum():,} bottoms")
    
    # Return everything
    return {
        'X_train': X_train, 'y_train': y_train, 'meta_train': meta_train,
        'X_val': X_val, 'y_val': y_val, 'meta_val': meta_val,
        'X_test': X_test, 'y_test': y_test, 'meta_test': meta_test,
        'feature_cols': feature_cols,
    }

# Usage
data = complete_data_pipeline('stock_data.parquet')

# Now ready to train:
model = lgb.LGBMClassifier(scale_pos_weight=250, ...)
model.fit(data['X_train'], data['y_train'],
          eval_set=[(data['X_val'], data['y_val'])],
          early_stopping_rounds=100)
```

---

## 4.6 Data Storage Recommendations

### Option 1: Single File (Small to Medium Dataset)

```python
# Save entire processed dataset
df.to_parquet('data/processed_stock_data.parquet', compression='snappy')

# Load and split
df = pd.read_parquet('data/processed_stock_data.parquet')
train, val, test = temporal_split(df)
```

### Option 2: Pre-Split Files (Large Dataset)

```python
# Save splits separately (saves memory during training)
train_df.to_parquet('data/train.parquet')
val_df.to_parquet('data/val.parquet')
test_df.to_parquet('data/test.parquet')

# Load only what you need
train_df = pd.read_parquet('data/train.parquet')
X_train, y_train, meta_train, feature_cols = prepare_feature_matrix(train_df)
del train_df  # Free memory
```

### Option 3: Numpy Arrays (For Training Only)

```python
# After feature extraction, save as numpy (fastest for training)
np.save('data/X_train.npy', X_train.values)
np.save('data/y_train.npy', y_train.values)
np.save('data/X_val.npy', X_val.values)
np.save('data/y_val.npy', y_val.values)

# Save feature names separately
with open('data/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

# Load for training (very fast)
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
```

---

## SUMMARY: Dataset Structure Checklist

```python
✓ Raw data:
  - Columns: date, stock_id, open, high, low, close, volume
  - Sorted by date (optional: then by stock_id)
  - No duplicates
  - All stocks mixed (not grouped)
  
✓ After feature engineering:
  - Added: ~45 feature columns
  - Added: 'label' column (0 or 1)
  - Dropped: NaN rows from rolling windows
  - Still interleaved by date
  
✓ Train/Val/Test:
  - Temporal split (not random)
  - Train: 2015-2021 (70%)
  - Val: 2022 (10%)
  - Test: 2023-2024 (20%)
  - Each split maintains interleaved structure
  
✓ Feature matrix:
  - X: (n_samples, n_features) - all numeric
  - y: (n_samples,) - binary 0/1
  - metadata: (n_samples, 3) - date, stock_id, close (for analysis)
  
✓ Label distribution:
  - ~1:250 imbalance
  - ~15,000 bottoms in train
  - ~2,000 bottoms in val
  - ~4,000 bottoms in test
```

**The key principle: INTERLEAVE BY DATE, NOT GROUP BY STOCK**

This ensures the model learns general patterns across diverse stocks, not stock-specific quirks.

---

# PHASE 5: HANDLING IMBALANCE (1:250)

## 5.1 Multi-Pronged Strategy

### ✅ DO: Combine multiple techniques

**1. Class weights (always use):**
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    scale_pos_weight=250,  # Imbalance ratio
    objective='binary',
    # Other params...
)
```

**2. SMOTE (carefully):**
```python
from imblearn.over_sampling import SMOTE

# Apply per stock to avoid mixing stocks
def smote_per_stock(df, target_ratio=0.1):
    """
    Oversample minority class per stock
    Target: 1:10 instead of 1:250
    """
    resampled_dfs = []
    
    for stock_id in df['stock_id'].unique():
        stock_df = df[df['stock_id'] == stock_id]
        
        X = stock_df[feature_cols]
        y = stock_df['label']
        
        # Only if stock has at least 5 positive samples
        if y.sum() >= 5:
            smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Reconstruct dataframe
            resampled_df = pd.DataFrame(X_resampled, columns=feature_cols)
            resampled_df['label'] = y_resampled
            resampled_df['stock_id'] = stock_id
            
            resampled_dfs.append(resampled_df)
    
    return pd.concat(resampled_dfs)
```

**3. Threshold tuning (post-training):**
```python
from sklearn.metrics import precision_recall_curve

def tune_threshold(model, X_val, y_val, target_recall=0.6):
    """
    Find threshold that gives desired recall
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Find threshold for target recall
    idx = np.argmin(np.abs(recall - target_recall))
    best_threshold = thresholds[idx]
    
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  Precision: {precision[idx]:.3f}")
    print(f"  Recall: {recall[idx]:.3f}")
    
    return best_threshold
```

### ❌ DO NOT:

**1. Don't use accuracy as metric:**
```python
# ❌ WRONG - useless with imbalance
accuracy = (predictions == labels).sum() / len(labels)
# Can get 99.6% by predicting all zeros (1/250 = 0.4% error)

# ✅ RIGHT - use precision, recall, F1
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

**2. Don't undersample majority class:**
```python
# ❌ WRONG - throws away 99% of data
# You have limited positive samples, can't afford to throw away negatives

# ✅ RIGHT - oversample minority or use weights
```

**3. Don't assume default threshold (0.5) works:**
```python
# ❌ WRONG
predictions = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
# With 1:250 imbalance, model outputs probabilities like 0.001, 0.002
# Threshold of 0.5 never triggers

# ✅ RIGHT
predictions = (model.predict_proba(X)[:, 1] > 0.02).astype(int)
# Lower threshold appropriate for imbalanced data
```

---

# PHASE 6: MODEL TRAINING

## 6.1 Model Selection Philosophy

### ✅ DO: Start Simple, Add Complexity Only If Needed

**Recommended progression:**

```
Level 1 (Start Here): LightGBM / XGBoost
├─ Pros: Fast, handles imbalance well, interpretable
├─ Cons: None for tabular data
└─ Expected F1: 0.15-0.25

Level 2 (If Level 1 plateaus): Random Forest Ensemble
├─ Pros: Different algorithm, can ensemble with GBM
├─ Cons: Slower than GBM
└─ Expected improvement: +0.02-0.05 F1

Level 3 (Only if desperate): LSTM/Transformer
├─ Pros: Learns temporal patterns directly
├─ Cons: Slow, hard to train, needs lots of data
└─ Expected improvement: +0.03-0.07 F1 (maybe)
```

**Start with this:**
```python
import lightgbm as lgb

# Baseline model
model = lgb.LGBMClassifier(
    objective='binary',
    scale_pos_weight=250,  # Imbalance ratio
    max_depth=7,           # Prevent overfitting
    num_leaves=63,         # 2^7 - 1
    learning_rate=0.01,    # Small for stability
    n_estimators=5000,     # Many trees, rely on early stopping
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Feature sampling
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    random_state=42
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=100,
    verbose=100
)
```

### ❌ DO NOT:

**1. Don't start with complex models:**
```python
# ❌ WRONG - jumping to neural networks first
# Hard to debug, slow to train, usually unnecessary
# Start simple

# ✅ RIGHT - start with GBM, only go neural if GBM fails
```

**2. Don't train without validation set:**
```python
# ❌ WRONG
model.fit(X_train, y_train)  # No validation
# Can't detect overfitting, no early stopping

# ✅ RIGHT
model.fit(X_train, y_train, 
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=100)
```

**3. Don't ignore feature importance:**
```python
# After training, ALWAYS check:
import matplotlib.pyplot as plt

lgb.plot_importance(model, max_num_features=20)
plt.title('Top 20 Features')
plt.show()

# If top feature is "day_of_week" → something is wrong
# If top features make sense (RSI, volume_z, drawdown) → good sign
```

---

## 6.2 Preventing Overfitting

### ✅ DO:

**1. Use regularization:**
```python
model = lgb.LGBMClassifier(
    max_depth=7,           # Limit tree depth
    min_child_samples=20,  # Min samples per leaf
    reg_alpha=0.1,         # L1 (feature selection)
    reg_lambda=1.0,        # L2 (weight penalty)
    subsample=0.8,         # Don't use all rows
    colsample_bytree=0.8,  # Don't use all features
)
```

**2. Monitor train vs validation performance:**
```python
def check_overfitting(model, X_train, y_train, X_val, y_val):
    """
    Compare performance on train vs val
    """
    from sklearn.metrics import f1_score
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    
    print(f"Train F1: {train_f1:.3f}")
    print(f"Val F1:   {val_f1:.3f}")
    print(f"Gap:      {train_f1 - val_f1:.3f}")
    
    if train_f1 - val_f1 > 0.10:
        print("⚠️  Large gap - overfitting detected")
        print("   Try: Lower max_depth, increase min_child_samples")
    else:
        print("✓ No obvious overfitting")
```

**3. Use early stopping:**
```python
# Stops training when val performance stops improving
early_stopping_rounds=100  # Stop if no improvement for 100 rounds
```

### ❌ DO NOT:

**1. Don't tune on test set:**
```python
# ❌ WRONG
for max_depth in [3, 5, 7, 9]:
    model = train(max_depth=max_depth)
    score = evaluate(model, X_test, y_test)  # Using test set!
    # This is indirect overfitting on test set

# ✅ RIGHT
for max_depth in [3, 5, 7, 9]:
    model = train(max_depth=max_depth)
    score = evaluate(model, X_val, y_val)  # Use validation set
    # Test set only used ONCE at the very end
```

**2. Don't add features without validation:**
```python
# ❌ WRONG
# Add 100 new features
# Train model
# If test F1 improves, keep them
# This is overfitting

# ✅ RIGHT
# Add features based on domain knowledge
# Check if val F1 improves (not test!)
# Only check test ONCE at end
```

---

# PHASE 7: EVALUATION

## 7.1 Metrics That Matter

### ✅ DO: Use Multiple Metrics

```python
def comprehensive_evaluation(model, X, y, threshold=0.5):
    """
    Complete evaluation for imbalanced classification
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_curve, roc_auc_score,
        average_precision_score
    )
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y, y_pred, target_names=['Not Bottom', 'Bottom']))
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    
    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.3f} (Of predicted bottoms, how many are real?)")
    print(f"Recall:    {recall:.3f} (Of real bottoms, how many did we catch?)")
    print(f"F1 Score:  {f1:.3f} (Harmonic mean of precision and recall)")
    print(f"ROC-AUC:   {roc_auc_score(y, y_pred_proba):.3f}")
    print(f"PR-AUC:    {average_precision_score(y, y_pred_proba):.3f}")
    
    # Precision @ Top K
    top_k_pct = 0.01  # Top 1% of predictions
    k = int(len(y) * top_k_pct)
    top_k_indices = np.argsort(y_pred_proba)[-k:]
    precision_at_k = y.iloc[top_k_indices].sum() / k
    
    print(f"\nPrecision @ Top 1%: {precision_at_k:.3f}")
    print(f"(Of top 1% highest probability predictions, {precision_at_k:.1%} are actual bottoms)")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'pr_auc': average_precision_score(y, y_pred_proba),
        'precision_at_1pct': precision_at_k
    }
```

**What good performance looks like:**
```
For 1:250 imbalance:

Baseline (predict all 0): F1 = 0.00, Precision = 0.00, Recall = 0.00
Random guessing:          F1 = 0.01, Precision = 0.004, Recall = 0.50

Good model:
├─ F1 Score:        0.15 - 0.25
├─ Precision:       0.10 - 0.20 (10-20% of predictions are correct)
├─ Recall:          0.40 - 0.60 (catch 40-60% of bottoms)
├─ Precision@1%:    0.30 - 0.50 (top predictions very reliable)
└─ PR-AUC:          0.15 - 0.30

Excellent model:
├─ F1 Score:        0.25+
├─ Precision:       0.20+
├─ Recall:          0.60+
├─ Precision@1%:    0.50+
└─ PR-AUC:          0.30+
```

### ❌ DO NOT:

**1. Don't rely on accuracy:**
```python
# ❌ WRONG
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
# With 1:250 imbalance, 99.6% accuracy by predicting all 0s
# Meaningless metric
```

**2. Don't use ROC-AUC as primary metric:**
```python
# ⚠️  MISLEADING for severe imbalance
# ROC-AUC treats both classes equally
# Can have high ROC-AUC but terrible precision

# ✅ BETTER: Use PR-AUC (Precision-Recall AUC)
# Focuses on positive class performance
```

**3. Don't forget business context:**
```python
# What matters in trading:
# - Precision: Avoid false alarms (buying non-bottoms)
# - Recall: Don't miss real bottoms
# - Precision@K: Top predictions should be very reliable

# Example interpretation:
# F1=0.20, Precision=0.15, Recall=0.50, Precision@1%=0.40

# Translation:
# - Catches 50% of bottoms (recall=0.50)
# - 15% of all bottom predictions are correct (precision=0.15)
# - Top 1% predictions are 40% accurate (good for filtering)

# Is this useful? Depends on strategy:
# - High frequency: Need higher precision
# - Long-term: Can tolerate lower precision if recall is good
```

---

## 7.2 Visualization

### ✅ DO: Visualize model decisions

```python
def plot_model_predictions(df, model, stock_id, start_date, end_date):
    """
    Visual check: Are predictions sensible?
    """
    stock_data = df[(df['stock_id'] == stock_id) & 
                    (df['date'] >= start_date) & 
                    (df['date'] <= end_date)]
    
    # Get predictions
    X = stock_data[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Top plot: Price + actual bottoms + predicted bottoms
    ax1.plot(stock_data['date'], stock_data['close'], label='Price', linewidth=2)
    
    actual_bottoms = stock_data[stock_data['label'] == 1]
    ax1.scatter(actual_bottoms['date'], actual_bottoms['close'],
               color='green', s=100, label='Actual Bottom', zorder=5, marker='o')
    
    predicted_bottoms = stock_data[probabilities > 0.5]  # Adjust threshold
    ax1.scatter(predicted_bottoms['date'], predicted_bottoms['close'],
               color='red', s=100, label='Predicted Bottom', zorder=5, marker='x')
    
    ax1.set_ylabel('Price')
    ax1.set_title(f'{stock_id} - Bottom Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Prediction probability over time
    ax2.plot(stock_data['date'], probabilities, label='Bottom Probability', color='purple')
    ax2.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    ax2.fill_between(stock_data['date'], 0, probabilities, alpha=0.3, color='purple')
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Date')
    ax2.set_title('Model Confidence Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

**What to look for:**
- ✅ Red X's (predictions) near green O's (actual bottoms)
- ✅ High probability spikes at actual bottoms
- ❌ Random predictions scattered everywhere
- ❌ High probability during uptrends (model is confused)

---

# PHASE 8: HYPERPARAMETER TUNING

## 8.1 Systematic Approach

### ✅ DO: Grid search or Bayesian optimization

```python
from sklearn.model_selection import GridSearchCV
import optuna  # For Bayesian optimization

# Option 1: Grid Search (exhaustive but slow)
def grid_search_tuning(X_train, y_train, X_val, y_val):
    """
    Try all combinations systematically
    """
    param_grid = {
        'max_depth': [5, 7, 9],
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.7, 0.8, 0.9],
    }
    
    # This would take days, so narrow down:
    # First tune depth, then leaves, etc.
    
    best_params = {}
    best_score = 0
    
    # Tune max_depth first
    for max_depth in param_grid['max_depth']:
        model = lgb.LGBMClassifier(
            max_depth=max_depth,
            scale_pos_weight=250,
            # ... other defaults
        )
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=100,
                 verbose=False)
        
        val_pred = model.predict(X_val)
        score = f1_score(y_val, val_pred)
        
        if score > best_score:
            best_score = score
            best_params['max_depth'] = max_depth
    
    # Continue for other params...
    return best_params


# Option 2: Bayesian Optimization (smarter, faster)
def bayesian_tuning(X_train, y_train, X_val, y_val):
    """
    Let Optuna find optimal hyperparameters
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': 250,
        }
        
        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=50,
                 verbose=False)
        
        val_pred = model.predict(X_val)
        return f1_score(y_val, val_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print("Best hyperparameters:", study.best_params)
    print("Best F1 score:", study.best_value)
    
    return study.best_params
```

### ❌ DO NOT:

**1. Don't tune on test set:**
```python
# ❌ WRONG
best_params = tune_hyperparameters(X_test, y_test)
# This is overfitting on test set

# ✅ RIGHT
best_params = tune_hyperparameters(X_val, y_val)
# Test set never seen until final evaluation
```

**2. Don't tune everything at once:**
```python
# ❌ WRONG - too many combinations
param_grid = {
    'max_depth': [3, 5, 7, 9, 11],
    'num_leaves': [20, 31, 50, 63, 100, 127],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    # ... 10 more parameters
}
# 5 × 6 × 5 × ... = millions of combinations

# ✅ RIGHT - tune in stages or use Bayesian
# Or just use good defaults and only tune 2-3 key params
```

**3. Don't spend forever tuning:**
```python
# Diminishing returns:
# Default params:     F1 = 0.18
# 1 hour tuning:      F1 = 0.20  (+0.02)
# 10 hours tuning:    F1 = 0.21  (+0.01)
# 100 hours tuning:   F1 = 0.215 (+0.005)

# Better use of time: Improve features, fix data quality
```

---

# PHASE 9: FINAL CHECKLIST BEFORE DEPLOYMENT

## ✅ Pre-Deployment Audit

```python
def pre_deployment_audit(model, df_train, df_val, df_test, feature_cols):
    """
    Final verification before using model
    """
    print("=" * 80)
    print("PRE-DEPLOYMENT AUDIT")
    print("=" * 80)
    
    # 1. Data leakage check
    print("\n1. CHECKING FOR DATA LEAKAGE...")
    check_for_leakage(df_train, df_val, df_test)
    
    # 2. Temporal consistency
    print("\n2. CHECKING TEMPORAL ORDERING...")
    assert df_train['date'].max() < df_val['date'].min(), "Train/val overlap!"
    assert df_val['date'].max() < df_test['date'].min(), "Val/test overlap!"
    print("✓ Temporal splits correct")
    
    # 3. Survivorship bias
    print("\n3. CHECKING SURVIVORSHIP BIAS...")
    all_stocks = df_train['stock_id'].unique()
    delisted_pct = check_survivorship_bias(all_stocks)
    if delisted_pct < 0.10:
        print("⚠️  WARNING: Possible survivorship bias")
    else:
        print("✓ Survivorship bias check passed")
    
    # 4. Feature consistency
    print("\n4. CHECKING FEATURE CONSISTENCY...")
    for col in feature_cols:
        train_mean = df_train[col].mean()
        val_mean = df_val[col].mean()
        test_mean = df_test[col].mean()
        
        # Means shouldn't be wildly different (unless regime change)
        if abs(val_mean - train_mean) / abs(train_mean) > 2.0:
            print(f"⚠️  WARNING: Feature {col} has very different distribution in val")
    
    print("✓ Feature distributions reasonable")
    
    # 5. Label quality
    print("\n5. CHECKING LABEL QUALITY...")
    train_imbalance = (df_train['label'] == 0).sum() / (df_train['label'] == 1).sum()
    print(f"Train imbalance: 1:{train_imbalance:.0f}")
    if train_imbalance < 100 or train_imbalance > 500:
        print("⚠️  WARNING: Unusual imbalance ratio")
    else:
        print("✓ Label imbalance reasonable")
    
    # 6. Model performance
    print("\n6. CHECKING MODEL PERFORMANCE...")
    metrics_val = comprehensive_evaluation(model, df_val[feature_cols], df_val['label'])
    metrics_test = comprehensive_evaluation(model, df_test[feature_cols], df_test['label'])
    
    print(f"\nValidation F1: {metrics_val['f1']:.3f}")
    print(f"Test F1:       {metrics_test['f1']:.3f}")
    
    if abs(metrics_val['f1'] - metrics_test['f1']) > 0.05:
        print("⚠️  WARNING: Large performance gap between val and test")
    else:
        print("✓ Consistent performance on val/test")
    
    # 7. Feature importance sanity check
    print("\n7. CHECKING FEATURE IMPORTANCE...")
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    
    print("Top 10 features:")
    for idx in reversed(top_features):
        print(f"  {feature_cols[idx]}: {feature_importance[idx]:.3f}")
    
    # Check if top features make sense
    suspicious_features = ['day_of_week', 'month', 'year']
    top_feature_names = [feature_cols[idx] for idx in top_features]
    
    if any(f in top_feature_names for f in suspicious_features):
        print("⚠️  WARNING: Time features are top importance (possible overfitting)")
    else:
        print("✓ Feature importance looks reasonable")
    
    # 8. Out-of-sample test
    print("\n8. CHECKING TEST SET PERFORMANCE...")
    if metrics_test['f1'] < 0.10:
        print("❌ FAILED: Test F1 too low, model not useful")
        return False
    elif metrics_test['precision'] < 0.05:
        print("❌ FAILED: Precision too low, too many false positives")
        return False
    elif metrics_test['recall'] < 0.20:
        print("❌ FAILED: Recall too low, missing most bottoms")
        return False
    else:
        print("✓ Test performance acceptable")
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    
    return True
```

---

# PHASE 10: WHAT NOT TO DO (SUMMARY)

## ❌ FATAL MISTAKES TO AVOID

### 1. **Data Leakage (Most Common Failure)**
```python
# ❌ Using future data in features
df['future_return'] = df['close'].pct_change(-10)

# ❌ Global normalization (sees test data)
scaler.fit(entire_dataset)

# ❌ Random train/test split (future in train)
train_test_split(df, shuffle=True)
```

### 2. **Survivorship Bias**
```python
# ❌ Training only on companies that exist today
stocks = get_current_sp500()

# ❌ Filtering based on current metrics
stocks = stocks[stocks['current_market_cap'] > 1e9]
```

### 3. **Wrong Evaluation**
```python
# ❌ Using accuracy with imbalance
score = accuracy_score(y_true, y_pred)

# ❌ Tuning on test set
for param in params:
    score = evaluate_on_test_set(param)

# ❌ Peeking at test set during development
if test_f1 > 0.30:  # Checking test periodically
    keep_changes()
```

### 4. **Overfitting**
```python
# ❌ Too many features vs samples
# 1000 features, 500 positive samples → overfits

# ❌ No regularization
model = LGBMClassifier(max_depth=20, min_child_samples=1)

# ❌ Training too long without early stopping
model.fit(X, y, n_estimators=10000)  # No early stopping
```

### 5. **Poor Feature Engineering**
```python
# ❌ Using raw prices
features = ['close', 'open', 'high', 'low']

# ❌ Using volume without normalization
features = ['volume']

# ❌ Creating target-leaked features
features = ['days_until_next_bottom']  # This is future!
```

---

# FINAL PHILOSOPHY SUMMARY

## The 10 Commandments of ML for Stock Bottoms

1. **Thou shalt not use future information** - Every feature must be calculable in real-time

2. **Thou shalt not ignore survivorship bias** - Include failed companies (10-20% of dataset)

3. **Thou shalt split by time, not randomly** - Train on past, test on future

4. **Thou shalt use appropriate metrics** - F1, precision, recall; not accuracy

5. **Thou shalt regularize** - max_depth, min_samples, L1/L2, early stopping

6. **Thou shalt validate before testing** - Tune on validation, test only once

7. **Thou shalt normalize per stock** - Each stock is its own universe

8. **Thou shalt start simple** - GBM first, not deep learning

9. **Thou shalt visualize** - Plot predictions, check feature importance

10. **Thou shalt be skeptical** - If F1 > 0.40, something is probably wrong

---

## Expected Outcomes

**Realistic expectations:**
```
Time investment: 2-4 weeks for solid model
Final test F1: 0.15 - 0.25 (good), 0.25 - 0.35 (excellent)
Precision: 10-20% (1 in 5-10 predictions correct)
Recall: 40-60% (catch 40-60% of bottoms)

Translation: Model identifies potential bottoms, but:
- Needs human verification
- Combine with other signals
- Use for filtering, not blind trading
```

**If you achieve:**
- F1 > 0.35 → Likely data leakage, check carefully
- Precision > 0.30 → Impressive, verify no overfitting
- Recall > 0.70 → Excellent, but check false positive rate

**This is a filtering tool, not a crystal ball.**

---

*Good luck! 🚀*