# Stock Bottom Prediction Model: Complete Development Philosophy

## GUIDING PRINCIPLES

**Core Truth:** Most ML projects fail not from bad algorithms, but from data leakage, poor evaluation, and overfitting. Our philosophy prioritizes **correctness over complexity**.

**Important Note:** What is presented here is MERELY A GUIDELINE and should be treated as such. Model choices, thresholds, and specific techniques are suggestions -- adapt to your data and use case.

---

# PHASE 1: DATA PREPARATION

## 1.1 Stock Universe

**What you did:** 13,000 tickers -> 1,500 stocks via filtering

**CRITICAL VERIFICATION CHECKLIST:**

```
[] Do your stocks include DELISTED/BANKRUPT companies?
  -> If NO: You have survivorship bias. Must fix.
  -> If YES: Good. The exact percentage depends on your universe and timeframe.

[] Did you filter based on CURRENT liquidity/market cap?
  -> If YES: Wrong. A stock delisted in 2020 had low liquidity in 2021 (doesn't exist).
         Use point-in-time filtering: "Was liquid DURING its existence"
  -> If NO: Good.

[] Are sectors reasonably balanced?
  -> If >30% of stocks in one sector (e.g., Tech): Consider rebalancing.

[] Market cap distribution?
  -> Should roughly reflect your deployment universe.
  -> If all large-cap: Model won't generalize to small-caps.
```

**KEY PRINCIPLE:** Define the *live deployment universe* first, then reconstruct that universe point-in-time, including names that later delisted. Don't force bankruptcies or small caps if you won't trade them. The goal is matching your actual trading universe, not hitting arbitrary delisted percentages.

**ACTION ITEMS:**

1. **Check for survivorship bias:**
```python
def audit_survivorship_bias(stock_list):
    """
    Verify the universe includes companies that were delisted
    during the study period.
    """
    currently_trading = [s for s in stock_list if is_currently_trading(s)]
    delisted = [s for s in stock_list if not is_currently_trading(s)]

    delisted_pct = len(delisted) / len(stock_list)

    print(f"Currently trading: {len(currently_trading)}")
    print(f"Delisted/bankrupt: {len(delisted)}")
    print(f"Delisted percentage: {delisted_pct:.1%}")

    if delisted_pct == 0:
        print("WARNING: No delisted companies -- survivorship bias is certain")
        return False
    else:
        print("Delisted companies present -- review whether proportion is reasonable for your universe")
        return True
```

2. **If you lack delisted stocks:** Source data from a provider that includes delistings natively (e.g., CRSP, Sharadar, EOD Historical Data) rather than manually adding tickers. Manual addition is fragile and incomplete.

3. **If delisting returns are available from your source, include them.** Otherwise failure outcomes remain understated.

4. **Identifier hygiene:** Use permanent IDs (CUSIP, FIGI, Permco) when possible, not only tickers. Tickers get reused (e.g., old ticker reassigned to a new company).

---

## 1.2 Data Quality & Cleaning

### DO:

**1. Point-in-time data integrity:**
```python
def verify_data_integrity(df):
    """
    Each row must reflect information AVAILABLE at that date.
    """
    # Check 1: Corporate actions properly adjusted
    splits = detect_stock_splits(df)
    if len(splits) > 0:
        print(f"Found {len(splits)} splits - verify adjusted prices used")

    # Check 2: No impossible values
    assert (df['close'] > 0).all(), "Negative/zero prices detected"
    assert (df['volume'] >= 0).all(), "Negative volume detected"
    assert (df['high'] >= df['low']).all(), "High < Low detected"
    assert (df['high'] >= df['close']).all(), "Close > High detected"
    assert (df['low'] <= df['close']).all(), "Close < Low detected"

    return True
```

**IMPORTANT: Adjusted close handling.** Corporate-action handling must be point-in-time consistent. Do not mix adjusted close with raw OHLC/volume. If you use adjusted close for returns, all OHLCV columns should be adjusted consistently.

**2. Handle missing data correctly:**
```python
def handle_missing_data(df):
    """
    Missing data philosophy: Conservative.
    CRITICAL: Do NOT forward-fill prices or volume.
    """
    # Rule 1: If >5% of a stock's data is missing, DROP the stock
    missing_pct = df.groupby('stock_id').apply(
        lambda x: x.isnull().sum().sum() / (len(x) * len(x.columns))
    )
    stocks_to_drop = missing_pct[missing_pct > 0.05].index
    print(f"Dropping {len(stocks_to_drop)} stocks with >5% missing data")
    df = df[~df['stock_id'].isin(stocks_to_drop)]

    # Rule 2: For OHLCV columns, do NOT forward-fill.
    # Forward-filling prices fabricates market states (e.g., a stock
    # halted for 3 days then gapping would get smoothed over).
    # Re-source the bar, or drop that stock-date.
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df.dropna(subset=price_cols)

    # Rule 3: For slow-moving non-price metadata (sector, market cap bucket),
    # forward-fill is acceptable with a short limit.
    metadata_cols = [c for c in df.columns if c not in price_cols
                     and c not in ['date', 'stock_id', 'label']]
    if metadata_cols:
        df[metadata_cols] = df.groupby('stock_id')[metadata_cols].ffill(limit=5)

    # Rule 4: Remaining NaNs -> drop those rows
    df = df.dropna()

    return df
```

**3. Detect and fix data errors:**
```python
def detect_data_errors(df):
    """
    Common data quality issues.
    """
    # Error 1: Unadjusted stock splits (overnight gap >50%)
    prev_close = df.groupby('stock_id')['close'].shift(1)
    df['overnight_return'] = df['open'] / prev_close - 1

    suspicious = df[abs(df['overnight_return']) > 0.5]
    if len(suspicious) > 0:
        print(f"WARNING: Found {len(suspicious)} suspicious overnight gaps >50%")
        print("   Likely unadjusted splits - verify with data provider")

    # Error 2: Days with zero volume (illiquid or data error)
    zero_vol = df[df['volume'] == 0]
    if len(zero_vol) > 0:
        print(f"WARNING: Found {len(zero_vol)} days with zero volume")
        df = df[df['volume'] > 0]

    # Error 3: Extreme daily returns (>75% in one day)
    df['daily_return'] = df.groupby('stock_id')['close'].pct_change()
    extreme = df[abs(df['daily_return']) > 0.75]
    if len(extreme) > 0:
        print(f"WARNING: Found {len(extreme)} extreme returns >75%")
        print("   Manual review required")

    return df
```

### DO NOT:

1. **DON'T drop NaNs globally:**
```python
# WRONG - drops entire stock if any NaN
df = df.dropna()

# RIGHT - handle per stock, per situation
df = handle_missing_data(df)
```

2. **DON'T use data that didn't exist at that time:**
```python
# WRONG - using future sector classification
df['sector'] = current_sector_map[df['stock_id']]  # Sector in 2024

# RIGHT - use sector at time of observation
df['sector'] = historical_sector_map[df['date']][df['stock_id']]
```

3. **DON'T forward-fill OHLCV data:**
```python
# WRONG - fabricates market states
df = df.groupby('stock_id').ffill(limit=5)

# RIGHT - drop missing bars or re-source them
df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
```

4. **DON'T trust data blindly:**
- Always plot random stock samples
- Check for obvious errors
- Verify adjusted prices used consistently

---

# PHASE 2: LABELING AND DATASET STRUCTURE

## 2.1 Labeling Philosophy - CRITICAL UNDERSTANDING

### The Pivot Labeling Approach is CORRECT

**Your pivot labeling does NOT create lookahead bias. Here's why:**

```python
# Your labeling process:
# IMPORTANT: Compute per stock, not on the mixed dataframe
df = df.sort_values(['stock_id', 'date'])
df['label'] = df.groupby('stock_id')['low'].transform(
    lambda s: s.rolling(17, center=True).apply(is_pivot_low, raw=False)
)

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
   -> Human used their eyes (any method) to create labels

2. Model trains on pixel features to predict those labels
   -> Model only sees pixels, not how human labeled

Stock bottoms:
1. Pivot algorithm looks at 17-day window, labels bottoms
   -> Algorithm used future data (days +8) to create labels

2. Model trains on RSI/volume/etc to predict those labels
   -> Model only sees RSI/volume/etc, not how pivot labeled

THIS IS FINE
```

### The Boundary Problem (CRITICAL)

With a centered pivot window of `lb=8, rb=8`, the last `rb` rows before any split boundary have incomplete pivot windows -- their labels depend on data from the next split. These rows must be **dropped**, not defaulted to 0:

```python
def create_labels_with_boundary_handling(df, lb=8, rb=8):
    """
    Create pivot labels and handle boundary rows.
    Rows where the centered window is incomplete get NaN labels.
    """
    df = df.sort_values(['stock_id', 'date'])
    window = lb + rb + 1

    df['label'] = df.groupby('stock_id')['low'].transform(
        lambda s: s.rolling(window, center=True).apply(is_pivot_low, raw=False)
    )
    # The rolling(center=True) naturally produces NaN for the first lb
    # and last rb rows of each stock. These must be dropped before splitting.
    # Do NOT fill them with 0.
    return df
```

At train/val/test split boundaries, the last `rb` training rows have labels that depend on prices from the validation period. These rows must be purged. See Phase 4 for the embargo gap solution.

### The Real Lookahead Check

```python
def verify_no_leakage(df):
    """
    What ACTUALLY matters: Features must be backward-looking.
    Labels can be created however (including using future data).
    """
    # FEATURES - These must ALL be backward-looking
    features_ok = {
        'rsi_14': True,  # Uses past 14 days
        'volume_z': True,  # Uses past 60 days
        'ret_1d': True,  # Yesterday's return
        'ma_20': True,  # Past 20 days

        # These would be leakage:
        'future_return': False,  # Uses tomorrow's price
        'will_recover': False,  # Uses next week's data
    }

    # LABELS - Can use any method
    labels_ok = {
        'pivot_low_center_window': True,  # Uses +/- 8 days
        'manual_annotation': True,  # Human marked it
        'lowest_in_month': True,  # Used full month
    }

    print("Label creation method doesn't matter")
    print("Feature calculation is what we must verify")
```

### Feature-Availability Audit

Any feature using pivots, divergences, support tests, "days since pivot", etc. must prove it only uses information that would have been known at decision time. This is especially tricky for features derived from label-like quantities:

```python
# WRONG - days_since_pivot uses pivot detection which needs future data
df['days_since_pivot'] = ...  # Only valid if computed from PAST pivots

# RIGHT - use a lagged pivot detection (only confirms pivot after rb days)
# A pivot at date T is only confirmed at T+rb, so shift the detection forward:
def confirmed_pivot_days(group, lb=8, rb=8):
    """Compute days since last CONFIRMED pivot low for a single stock."""
    window = lb + rb + 1
    raw_pivots = group['low'].rolling(window, center=True).apply(is_pivot_low, raw=False)
    # Shift by rb: pivot at T is only known at T+rb
    confirmed = raw_pivots.shift(rb).fillna(0)
    # Count days since last confirmed pivot
    is_pivot = confirmed == 1
    groups = is_pivot.cumsum()
    days_since = is_pivot.groupby(groups).cumcount()
    return days_since

df['days_since_confirmed_pivot'] = df.groupby('stock_id').apply(
    confirmed_pivot_days
).reset_index(level=0, drop=True)
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
# - Total rows: ~1,500 stocks x 2,500 days = 3,750,000 rows
```

**Verification:**
```python
def verify_raw_data_structure(df):
    required_cols = ['date', 'stock_id', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_cols), "Missing required columns"

    df['date'] = pd.to_datetime(df['date'])

    # No duplicates
    duplicates = df.duplicated(subset=['date', 'stock_id'])
    assert duplicates.sum() == 0, f"Found {duplicates.sum()} duplicate rows"

    assert df['close'].dtype in [float, np.float64], "Price data should be float"
    assert df['volume'].dtype in [int, np.int64, float, np.float64], "Volume should be numeric"

    print(f"Dataset summary:")
    print(f"  Stocks: {df['stock_id'].nunique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Avg days per stock: {len(df) / df['stock_id'].nunique():.0f}")

    return True
```

---

## 2.3 Data Organization

**CRITICAL RULE:** Before any `rolling / pct_change / shift / label` computation, sort by `stock_id, date` and compute strictly within `groupby('stock_id')`. This prevents cross-stock contamination regardless of row ordering.

```python
def prepare_data_for_computation(df):
    """
    Sort correctly before any per-stock computation.
    """
    df = df.sort_values(['stock_id', 'date']).reset_index(drop=True)
    return df
```

**For training data ordering:** Tree-based models (gradient boosting, random forests) don't care about row order -- they build trees on the full dataset, not sequentially through batches. Row ordering only matters for SGD-based models (neural nets) that train in mini-batches. For tree-based models, interleaved vs. grouped ordering is irrelevant as long as features were computed correctly within each stock.

If you're using neural networks or other batch-based training, interleaving by date ensures diverse stocks in each batch:

```python
def organize_for_batch_training(df):
    """
    For SGD-based models: interleave stocks by date.
    For tree-based models: ordering doesn't matter.
    """
    df = df.sort_values(['date', 'stock_id']).reset_index(drop=True)
    return df
```

---

# PHASE 3: FEATURE ENGINEERING

## 3.1 The Golden Rule: NO FUTURE INFORMATION

**Every feature must answer: "Could I calculate this in real-time trading?"**

### DO:

**1. Use rolling windows (backward-looking only):**
```python
# CORRECT - only uses past data
df['rsi_14'] = df.groupby('stock_id')['close'].transform(
    lambda x: talib.RSI(x, timeperiod=14)
)

df['ma_20'] = df.groupby('stock_id')['close'].transform(
    lambda x: x.rolling(20).mean()
)

df['volume_z'] = df.groupby('stock_id')['volume'].transform(
    lambda x: (x - x.rolling(60).mean()) / x.rolling(60).std()
)
```

**2. Per-stock normalization (avoid cross-stock contamination):**
```python
def normalize_per_stock(df, column, window=252):
    grouped = df.groupby('stock_id')
    df[f'{column}_norm'] = grouped[column].transform(
        lambda x: (x - x.rolling(window, min_periods=20).mean()) /
                  x.rolling(window, min_periods=20).std()
    )
    return df
```

**3. Only use "shift" to create lags, never negative shifts:**
```python
# CORRECT - using past values
df['close_lag1'] = df.groupby('stock_id')['close'].shift(1)
df['close_lag5'] = df.groupby('stock_id')['close'].shift(5)

# WRONG - looking into future
df['close_future'] = df.groupby('stock_id')['close'].shift(-1)
```

### DO NOT:

**1. NEVER normalize across entire dataset:**
```python
# WRONG - uses global statistics including future
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['price_scaled'] = scaler.fit_transform(df[['close']])
# This uses mean/std of ALL data (train + test)

# RIGHT - fit on train only, transform separately
scaler = StandardScaler()
train['price_scaled'] = scaler.fit_transform(train[['close']])
test['price_scaled'] = scaler.transform(test[['close']])
```

**2. NEVER use `.min()`, `.max()`, `.mean()` without `.rolling()`:**
```python
# WRONG - sees entire series including future
df['price_normalized'] = df['close'] / df['close'].max()

# RIGHT - only uses past
df['price_normalized'] = df.groupby('stock_id')['close'].transform(
    lambda x: x / x.rolling(252, min_periods=50).max()
)
```

**3. NEVER create features that "knew the answer":**
```python
# WRONG - this is literally the target
df['will_go_up'] = (df['close'].shift(-20) > df['close']).astype(int)

# Even subtle versions are wrong:
df['momentum_future'] = df['close'].pct_change(-10)  # 10-day forward return
```

---

## 3.2 Feature Categories

### DO: Include these feature types

**1. Price-based (returns, not absolute prices):**
```python
df['ret_1d'] = df.groupby('stock_id')['close'].pct_change(1)
df['ret_5d'] = df.groupby('stock_id')['close'].pct_change(5)
df['ret_10d'] = df.groupby('stock_id')['close'].pct_change(10)
df['ret_20d'] = df.groupby('stock_id')['close'].pct_change(20)

df['price_to_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
df['price_to_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
```

**2. Volume features:**
```python
df['volume_ratio'] = df['volume'] / df['volume_ma20']

df['volume_z'] = df.groupby('stock_id')['volume_ratio'].transform(
    lambda x: (x - x.rolling(60).mean()) / x.rolling(60).std()
)
```

**3. Technical indicators:**
```python
# RSI, MACD, Bollinger Bands, ATR, etc.
# All backward-looking by construction
```

**4. Pattern features (divergences, exhaustion):**
```python
# Multi-indicator divergence, panic selling, volume exhaustion, etc.
```

**5. Contextual features:**
```python
# Drawdown from highs
df['drawdown'] = (df['close'] - df['high_252']) / df['high_252']

# Time features
df['day_of_week'] = df['date'].dt.dayofweek
```

**6. Market-level / regime features (consider adding):**
```python
# A bottom in a bear market looks different from a pullback in a bull market
df['spy_drawdown'] = ...  # SPY drawdown from 52-week high
df['vix_level'] = ...  # VIX at that date
df['market_breadth'] = ...  # % of stocks above 200-day MA
```

### DO NOT:

**1. Don't use raw prices as features:**
```python
# WRONG
features = ['close', 'open', 'high', 'low']
# $5 stock vs $500 stock - incomparable

# RIGHT
features = ['ret_1d', 'price_to_sma20', 'drawdown']
# All scale-invariant
```

**2. Don't use volume without normalization:**
```python
# WRONG
features = ['volume']
# Apple: 50M shares vs penny stock: 10K shares

# RIGHT
features = ['volume_ratio', 'volume_z']
# Both measure "how unusual is today's volume FOR THIS STOCK"
```

**3. Don't create redundant features:**
```python
# WRONG - all measure same thing
features = ['sma_20', 'sma_21', 'sma_22', 'sma_23']

# RIGHT - different timeframes
features = ['sma_20', 'sma_50', 'sma_200']
```

## 3.3 Multi-Collinearity

RSI, drawdown, price-to-SMA, and ret_20d are often highly correlated. Gradient boosting handles this okay (it picks one and ignores the rest), but high collinearity inflates feature count and muddies feature importance.

**What to do:**
- Compute pairwise correlation matrix of features; drop one from any pair with |r| > 0.95
- Optionally use VIF (Variance Inflation Factor) to identify redundant features
- Don't obsess over moderate correlations (|r| < 0.8) for tree models -- they handle it

```python
def check_collinearity(X, y, threshold=0.95):
    """
    Find highly correlated pairs. When dropping one from a pair,
    prefer to keep the one with higher univariate predictive power.
    """
    from sklearn.metrics import average_precision_score

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for other in correlated:
            if other in to_drop or col in to_drop:
                continue
            # Keep the feature with higher standalone predictive power.
            # Use max(AP(y, feat), AP(y, -feat)) because AP assumes higher
            # values = more positive. Features like drawdown (more negative
            # near bottoms) would get near-zero AP without the negation check.
            ap_col = max(average_precision_score(y, X[col]),
                         average_precision_score(y, -X[col]))
            ap_other = max(average_precision_score(y, X[other]),
                           average_precision_score(y, -X[other]))
            drop = other if ap_col >= ap_other else col
            to_drop.add(drop)
            print(f"  {col} <-> {other} (r={corr.loc[other, col]:.3f}): drop {drop}")

    print(f"\nFeatures to drop (|r| > {threshold}): {sorted(to_drop)}")
    return sorted(to_drop)
```

---

## 3.4 Feature Engineering Output

**After applying all indicators and features:**

```python
# df_features (after feature engineering)
      date       stock_id   close  volume    ret_1d  ret_5d  rsi_14  macd  volume_z  drawdown  ...  label
0   2015-01-20   AAPL      110.22  51.2M    -0.021   -0.045   42.3   -1.2    0.8      -0.15   ...    0
1   2015-01-21   AAPL      109.55  48.9M    -0.006   -0.051   39.1   -1.5    0.3      -0.16   ...    0
...
1000 2015-02-15  TSLA      210.45   5.1M    -0.085   -0.142   22.1   -3.5    2.8      -0.28   ...    1  <- Bottom!
```

**Critical: Handle NaN from rolling windows**
```python
def handle_feature_nans(df, feature_cols, label_col='label'):
    """
    NaN values appear at start of each stock's history
    (not enough data for rolling windows).
    Only drop rows with NaN in feature or label columns,
    not metadata columns you don't care about.
    """
    required_cols = feature_cols + [label_col]
    before = len(df)

    df_clean = df.dropna(subset=required_cols)

    print(f"Rows before dropna: {before:,}")
    print(f"Rows after dropna: {len(df_clean):,}")
    print(f"Lost: {before - len(df_clean):,} rows ({(before - len(df_clean))/before:.1%})")

    return df_clean
```

---

# PHASE 4: TRAIN/VAL/TEST SPLIT

## 4.1 The ONLY Correct Way: Temporal Split

### DO:

```python
def temporal_split(df, train_end='2021-12-31', val_end='2022-12-31'):
    """
    Split by time, not randomly. Prevents lookahead bias.
    """
    train = df[df['date'] <= train_end]
    val = df[(df['date'] > train_end) & (df['date'] <= val_end)]
    test = df[df['date'] > val_end]

    print(f"Train: {train['date'].min()} to {train['date'].max()}")
    print(f"Val:   {val['date'].min()} to {val['date'].max()}")
    print(f"Test:  {test['date'].min()} to {test['date'].max()}")

    assert train['date'].max() < val['date'].min()
    assert val['date'].max() < test['date'].min()

    return train, val, test
```

**Why this matters:**
```
Correct temporal split:
Train on 2015-2021 -> Validate on 2022 -> Test on 2023-2024
Simulates real trading: Use past to predict future

Wrong random split:
Train on [2015 days + 2020 days + 2023 days]
Test on [2016 days + 2021 days + 2024 days]
Model sees future during training!
```

### DO NOT:

**1. NEVER use random split:**
```python
# WRONG - catastrophic for time series
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

**2. NEVER use shuffled K-Fold cross-validation:**
```python
# WRONG - shuffled CV has the same issue as random split
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
```

**However, time-series-aware CV is valid and often better** than a single temporal split, since it gives you variance estimates.

**WARNING:** sklearn's `TimeSeriesSplit(gap=N)` splits by **row index**, not by date. With pooled panel data (multiple stocks per date), `gap=8` skips 8 *rows*, not 8 trading days. With 1,500 stocks, 8 rows is ~0.005 days -- useless as an embargo. For panel data, split on **unique dates**, then map dates back to rows:

```python
def panel_time_series_cv(df, n_splits=5, embargo_sessions=8):
    """
    Time-series CV that splits by trading date, not row index.
    Correct for pooled stock-day panels.
    """
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // (n_splits + 1)

    for i in range(n_splits):
        train_end_idx = fold_size * (i + 1)
        test_start_idx = train_end_idx + embargo_sessions  # embargo in sessions
        test_end_idx = test_start_idx + fold_size

        if test_end_idx > n_dates:
            break

        train_dates = set(unique_dates[:train_end_idx - embargo_sessions])
        test_dates = set(unique_dates[test_start_idx:test_end_idx])

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'].isin(test_dates)

        yield df.index[train_mask], df.index[test_mask]
```

Purged K-fold with embargo gaps is also valid. The key rule: **never shuffle time-series data across folds**, and always embargo in **trading sessions**, not row counts.

**3. NEVER split within stocks:**
```python
# WRONG - splits Apple's history randomly
apple_train, apple_test = random_split(apple_data)

# RIGHT - all stocks split at same date
train = all_stocks[all_stocks['date'] <= '2021-12-31']
test = all_stocks[all_stocks['date'] > '2021-12-31']
```

---

## 4.2 Embargo / Purge Gap (CRITICAL)

With an 8-day pivot window (`rb=8`), the last 8 days of training data have labels computed using prices from the validation period. Similarly at the val/test boundary. This creates label leakage across splits.

**Solution: Insert an embargo gap >= `rb` trading sessions on BOTH sides of each boundary.** The leakage is from the last `rb` rows of the earlier split (whose labels depend on prices from the next split). Drop rows from both sides of every boundary to be safe.

**Use trading sessions, not calendar days** -- `pd.Timedelta(days=8)` counts weekends/holidays; use actual trading dates from your data instead.

```python
def temporal_split_with_embargo(df, train_end, val_end, embargo_sessions=8):
    """
    Split with embargo gap at each boundary.
    Embargo is in trading sessions (actual dates in the data), not calendar days.

    The leakage direction is one-way: the last `rb` rows of the earlier split
    have labels depending on the next split's prices. The first rows of the
    later split have labels depending on *past* prices, which is fine (labels
    are allowed to use any data). Purging both sides is over-conservative but
    safe -- at the cost of losing ~2*embargo_sessions of data per boundary.
    If data is scarce, only purge the earlier side.
    """
    dates = sorted(df['date'].unique())

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    # Find the actual trading dates closest to boundaries
    train_end_idx = np.searchsorted(dates, train_end_ts, side='right') - 1
    val_end_idx = np.searchsorted(dates, val_end_ts, side='right') - 1

    # Embargo: drop embargo_sessions on each side of each boundary
    train_cutoff = dates[max(0, train_end_idx - embargo_sessions)]
    val_start = dates[min(len(dates) - 1, train_end_idx + embargo_sessions + 1)]
    val_cutoff = dates[max(0, val_end_idx - embargo_sessions)]
    test_start = dates[min(len(dates) - 1, val_end_idx + embargo_sessions + 1)]

    train = df[df['date'] <= train_cutoff]
    val = df[(df['date'] >= val_start) & (df['date'] <= val_cutoff)]
    test = df[df['date'] >= test_start]

    print(f"Train: {train['date'].min()} to {train['date'].max()}")
    print(f"  (embargo: {embargo_sessions} sessions purged before boundary)")
    print(f"Val:   {val['date'].min()} to {val['date'].max()}")
    print(f"  (embargo: {embargo_sessions} sessions purged on each side)")
    print(f"Test:  {test['date'].min()} to {test['date'].max()}")

    return train, val, test
```

## 4.3 Rolling Features Across Split Boundaries

Backward-looking rolling features (e.g., 252-day rolling max) computed *before* splitting will use pre-split (train) data when computing early val/test rows. **This is NOT leakage** -- it's exactly what happens in production: early validation rows *should* use earlier history, because that history would be available in real-time.

The concern only arises if you compute rolling stats using *future* data within the same split (e.g., a centered rolling window on features, which you should never do).

If you compute features *after* splitting instead, carry a **warm-up history buffer** from pre-split dates so early rows still get valid rolling computations, then score only post-split rows.

## 4.4 Train/Val/Test Structure Details

```python
def create_train_val_test_splits(df):
    train_end = '2021-12-31'
    val_end = '2022-12-31'

    train_df, val_df, test_df = temporal_split_with_embargo(
        df, train_end, val_end, embargo_sessions=8
    )

    # DON'T check "all stocks appear in all splits" -- that's wrong.
    # IPOs and delistings mean many names should only exist in some splits.
    # The right check is point-in-time eligibility.
    print("Stocks in train:", train_df['stock_id'].nunique())
    print("Stocks in val:", val_df['stock_id'].nunique())
    print("Stocks in test:", test_df['stock_id'].nunique())

    print(f"\nLabel distribution:")
    print(f"Train: {(train_df['label']==1).sum()} positives / {len(train_df)} total")
    print(f"Val:   {(val_df['label']==1).sum()} positives / {len(val_df)} total")
    print(f"Test:  {(test_df['label']==1).sum()} positives / {len(test_df)} total")

    return train_df, val_df, test_df
```

---

## 4.5 Walk-Forward Validation (Advanced Alternative)

### DO (if you want to be extra rigorous):

```python
def walk_forward_validation(df, train_years=5, test_years=1, embargo_sessions=8):
    """
    More realistic: Retrain model periodically.

    Split 1: Train 2015-2019, Test 2020
    Split 2: Train 2016-2020, Test 2021
    ...
    Average performance across all splits.
    """
    results = []
    start_year = df['date'].min().year
    end_year = df['date'].max().year

    for test_year in range(start_year + train_years, end_year + 1):
        train_start = test_year - train_years
        train_end_date = f'{test_year - 1}-12-31'
        test_start_date = f'{test_year}-01-01'

        # Embargo on both sides of the boundary (in trading sessions)
        dates = sorted(df['date'].unique())
        boundary_idx = np.searchsorted(dates, pd.Timestamp(train_end_date), side='right') - 1
        train_cutoff = dates[max(0, boundary_idx - embargo_sessions)]
        test_start = dates[min(len(dates) - 1, boundary_idx + embargo_sessions + 1)]

        train = df[(df['date'].dt.year >= train_start) &
                   (df['date'] <= train_cutoff)]
        test = df[(df['date'] >= test_start) &
                  (df['date'].dt.year == test_year)]

        model = train_model(train)
        metrics = evaluate_model(model, test)
        results.append({'test_year': test_year, 'metrics': metrics})

    return results
```

**Why walk-forward is better:**
- Tests model on multiple future periods
- Accounts for regime changes (bull/bear markets)
- More conservative performance estimate
- Gives variance estimates across periods

---

## 4.6 Decision-Time / Execution-Time Definition (CRITICAL)

**Biggest missing piece in many pipelines.** If you use daily close/high/low/final volume as features, the prediction is only actionable *after the session closes* or at next open. If you want to trade *before* the close, those features are illegal.

Define clearly:
- **Decision time:** When does the model make a prediction? (e.g., after market close)
- **Execution time:** When does the trade happen? (e.g., next market open)

```python
# If decision_time = "after close on day T":
# - Features can use OHLCV of day T and earlier
# - Execution happens at open of day T+1
# - Returns should be measured from open T+1

# If decision_time = "before close on day T":
# - Features can ONLY use data up to day T-1
# - Close/High/Low of day T are NOT available
```

---

## 4.7 Feature Matrix Preparation

```python
def prepare_feature_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    y = df['label'].copy()
    metadata = df[['date', 'stock_id', 'close']].copy()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")

    return X, y, metadata
```

---

## 4.8 Complete Pipeline Example

```python
def complete_data_pipeline(raw_data_path):
    # Step 1: Load raw data
    df = pd.read_parquet(raw_data_path)
    verify_raw_data_structure(df)

    # Step 2: Sort for per-stock computation
    df = df.sort_values(['stock_id', 'date']).reset_index(drop=True)

    # Step 3: Feature engineering (all within groupby)
    df = create_ml_features(df)

    # Step 4: Create labels (per stock, centered window)
    df = create_labels_with_boundary_handling(df, lb=8, rb=8)

    # Step 5: Handle NaNs
    df = handle_feature_nans(df)

    # Step 6: Temporal split WITH embargo
    train_df, val_df, test_df = temporal_split_with_embargo(
        df, '2021-12-31', '2022-12-31', embargo_sessions=8
    )

    # Step 7: Prepare feature matrices
    X_train, y_train, meta_train = prepare_feature_matrix(train_df, feature_cols)
    X_val, y_val, meta_val = prepare_feature_matrix(val_df, feature_cols)
    X_test, y_test, meta_test = prepare_feature_matrix(test_df, feature_cols)

    return {
        'X_train': X_train, 'y_train': y_train, 'meta_train': meta_train,
        'X_val': X_val, 'y_val': y_val, 'meta_val': meta_val,
        'X_test': X_test, 'y_test': y_test, 'meta_test': meta_test,
        'feature_cols': feature_cols,
    }
```

---

## 4.9 Data Storage Recommendations

### Option 1: Single File (Small to Medium Dataset)
```python
df.to_parquet('data/processed_stock_data.parquet', compression='snappy')
```

### Option 2: Pre-Split Files (Large Dataset)
```python
train_df.to_parquet('data/train.parquet')
val_df.to_parquet('data/val.parquet')
test_df.to_parquet('data/test.parquet')
```

### Option 3: Numpy Arrays (For Training Only)
```python
np.save('data/X_train.npy', X_train.values)
np.save('data/y_train.npy', y_train.values)
```

---

# PHASE 5: HANDLING IMBALANCE (1:250)

## 5.1 Strategy

### DO: Use these techniques (pick what works, don't stack blindly)

**1. Class weights (simplest, start here):**
```python
# Compute dynamically from the data, don't hardcode
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
class_weight_ratio = neg_count / pos_count

# For gradient boosting frameworks:
# scale_pos_weight=class_weight_ratio (or similar parameter name)
# For sklearn estimators:
# class_weight={0: 1, 1: class_weight_ratio}
```

**IMPORTANT:** Class weights affect probability calibration. Models trained with large class weights produce uncalibrated probability estimates. If you use probabilities for ranking, sizing, or threshold tuning, you **must** calibrate on validation data afterward (see Phase 7.3).

**2. Random oversampling (if you need more positive samples):**
```python
from imblearn.over_sampling import RandomOverSampler

# Duplicates real positive samples -- preserves actual market states
ros = RandomOverSampler(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

**Why NOT SMOTE:** SMOTE creates synthetic samples by interpolating between neighbors. For time-series financial data, this generates feature combinations that never occurred in markets -- non-physical states. Stick to duplicating real samples or using class weights.

**3. Threshold tuning (post-training):**
```python
from sklearn.metrics import precision_recall_curve

def tune_threshold(model, X_val, y_val, target_recall=0.6):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

    # precision_recall_curve returns precision/recall of length n+1
    # but thresholds of length n. The last precision/recall entry has
    # no corresponding threshold, so slice it off.
    recall_at_thresholds = recall[:-1]
    precision_at_thresholds = precision[:-1]

    idx = np.argmin(np.abs(recall_at_thresholds - target_recall))
    best_threshold = thresholds[idx]

    print(f"Best threshold: {best_threshold:.4f}")
    print(f"  Precision: {precision_at_thresholds[idx]:.3f}")
    print(f"  Recall: {recall_at_thresholds[idx]:.3f}")

    return best_threshold
```

**4. Undersampling + bagging (valid for extreme imbalance):**

Contrary to the naive "don't undersample" advice, undersampling the majority class inside a bagging ensemble works well. Each base estimator sees a balanced subsample, and the ensemble averages out the information loss.

```python
# BalancedRandomForest or EasyEnsemble approaches
# Each base learner trains on all positives + a random subsample of negatives
# Ensemble aggregates predictions from many such learners
```

### DO NOT:

**1. Don't use accuracy as metric:**
```python
# WRONG - useless with imbalance
accuracy = (predictions == labels).sum() / len(labels)
# Can get 99.6% by predicting all zeros

# RIGHT - use precision, recall, F1, AP
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

**2. Don't combine class weights AND oversampling without careful calibration:**
```python
# WRONG - double-corrects for imbalance, model massively over-predicts positives
model = SomeClassifier(scale_pos_weight=250)
X_resampled, y_resampled = RandomOverSampler().fit_resample(X, y)
model.fit(X_resampled, y_resampled)

# RIGHT - pick ONE approach, or calibrate carefully if combining
```

**3. Don't assume default threshold (0.5) works:**
```python
# WRONG
predictions = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
# With 1:250 imbalance, probabilities are often << 0.5

# RIGHT - tune threshold on validation data
predictions = (model.predict_proba(X)[:, 1] > tuned_threshold).astype(int)
```

**4. Don't use `model.predict()` for optimization objectives:**
```python
# WRONG - predict() uses fixed 0.5 threshold, returns ~all zeros at 1:250
val_pred = model.predict(X_val)
score = f1_score(y_val, val_pred)  # F1 ~ 0

# RIGHT - use predict_proba and tune threshold, or use a ranking metric
y_proba = model.predict_proba(X_val)[:, 1]
score = average_precision_score(y_val, y_proba)  # threshold-free
```

---

# PHASE 6: MODEL TRAINING

## 6.1 Model Selection Philosophy

### DO: Start Simple, Add Complexity Only If Needed

**Recommended progression (all model choices are optional):**

```
Level 1 (Start Here): Gradient boosting (e.g., LightGBM, XGBoost, CatBoost)
  Pros: Fast, handles imbalance well, interpretable, strong on tabular data
  Good starting point for structured/tabular financial features

Level 2 (If Level 1 plateaus): Ensemble of multiple model types
  Combine different algorithms (stacking, blending)
  Different models make different errors

Level 3 (Only if needed): Sequence models (LSTM, Transformer)
  Pros: Learns temporal patterns directly
  Cons: Slow, hard to train, needs lots of data
```

**Baselines (always establish these first):**
```python
# Before any ML model, establish naive baselines:
# 1. RSI < 30 (simple oversold signal)
# 2. Drawdown > X% + volume spike
# 3. Logistic regression on top features
# 4. Class-prior dummy (predicts positive at base rate)
#
# If your ML model can't beat these, the features or pipeline have issues.
```

**Example starter model:**
```python
# This is ONE possible starting point -- adapt to your needs
import lightgbm as lgb

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()

model = lgb.LGBMClassifier(
    objective='binary',
    scale_pos_weight=neg_count / pos_count,
    max_depth=7,
    num_leaves=63,
    learning_rate=0.01,
    n_estimators=5000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)
```

### DO NOT:

**1. Don't start with complex models before establishing baselines.**

**2. Don't train without validation set:**
```python
# WRONG
model.fit(X_train, y_train)  # No validation, no early stopping

# RIGHT
model.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         ...)  # With early stopping
```

**3. Don't ignore feature importance:**
```python
# After training, ALWAYS check feature importance
# If top feature is "day_of_week" -> something is wrong
# If top features are RSI, volume_z, drawdown -> good sign
```

---

## 6.2 Preventing Overfitting

### DO:

**1. Use regularization** (specifics depend on model choice):
- Limit tree depth / number of leaves
- Minimum samples per leaf
- L1/L2 regularization
- Row and column subsampling
- Early stopping on validation metric

**2. Monitor train vs validation performance:**
```python
def check_overfitting(train_metric, val_metric, metric_name='F1'):
    gap = train_metric - val_metric
    print(f"Train {metric_name}: {train_metric:.3f}")
    print(f"Val {metric_name}:   {val_metric:.3f}")
    print(f"Gap:      {gap:.3f}")

    if gap > 0.10:
        print("WARNING: Large gap - overfitting detected")
    else:
        print("No obvious overfitting")
```

### DO NOT:

**1. Don't tune on test set:**
```python
# WRONG
for param in param_grid:
    score = evaluate(model, X_test, y_test)  # Indirect overfitting on test

# RIGHT
for param in param_grid:
    score = evaluate(model, X_val, y_val)  # Test set only used ONCE at end
```

**2. Don't add features without validation:**
- Add features based on domain knowledge
- Check if val metric improves (not test)
- Run feature ablation: remove each feature block and measure impact

---

# PHASE 7: EVALUATION

## 7.0 What "Good" Actually Means

Standard classification metrics don't capture what you actually care about. You don't care if the model identifies the *exact* pivot low. You care:

1. Did we buy near a low?
2. Did price go up after we bought?
3. Did we avoid buying into continued downtrends (catching falling knives)?

**Don't optimize classification accuracy. Optimize: "when this model says buy, do I make money?"**

Precision/recall are proxies. Forward returns are the truth. A model with 10% precision but 5% avg 10d return on signals crushes a model with 25% precision but 0.5% avg return.

### Tiered Evaluation Framework

The tiers are NOT different loss functions. They're a **selection funnel** applied after training:

```
Model finishes training
    |
    +-- Tier 1: Check val PR-AUC, precision, recall
    |   +-- AP < 0.05? -> trash, don't bother continuing
    |   +-- AP >= 0.05? -> proceed
    |
    +-- Tier 2: Compute forward returns on val set
    |   +-- Mean 10d return < 0%? -> trash
    |   +-- Win rate < 50%? -> trash
    |   +-- Profitable? -> proceed
    |
    +-- Tier 3: Composite score on val set
        +-- Rank all surviving models by risk-adjusted score
            +-- Pick best -> run ONCE on test set -> final verdict
```

**During training:** The model optimizes its own loss (e.g., binary cross-entropy with class weights). You don't touch this. Early stopping monitors val AP (Tier 1). This is automatic.

**After training completes (once per model variant):** Run the tiered evaluation funnel below.

### Tier 1 -- Classification (Sanity Check Only)

Precision, recall, AP. Just to confirm the model isn't random. Not the final judge.

### Tier 2 -- Forward Returns (The Real Test)

For every positive prediction, measure the return N days later:

```python
def forward_return_eval(df, y_pred, horizons=[5, 10, 20]):
    """
    The metric that actually matters.
    Uses next-open entry to match decision-time rule.
    """
    df = df.copy()
    df['pred'] = y_pred

    buy_signals = df[df['pred'] == 1]
    results = {}

    for h in horizons:
        # Entry at next open, exit at open h days later
        entry = df.groupby('stock_id')['open'].shift(-1)
        exit_ = df.groupby('stock_id')['open'].shift(-(h + 1))
        fwd_ret = exit_ / entry - 1

        signal_returns = fwd_ret.loc[buy_signals.index].dropna()

        wins = signal_returns[signal_returns > 0]
        losses = signal_returns[signal_returns < 0]

        results[f'mean_return_{h}d'] = signal_returns.mean()
        results[f'win_rate_{h}d'] = (signal_returns > 0).mean()
        results[f'avg_win_{h}d'] = wins.mean() if len(wins) > 0 else 0
        results[f'avg_loss_{h}d'] = losses.mean() if len(losses) > 0 else 0
        results[f'profit_factor_{h}d'] = (
            wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
        )

    return results
```

**What good looks like:**
- Mean 10d forward return: >2% (vs ~0.4% for random days)
- Win rate at 10d: >60%
- Profit factor: >1.5
- Max drawdown on any single signal: manageable

### Tier 3 -- Penalty for Bad Behavior

This is what most people miss. Define a composite score that rewards good entries and penalizes blowups:

```python
def composite_score(df, y_pred):
    """
    Risk-adjusted composite score. Penalizes falling knives hard.
    """
    df = df.copy()
    df['pred'] = y_pred

    # Forward return at next open -> open 10 days later
    entry = df.groupby('stock_id')['open'].shift(-1)
    exit_ = df.groupby('stock_id')['open'].shift(-11)
    df['fwd_10d'] = exit_ / entry - 1

    signal_returns = df.loc[df['pred'] == 1, 'fwd_10d'].dropna()

    if len(signal_returns) == 0:
        return float('-inf')

    # Reward
    mean_return = signal_returns.mean()
    win_rate = (signal_returns > 0).mean()

    # Penalty: how bad are the losses?
    worst_decile = signal_returns.quantile(0.1)  # Bottom 10% of trades

    # Penalty: catching falling knives
    # Did price keep dropping >5% after our "bottom" call?
    knife_rate = (signal_returns < -0.05).mean()

    score = (
        0.4 * mean_return * 100       # reward good entries
        + 0.3 * win_rate              # reward consistency
        - 0.2 * abs(worst_decile)*100 # penalize blowups
        - 0.1 * knife_rate * 100      # penalize falling knives
    )
    return score
```

### Practical Example -- Hyperparameter Selection

```
Model A: max_depth=5, lr=0.01
  Tier 1: AP=0.08    pass
  Tier 2: mean 10d return=1.2%, win_rate=55%    pass
  Tier 3: composite=3.2

Model B: max_depth=7, lr=0.01
  Tier 1: AP=0.12    pass
  Tier 2: mean 10d return=2.8%, win_rate=62%    pass
  Tier 3: composite=5.1    <-- winner

Model C: max_depth=9, lr=0.05
  Tier 1: AP=0.15    pass (looks great!)
  Tier 2: mean 10d return=-0.3%, win_rate=48%    FAIL
  -> High precision but picks bottoms that don't recover. Killed here.
```

Model C is the important case -- great classification metrics, terrible trading outcome. Without Tier 2 you'd have shipped it.

### Tier 4 -- RL (Future Work)

Different paradigm entirely. Instead of "classify this row as bottom or not," the agent learns a policy: "given current market state, should I buy, hold, or do nothing?" The reward function directly encodes profit/loss. No labels needed -- the agent learns from outcomes.

Example reward structure:
```
+1.0  if 10d forward return > 3%
+0.3  if 10d forward return 0-3%
-0.5  if 10d forward return 0 to -3%
-2.0  if 10d forward return < -3%   (asymmetric -- punish knives hard)
-0.01 per day with no signal        (small cost for inaction)
```

Asymmetric penalties are key. Buying a falling knife is worse than missing a bottom.

**Skip Tier 4 for now.** It's a separate project. Stick with supervised learning (Tiers 1-3).

---

## 7.1 Metrics That Matter

### DO: Use Multiple Metrics

```python
def comprehensive_evaluation(model, X, y, threshold):
    """
    Complete evaluation for imbalanced classification.
    IMPORTANT: threshold must be provided (tuned on validation set).
    Do NOT tune threshold on the same data you evaluate -- that leaks info.
    Use tune_threshold() on validation data first, then pass the frozen
    threshold here for both val and test evaluation.
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_curve, roc_auc_score,
        average_precision_score, brier_score_loss
    )

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y, y_pred, target_names=['Not Bottom', 'Bottom']))

    print("=" * 60)
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
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

    # average_precision_score computes Average Precision (AP): the weighted
    # mean of precisions at each threshold, where the weight is the increase
    # in recall. This is NOT trapezoidal area under the PR curve -- sklearn
    # explicitly documents that AP and trapezoidal PR-AUC differ.
    ap_score = average_precision_score(y, y_pred_proba)

    print(f"Precision: {precision_val:.3f}")
    print(f"Recall:    {recall_val:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y, y_pred_proba):.3f}")
    print(f"AP (Average Precision): {ap_score:.3f}")
    print(f"Brier Score: {brier_score_loss(y, y_pred_proba):.4f}")
    print(f"Threshold used: {threshold:.4f}")

    # Precision @ Top K
    top_k_pct = 0.01
    k = int(len(y) * top_k_pct)
    top_k_indices = np.argsort(y_pred_proba)[-k:]
    precision_at_k = y.iloc[top_k_indices].sum() / k

    print(f"\nPrecision @ Top 1%: {precision_at_k:.3f}")

    return {
        'precision': precision_val,
        'recall': recall_val,
        'f1': f1,
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'ap': ap_score,
        'brier': brier_score_loss(y, y_pred_proba),
        'precision_at_1pct': precision_at_k,
        'threshold': threshold,
    }
```

**What good performance looks like:**
```
For 1:250 imbalance, realistic expectations:

Baseline (predict all 0): F1 = 0.00
Random guessing:          F1 ~ 0.01

Good model:
  F1 Score:        0.15 - 0.25
  Precision:       0.10 - 0.20
  Recall:          0.40 - 0.60
  Precision@1%:    0.30 - 0.50
  AP:              0.15 - 0.30
```

### DO NOT:

**1. Don't rely on accuracy** (99.6% by predicting all zeros).

**2. Don't use ROC-AUC as primary metric** -- misleading for severe imbalance. Use AP (Average Precision) or precision/recall-based metrics.

**3. Don't confuse AP with PR-AUC.** `average_precision_score()` computes Average Precision (weighted mean of precisions at each threshold), which approximates but is not identical to the area under the PR curve.

---

## 7.2 Per-Date Ranking Metrics

Global `precision@1%` can be distorted by crash clusters (many bottoms on the same dates inflate the score). Also report daily/top-K metrics:

```python
def per_date_metrics(df, y_pred_proba, y_true, top_k=10):
    """
    Report precision of top-K predictions per day, averaged across days.
    """
    df = df.copy()
    df['pred_proba'] = y_pred_proba
    df['label'] = y_true

    daily_precision = []
    for date, group in df.groupby('date'):
        if len(group) < top_k:
            continue
        top_k_idx = group['pred_proba'].nlargest(top_k).index
        prec = group.loc[top_k_idx, 'label'].mean()
        daily_precision.append(prec)

    avg_daily_prec = np.mean(daily_precision)
    print(f"Avg daily precision@{top_k}: {avg_daily_prec:.3f}")
    print(f"  (averaged across {len(daily_precision)} trading days)")
    return avg_daily_prec
```

---

## 7.3 Probability Calibration

Models trained with class weights or oversampling produce **uncalibrated** probability estimates. A "0.8 probability" does not mean ~80% chance of being a bottom. For trading signals, well-calibrated probabilities matter.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator

def calibrate_and_evaluate(model, X_val, y_val, X_test, y_test):
    """
    Calibrate probabilities using validation data, evaluate on test.
    X_val/y_val must be disjoint from the data used to fit `model`.
    """
    # Wrap the already-fitted model so CalibratedClassifierCV doesn't refit it.
    # FrozenEstimator prevents refitting during internal CV.
    # Option 1: Platt scaling (logistic regression on raw scores)
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method='sigmoid'
    )
    calibrated.fit(X_val, y_val)

    # Option 2: Isotonic regression (more flexible, needs more data)
    # calibrated = CalibratedClassifierCV(
    #     estimator=FrozenEstimator(model),
    #     method='isotonic'
    # )

    # Evaluate calibration with reliability curve
    y_proba_raw = model.predict_proba(X_test)[:, 1]
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]

    # Brier score (lower is better, proper scoring rule)
    from sklearn.metrics import brier_score_loss
    print(f"Brier score (raw):        {brier_score_loss(y_test, y_proba_raw):.4f}")
    print(f"Brier score (calibrated): {brier_score_loss(y_test, y_proba_cal):.4f}")

    # Plot reliability curve
    fraction_pos_raw, mean_pred_raw = calibration_curve(y_test, y_proba_raw, n_bins=10)
    fraction_pos_cal, mean_pred_cal = calibration_curve(y_test, y_proba_cal, n_bins=10)

    return calibrated
```

---

## 7.4 Economic Evaluation / Backtest

Classification metrics alone don't tell you if the model is tradeable. Add a simple backtest:

```python
def simple_backtest(df_test, y_pred_proba, threshold, hold_days=10):
    """
    If you buy when model predicts bottom, what's the return after N days?

    Assumes decision_time = "after close on day T", so:
    - Entry at open of T+1
    - Exit at open of T+1+hold_days
    This matches the decision-time rule: features use day T's OHLCV,
    but execution is next open.
    """
    df = df_test.copy()
    df['pred_proba'] = y_pred_proba
    df['signal'] = (df['pred_proba'] > threshold).astype(int)

    # Forward return: buy at next open, sell at open hold_days later
    # (computed on test set, NOT used in training)
    entry_price = df.groupby('stock_id')['open'].shift(-1)              # open at T+1
    exit_price = df.groupby('stock_id')['open'].shift(-(hold_days + 1)) # open at T+1+hold_days
    df['fwd_return'] = exit_price / entry_price - 1

    signals = df[df['signal'] == 1]
    signals_with_return = signals.dropna(subset=['fwd_return'])
    signals_no_return = len(signals) - len(signals_with_return)

    if len(signals) == 0:
        print("No signals generated")
        return

    print(f"Backtest Results ({hold_days}-day hold, entry at next open):")
    print(f"  Total signals: {len(signals)}")
    if signals_no_return > 0:
        print(f"  Signals without forward return (near end of test): {signals_no_return}")
    print(f"  Avg return per trade: {signals_with_return['fwd_return'].mean():.3%}")
    print(f"  Median return: {signals_with_return['fwd_return'].median():.3%}")
    print(f"  Hit rate (>0%): {(signals_with_return['fwd_return'] > 0).mean():.1%}")
    print(f"  Worst trade: {signals_with_return['fwd_return'].min():.3%}")
    print(f"  Best trade: {signals_with_return['fwd_return'].max():.3%}")

    # Per-period stats
    signals_per_day = signals.groupby('date').size()
    print(f"  Avg signals/day: {signals_per_day.mean():.1f}")
    print(f"  Max concurrent: {signals_per_day.max()}")
```

**What to report:**
- Avg trade return, hit rate
- Turnover, max concurrent positions
- Slippage/spread assumptions
- Max drawdown of the strategy
- Sharpe/Sortino ratio
- Capacity estimate

---

## 7.5 Slice Stability

Don't just report one aggregate test score. Report metrics by:
- **Year** (does model degrade over time?)
- **Market regime** (bull/bear/sideways)
- **Sector** (does it only work for tech?)
- **Market cap bucket** (large/mid/small)
- **Liquidity bucket**

```python
def slice_metrics(df_test, y_pred_proba, y_true, slice_col):
    """
    Report AP by slice.
    """
    df = df_test.copy()
    df['pred_proba'] = y_pred_proba
    df['label'] = y_true

    for name, group in df.groupby(slice_col):
        if group['label'].sum() < 5:
            continue
        ap = average_precision_score(group['label'], group['pred_proba'])
        n_pos = group['label'].sum()
        print(f"  {name}: AP={ap:.3f} (n_pos={n_pos})")
```

---

## 7.6 Target / Trade Alignment

Exact-day pivot labels are noisy. Consider adding:

1. **Event-tolerant metric:** A prediction within +/-1-3 days of a bottom counts as correct.
2. **Trading label:** "Buying at next open after signal leads to acceptable 5/10/20-day return with bounded drawdown."

```python
def event_tolerant_precision(df, y_pred, tolerance_days=2):
    """
    A prediction is 'correct' if there's a true bottom within +/- tolerance_days
    for the SAME stock. Must be computed per-stock using date distance,
    not row index distance (row indices cross stock boundaries in pooled data).
    """
    df = df.copy()
    df['pred'] = y_pred

    correct = 0
    total_preds = 0

    for stock_id, group in df.groupby('stock_id'):
        group = group.sort_values('date')
        pred_dates = group.loc[group['pred'] == 1, 'date'].values
        true_dates = group.loc[group['label'] == 1, 'date'].values

        if len(pred_dates) == 0:
            continue

        total_preds += len(pred_dates)
        for pd_date in pred_dates:
            # Check date distance, not row distance
            day_diffs = np.abs((true_dates - pd_date) / np.timedelta64(1, 'D'))
            if len(day_diffs) > 0 and np.min(day_diffs) <= tolerance_days:
                correct += 1

    precision = correct / total_preds if total_preds > 0 else 0
    print(f"Event-tolerant precision (+/-{tolerance_days} days): {precision:.3f}")
    print(f"  ({correct}/{total_preds} predictions near a true bottom)")
    return precision
```

---

## 7.7 Visualization

### DO: Visualize model decisions

```python
def plot_model_predictions(df, model, feature_cols, stock_id, start_date, end_date):
    stock_data = df[(df['stock_id'] == stock_id) &
                    (df['date'] >= start_date) &
                    (df['date'] <= end_date)]

    X = stock_data[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    ax1.plot(stock_data['date'], stock_data['close'], label='Price', linewidth=2)

    actual_bottoms = stock_data[stock_data['label'] == 1]
    ax1.scatter(actual_bottoms['date'], actual_bottoms['close'],
               color='green', s=100, label='Actual Bottom', zorder=5, marker='o')

    threshold = 0.02  # Use tuned threshold, not 0.5
    predicted_bottoms = stock_data[probabilities > threshold]
    ax1.scatter(predicted_bottoms['date'], predicted_bottoms['close'],
               color='red', s=100, label='Predicted Bottom', zorder=5, marker='x')

    ax1.set_ylabel('Price')
    ax1.set_title(f'{stock_id} - Bottom Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(stock_data['date'], probabilities, label='Bottom Probability', color='purple')
    ax2.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
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
- Predicted bottoms near actual bottoms
- High probability spikes at actual bottoms
- NOT random predictions scattered everywhere
- NOT high probability during uptrends (model confused)

---

# PHASE 8: HYPERPARAMETER TUNING

## 8.1 Systematic Approach

### DO: Use Bayesian optimization or structured search

```python
import optuna

def bayesian_tuning(X_train, y_train, X_val, y_val):
    """
    Use Bayesian optimization to find hyperparameters.
    Model-agnostic framework -- adapt param space to your model.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    def objective(trial):
        # Example param space for gradient boosting -- adapt as needed
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': neg_count / pos_count,
        }

        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50)],
                 verbose=False)

        # Use AP (threshold-free) as objective, NOT F1 with default threshold
        y_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_proba)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters:", study.best_params)
    print("Best AP score:", study.best_value)

    return study.best_params
```

### DO NOT:

**1. Don't tune on test set.**

**2. Don't use F1 with `model.predict()` as the tuning objective:**
```python
# WRONG -- predict() uses 0.5 threshold, returns all zeros at 1:250
val_pred = model.predict(X_val)
return f1_score(y_val, val_pred)  # Always ~0

# RIGHT -- use threshold-free metric
y_proba = model.predict_proba(X_val)[:, 1]
return average_precision_score(y_val, y_proba)
```

**3. Don't spend forever tuning:**
```
Diminishing returns:
Default params:     AP = 0.18
1 hour tuning:      AP = 0.20  (+0.02)
10 hours tuning:    AP = 0.21  (+0.01)
100 hours tuning:   AP = 0.215 (+0.005)

Better use of time: Improve features, fix data quality, add baselines.
```

---

## 8.2 Model Ensembling

Ensembling different model types is low-hanging fruit once individual models are tuned:

```python
def simple_ensemble(models, X, weights=None):
    """
    Average probabilities from multiple models.
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    proba = np.zeros(len(X))
    for model, w in zip(models, weights):
        proba += w * model.predict_proba(X)[:, 1]

    return proba

# For more sophisticated ensembling:
# - Stacking with a simple logistic meta-learner
# - Blend predictions on validation set to find optimal weights
# - Use diverse model types (different algorithms make different errors)
```

---

# PHASE 9: FINAL CHECKLIST BEFORE DEPLOYMENT

## Pre-Deployment Audit

```python
def pre_deployment_audit(model, df_train, df_val, df_test, feature_cols, all_trading_dates):
    """
    all_trading_dates: sorted list of ALL trading dates from the original
    (pre-split) dataframe. Must be passed in because the splits have the
    embargo dates removed -- counting gaps from the splits themselves
    would always show 0.
    """
    print("=" * 80)
    print("PRE-DEPLOYMENT AUDIT")
    print("=" * 80)

    # 1. Temporal consistency
    print("\n1. CHECKING TEMPORAL ORDERING...")
    assert df_train['date'].max() < df_val['date'].min(), "Train/val overlap!"
    assert df_val['date'].max() < df_test['date'].min(), "Val/test overlap!"
    print("OK: Temporal splits correct")

    # 2. Embargo gap (count in trading sessions, not calendar days)
    # Use the original date list, not the splits (embargo removes those dates)
    print("\n2. CHECKING EMBARGO GAP...")
    gap_tv_sessions = len([d for d in all_trading_dates
                           if df_train['date'].max() < d < df_val['date'].min()])
    gap_vt_sessions = len([d for d in all_trading_dates
                           if df_val['date'].max() < d < df_test['date'].min()])
    print(f"Train-Val gap: {gap_tv_sessions} trading sessions")
    print(f"Val-Test gap: {gap_vt_sessions} trading sessions")
    if gap_tv_sessions < 8 or gap_vt_sessions < 8:
        print("WARNING: Embargo gap < pivot window (8 sessions). Label leakage possible.")

    # 3. Survivorship bias
    print("\n3. CHECKING SURVIVORSHIP BIAS...")
    audit_survivorship_bias(df_train['stock_id'].unique().tolist())

    # 4. Feature distribution stability
    print("\n4. CHECKING FEATURE DISTRIBUTIONS...")
    check_feature_stability(df_train, df_val, feature_cols)

    # 5. Label quality
    print("\n5. CHECKING LABEL QUALITY...")
    train_imbalance = (df_train['label'] == 0).sum() / max((df_train['label'] == 1).sum(), 1)
    print(f"Train imbalance: 1:{train_imbalance:.0f}")

    # 6. Model performance
    # Tune threshold on val ONLY, then freeze it for test
    print("\n6. CHECKING MODEL PERFORMANCE...")
    threshold = tune_threshold(model, df_val[feature_cols], df_val['label'])
    metrics_val = comprehensive_evaluation(model, df_val[feature_cols], df_val['label'], threshold)
    metrics_test = comprehensive_evaluation(model, df_test[feature_cols], df_test['label'], threshold)

    if abs(metrics_val['f1'] - metrics_test['f1']) > 0.05:
        print("WARNING: Large performance gap between val and test")

    # 7. Feature importance sanity check
    print("\n7. CHECKING FEATURE IMPORTANCE...")
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:]
        print("Top 10 features:")
        for idx in reversed(top_indices):
            print(f"  {feature_cols[idx]}: {feature_importance[idx]:.3f}")

    # 8. Calibration check
    print("\n8. CHECKING PROBABILITY CALIBRATION...")
    y_proba = model.predict_proba(df_val[feature_cols])[:, 1]
    brier = brier_score_loss(df_val['label'], y_proba)
    print(f"Brier score (validation): {brier:.4f}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    return True


def check_feature_stability(df_train, df_val, feature_cols):
    """
    Use KS-test or PSI for distribution drift detection.
    Crude mean comparison is insufficient.
    """
    from scipy.stats import ks_2samp

    drifted = []
    for col in feature_cols:
        train_vals = df_train[col].dropna()
        val_vals = df_val[col].dropna()

        stat, pvalue = ks_2samp(train_vals, val_vals)
        if pvalue < 0.001:  # Significant drift
            drifted.append((col, stat, pvalue))

    if drifted:
        print(f"WARNING: {len(drifted)} features show significant distribution drift:")
        for col, stat, pval in sorted(drifted, key=lambda x: -x[1])[:5]:
            print(f"  {col}: KS={stat:.3f}, p={pval:.2e}")
    else:
        print("OK: No significant feature drift detected")
```

---

## Feature Staleness in Production

No discussion of how features get computed in real-time inference is complete without addressing:

- **Rolling windows need warm-up:** A 252-day rolling feature needs 252 days of history before it produces valid values. Your inference pipeline must maintain this history.
- **Cumulative features (OBV, etc.) need full history** from the stock's start.
- **Feature computation must match training exactly:** Same indicators, same parameters, same handling of missing data.

```
Production inference pipeline:
1. Maintain rolling buffer of last N days per stock (N = max lookback)
2. On new bar arrival, update buffer, compute features
3. Feed features to model, get probability
4. Apply tuned threshold -> signal
5. Log features + prediction for monitoring
```

---

# PHASE 10: WHAT NOT TO DO (SUMMARY)

## FATAL MISTAKES TO AVOID

### 1. Data Leakage (Most Common Failure)
```python
# Using future data in features
df['future_return'] = df['close'].pct_change(-10)

# Global normalization (sees test data)
scaler.fit(entire_dataset)

# Random train/test split (future in train)
train_test_split(df, shuffle=True)
```

### 2. Survivorship Bias
```python
# Training only on companies that exist today
stocks = get_current_sp500()

# Filtering based on current metrics
stocks = stocks[stocks['current_market_cap'] > 1e9]
```

### 3. Wrong Evaluation
```python
# Using accuracy with imbalance
score = accuracy_score(y_true, y_pred)

# Tuning on test set
for param in params:
    score = evaluate_on_test_set(param)

# Using F1(predict()) as tuning objective with extreme imbalance
score = f1_score(y_val, model.predict(X_val))  # Always ~0
```

### 4. Overfitting
```python
# Too many features vs positive samples
# No regularization
# Training without early stopping
```

### 5. Poor Feature Engineering
```python
# Using raw prices
features = ['close', 'open', 'high', 'low']

# Using volume without normalization
features = ['volume']

# Forward-filling prices (fabricates states)
df.ffill(limit=5)  # for OHLCV columns
```

### 6. No Embargo Gap
```python
# Splitting without purge gap at boundaries
# Label leakage from centered pivot window
```

### 7. Double-Correcting for Imbalance
```python
# Using SMOTE + scale_pos_weight simultaneously
# Massive over-prediction of positive class
```

---

# FINAL PHILOSOPHY SUMMARY

## Key Principles

1. **No future information in features** -- Every feature must be calculable at decision time.

2. **Include delisted companies** -- Reconstruct the point-in-time universe, not just current survivors.

3. **Split by time with embargo gaps** -- Train on past, test on future, with gaps >= pivot window.

4. **Use appropriate metrics** -- AP, precision, recall, Brier score; not accuracy. Tune thresholds on validation.

5. **Regularize** -- Limit complexity, use early stopping, monitor train-val gap.

6. **Validate before testing** -- Tune on validation, test only once at the end.

7. **Normalize per stock** -- Each stock is its own universe.

8. **Establish baselines first** -- Beat naive rules before claiming model success.

9. **Calibrate probabilities** -- Raw model scores are not probabilities, especially with class weights.

10. **Evaluate economically** -- Classification metrics alone are insufficient. Simulate trades.

11. **Report slice stability** -- Metrics by year, regime, sector, cap bucket. One aggregate score hides problems.

12. **If F1 > 0.40, audit carefully** -- Not necessarily wrong, but warrants investigation for leakage or overfitting. Treat as an audit trigger, not a law.

---

## Expected Outcomes

**Realistic expectations:**
```
Final test F1: 0.15 - 0.25 (good), 0.25 - 0.35 (excellent)
Precision: 10-20% (1 in 5-10 predictions correct)
Recall: 40-60% (catch 40-60% of bottoms)

Translation: Model identifies potential bottoms, but:
- Needs human verification
- Combine with other signals
- Must pass economic backtest
- Use for filtering, not blind trading
```

**This is a filtering tool, not a crystal ball.**
