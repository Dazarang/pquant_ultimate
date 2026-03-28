# Metrics Reference

## Composite Score (Ratchet Metric)

The single number the autoresearch loop optimizes. Higher = better.

```
score = 0.4 * mean_10d_return * 100
      + 0.3 * win_rate
      - 0.2 * |worst_decile| * 100
      - 0.1 * knife_rate * 100
```

## Tier 1: Classification (Sanity Check)

Must pass avg_precision > 0.05 or the iteration is rejected.

| Metric | Description |
|--------|-------------|
| **Precision** | Of predicted bottoms, what % were actual bottoms |
| **Recall** | Of actual bottoms, what % the model caught |
| **F1** | Harmonic mean of precision and recall |
| **ROC-AUC** | Ability to rank bottoms higher than non-bottoms (threshold-free) |
| **Avg Precision (AP)** | Area under precision-recall curve. Best single metric for imbalanced data |

## Tier 2: Forward Returns (The Real Test)

Must have mean 10d return > 0% or the iteration is rejected.

Entry: next-day open after signal. Exit: open N days later.

| Metric | Description |
|--------|-------------|
| **Mean Nd return** | Average return N trading days after buying. Measured at 5d, 10d, 20d |
| **Win rate Nd** | % of signals where the stock went up after N days |
| **Profit factor Nd** | Total wins / total losses. >1 profitable, >2 good |
| **N signals** | Number of buy signals with valid forward data |

## Tier 3: Composite Score Components

| Component | Weight | What it rewards/penalizes |
|-----------|--------|--------------------------|
| **Mean 10d return** | +40% | Profitable signals. Higher average return = better |
| **Win rate** | +30% | Consistency. More winners = better |
| **Worst decile** | -20% | Penalizes blowups. Average return of bottom 10% of trades |
| **Knife rate** | -10% | Penalizes falling knives. % of signals where 10d return < -5% |

## Backtest Metrics (backtest_quick)

Simulates actual trading with target/stop/max-hold rules.

| Metric | Description |
|--------|-------------|
| **Win rate** | % of trades with positive return |
| **Avg return** | Mean return across all trades |
| **Med return** | Median return (less sensitive to outliers) |
| **Exit reasons** | target (hit +10%), stop (hit -5%), max_hold (30 days, closed at close price) |

## Dataset Properties

| Property | Value |
|----------|-------|
| Date range | 2015-10-15 to 2026-01-07 |
| Stocks | 1,336 |
| Rows | ~3M |
| Features | 231 across 7 groups |
| Label | PivotLow (binary, ~3% positive, ~1:32 imbalance) |
| Embargo | 13 trading sessions at each split boundary |
