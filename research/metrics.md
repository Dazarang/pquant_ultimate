# Metrics Reference

## Composite Score (Ratchet Metric)

The single number the autoresearch loop optimizes. Higher = better.

```
score = 0.30 * excess_return * 100         (alpha over equal-weight market)
      + 0.25 * (win_rate - 0.5) * 100      (consistency edge vs coin flip)
      - 0.20 * |worst_decile| * 100        (tail risk penalty)
      - 0.10 * knife_rate * 100            (falling knife penalty, >5% loss)
      - 0.15 * |mean_mae| * 100            (path risk: avg max adverse excursion)
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

At least one horizon (5d/10d/20d) must show positive excess return vs market, or the iteration is rejected.

Entry: next-day open after signal. Exit: open N days later.

| Metric | Description |
|--------|-------------|
| **Mean Nd return** | Average return N trading days after buying. Measured at 5d, 10d, 20d |
| **Excess Nd** | Signal return minus equal-weight market return over same window |
| **Win rate Nd** | % of signals where the stock went up after N days |
| **Profit factor Nd** | Total wins / total losses. >1 profitable, >2 good |
| **MAE Nd** | Mean max adverse excursion: avg worst drawdown during holding period |
| **Worst MAE Nd** | Single worst drawdown across all signals |
| **N signals** | Number of buy signals with valid forward data |
| **Effective N** | Unique signal dates (proxy for independent bets; clustered signals inflate raw N) |

## Tier 3: Composite Score Components

| Component | Weight | What it rewards/penalizes |
|-----------|--------|--------------------------|
| **Excess return** | +30% | Alpha over equal-weight market benchmark |
| **Win rate edge** | +25% | Consistency above 50% baseline (centered on coin flip) |
| **Worst decile** | -20% | Penalizes blowups. 10th percentile of trade returns |
| **Knife rate** | -10% | Penalizes falling knives. % of signals where 10d return < -5% |
| **Mean MAE** | -15% | Path risk. Average max adverse excursion during holding period |

After Tier 3, a **regime breakdown** (bull/bear based on trailing 20d market return) reports signal quality per market environment. Informational only, no gating.

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
