# Metrics Reference

## Multi-Budget Composite Score (Ratchet Metric)

The single number the autoresearch loop optimizes. Higher = better.

Evaluation is **threshold-free**. The model outputs probabilities; the judge selects signals at 6 budget levels and evaluates at 3 horizons.

**Budgets:** top 0.05%, 0.10%, 0.25%, 0.50%, 1.00%, 2.00% of predictions by probability
**Horizons:** 5d, 10d, 20d

Per (budget, horizon) cell:
```
raw = 0.50 * excess_return * 100         (alpha over equal-weight market)
    + 0.15 * (win_rate - 0.5) * 100      (consistency edge vs coin flip)
    - 0.15 * |worst_decile| * 100        (tail risk penalty)
    - 0.10 * knife_rate * 100            (falling knife penalty, >5% loss)
    - 0.10 * |mean_mae| * 100            (path risk: avg max adverse excursion)

W = sqrt(effective_n / (effective_n + 20))   (soft evidence scaling)
```

Final score = mean of W * raw across all 18 cells. Missing cells count as 0.

## Tier 1: Ranking Quality (Gate)

Must pass avg_precision > 0.05 or the iteration is rejected.

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Ability to rank bottoms higher than non-bottoms (threshold-free) |
| **Avg Precision (AP)** | Area under precision-recall curve. Summarizes precision-recall tradeoff |

## Tier 2: Multi-Budget Composite

The model is evaluated at 6 x 3 = 18 operating points. Each cell computes:

| Component | Weight | What it rewards/penalizes |
|-----------|--------|--------------------------|
| **Excess return** | +50% | Alpha over equal-weight market benchmark |
| **Win rate edge** | +15% | Consistency above 50% baseline (centered on coin flip) |
| **Worst decile** | -15% | Penalizes blowups. 10th percentile of trade returns |
| **Knife rate** | -10% | Penalizes falling knives. % of signals where 10d return < -5% |
| **Mean MAE** | -10% | Path risk. Average max adverse excursion during holding period |

Each cell is then scaled by W = sqrt(effective_n / (effective_n + 20)):
- N=1: W=0.22 (nearly zeroed)
- N=20: W=0.71
- N=100: W=0.91
- N=500: W=0.98

Soft evidence scaling; see W values above.

## Forward Return Metrics (per budget, per horizon)

Entry: next-day open after signal. Exit: open N days later.

| Metric | Description |
|--------|-------------|
| **Mean Nd return** | Average return N trading days after buying |
| **Excess Nd** | Signal return minus equal-weight market return over same window |
| **Win rate Nd** | % of signals where the stock went up after N days |
| **Profit factor Nd** | Total wins / total losses. >1 net positive, >2 wins outweigh losses 2:1 |
| **MAE Nd** | Mean max adverse excursion: avg worst drawdown during holding period |
| **Worst MAE Nd** | Single worst drawdown across all signals |
| **N signals** | Number of buy signals with valid forward data |
| **Effective N** | Unique signal dates (proxy for independent bets) |

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
| Label | PivotLow (binary, ~5% positive, ~1:20 imbalance) |
| Embargo | 13 trading sessions at each split boundary |
