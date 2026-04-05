# Metrics Reference

## Multi-Budget Composite Score (Ratchet Metric)

The single number the autoresearch loop optimizes. Higher = better.

Evaluation is **threshold-free**. The model outputs probabilities; the judge selects signals at 5 budget levels and evaluates at 3 horizons.

The evaluator is **event-aware**:
- Tier 1 still evaluates row ranking on the expanded `PivotLow` label.
- Trading metrics collapse duplicate predictions inside the same true bottom event to one earliest tradable entry.
- False positives remain separate entries.
- Event diagnostics report both exact-center hits and buyable-zone coverage.

Historical composite scores from the earlier row-based evaluator are **not directly comparable** to this one, because the scoring unit and penalties changed.

**Budgets:** top 0.10%, 0.25%, 0.50%, 1.00%, 2.00% of predictions by probability
**Horizons:** 5d, 10d, 20d

Per (budget, horizon) cell:
```
raw = 0.40 * excess_return * 100         (alpha over equal-weight market)
    + 0.20 * (win_rate - 0.5) * 100      (consistency edge vs coin flip)
    - 0.10 * |worst_decile| * 100        (tail risk penalty)
    - 0.10 * knife_rate * 100            (falling knife penalty, >5% loss)
    - 0.05 * |tail_mae| * 100            (tail path risk: 25th percentile MAE)
    - 0.15 * entry_slippage * 100        (timing quality: gap to best buyable entry in event zone)

W = effective_n / (effective_n + 50)       (linear evidence scaling)
```

Final score = mean of W * raw across all valid cells (5 budgets x 3 horizons = 15 possible). Missing cells are excluded from the average, not counted as 0.

`tail_mae` = 25th percentile of per-signal MAE (worst quartile). Targets the truly bad entries without penalizing harmless path noise already captured by worst_decile and knife_rate. `entry_slippage` = gap between entry and best available price in the true bottom zone; 0 when no events are hit.

## Score Bands

Higher is better. Unbounded. Derived from the formula applied to representative per-cell metric profiles (W = 0.91):

| Score | Label | Typical per-cell profile |
|-------|-------|-------------------------|
| < -4 | no skill | ~0% excess, ~50% win, ~18% knife, ~9% worst decile |
| -4 to -3 | noise | Some signal but penalties dominate |
| -3 to -2 | weak | ~0.5% excess, ~52% win, ~15% knife, ~7% worst decile |
| -2 to -1 | moderate | ~1% excess, ~53% win, ~12% knife, ~6% worst decile |
| -1 to 0 | good | ~2% excess, ~55% win, ~10% knife, ~5% worst decile |
| 0 to 1 | strong | ~3% excess, ~57% win, ~8% knife, ~4% worst decile |
| > 1 | exceptional | >= 5% excess, >= 60% win, <= 5% knife |

A score of 0 means rewards and penalties are exactly balanced across all 15 cells on average. Positive scores indicate net positive risk-adjusted utility.

**Good model threshold: > -1.** A model scoring above -1 consistently delivers positive excess return with controlled downside across most budget/horizon combinations. This corresponds to approximately 2% excess return at 55% win rate with 10% knife rate.

## Label Mechanics

`PivotLow` is not a fixed `[-1, +1]` label.

For each stock:
- Find the base pivot center on `close` with `lb=8`, `rb=13`.
- Mark that center as `PivotLow_base = 1`.
- Expand to adjacent `-1` / `+1` rows **only** when that adjacent close is within `1%` of the base pivot close.
- Assign one `PivotLow_event_id` per base pivot and a signed `PivotLow_event_offset` (`-1`, `0`, `+1`) within the event zone.

This means a true bottom event can be:
- center only
- center plus previous day
- center plus next day
- all three

## Tier 1: Ranking Quality (Gate)

Must pass avg_precision > 0.05 or the iteration is rejected.

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Ability to rank bottoms higher than non-bottoms (threshold-free) |
| **Avg Precision (AP)** | Area under precision-recall curve on expanded `PivotLow` |
| **Base ROC-AUC** | Ranking quality on exact pivot centers only (`PivotLow_base`) |
| **Base AP** | Precision-recall on exact pivot centers only |

## Tier 2: Multi-Budget Composite

The model is evaluated at 5 x 3 = 15 operating points.

For each (budget, horizon) cell:
- select the top-budget rows by probability
- discard rows without a valid next-open to horizon-open window
- collapse duplicate selected rows inside the same true bottom event to one earliest tradable entry
- cap total credit per true event at 1.0 signal mass
- keep false positives as separate entries

The raw cell score then computes:

| Component | Weight | What it rewards/penalizes |
|-----------|--------|--------------------------|
| **Excess return** | +40% | Alpha over equal-weight market benchmark |
| **Win rate edge** | +20% | Consistency above 50% baseline (centered on coin flip) |
| **Worst decile** | -10% | Penalizes blowups. 10th percentile of trade returns |
| **Knife rate** | -10% | Penalizes falling knives. % of signals where 10d return < -5% |
| **Tail MAE** | -5% | Tail path risk. 25th percentile of max adverse excursion (worst quartile) |
| **Entry slippage** | -15% | Timing quality. Gap between entry price and best available in the event zone |

Each cell is then scaled by W = effective_n / (effective_n + 50):
- N=1: W=0.02
- N=20: W=0.29
- N=50: W=0.50
- N=100: W=0.67
- N=500: W=0.91

Linear evidence scaling; see W values above.

## Forward Return Metrics (per budget, per horizon)

Entry: next-day open after the evaluated row. Exit: open N days later.

| Metric | Description |
|--------|-------------|
| **Mean Nd return** | Average return N trading days after buying |
| **Excess Nd** | Signal return minus equal-weight market return over same window |
| **Win rate Nd** | % of signals where the stock went up after N days |
| **Profit factor Nd** | Total wins / total losses. >1 net positive, >2 wins outweigh losses 2:1 |
| **MAE Nd** | Mean max adverse excursion: avg worst drawdown during holding period |
| **Worst MAE Nd** | Single worst drawdown across all signals |
| **N signals** | Number of evaluated entries after event collapse |
| **Effective N** | Unique evaluated signal dates (proxy for independent bets) |
| **Event recall** | Fraction of true bottom events hit at least once |
| **Exact center recall** | Fraction of exact base pivot centers hit |
| **Zone precision** | Share of evaluated entry mass that lands inside a true bottom event |
| **Duplicate rows / mass** | Selected rows or signal mass wasted by repeated calls inside one event |
| **Avg entry offset** | Average signed timing relative to exact pivot center |
| **Avg entry slippage** | Average gap between chosen entry and the best next-open available inside that true event zone |

## Backtest Metrics (backtest_quick)

Simulates actual trading with target/stop/max-hold rules. Duplicate predictions inside the same true bottom event are collapsed to one earliest entry before backtesting.

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
| Label | `PivotLow` expanded buy zone, plus `PivotLow_base` / `PivotLow_event_id` / `PivotLow_event_offset` metadata in newly built datasets |
| Embargo | 13 trading sessions at each split boundary |
