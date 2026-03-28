# Knowledge Base

Things to understand about this project's ML pipeline, autoresearch, and the decisions behind them.

## Two Loss Functions

There are two levels of optimization happening simultaneously:

**Level 1 -- Model training (inside `model.fit()`)**
XGBoost minimizes binary cross-entropy (logloss) to learn "what does a stock bottom look like?" This is the model's internal objective. It runs every time you train.

**Level 2 -- Research loop (the gate)**
Composite score asks "when this model says buy, do I make money?" This determines keep/discard in the autoresearch loop. The outer loop never sees logloss.

These MUST be different. A model can have great logloss (accurately classifies bottoms) but terrible composite score (the "bottoms" it finds are falling knives that keep dropping).

## The Ratchet

A ratchet only moves in one direction. The composite score is the ratchet metric:
- Score improves? Keep the commit, advance the branch.
- Score stays same or drops? Revert, log to combat log.

The branch can never get worse. Each winning iteration builds on the previous winner.

## Composite Score Breakdown

```
score = 0.4 * mean_10d_return * 100    (are signals profitable?)
      + 0.3 * win_rate                 (how consistent?)
      - 0.2 * |worst_decile| * 100     (how bad are worst trades?)
      - 0.1 * knife_rate * 100         (how often buy into >5% drops?)
```

| Score | Meaning |
|-------|---------|
| < 0 | Losing money or too many falling knives |
| 0-1 | Marginal edge |
| 1-2 | Decent |
| 2-3 | Good |
| 3+ | Excellent (validate to rule out overfitting) |

Scores above 3-4 on a small stock universe (5 stocks) should be validated on more stocks and with `benchmark_random_entry()` to confirm genuine skill.

## 3-Tier Evaluation System

| Tier | What | Hard gate? |
|------|------|------------|
| Tier 1: Classification | Precision, recall, F1, ROC-AUC, avg_precision | AP must be > 0.05 |
| Tier 2: Forward Returns | Mean return, win rate, profit factor at 5/10/20 days | Mean 10d return must be > 0% |
| Tier 3: Composite Score | Single number combining return, consistency, risk | This is the ratchet metric |

Fail Tier 1 or 2 and the iteration is rejected regardless of composite score.

## Benchmark vs Tiers

The `benchmark_random_entry()` is NOT part of the research loop. It's a post-research validation you run once on the winner.

| | Tiers (every iteration) | Benchmark (once at end) |
|---|---|---|
| Question | "Is this experiment better than the last?" | "Does this model have real skill vs random?" |
| Speed | Fast (~seconds) | Slow (~30-60s, 500 simulations) |
| Output | Score to maximize | Pass/fail (p < 0.05) |
| When | Every gate.sh run | After research completes |

The benchmark picks random entry dates on the same stocks and compares returns. If your model can't beat random entry, it has no timing skill -- it's just picking volatile stocks.

## Class Imbalance

PivotLow is ~3% positive (1:32 ratio). This means:
- Default threshold 0.5 often produces zero signals (model never predicts positive)
- Must use `scale_pos_weight` in XGBoost (= neg_count / pos_count)
- Lower thresholds (0.2-0.4) trade precision for more signals
- More signals with decent forward returns can beat fewer signals with high precision

## Embargo Gap

With pivot window `lb=8, rb=13`, the last 13 training days have labels that depend on validation-period prices. `temporal_split` drops 13 trading sessions at each boundary by default to prevent this label leakage.

## Forward Returns Methodology

- Decision time: after market close on day T
- Entry: next-day open (day T+1)
- Exit: open N days later (day T+1+N)
- This matches real trading: you can't act on today's close until tomorrow

## Knife Rate

Percentage of signals where the 10-day forward return is worse than -5%. Measures how often the model buys into continued downtrends ("catching a falling knife"). A knife rate above 10% is concerning.

## Feature Groups

| Group | Count | What |
|-------|-------|------|
| base | 54 | Price returns, SMAs, RSI, MACD, BBands, ATR, OBV, drawdown |
| advanced | 22 | Divergence, exhaustion, panic selling, support tests, calendar |
| lag | 70 | 14 features x 5 lag periods (1,2,3,5,10) |
| rolling | 64 | 4 features x 4 stats x 4 windows |
| roc | 5 | Rate of change features |
| percentile | 2 | Close and RSI percentile ranks |
| interaction | 3 | Cross-feature products |

All features are backward-looking (no future data). Features are defined in `lib/features.py` with the `FEATURES` catalog.

## macOS Dependencies

XGBoost and LightGBM require OpenMP for parallel tree building:
```bash
brew install libomp
```
Without it, XGBoost's compiled C++ library (`libxgboost.dylib`) cannot load -- it needs `libomp.dylib` to parallelize across CPU cores. This is a macOS-specific requirement; Linux ships with libgomp.

## Anti-Gaming

The autoresearch gate prevents the agent from "cheating" by:
- Making `lib/` immutable (can't change how metrics are computed)
- Making `gate.sh` and `baseline.py` immutable
- Checking git diff before running (revert if protected files changed)
- Using a fixed evaluation on a fixed validation set

This is the "move the judge out of the arena" principle from the tennis-xgboost-autoresearch repo.
