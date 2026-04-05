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

## Multi-Budget Composite Score

Evaluation is **threshold-free**. The model outputs probabilities; the judge selects signals at 5 budget levels (top 0.10% to 2%) and evaluates at 3 horizons (5d, 10d, 20d).

Per (budget, horizon) cell:
```
raw = 0.40 * excess * 100 + 0.20 * (win-0.5) * 100 - 0.10 * |worst_decile| * 100 - 0.10 * knife * 100 - 0.05 * |tail_mae| * 100 - 0.15 * entry_slippage * 100
W = effective_n / (effective_n + 50)
```

Final score = mean of W * raw across all valid cells (5 budgets x 3 horizons = 15 cells). Missing cells excluded from average.

The multi-budget approach prevents gaming: the model must rank well at both sparse (high-conviction) and dense (screening) signal rates simultaneously.

## 2-Tier Evaluation System

| Tier | What | Hard gate? |
|------|------|------------|
| Tier 1: Ranking Quality | ROC-AUC, avg_precision | AP must be > 0.05 |
| Tier 2: Multi-Budget Composite | 5 budgets x 3 horizons = 15 cells, linear N-scaling | This is the ratchet metric |

Fail Tier 1 and the iteration is rejected.

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

PivotLow is ~5% positive (1:20 ratio). This means:
- Must use `scale_pos_weight` in XGBoost (= neg_count / pos_count) or `class_weight="balanced"`
- Probability calibration can improve ranking quality
- The multi-budget eval handles the threshold question -- the model just needs good probability ranking

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

## Pipeline Phases

| Phase | What | When |
|-------|------|------|
| 1. Autoresearch | Ratchet loop explores model/feature/hyperparameter space (open-ended) | Now |
| 2. Validation | Test set eval, `benchmark_random_entry()`, regime breakdown, signal clustering | After ratchet converges |
| 3. Optuna | Fine-tune hyperparams on winning architecture (defined search space) | After validation confirms skill |
| 4. Production | Extract model outside research/, daily pipeline, signal generation, position sizing | After Optuna |

Autoresearch finds the architecture, Optuna squeezes it. Sequential, not parallel.

## How the Threshold-Free System Works (End to End)

Research and deployment are deliberately separated. Research finds the best **ranker** (model that puts profitable bottoms at the top of its probability list). Deployment finds the best **decision rule** for that ranker (how many signals to trade, position sizing, stops).

### Concrete example

**During research** -- the agent improves the model. Suppose after 50 iterations it finds an ensemble that scores +2.5 on the multi-budget composite. The model outputs probabilities like:

```
Row 1:  proba=0.91  (true bottom, +8% 10d return)
Row 2:  proba=0.87  (true bottom, +5% 10d return)
Row 3:  proba=0.85  (NOT a bottom, -3% 10d return)
Row 4:  proba=0.72  (true bottom, +2% 10d return)
...
Row 300K: proba=0.01  (NOT a bottom)
```

The multi-budget eval scores this model at 5 budget levels:
- Top 0.10% (300 signals): mostly true bottoms, high excess return
- Top 0.25% (750 signals): still good, some noise
- Top 2% (6000 signals): diluted, weaker edge

The model's composite score (+2.5) tells us: "this model ranks profitable bottoms well across a range of signal volumes." There is no threshold -- we just measured how good the ranking is.

**After research -- choosing how to trade it:**

The model is frozen. Now on the validation set, you answer: "at what probability cutoff do I get the best risk-adjusted trades?"

```python
# Option A: fixed budget -- "buy top 3 signals per day"
y_pred = select_top_frac(y_proba, 0.0025)

# Option B: probability threshold -- find the cutoff that maximizes Sharpe
for threshold in np.arange(0.3, 0.9, 0.01):
    y_pred = (y_proba > threshold).astype(int)
    sharpe = compute_sharpe(val, y_pred)  # backtest this threshold

# Option C: Optuna sweeps threshold + stop_loss + position_size jointly
```

If Option B finds that threshold=0.65 maximizes Sharpe on the validation set, you freeze that and test on the held-out test set (2024+). If it holds, that becomes your production threshold.

**The key principle:** the research agent never sees a threshold, so it can't game it. It only optimizes "is my ranking good?" The deployment threshold is derived afterward from the winning model's probability distribution.

### What makes a "good" model

| Composite Score | Meaning | Next step |
|-----------------|---------|-----------|
| < -2 | No useful ranking skill | Keep researching |
| -2 to 0 | Some ranking ability but path risk dominates | Research or feature engineering |
| 0 to +2 | Decent ranker, worth validating | Run test_and_plot.py, check test set |
| +2 to +4 | Good ranker, profitable at multiple budgets | Validate, then tune deployment with Optuna |
| +4 to +8 | Excellent (close to oracle=8.4) | Validate carefully for overfitting |

### After research: concrete steps

1. **Validate**: `uv run python research/utils/backtest_and_plot.py` -- evaluate on test set (2024+), run benchmark_random_entry (p < 0.05?), visually inspect signal plots
2. **Tune deployment**: sweep threshold / budget / stop-loss on validation set to find the best trading rule for this specific model. Use Optuna or grid search.
3. **Freeze and test**: apply the frozen deployment config to the test set. This is the final out-of-sample check.
4. **Productionize**: hardcode model config + deployment threshold into `training/train.py` + `prediction/predict.py`

## Optuna + Scoring System

Two nested optimization loops, model-agnostic (any `.fit()` + `.predict_proba()` model):

- **Inner (model.fit):** Each model minimizes its own loss to learn P(bottom). XGBoost: logloss. RandomForest: Gini. Neural nets: BCE/focal loss. The model knows nothing about composite score.
- **Outer (Optuna):** Maximizes multi-budget composite score -- evaluates probability ranking across 5 budgets x 3 horizons = 15 cells.

The tiers map directly to Optuna:

| Tier | Optuna role |
|------|-------------|
| Tier 1 (AP > 0.05) | Pruning -- kill trial early, cheap |
| Tier 2 (multi-budget composite) | Objective -- the value Optuna maximizes |

`tiered_eval` already stops early on tier failure -- return `-inf` for failed trials. Multi-objective (excess return vs knife rate separately) is an alternative for Pareto exploration.

For neural nets, Optuna can also tune the loss function itself (e.g., focal loss gamma for class imbalance).

## Anti-Gaming

The autoresearch gate prevents the agent from "cheating" by:
- Making `lib/` immutable (can't change how metrics are computed)
- Making `gate.sh` and `utils/baseline.py` immutable
- Checking git diff before running (revert if protected files changed)
- Using a fixed evaluation on a fixed validation set

This is the "move the judge out of the arena" principle from the tennis-xgboost-autoresearch repo.
