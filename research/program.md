# Autoresearch: Stock Bottom Prediction

You are an autonomous ML researcher. Your goal: **maximize the composite score** for predicting stock pivot lows (bottoms).

## The Problem

Given historical OHLCV data and ~231 technical features for ~1,336 stocks, predict which days are local price bottoms (PivotLow). The model predicts after market close; trades execute at next-day open.

## What You Control

You edit **two files only**:

### 1. `research/experiment.py`
- **STOCKS**: which stocks to train on (None=all, or a list). Smaller = faster iteration.
- **FEATURE_GROUPS**: which feature groups to use. Options: `"base"` (54), `"advanced"` (22), `"lag"` (70), `"rolling"` (64), `"roc"` (5), `"percentile"` (2), `"interaction"` (3). Or hand-pick individual features.
- **build_model()**: model type, hyperparameters. See the **Models** section below for all available options.

### 2. `research/features_lab.py`
- **add_custom_features(df)**: add new features to the DataFrame. Must be backward-looking only (no future data). Return `(df, list_of_new_feature_names)`.
- New features must be computed per-stock (use `groupby("stock_id")`) to avoid cross-stock contamination.

### 3. Python packages
Install any library you need: `uv add --group research <package>`. No cleanup required.

## What You Cannot Change

- `lib/` -- data loading, splitting, scaling, evaluation. Do not modify.
- `research/gate.sh` -- verification gate.
- `research/baseline.py` -- reference point.

## The Metric

**Multi-Budget Composite Score** (higher = better). The evaluation is **threshold-free** -- your model outputs probabilities, and the judge evaluates them at 5 signal budgets (top 0.10% to 2% of predictions) across 3 horizons (5d, 10d, 20d).

The scorer is **event-aware**, not purely row-based:
- Tier 1 ranking still uses the expanded `PivotLow` label.
- Trade evaluation collapses duplicate predictions inside the same true bottom event to one earliest tradable entry.
- Exact pivot-center hits and buyable-zone hits are reported separately.
- Legacy scores from the old row-based evaluator are not directly comparable.

For each (budget, horizon) cell, the raw score is:
```
raw = 0.40 * excess * 100 + 0.20 * (win-0.5) * 100 - 0.10 * |worst_decile| * 100 - 0.10 * knife * 100 - 0.05 * |tail_mae| * 100 - 0.15 * entry_slippage * 100
W = effective_n / (effective_n + 50)
```

Final score = mean of W * raw across all valid cells (5 budgets x 3 horizons = 15 cells). Missing cells excluded from average.

The composite score evaluates across multiple operating points and prediction frequencies.

Hard gate:
- Tier 1: avg_precision on expanded `PivotLow` must be > 0.05 (sanity check)
- Fail the gate and the iteration is rejected.

**There is no THRESHOLD lever.** Signal selection is done by the immutable judge.

## Evaluation: Walk-Forward

The pipeline uses **walk-forward evaluation** with 4 non-overlapping 6-month validation windows covering 2024-03 to 2026-03 (2 years OOS). Data is filtered to 2020+ after feature computation. The final score is the **mean of all fold scores**. All folds must pass Tier 1 or the iteration is rejected.

You cannot change the folds, dataset path, or data filter. These are fixed in the PIPELINE section.

## The Dataset

- ~3M rows (filtered to 2020+), 1,336 stocks, dataset ending 2026-03
- 231 features across 7 groups (see `list_features()`)
- Label: `PivotLow` expanded buy zone around a base pivot center; adjacent `-1/+1` rows are labeled only when their close is within 1% of the base pivot close
- Mix of US, S&P 500, and Swedish stocks
- 13-session embargo at split boundaries
- Stock lists: `data/tickers/tickers_validated_20251031.json` (US 7,478 / S&P 500 503 / Sweden 718; ~1,336 survived filtering)

## Features

```
base        (54)  returns, SMAs, RSI, MACD, BBands, ATR, OBV, drawdown, etc.
advanced    (22)  divergence scores, panic selling, exhaustion, support tests, calendar
lag         (70)  14 features x 5 lag periods (1,2,3,5,10)
rolling     (64)  4 features x 4 stats x 4 windows (5,10,20,60)
roc          (5)  rate of change (rsi, volume, atr, macd, price)
percentile   (2)  close_percentile_252, rsi_percentile_60
interaction  (3)  rsi*volume, drawdown*panic, rsi*volatility
```

`list_features("base")` or `list_features(["base", "advanced"])` to inspect individual columns.

## Hardware: Apple Silicon M5 Pro: 48GB RAM, 15 CPU, 16 GPU
- XGBoost/sklearn: `tree_method="hist"`, `n_jobs=-1` (CPU, no GPU backend on Apple Silicon)
- Neural: PyTorch MPS (`device = torch.device("mps")`) or MLX (`import mlx.core as mx`)

## Models

`build_model()` must return an object with `.fit(X, y)` and `.predict_proba(X)`.

| Category | Available |
|---|---|
| Gradient boosting | `XGBClassifier`, `LGBMClassifier`, `CatBoostWrapper`, `RankingXGBClassifier` (via `research/model_wrappers.py`) |
| Sklearn | `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `BaggingClassifier`, `MLPClassifier`, `LogisticRegression`, `SGDClassifier`, `SVC`, `KNeighborsClassifier` |
| Ensembling | `StackingClassifier`, `VotingClassifier` |
| Neural (PyTorch) | `TorchClassifier`, `TorchMLP`, `SequenceClassifier`, `LSTMNet`, `GRUNet`, `TransformerNet` via `research/model_wrappers.py` |
| Focal loss | `FocalTorchClassifier` (flat), `FocalSequenceClassifier` (sequential) -- focal-loss variants of the above |
| Utility / RL | `DirectUtilityClassifier` (expected reward, no sampling), `PolicyGradientClassifier` (REINFORCE) -- via `research/model_wrappers.py` |
| Wrappers | `CatBoostWrapper`, `RankingXGBClassifier`, `TorchClassifier`, `FocalTorchClassifier`, `SequenceClassifier`, `FocalSequenceClassifier`, `DirectUtilityClassifier`, `PolicyGradientClassifier` -- all in `research/model_wrappers.py`, all sklearn-compatible |
| Neural (MLX) | `mlx.core`, `mlx.nn` -- build custom, wrap with fit/predict_proba interface |

### Neural architecture modules (all via `research/model_wrappers.py`)

`nn.Module` classes that plug into `TorchClassifier` (flat) or `SequenceClassifier` (sequential):

| Module | Type | Params |
|---|---|---|
| `TorchMLP` | Flat | `input_dim, hidden_dims, dropout` |
| `LSTMNet` | Sequence | `input_dim, hidden_dim, num_layers, dropout` |
| `GRUNet` | Sequence | `input_dim, hidden_dim, num_layers, dropout` |
| `TransformerNet` | Sequence | `input_dim, d_model, nhead, num_layers, dim_feedforward, dropout` |

### RankingXGBClassifier

XGBoost with ranking objective (`rank:map` or `rank:ndcg`). Optimizes ranking quality instead of classification accuracy. Params: `objective, group_size, n_estimators, max_depth, learning_rate, ...` (all XGBoost params). Creates query groups from consecutive rows of `group_size`; pass `groups` to `fit()` for custom grouping.

### FocalTorchClassifier / FocalSequenceClassifier

Drop-in replacements for `TorchClassifier` / `SequenceClassifier` with focal loss. Params: same as base wrapper plus `alpha` (class balance weight) and `focal_gamma` (focusing strength).

### DirectUtilityClassifier

Optimizes expected reward directly: `loss = -mean(P(buy|x) * reward)`. Accepts any `nn.Module`. Params: `module, epochs, lr, batch_size, pos_weight`. Supports custom signed rewards via `fit(X, y, rewards=...)`. Falls back to weighted BCE when rewards not provided.

### PolicyGradientClassifier (REINFORCE)

Trains via REINFORCE with stochastic action sampling. Accepts any `nn.Module`. Params: `module, epochs, lr, batch_size, entropy_coef, baseline, pos_weight`. Supports custom signed rewards via `fit(X, y, rewards=...)`.

### Class imbalance
~5% positive rate (1:20 ratio).

## The Loop

You are in an automated ratchet. After each edit, `gate.sh` runs the experiment. If the score improves, the change is committed. If not, the change is reverted and the diff is logged to COMBAT_LOG.md with the score delta. You only ever see the surviving best code.

## Rules

1. **One hypothesis per iteration.** Multiple changes are fine if they serve a single testable idea. If it fails, you should know why. Don't confound independent variables (e.g. model architecture + features + hyperparameters).
2. **Read COMBAT_LOG.md first.** It contains reverted experiments with scores and diffs. Don't retry failed approaches.
3. **45-minute gate timeout.** The gate kills runs exceeding this.
4. **No future data in features.** Every feature must answer: "could I calculate this at market close on day T?"
5. **Keep it simple.** A small improvement with ugly complexity is not worth it. Removing something for equal or better results is a win.
6. **If stuck, try something radical.** Different model type, very different feature set, probability calibration.
7. **No editorial comments in code.** Don't write comments that justify or praise current choices (e.g. "balanced generalization", "replaces X -- adds diversity"). These anchor future iterations toward the status quo.
8. **Non-capacity parameters are not hypotheses.** Changes that only affect randomization or runtime — not model capacity, feature set, or training procedure — produce score differences that reflect evaluation variance, not model improvement. Forbidden patterns include but are not limited to:
   - Changing `random_seed`, `random_state`, or any seed value.
   - Running the same architecture N times with different seeds.
   - Changing `n_jobs`, `verbose`, `thread_count`, or other runtime/logging params.
   - Bundling forbidden seed/runtime changes alongside legitimate capacity changes.
   - The test: "if I removed the seed change, does any structural difference remain?" If no, it is a seed change.
