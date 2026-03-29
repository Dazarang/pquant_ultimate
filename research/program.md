# Autoresearch: Stock Bottom Prediction

You are an autonomous ML researcher. Your goal: **maximize the composite score** for predicting stock pivot lows (bottoms).

## The Problem

Given historical OHLCV data and ~231 technical features for ~1,336 stocks, predict which days are local price bottoms (PivotLow). The model predicts after market close; trades execute at next-day open.

## What You Control

You edit **two files only**:

### 1. `research/experiment.py`
- **STOCKS**: which stocks to train on (None=all, or a list). Smaller = faster iteration.
- **FEATURE_GROUPS**: which feature groups to use. Options: `"base"` (54), `"advanced"` (22), `"lag"` (70), `"rolling"` (64), `"roc"` (5), `"percentile"` (2), `"interaction"` (3). Or hand-pick individual features.
- **TRAIN_END / VAL_END**: temporal split boundaries.
- **build_model()**: model type, hyperparameters. You can use: sklearn, xgboost, lightgbm. Import what you need.
- **threshold**: prediction threshold (default 0.5). With 1:32 class imbalance, lower thresholds may help.

### 2. `research/features_lab.py`
- **add_custom_features(df)**: add new features to the DataFrame. Must be backward-looking only (no future data). Return `(df, list_of_new_feature_names)`.
- New features must be computed per-stock (use `groupby("stock_id")`) to avoid cross-stock contamination.

## What You Cannot Change

- `lib/` -- data loading, splitting, scaling, evaluation. The judge is not in the arena.
- `research/gate.sh` -- verification gate.
- `research/baseline.py` -- reference point.

## The Metric

**Composite Score** (higher = better). Formula:
```
0.30 * excess_return * 100         (alpha over equal-weight market)
+ 0.25 * (win_rate - 0.5) * 100     (consistency edge vs coin flip)
- 0.20 * |worst_decile| * 100       (tail risk penalty)
- 0.10 * knife_rate * 100           (falling knife penalty, >5% loss)
- 0.15 * |mean_mae| * 100           (path risk: avg max adverse excursion)
```

But there are **hard gates** before composite score:
- Tier 1: avg_precision must be > 0.05 (sanity check)
- Tier 2: at least one horizon (5d/10d/20d) must show positive excess return vs market
- Fail either gate and the iteration is rejected.

## The Dataset

- ~3M rows, 1,336 stocks, 2015-2026
- 231 features across 7 groups (see `list_features()`)
- Label: PivotLow (binary, ~5% positive rate, ~1:20 imbalance, window [-1,+1] @ 1% tolerance)
- Mix of US, S&P 500, and Swedish stocks
- Temporal split with 13-session embargo at boundaries
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
| Gradient boosting | `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier` |
| Sklearn | `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `BaggingClassifier`, `MLPClassifier`, `LogisticRegression`, `SGDClassifier`, `SVC`, `KNeighborsClassifier` |
| Ensembling | `StackingClassifier`, `VotingClassifier` |
| Neural (PyTorch) | `TorchClassifier`, `TorchMLP`, `SequenceClassifier`, `LSTMNet` via `research/model_wrappers.py` |
| Neural (MLX) | `mlx.core`, `mlx.nn` -- build custom, wrap with fit/predict_proba interface |

### Class imbalance
~5% positive rate (1:20 ratio). Address with `scale_pos_weight`, `class_weight`, threshold tuning, or other techniques.

## The Loop

You are in an automated ratchet. After each edit, `gate.sh` runs the experiment. If the score improves, the change is committed. If not, the change is reverted and the diff is logged to COMBAT_LOG.md with the score delta. You only ever see the surviving best code.

## Rules

1. **One hypothesis per iteration.** Multiple changes are fine if they serve a single testable idea. If it fails, you should know why. Don't confound independent variables (e.g. model + features + threshold).
2. **Read COMBAT_LOG.md first.** It contains reverted experiments with scores and diffs. Don't retry failed approaches.
3. **30-minute gate timeout.** The gate kills runs exceeding this.
4. **No future data in features.** Every feature must answer: "could I calculate this at market close on day T?"
5. **Keep it simple.** A small improvement with ugly complexity is not worth it. Removing something for equal or better results is a win.
6. **If stuck, try something radical.** Different model type, very different feature set, extreme threshold.

## Dead Ends (updated as research progresses)

Check COMBAT_LOG.md for the full list. Common dead ends:
- (none yet -- you are the first researcher)
