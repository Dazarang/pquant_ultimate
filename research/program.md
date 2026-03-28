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
0.4 * mean_10d_return * 100
+ 0.3 * win_rate
- 0.2 * abs(worst_decile_return) * 100
- 0.1 * knife_rate * 100
```

But there are **hard gates** before composite score:
- Tier 1: avg_precision must be > 0.05 (sanity check)
- Tier 2: mean 10d forward return must be > 0% (does buying on signals make money?)
- Fail either gate and the iteration is rejected.

## The Dataset

- ~3M rows, 1,336 stocks, 2015-2026
- 231 features across 7 groups (see `list_features()`)
- Label: PivotLow (binary, ~5% positive rate, ~1:20 imbalance, window [-1,+1] @ 1% tolerance)
- Mix of US, S&P 500, and Swedish stocks

**Available stocks**: 1,336 in the dataset. Full ticker lists in `data/tickers/`:
- `tickers_validated_20251031.json` -- US (7,478), S&P 500 (503), Sweden (718)
- Only ~1,336 made it into the dataset after filtering and validation
- To see all stocks in the dataset: `uv run python -c "import pandas as pd; print(sorted(pd.read_parquet('data/datasets/20260115/dataset.parquet', columns=['stock_id'])['stock_id'].unique()))"`
- Temporal split with 13-session embargo at boundaries

**For fast iteration, use a small stock subset (5-20 stocks).** You can try larger sets once you have a strong model.

## Available Feature Groups

```
base        (54): returns, SMAs, RSI, MACD, BBands, ATR, OBV, drawdown, etc.
advanced    (22): divergence scores, panic selling, exhaustion, support tests, calendar
lag         (70): 14 features x 5 lag periods (1,2,3,5,10)
rolling     (64): 4 features x 4 stats (mean,std,min,max) x 4 windows (5,10,20,60)
roc          (5): rate of change (rsi, volume, atr, macd, price)
percentile   (2): close_percentile_252, rsi_percentile_60
interaction  (3): rsi*volume, drawdown*panic, rsi*volatility
```

Call `list_features("base")` to get the list, or `list_features(["base", "advanced"])` for combined.

## Hardware: Apple Silicon M5 Pro
- 15 CPU cores, 16 GPU cores (Metal), 48 GB unified RAM
- XGBoost: `tree_method="hist"` with `n_jobs=-1` (no GPU backend on Apple Silicon; uses all CPU cores)
- `n_jobs=-1` uses all CPU cores for sklearn/xgboost parallelism
- If exploring neural approaches: use MLX (`import mlx.core as mx`) for native Apple Silicon or PyTorch MPS (`device = torch.device("mps")`)
- LightGBM is available in dependencies and may outperform XGBoost on wide feature sets
- 48 GB RAM supports large feature universes -- don't hesitate to use all feature groups

## Research Strategy

### Start simple, add complexity only if it helps
1. Start with base features, tune hyperparameters first
2. Try different feature sets (add lag, rolling, advanced, or hand-pick)
3. Try feature selection (drop low-importance features)
4. Try custom features in features_lab.py
5. Try different models (see model menu below)
6. Try ensemble/stacking of models
7. Try threshold tuning

### Model menu

**Gradient boosting (installed):**
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier      # verbose=0 to suppress output
```

**Sklearn (installed):**
```python
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, StackingClassifier, VotingClassifier,
    BaggingClassifier, AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC  # probability=True for predict_proba
from sklearn.neighbors import KNeighborsClassifier
```

**Ensembling:**
```python
from sklearn.ensemble import StackingClassifier, VotingClassifier
# Combine any models. Final estimator blends predictions.
```

**Neural networks (torch installed, runs on MPS/Apple Silicon):**
```python
# Ready-to-use wrappers in research/model_wrappers.py:
from research.model_wrappers import TorchClassifier, TorchMLP

def build_model(y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    return TorchClassifier(
        module=TorchMLP(input_dim=54, hidden_dims=(128, 64)),
        epochs=50, lr=1e-3, batch_size=512,
        pos_weight=neg / pos,
    )
```

**Sequence models (LSTM, GRU -- torch installed):**
```python
# Data is sequential per stock. Skip lag features, feed raw windows instead.
from research.model_wrappers import SequenceClassifier, LSTMNet

def build_model(y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    return SequenceClassifier(
        module=LSTMNet(input_dim=54, hidden_dim=64, num_layers=2),
        window=10, epochs=30, lr=1e-3, batch_size=256,
        pos_weight=neg / pos,
    )
```

**Custom PyTorch models:**
```python
# Build any nn.Module, wrap with TorchClassifier. Must output single logit.
# TorchClassifier handles training loop, MPS device, and predict_proba.
```

**MLX (installed, native Apple Silicon):**
```python
import mlx.core as mx
import mlx.nn as nn
# Build custom MLX models. Wrap with fit()/predict_proba() interface.
```

**Pipeline constraint:** `build_model()` must return an object with `.fit(X, y)` and `.predict_proba(X)` methods. All wrappers in `research/model_wrappers.py` satisfy this.

**Hyperparameters are fully yours to control.** Every parameter shown in examples above (hidden_dims, epochs, lr, batch_size, window, num_layers, dropout, pos_weight, etc.) can be changed, added, or removed. The wrappers accept any valid arguments for their underlying models. You can also pass different constructor args to the nn.Module classes (e.g. different hidden_dim, add layers, change activation). The examples are starting points, not constraints.

### Class imbalance is critical
- ~5% positive rate (1:20 ratio). Default threshold 0.5 may produce too few signals.
- Use `scale_pos_weight` (XGBoost/LightGBM) or `class_weight` (sklearn).
- Tune threshold on validation set (try 0.1-0.4 range).
- More signals with lower precision can beat fewer signals with higher precision if forward returns are positive.

### What matters for composite score
- **Mean 10d return** (40% weight): are your buy signals profitable?
- **Win rate** (30% weight): what fraction of signals go up?
- **Worst decile** (20% penalty): how bad are your worst trades?
- **Knife rate** (10% penalty): how often do you buy into >5% continued drops?

You want signals that catch actual bottoms, not falling knives. Precision matters more than recall here.

## Rules

1. **ONE change per iteration.** Don't change model AND features AND threshold at once. Isolate variables.
2. **Read COMBAT_LOG.md first.** Don't retry failed approaches.
3. **Time budget: 15 minutes.** If training takes longer, use fewer stocks or simpler model.
4. **No future data in features.** Every feature must answer: "could I calculate this at market close on day T?"
5. **Keep it simple.** A small improvement with ugly complexity is not worth it.
6. **If stuck, try something radical.** Different model type, very different feature set, extreme threshold.

## Dead Ends (updated as research progresses)

Check COMBAT_LOG.md for the full list. Common dead ends:
- (none yet -- you are the first researcher)
