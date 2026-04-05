# Autoresearch: How It Works

Autonomous ML research loop inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). An LLM agent iteratively edits experiment code, runs it, keeps improvements, reverts failures, and documents everything.

## Architecture

```
research/
├── experiment.py       # MUTABLE -- model, hyperparams, features, stocks
├── features_lab.py     # MUTABLE -- custom feature engineering (accumulates on wins)
├── logs/
│   ├── COMBAT_LOG.md   # What failed and why (knowledge preservation)
│   ├── RESEARCH_LOG.md # Human-readable iteration log
│   └── run.log         # Last experiment stdout
├── utils/
│   ├── backtest_and_plot.py
│   ├── baseline.py     # IMMUTABLE -- logistic regression reference point
│   ├── diagnostics.py
│   ├── metrics.md      # Metric definitions reference
│   ├── model_wrappers.py
│   └── plot.py         # Generates plots/progress.png from results.tsv
├── plots/              # Generated plot images (progress.png, etc.)
├── program.md          # Agent instructions, constraints, dead ends
├── gate.sh             # Verification gate (immutability, timeout, sanity)
├── run.sh              # Outer loop orchestrator
├── results.tsv         # Score log for plotting (iter, score, status, description)
└── .best_score         # Current best composite score
```

## The Loop

```
./research/run.sh 20

for each iteration:
  1. Agent reads: program.md, COMBAT_LOG.md, experiment.py, features_lab.py
  2. Agent makes ONE focused edit to experiment.py and/or features_lab.py
  3. gate.sh runs:
     a. Verify lib/ unchanged (anti-gaming -- judge stays out of arena)
     b. Run experiment.py (30 min timeout)
     c. Extract COMPOSITE_SCORE from stdout
     d. Check PASSED flag (AP gate)
  4. If score > best → commit, update .best_score, log to results.tsv (keep)
  5. If score <= best → log diff to COMBAT_LOG.md, revert, log to results.tsv (discard)
  6. Update plots/progress.png
```

## What the Agent Controls (6 levers)

| Lever | Location | Example |
|-------|----------|---------|
| Stock universe | experiment.py `STOCKS` | `["AAPL", "MSFT"]` or `None` for all |
| Feature selection | experiment.py `FEATURE_GROUPS` | `["base", "advanced"]` or hand-picked list |
| Custom features | features_lab.py | New backward-looking features |
| Split boundaries | experiment.py `TRAIN_END/VAL_END` | `"2023-06-30"` |
| Model + hyperparams | experiment.py `build_model()` | XGBoost, LightGBM, depth, lr, etc. |

## What's Immutable (anti-gaming)

| File | Why |
|------|-----|
| `lib/data.py` | Can't change how data loads/splits |
| `lib/eval.py` | Can't change how metrics are computed |
| `lib/features.py` | Feature catalog is reference only |
| `research/gate.sh` | Can't weaken verification |
| `research/utils/baseline.py` | Fixed comparison point |

## The Metric: Multi-Budget Composite Score

Evaluation is **threshold-free**. The model outputs probabilities; the judge selects signals at 5 budget levels (top 0.10% to 2%) and evaluates at 3 horizons (5d, 10d, 20d).

The scorer is event-aware:
- row-level AP still uses the expanded `PivotLow` label
- trading metrics collapse duplicate predictions inside the same true bottom event
- exact pivot-center hits and buyable-zone hits are tracked separately
- legacy scores from the old row-based evaluator are not directly comparable

Per (budget, horizon) cell:
```
raw = 0.40 * excess * 100 + 0.20 * (win-0.5) * 100 - 0.10 * |worst_decile| * 100 - 0.10 * knife * 100 - 0.05 * |tail_mae| * 100 - 0.15 * entry_slippage * 100
W = effective_n / (effective_n + 50)
```

Final score = mean of W * raw across all valid cells (5 budgets x 3 horizons = 15 cells). Missing cells excluded from average.

Hard gate: avg_precision > 0.05 on expanded `PivotLow` (model better than random)

## Two Loss Functions

| What | Purpose | Used by |
|------|---------|---------|
| Model loss (logloss/gini/etc) | Train the model to learn patterns | `model.fit()` |
| Multi-budget composite score | Judge if probability ranking translates to profitable trades | `gate.sh` keep/discard |

These are intentionally different. The model optimizes its internal loss to find patterns. We evaluate whether those patterns make money across multiple signal budgets. A model can have great logloss but terrible composite score (predicts labels accurately but the "bottoms" it ranks highly keep dropping).

## From Research to Production

```
PHASE 1: Autoresearch
    Run ./research/run.sh 50 overnight
    Output: winning experiment.py + features_lab.py

PHASE 2: Validate winner
    Run benchmark_random_entry() -- statistical significance test
    Run tiered_eval on TEST set (never seen during research)
    Run on more stocks -- does the edge hold at scale?

PHASE 3: Promote features
    Copy proven features from features_lab.py → lib/features.py
    Update FEATURES catalog
    Rebuild dataset with new features

PHASE 4: Productionize (PLAN.md Phase 3)
    Hardcode winning config into training/train.py + config.yaml
    Input: dataset.parquet → Output: runs/YYYYMMDD/model.pkl

PHASE 5: Deploy (PLAN.md Phases 4-5)
    prediction/predict.py -- live signals
    backtesting/backtest.py -- full historical backtest

PHASE 6: Re-run periodically
    New data → rebuild dataset → run autoresearch again
    Research is never "done"
```

## Mapping to Karpathy's Design

| Concept | Karpathy | Ours |
|---------|----------|------|
| Mutable code | `train.py` (1 file) | `experiment.py` + `features_lab.py` (2 files) |
| Immutable code | `prepare.py` | `lib/` (data, eval, features) |
| Metric | val_bpb (lower=better) | composite_score (higher=better) |
| Agent loop | Internal (agent never stops) | Bash outer loop (`run.sh`) |
| Tracking | `results.tsv` (untracked) | `results.tsv` + `RESEARCH_LOG.md` |
| Knowledge preservation | None for failures | `COMBAT_LOG.md` (diffs + analysis) |
| Anti-gaming | Fixed eval function | Immutable lib/ + gate diff checks |
| Visualization | `analysis.ipynb` → `progress.png` | `utils/plot.py` → `plots/progress.png` |
| Time budget | 5 min/experiment | 15 min/experiment |
