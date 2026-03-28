# Running Autoresearch

## Prerequisites

```bash
brew install libomp    # Required for XGBoost on macOS
uv sync                # Install Python dependencies
```

## Quick Start

```bash
# Run 20 iterations with Claude Code agent (on current branch)
./research/run.sh

# Run 50 iterations
./research/run.sh 50

# Run on a separate branch (safe, merge later)
./research/run.sh 20 --branch

# Manual mode (you edit experiment.py, press Enter, loop scores it)
./research/run.sh manual
```

## What Happens

Each iteration (~3-5 min with 5 stocks):

1. Claude Code agent reads current best + combat log
2. Makes one edit to `research/experiment.py` or `research/features_lab.py`
3. `gate.sh` verifies nothing in `lib/` was touched, runs the experiment
4. If score improves: commits the change
5. If not: reverts and logs what failed to `COMBAT_LOG.md`
6. Updates `progress.png` plot

## Monitoring

While running:
```bash
# Watch the score progress
tail -f research/results.tsv

# View the plot (updates after each iteration)
open research/progress.png

# Check what's been tried
cat research/COMBAT_LOG.md
```

## After Research

### 1. View results
```bash
# Generate/refresh plot
uv run python research/plot.py
open research/progress.png

# Check final score
cat research/.best_score
```

### 2. Validate the winner
```bash
uv run python -c "
from lib.data import load_dataset, temporal_split, scale, LABEL_COL
from lib.eval import tiered_eval, benchmark_random_entry

# Load with same config as winning experiment
df, fc = load_dataset('data/datasets/20260115/dataset.parquet', stocks=['AAPL','MSFT','GOOG','AMZN','TSLA'])
train, val, test = temporal_split(df)
train_s, val_s, test_s, scaler = scale(train, val, test, fc)

# Train winning model (import from experiment.py)
from research.experiment import build_model, THRESHOLD
import numpy as np
X_train, y_train = train_s[fc].values, train_s[LABEL_COL].values
X_test, y_test = test_s[fc].values, test_s[LABEL_COL].values

model = build_model(y_train)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > THRESHOLD).astype(int)

# Tier eval on TEST set (never seen during research)
print('=== TEST SET (unseen) ===')
results = tiered_eval(test, y_test, y_pred, y_proba)

# Benchmark against random entry
print('\n=== BENCHMARK ===')
bm = benchmark_random_entry(test, y_pred)
"
```

### 3. If on a branch, merge
```bash
git checkout main
git merge autoresearch/YYYYMMDD_HHMMSS
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `XGBoostError: libomp.dylib not found` | `brew install libomp` |
| Gate timeout (>15 min) | Use fewer stocks or simpler model |
| All iterations fail | Check `research/run.log` for errors |
| Score stuck | Read `COMBAT_LOG.md`, try radical changes |
| 5 consecutive knowledge-only | Search space exhausted, review program.md |
| 10 consecutive gate failures | Circuit breaker, check experiment.py for bugs |

## Files Reference

| File | Editable | Purpose |
|------|----------|---------|
| `research/experiment.py` | By agent | Model, params, features, stocks |
| `research/features_lab.py` | By agent | Custom features |
| `research/program.md` | By you | Agent instructions |
| `research/gate.sh` | No | Verification gate |
| `research/run.sh` | No | Outer loop |
| `research/baseline.py` | No | Reference point |
| `research/results.tsv` | Auto | Score data for plot |
| `research/progress.png` | Auto | Visual progress chart |
| `research/RESEARCH_LOG.md` | Auto | Iteration history |
| `research/COMBAT_LOG.md` | Auto + agent | Failure analysis |
| `research/.best_score` | Auto | Current best score |
| `research/run.log` | Auto | Last experiment stdout |
