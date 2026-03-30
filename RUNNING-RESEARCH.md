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
# Use the dedicated test script (trains model, evaluates on held-out test set,
# runs benchmark and backtest, generates signal plots)
uv run python research/test_and_plot.py
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
