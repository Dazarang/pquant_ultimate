# Project Journal

## How This Works

This journal has three sections:

1. **Current State** - Living documentation of what exists NOW. Update when adding/removing major components. Keep entries brief with pointers to READMEs. No historical info here.

2. **Journal Entries** - Chronological log of changes. Newest entries at TOP. Format:
   ```
   ### YYYY-MM-DD
   Brief description of what was done. Be extremely concise. Sacrifice grammar for the sake of concision.
   ```

---

## Current State

### Data Pipeline (`data/`)
- `run.py` - Unified entry point (interactive menu or --flags)
- `scripts/` - Pipeline steps (1-6)
- `tickers/` - Ticker JSON files
- `datasets/YYYYMMDD/` - Output (ohlcv.parquet, dataset.parquet)

See `data/README.md` for usage.

### Pipeline (`pipeline/`) - DEPRECATED
To be replaced by `lib/` + `training/`. See `PLAN.md`.

### Indicators (`indicators/`)
Technical indicator library organized by category:
- `momentum.py`, `trend.py`, `volatility.py`, `volume.py`
- `pattern.py` - Pattern recognition (divergences, pivots)
- `cycle.py`, `advanced.py` - Specialized indicators
- `notebooks/` - Interactive examples per category

See `indicators/README_INDICATORS.md` for usage.

---

## Journal Entries

### 2026-01-15
Designed new project structure. Created README.md and PLAN.md. Key decisions:
- Restructure data/ (scripts subfolder, unified run.py entry point)
- Add lib/ for shared utilities (data, features, eval)
- Add training/ with config-driven model selection
- Add backtesting/, prediction/, runs/
- Deprecate pipeline/ (over-engineered)

### 2026-01-15
Initial journal setup.

### 2024-11-02
Added training dataset builder with feature extraction. Implemented backward-looking local extrema detection for indicators. Enhanced RSI divergence detection. Added incremental training data update script.

### 2024-10-31
Added indicator notebooks (momentum, trend, volatility, volume, pattern, cycle, advanced). Implemented pivot detection with price tolerance. Streamlined visualization code with Plotly.
