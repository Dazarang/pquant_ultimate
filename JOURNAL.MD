# Project Journal

## How This Works

This journal has three sections:

1. **Current State** - Living documentation of what exists NOW. Update when adding/removing major components. Keep entries brief with pointers to READMEs. No historical info here.

2. **Journal Entries** - Chronological log of changes. Newest entries at TOP. Format:
   ```
   ### YYYY-MM-DD
   Brief description of what was done.
   ```

---

## Current State

### Data Pipeline (`data/`)
Scripts 1-5 for ticker acquisition and training data generation:
- `1_get_tickers.py` - Fetch tickers from various sources
- `2_filter_tickers.py` - Filter by criteria (volume, price, etc.)
- `3_validate_tickers.py` - Validate data quality
- `4_build_training_set.py` - Build training dataset with features
- `5_update_training_data.py` - Incremental updates to training data

See `data/README.md` for usage, `data/docs/` for deep dives.

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
