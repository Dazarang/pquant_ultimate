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

### Lib (`lib/`)
Shared utilities:
- `features.py` - build_features() for ML dataset creation

### Pipeline (`pipeline/`) - DEPRECATED
To be deleted in Phase 7. See `PLAN.md`.

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
Added validate_incremental.py to sanity check incremental updates. Validates: overlapping data consistency, new data exists, column match, no data loss.

### 2026-01-15
Fixed incremental mode bugs: (1) load_ohlcv now filters extra columns (adj close was 100% NaN causing all rows to drop), (2) check_incremental_possible now looks for previous dataset.parquet when current folder only has ohlcv, (3) filters OHLCV columns in incremental path same as full build.

### 2026-01-15
Added --test flag to 6_build_features.py for quick testing with N random stocks (default 100). Updated run.py with --features-inc, --features-test flags and expanded interactive menu (options 1-6).

### 2026-01-15
Added --incremental flag to 6_build_features.py. Incremental mode: keeps rows before cutoff (dataset_max - 365 days), recalculates features for cutoff onwards using 365-day lookback buffer for indicator warmup, then concats. Falls back to full build if no existing dataset.

### 2026-01-15
Added tqdm progress bars to build_features() and 6_build_features.py. Added min_rows filter (252) to skip stocks with insufficient data. Added total timing display.

### 2026-01-15
Optimized indicators/advanced.py using numba JIT compilation. Created indicators/_numba_core.py with optimized functions for: count_support_tests, count_consecutive_down, days_since_last_low, detect_divergence_pairs, detect_hidden_divergence. Changed O(n*m) loops to vectorized numba calls. Added verbose timing to create_all_advanced_features().

### 2026-01-15
Fixed data cleaning: removed ffill (no fake data), drop invalid OHLCV rows instead. Added validation for OHLC relationships (high>=low, etc). Removed inf->0 masking in features.py - let NaN propagate to be cleaned.

### 2026-01-15
Added lib/features.py with build_features() extracted from training-dataset-structure.ipynb. Created data/scripts/6_build_features.py. ~199 features: base indicators, lagged, rolling, rate of change, percentile, interaction, pivot labels.

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
