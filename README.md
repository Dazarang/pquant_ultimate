# pQuant Ultimate

ML-based stock bottom detection system.

## Structure

```
pquant_ultimate/
├── data/                 # Data acquisition & feature engineering
│   ├── run.py            # Unified entry point
│   ├── scripts/          # Pipeline steps (1-6)
│   ├── tickers/          # Ticker lists (raw, filtered, validated)
│   └── datasets/         # Output
│       └── YYYYMMDD/
│           ├── ohlcv.parquet     # Raw price data
│           └── dataset.parquet   # ML-ready features
│
├── indicators/           # Technical indicator library
│
├── lib/                  # Shared utilities
│   ├── data.py           # load_dataset(), temporal_split(), scale()
│   ├── features.py       # build_features() - used by data/scripts/6
│   └── eval.py           # metrics(), plot_results()
│
├── training/             # Model training
│   ├── train.py          # Main training script
│   ├── config.yaml       # Model type, hyperparams
│   └── models/           # Model implementations
│       ├── xgboost.py
│       ├── ensemble.py
│       └── ...
│
├── backtesting/          # Strategy testing
│   ├── backtest.py
│   └── strategies/
│
├── prediction/           # Live predictions
│   └── predict.py
│
└── runs/                 # Timestamped outputs
    └── YYYYMMDD_HHMMSS/
        ├── model.pkl
        ├── scaler.pkl
        ├── config.yaml
        └── metrics.json
```

## Usage

### Data Pipeline

```bash
cd data/

# Interactive
uv run python run.py

# Or direct
uv run python run.py --full      # Full pipeline (1-6)
uv run python run.py --update    # Update OHLCV + features
uv run python run.py --features  # Build features only
```

### Training

```bash
# Edit config
vim training/config.yaml

# Train
uv run python training/train.py

# Compare runs
uv run python scripts/compare_runs.py
```

### Backtesting

```bash
uv run python backtesting/backtest.py --run runs/20260115_143022
```

### Prediction

```bash
uv run python prediction/predict.py --tickers AAPL,MSFT
```

## Key Files

- `data/README.md` - Data pipeline details
- `indicators/README_INDICATORS.md` - Indicator usage
- `training/config.yaml` - Model configuration
- `JOURNAL.md` - Development log
