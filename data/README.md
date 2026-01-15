# Data Pipeline

Ticker acquisition and OHLCV data generation.

## Structure

```
data/
├── run.py              # Unified entry point
├── scripts/            # Pipeline steps
│   ├── 1_get_tickers.py
│   ├── 2_filter_tickers.py
│   ├── 3_validate_tickers.py
│   ├── 4_build_ohlcv.py
│   ├── 5_update_ohlcv.py
│   └── 6_build_features.py   # TODO
├── tickers/            # Ticker lists
│   ├── tickers_YYYYMMDD.json
│   ├── tickers_filtered_YYYYMMDD.json
│   └── tickers_validated_YYYYMMDD.json
└── datasets/           # Output data
    └── YYYYMMDD/
        ├── ohlcv.parquet     # Raw price data
        ├── dataset.parquet   # ML-ready (after 6_build_features)
        └── metadata.json
```

## Usage

```bash
# Interactive menu
uv run python data/run.py

# Or with flags
uv run python data/run.py --full      # Full pipeline (1-5)
uv run python data/run.py --update    # Update OHLCV (5)
uv run python data/run.py --features  # Build features (6)
uv run python data/run.py --ohlcv     # OHLCV only (1-4)
```

## Daily Workflow

Once initial dataset exists, only update is needed:

```bash
uv run python data/run.py --update
```

## Pipeline Flow

```
1_get_tickers     → tickers/tickers_YYYYMMDD.json           (~13k raw)
2_filter_tickers  → tickers/tickers_filtered_YYYYMMDD.json  (~8.7k clean)
3_validate_tickers→ tickers/tickers_validated_YYYYMMDD.json (~6-7k valid)
4_build_ohlcv     → datasets/YYYYMMDD/ohlcv.parquet         (1.5k stocks)
5_update_ohlcv    → datasets/YYYYMMDD/ohlcv.parquet         (incremental)
6_build_features  → datasets/YYYYMMDD/dataset.parquet       (ML-ready)
```

## When to Re-run Full Pipeline

- Initial setup (first time)
- Adding new tickers (IPOs, expanding coverage)
- Major schema changes

See `docs/quick-start.md` for detailed documentation.
