# Data Pipeline

4-step pipeline: download tickers, filter junk, validate alive, build training set.

## Quick Start

```bash
cd data/

# Step 1: Download tickers (~30s, run once/month)
uv run python 1_get_tickers.py

# Step 2: Filter junk (instant, no API)
uv run python 2_filter_tickers.py

# Step 3: Validate alive (~1-2h, API calls)
uv run python 3_validate_tickers.py

# Step 4: Build training set (~1h, API calls)
uv run python 4_build_training_set.py

# Optional: Incremental update (fetches new data for existing set)
uv run python 5_update_training_data.py
```

## Daily Usage

Once you have a training set built, **only script 5 is needed** to keep data fresh:

```bash
uv run python 5_update_training_data.py
```

It auto-detects latest dataset, fetches only new data, merges, and skips if already up-to-date.

**When to re-run scripts 1-4:**
- Initial build (first time)
- Adding new tickers (new IPOs, expanding coverage)
- Full rebuild (data corruption, schema changes)

## Pipeline Flow

```
1_get_tickers.py       → tickers_data/tickers_YYYYMMDD.json          (~13k raw)
2_filter_tickers.py    → tickers_data/tickers_filtered_YYYYMMDD.json (~8.7k clean)
3_validate_tickers.py  → tickers_data/tickers_validated_YYYYMMDD.json (~6-7k valid)
4_build_training_set.py → training_data/YYYYMMDD/                     (1.5k stocks, 10yr)
```

## Output Files

`training_data/YYYYMMDD/` contains:
- `training_data.pkl` - Full dataset (~140MB)
- `training_stocks_data.parquet` - Parquet format (~95MB)
- `metadata.json` - Run configuration
- `selection_stats.json` - Stratification stats

## Key Design

- Filter BEFORE validate (saves 33% API calls)
- Market cap stratification (US vs Swedish thresholds)
- 10% failed stocks kept (anti-survivorship bias)

See `docs/quick-start.md` for detailed documentation.
