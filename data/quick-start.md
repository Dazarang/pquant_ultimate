# quick start guide

## problem fixed
build_training_set.py had critical bugs:
- zerodivisionerror crash
- 28% dead tickers wasting time
- rate limit errors
- missing market cap data

## solution
two-step process:

### step 1: clean tickers (run once, ~25 minutes)
```bash
cd data
uv run python clean_ticker_json.py
```

validates ~10k tickers, removes dead ones.

output: `tickers_validated_YYYYMMDD.json`

### step 2: build training set
update line 481 in build_training_set.py:
```python
TICKER_JSON_PATH = 'data/tickers_data/tickers_validated_20251029.json'
```

run it:
```bash
uv run python build_training_set.py
```

## what changed

### build_training_set.py fixes:
1. uses fast_info instead of info (more reliable)
2. fixed division by zero crash
3. fallback to volume stratification if no market cap
4. better rate limiting (1.0s between batches)

### new validator:
- pre-filters dead tickers
- saves 25%+ download time
- prevents rate limit errors

## expected results
before fixes:
- 2321 valid / 906 rejected (28% waste)
- rate limit errors after batch 22
- crash on stratified selection

after fixes:
- fewer rejected tickers (pre-filtered)
- no rate limit errors (better delays)
- successful stratification (fallback logic)
- no crashes

## testing shortcut
for quick test, edit build_training_set.py line 484:
```python
TARGET_US_SAMPLE = 500  # instead of 2000
```

faster download (~10 minutes instead of 30).
