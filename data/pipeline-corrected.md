# correct pipeline - professional architecture

## problem identified
wasteful pipeline validated ALL 13k tickers then filtered out ~5k:
- 2 hours wasted on ETFs/warrants
- ~5k wasted API calls
- unprofessional design

## correct pipeline per CLAUDE.md principles

### single responsibility principle
each script does ONE thing:
1. **filter_tickers.py** - pure filtering (no API)
2. **validate_tickers.py** - validation only (API calls)
3. **build_training_set.py** - training set construction (API calls)

### modular design
composable steps, clear inputs/outputs:

```
tickers_20251029.json (13k raw)
        ↓
[1] filter_tickers.py (instant)
        ↓
tickers_filtered_YYYYMMDD.json (~8k filtered)
        ↓
[2] validate_tickers.py (1-2 hours)
        ↓
tickers_validated_YYYYMMDD.json (~6k valid)
        ↓
[3] build_training_set.py (45-90 min)
        ↓
training_data.pkl + training_stocks_data.parquet
```

## step-by-step execution

### step 1: filter (instant, no API)
```bash
cd /Users/deaz/Developer/project_quant/pQuant_ultimate/data
uv run python filter_tickers.py
```

**what it does:**
- removes ETFs (SPY, QQQ, IWM, etc)
- removes warrants (ends with W, WW, .W)
- removes units (ends with U, .U)
- removes preferred shares (ends with P, PR, PRA, etc)
- removes derivatives (>6 chars, special symbols)

**output:**
- `tickers_filtered_20251029.json`
- ~8k tickers (from 13k)
- removed ~5k junk (38% reduction)

**time:** <1 second

---

### step 2: validate (1-2 hours, API calls)
```bash
uv run python validate_tickers.py
```

**what it does:**
- validates each ticker has recent data (3 days)
- removes delisted stocks
- removes stocks with timezone errors
- removes stocks that fail to download

**input:**
- `tickers_filtered_20251029.json` (8k filtered)

**output:**
- `tickers_validated_20251029.json`
- ~6k valid tickers
- removed ~2k dead (25% of filtered)

**time:** 1-2 hours (~8k API calls at 0.15s each)

---

### step 3: build training set (45-90 min, API calls)
```bash
uv run python build_training_set.py
```

**what it does:**
- downloads 10 years historical data (2015-2024)
- filters by price (>$1), volume (>50k/day)
- identifies delisted stocks (survivorship bias prevention)
- enriches with metadata (market cap, sector)
- stratifies by market cap (US vs Swedish thresholds)
- selects 1500 final stocks

**input:**
- `tickers_validated_20251029.json` (6k valid)

**output:**
- `training_data.pkl` (~500MB)
- `training_stocks_data.parquet` (~300MB)
- ~1,500 stocks
- ~3.75M rows (2,500 days × 1,500 stocks)

**time:** 45-90 minutes (depends on rate limits)

---

## efficiency gains

### old pipeline (broken)
```
13k tickers → validate all (2 hours) → filter (instant) → 8k usable
```
- **wasted:** 5k validations on junk (ETFs, warrants)
- **wasted time:** ~45 minutes
- **wasted API calls:** ~5k

### new pipeline (correct)
```
13k tickers → filter (instant) → 8k clean → validate (1.5 hours) → 6k usable
```
- **saved:** 5k unnecessary validations
- **saved time:** ~45 minutes
- **saved API calls:** ~5k

### total improvement
- **38% fewer API calls** (13k → 8k)
- **38% faster validation** (2 hours → 1.2 hours)
- **professional architecture** (single responsibility, modular)

---

## file sizes

| file | size | description |
|------|------|-------------|
| tickers_20251029.json | ~500KB | raw 13k tickers |
| tickers_filtered_YYYYMMDD.json | ~300KB | filtered 8k tickers |
| tickers_validated_YYYYMMDD.json | ~200KB | validated 6k tickers |
| training_data.pkl | ~500MB | 1.5k stocks, 10 years |
| training_stocks_data.parquet | ~300MB | compressed version |

---

## key architectural improvements

### single responsibility
- **filter_tickers.py:** filtering only, no validation
- **validate_tickers.py:** validation only, no filtering
- **build_training_set.py:** training set only, assumes pre-filtered

### no duplication
- filtering logic in ONE place (filter_tickers.py)
- validation logic in ONE place (validate_tickers.py)
- no redundant filtering in build_training_set.py

### clear contracts
- each script has defined input/output
- each step independent, testable
- can skip steps if input already exists

### professional design per CLAUDE.md
- modular (composable steps)
- single responsibility (one job per file)
- efficient (no wasted work)
- maintainable (clear separation)

---

## running the complete pipeline

```bash
cd /Users/deaz/Developer/project_quant/pQuant_ultimate/data

# step 1: filter (instant)
uv run python filter_tickers.py

# step 2: validate (1-2 hours)
uv run python validate_tickers.py

# step 3: build training set (45-90 min)
uv run python build_training_set.py
```

total time: ~2-3 hours (was 3-4 hours)

---

## verification

after each step, verify output:

```bash
# after filtering
python3 -c "import json; d=json.load(open('tickers_data/tickers_filtered_20251029.json')); print(f'US: {len(d.get(\"US\",[]))}, SP500: {len(d.get(\"SP500\",[]))}, Sweden: {len(d.get(\"Sweden\",[]))}')"

# after validation
python3 -c "import json; d=json.load(open('tickers_data/tickers_validated_20251029.json')); print(f'US: {len(d.get(\"US\",[]))}, SP500: {len(d.get(\"SP500\",[]))}, Sweden: {len(d.get(\"Sweden\",[]))}')"

# after training set
ls -lh training_data.pkl training_stocks_data.parquet
```

---

## summary

**before:** validate everything, filter later (wasteful)
**after:** filter first, validate clean list (efficient)

**improvement:** 38% fewer API calls, 38% faster, professional architecture

follows CLAUDE.md principles:
- single responsibility
- modular design
- no duplication
- efficiency first
