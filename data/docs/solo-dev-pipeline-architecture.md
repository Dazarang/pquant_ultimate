# solo dev pipeline architecture (no corporate bs)

## what to keep vs what to skip

**keep (makes you better engineer):**
- modular design
- configuration files
- versioning outputs
- basic testing
- simple monitoring
- documentation for future you

**skip (corporate overhead):**
- airflow/orchestration platforms
- kubernetes
- access control/iam
- audit logging
- compliance frameworks
- team collaboration tools

---

## 1. orchestration (solo dev style)

**skip:** airflow, prefect, dagster

**use:** simple bash scripts + cron

**pipeline.sh:**
```bash
#!/bin/bash
set -e  # exit on error

cd /Users/deaz/Developer/project_quant/pQuant_ultimate/data

echo "=== running data pipeline ==="
date

# step 1: filter tickers (instant)
uv run python filter_tickers.py || { echo "filter failed"; exit 1; }

# step 2: validate tickers (1-2 hours)
uv run python validate_tickers.py || { echo "validation failed"; exit 1; }

# step 3: build training set (45-90 min)
uv run python build_training_set.py || { echo "build failed"; exit 1; }

echo "=== pipeline complete ==="
date
```

**monthly_update.sh:**
```bash
#!/bin/bash
# run first of month to refresh ticker lists

cd /Users/deaz/Developer/project_quant/pQuant_ultimate/data
uv run python get_tickers.py
```

**schedule with cron:**
```bash
# monthly ticker refresh (1st of month, 2am)
0 2 1 * * /path/to/monthly_update.sh >> /tmp/ticker_refresh.log 2>&1

# weekly training data rebuild (sunday, 3am)
0 3 * * 0 /path/to/pipeline.sh >> /tmp/pipeline.log 2>&1
```

**why this works:**
- no complex orchestrator to learn
- simple to debug (just bash)
- logs to files you can check
- cron handles scheduling
- set-e ensures failures stop pipeline

---

## 2. configuration (minimal but powerful)

**config.yaml:**
```yaml
# single config file, environment-aware

data:
  date_range: ["2015-01-01", "2024-12-31"]
  target_stocks: 1500
  failed_stock_pct: 0.10

  market_cap_thresholds:
    us: [2e9, 10e9, 200e9, 1e15]
    swedish: [500e6, 2e9, 10e9, 1e15]

validation:
  lookback_days: 3
  rate_limit_delay: 1.0
  max_retries: 3

# test mode for quick iteration
test_mode:
  enabled: false
  sample_size: 200
  target_stocks: 500
```

**load in python:**
```python
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

# usage
config = load_config()
date_range = config['data']['date_range']
```

**why this works:**
- change parameters without editing code
- document your choices (config is self-documenting)
- easy to experiment (save config.yaml versions)

---

## 3. versioning (essential for learning)

**simple versioning structure:**
```
data/tickers_data/
  2025-10-29/
    raw.json
    filtered.json
    validated.json
    metadata.json
  2025-11-05/
    raw.json
    filtered.json
    validated.json
    metadata.json

data/training_data/
  v1_2025-10-29/
    training_data.pkl
    training_stocks_data.parquet
    metadata.json
  v2_2025-11-05/
    training_data.pkl
    training_stocks_data.parquet
    metadata.json
```

**metadata.json (lightweight tracking):**
```json
{
  "created_at": "2025-10-29T10:15:00",
  "counts": {
    "us": 5234,
    "sp500": 503,
    "swedish": 678,
    "total": 6415
  },
  "config": {
    "date_range": ["2015-01-01", "2024-12-31"],
    "target_stocks": 1500
  },
  "quality_checks": {
    "no_duplicates": true,
    "all_10_years": true,
    "price_volume_valid": true
  }
}
```

**why this works:**
- reproduce any experiment
- compare versions (did v2 improve over v1?)
- rollback if needed
- learn what worked and what didn't

---

## 4. testing (pragmatic approach)

**skip:** 80% coverage targets, integration test suites, ci/cd

**use:** critical path tests only

**test_critical.py:**
```python
"""test the stuff that breaks most often"""
import pytest
from filter_tickers import filter_junk_tickers, remove_duplicates_and_class_shares

def test_removes_dollar_signs():
    tickers = ['AAPL', 'ABR$D', 'MSFT', 'ACR$C']
    result = filter_junk_tickers(tickers)
    assert 'ABR$D' not in result
    assert 'ACR$C' not in result
    assert 'AAPL' in result

def test_removes_preferred_series():
    tickers = ['OXLC', 'OXLCG', 'OXLCI', 'OXLCL', 'AAPL']
    result = filter_junk_tickers(tickers)
    # should keep base, remove variants
    assert 'OXLC' in result
    assert 'OXLCG' not in result

def test_keeps_b_over_a():
    us = ['KELYA', 'KELYB', 'AAPL']
    sp500 = []
    result, count = remove_duplicates_and_class_shares(us, sp500)
    assert 'KELYB' in result
    assert 'KELYA' not in result
    assert count == 1

# run with: pytest test_critical.py
```

**why this works:**
- tests the bugs you actually hit
- fast to run (<5 seconds)
- catches regressions
- learn pytest basics without overhead

---

## 5. monitoring (solo dev style)

**skip:** grafana, datadog, prometheus

**use:** simple logging + notification script

**add to pipeline.sh:**
```bash
#!/bin/bash
set -e

LOG_FILE="/tmp/pipeline_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== pipeline start ==="
    date

    uv run python filter_tickers.py
    uv run python validate_tickers.py
    uv run python build_training_set.py

    echo "=== pipeline complete ==="
    date
} 2>&1 | tee "$LOG_FILE"

# check if training_data.pkl was created
if [ -f "training_data.pkl" ]; then
    echo "success: training_data.pkl created"
    ls -lh training_data.pkl
else
    echo "error: training_data.pkl not found"
    exit 1
fi
```

**optional: slack notification (if you want):**
```bash
# add to end of pipeline.sh
if [ $? -eq 0 ]; then
    curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"pipeline completed successfully"}' \
    YOUR_SLACK_WEBHOOK
fi
```

**why this works:**
- logs show what happened
- easy to grep logs for errors
- optional notifications without complex setup

---

## 6. data quality checks (lightweight)

**validate_output.py:**
```python
"""run after build_training_set.py to sanity check output"""
import pickle
import pandas as pd
from pathlib import Path

def validate_training_data():
    """quick checks to catch obvious problems"""

    # load data
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    checks = []

    # check 1: right number of stocks
    stock_count = len(data)
    checks.append(('stock_count', stock_count == 1500, f'got {stock_count}'))

    # check 2: all have 10 years data
    for ticker, df in data.items():
        years = df.index.year.nunique()
        if years < 10:
            checks.append((f'{ticker}_years', False, f'only {years} years'))

    # check 3: no all-nan columns
    for ticker, df in data.items():
        if df.isna().all().any():
            checks.append((f'{ticker}_nan', False, 'has all-nan column'))

    # check 4: price and volume valid
    for ticker, df in data.items():
        if (df['Close'] < 1).any():
            checks.append((f'{ticker}_price', False, 'has price < $1'))
        if (df['Volume'] < 50000).any():
            checks.append((f'{ticker}_volume', False, 'has volume < 50k'))

    # print results
    failures = [c for c in checks if not c[1]]
    if failures:
        print(f"validation failed: {len(failures)} checks")
        for check in failures:
            print(f"  {check[0]}: {check[2]}")
        return False
    else:
        print(f"validation passed: all {len(checks)} checks")
        return True

if __name__ == '__main__':
    success = validate_training_data()
    exit(0 if success else 1)
```

**add to pipeline.sh:**
```bash
uv run python build_training_set.py
uv run python validate_output.py || { echo "quality checks failed"; exit 1; }
```

**why this works:**
- catches bad data before you use it
- prevents wasted model training time
- learn what quality checks matter

---

## 7. experimentation workflow

**structure for learning:**
```
experiments/
  2025-10-29_baseline/
    config.yaml (copy of config used)
    training_data.pkl
    notes.md (what you tried, what worked)
  2025-11-05_more_stocks/
    config.yaml (target_stocks: 2000)
    training_data.pkl
    notes.md
  2025-11-12_shorter_window/
    config.yaml (date_range: 2020-2024)
    training_data.pkl
    notes.md
```

**notes.md template:**
```markdown
# experiment: more stocks (2000 instead of 1500)

## hypothesis
more training data → better model performance

## changes
- config: target_stocks 1500 → 2000
- pipeline ran successfully

## results
- training time: 75 min (vs 60 min baseline)
- validation loss: 0.42 (vs 0.45 baseline) ✓ improvement
- test accuracy: 73% (vs 71% baseline) ✓ improvement

## conclusion
keep 2000 stocks, worth the extra time

## next experiment
try swedish-only model
```

**why this works:**
- document your learning
- compare experiments objectively
- avoid repeating failed approaches
- build intuition about what works

---

## 8. debugging workflow

**when pipeline fails, systematic approach:**

**1. check logs:**
```bash
tail -f /tmp/pipeline.log
grep -i error /tmp/pipeline.log
```

**2. run step manually with verbose output:**
```bash
cd data
uv run python build_training_set.py 2>&1 | tee debug.log
```

**3. add debug mode to scripts:**
```python
# add to build_training_set.py
import sys
DEBUG = '--debug' in sys.argv

if DEBUG:
    print(f"processing {ticker}: market_cap={market_cap}, bucket={bucket}")
```

**4. test with small sample:**
```yaml
# config.yaml
test_mode:
  enabled: true
  sample_size: 10  # just 10 tickers
```

**why this works:**
- fast iteration when debugging
- learn to diagnose issues systematically
- no complex debugging infrastructure needed

---

## 9. backup strategy (simple but effective)

**automated backups:**
```bash
#!/bin/bash
# backup.sh - run daily

BACKUP_DIR="/Users/deaz/backups/pquant"
DATE=$(date +%Y%m%d)

mkdir -p "$BACKUP_DIR/$DATE"

# backup latest training data
cp -r data/training_data/v* "$BACKUP_DIR/$DATE/"

# backup configs
cp data/config.yaml "$BACKUP_DIR/$DATE/"

# keep only last 7 days
find "$BACKUP_DIR" -type d -mtime +7 -exec rm -rf {} +

echo "backup complete: $BACKUP_DIR/$DATE"
```

**schedule with cron:**
```bash
# daily backup at 1am
0 1 * * * /path/to/backup.sh
```

**why this works:**
- protect against accidental deletion
- no cloud service needed
- automatic cleanup (7 days retention)

---

## 10. git workflow (solo dev)

**what to commit:**
```
git add data/*.py
git add data/config.yaml
git add data/tests/
git add data/docs/
```

**what to gitignore:**
```gitignore
# gitignore
*.pkl
*.parquet
*.json
*.log
__pycache__/
.pytest_cache/
experiments/
training_data/
tickers_data/
```

**commit messages:**
```bash
git commit -m "fix: handle multiindex columns in validator"
git commit -m "feat: add market cap stratification"
git commit -m "refactor: split filter logic into separate functions"
git commit -m "docs: update quick-start with config usage"
```

**branching (optional):**
```bash
# working on new feature
git checkout -b experiment/swedish-only-model
# hack hack hack
git commit -m "feat: add swedish market cap thresholds"
# works? merge
git checkout main
git merge experiment/swedish-only-model
```

**why this works:**
- track what changed and why
- rollback bad changes
- learn git basics naturally

---

## practical implementation roadmap

**week 1: configuration**
- create config.yaml
- move hardcoded values to config
- add test_mode toggle
- result: easy to experiment

**week 2: versioning**
- add date folders for outputs
- create metadata.json writer
- save config snapshot with each run
- result: reproducible experiments

**week 3: automation**
- create pipeline.sh
- add error handling (set -e)
- schedule with cron
- result: hands-off execution

**week 4: quality**
- write validate_output.py
- add critical path tests
- integrate into pipeline.sh
- result: catch bad data early

**week 5: monitoring**
- structured logging
- simple notification (optional)
- experiment notes.md template
- result: track what works

---

## tools worth learning (solo dev)

**keep it simple:**
- bash scripting (orchestration)
- yaml (configuration)
- pytest (testing)
- git (version control)
- ripgrep/fd (debugging)

**skip (overkill for solo):**
- airflow/prefect
- kubernetes/docker
- ci/cd platforms
- monitoring dashboards
- database for metadata (just use json files)

---

## key principles (no bs version)

**1. modular code:**
- each script does one thing
- easy to test individual steps
- learn proper separation of concerns

**2. configuration over hardcoding:**
- change behavior without editing code
- document your choices
- experiment faster

**3. version everything:**
- save outputs with dates
- track what changed
- compare experiments objectively

**4. test critical paths:**
- not everything, just what breaks often
- fast feedback loop
- catch regressions

**5. automate boring stuff:**
- cron for scheduling
- bash for orchestration
- scripts for backups

**6. document for future you:**
- config.yaml documents choices
- metadata.json documents results
- notes.md documents learning

---

## bottom line

**your current 4-step pipeline is already professional:**
- modular design ✓
- single responsibility ✓
- efficient flow ✓

**add these for solo dev excellence:**
- config.yaml (experiment faster)
- versioned outputs (compare experiments)
- pipeline.sh + cron (automate execution)
- validate_output.py (catch bad data)
- test_critical.py (prevent regressions)
- experiments/ folder (document learning)

**skip the corporate stuff:**
- no orchestration platforms
- no monitoring dashboards
- no access control
- no complex infrastructure

**result:** professional system architecture without corporate overhead, optimized for learning and iteration.
