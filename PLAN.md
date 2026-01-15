# Refactor Plan

## Phase 1: Restructure data/

**Move scripts:**
```
data/1_get_tickers.py      → data/scripts/1_get_tickers.py
data/2_filter_tickers.py   → data/scripts/2_filter_tickers.py
data/3_validate_tickers.py → data/scripts/3_validate_tickers.py
data/4_build_training_set.py → data/scripts/4_build_ohlcv.py (rename)
data/5_update_training_data.py → data/scripts/5_update_ohlcv.py (rename)
```

**Rename folders:**
```
data/tickers_data/   → data/tickers/
data/training_data/  → data/datasets/
```

**Rename output files:**
```
training_stocks_data.parquet → ohlcv.parquet
```

**Add:**
- `data/scripts/6_build_features.py` - Creates dataset.parquet from ohlcv.parquet
- `data/run.py` - Unified entry point with interactive menu

**Update:**
- `data/README.md` - Reflect new structure
- All scripts - Update import paths

---

## Phase 2: Add lib/

**Create:**
```
lib/
├── __init__.py
├── data.py        # load_dataset(), temporal_split(), scale()
├── features.py    # build_features() - indicator + lag + label logic
└── eval.py        # evaluate(), plot_confusion(), backtest_quick()
```

**Source:** Extract from `indicators/notebooks/training-dataset-structure.ipynb`

---

## Phase 3: Add training/

**Create:**
```
training/
├── __init__.py
├── train.py       # Main script: load → split → scale → select → train → save
├── config.yaml    # model_type, hyperparams, feature_selection settings
├── tuner.py       # Optuna objective functions
└── models/
    ├── __init__.py
    ├── base.py      # BaseModel interface
    ├── xgboost.py   # XGBoost wrapper
    ├── ensemble.py  # Stacking ensemble
    └── lightgbm.py  # Optional
```

**Source:** Refactor from `reference_XGBOOST.ipynb`

---

## Phase 4: Add backtesting/

**Create:**
```
backtesting/
├── __init__.py
├── backtest.py    # Load model, run strategy, output metrics
└── strategies/
    ├── __init__.py
    └── pivot_bottom.py  # Entry on prediction, exit on target/stop
```

---

## Phase 5: Add prediction/

**Create:**
```
prediction/
├── __init__.py
└── predict.py     # Load model, fetch fresh data, generate signals
```

---

## Phase 6: Add runs/ structure

**Create:**
```
runs/
└── .gitkeep
```

Each training run saves to `runs/YYYYMMDD_HHMMSS/`:
- model.pkl
- scaler.pkl
- config.yaml (snapshot)
- metrics.json
- features.txt (selected features)

---

## Phase 7: Cleanup

**Delete:**
- `pipeline/` folder (replaced by lib/ + training/)
- `REFACTOR_PLAN.md` (replaced by this file, then delete this too)

**Keep:**
- `indicators/` - No changes needed
- `reference_XGBOOST.ipynb` - Keep as reference, move to `docs/` or delete later

---

## Order of Execution

1. [x] Phase 1: Restructure data/ (mostly file moves + run.py)
2. [ ] Phase 2: Add lib/ (extract from notebook)
3. [ ] Phase 3: Add training/ (refactor from old notebook)
4. [ ] Phase 4: Add backtesting/
5. [ ] Phase 5: Add prediction/
6. [ ] Phase 6: Add runs/
7. [ ] Phase 7: Cleanup

---

## Notes

- indicators/ stays as-is (already clean)
- Each phase should be a commit
- Test after each phase
