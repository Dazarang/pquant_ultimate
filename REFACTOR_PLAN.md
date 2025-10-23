# Complete Refactor Plan: pQuant ML Stock Prediction System

## Architecture Principles
- **Performance First**: Vectorization, parallel processing, caching, Numba JIT
- **Modular OOP**: Single responsibility, <600 lines/file, composition over inheritance
- **Future-Proof**: Pi deployment now, broker API/cloud ready

---

## Phase 1: Indicators Module (No ta-lib)
**Goal**: Custom implementations, 10x faster with vectorization

### File Structure
```
pquant_ultimate/indicators/
├── base.py              (~150 lines) - BaseIndicator abstract class, validation
├── trend.py             (~200 lines) - SMA, EMA, WMA, VWAP
├── momentum.py          (~250 lines) - RSI, MACD, ADX, ROC, MOM
├── volatility.py        (~200 lines) - BBands, ATR, ADR, APZ, SAR
├── volume.py            (~150 lines) - OBV, ADOSC
├── pattern.py           (~200 lines) - find_pivots, Hammer patterns
├── cycle.py             (~150 lines) - HT_SINE, HT_LEADSINE, HT_TRENDMODE
├── composite.py         (~300 lines) - enhanced_real_time_strength_index
└── calculator.py        (~250 lines) - IndicatorCalculator orchestrator
```

### Key Changes
- **NumPy/Pandas only** - remove all talib calls
- **Numba JIT** - @njit decorators for hot loops (5-10x speedup)
- **Vectorized operations** - pandas rolling/expanding for efficiency
- **Lazy evaluation** - calculate indicators on-demand with caching
- **Type hints** - full typing for IDE support and validation

---

## Phase 2: Data & Feature Pipeline
**Goal**: Reproducible, fast, config-driven

### File Structure
```
pquant_ultimate/
├── data/
│   ├── fetcher.py       (~200 lines) - YFinance wrapper, retry logic, rate limiting
│   ├── cache_manager.py (~150 lines) - Disk cache, expiry, invalidation
│   └── validator.py     (~100 lines) - Data quality checks, missing values
├── features/
│   ├── engineer.py      (~250 lines) - FeatureEngineer class, indicator integration
│   ├── lag_creator.py   (~150 lines) - Efficient lag generation (vectorized)
│   ├── selector.py      (~300 lines) - RFE, correlation filtering, feature importance
│   └── scaler.py        (~100 lines) - StandardScaler wrapper with persistence
├── config/
│   ├── indicators.yaml  - Which indicators, periods [5,8,21,55,89,200]
│   ├── features.yaml    - Lag windows, diff pairs, derived features
│   └── data.yaml        - Tickers, date ranges, validation rules
```

### Performance Optimizations
- **Parallel fetching** - asyncio for multi-ticker downloads (3x faster)
- **Incremental updates** - only fetch new data, not full history
- **Polars option** - 5-10x faster than pandas for large datasets
- **Feature caching** - reuse computed indicators across runs

---

## Phase 3: Model Training Pipeline
**Goal**: Keep stacked ensemble, optimize training speed

### File Structure
```
pquant_ultimate/models/
├── base_model.py        (~150 lines) - BaseModel interface
├── ensemble_manager.py  (~400 lines) - StackedEnsemble orchestrator
├── tuner.py             (~300 lines) - Optuna multi-objective optimization
├── trainer.py           (~350 lines) - TrainingPipeline with CV, SMOTEENN
├── evaluator.py         (~200 lines) - Metrics calculation, plots
├── registry.py          (~150 lines) - Model versioning, metadata tracking
└── config/
    ├── xgboost.yaml     - XGBoost hyperparameter ranges
    ├── randomforest.yaml
    ├── catboost.yaml
    ├── svm.yaml
    └── ensemble.yaml    - Meta-learner config, CV strategy
```

### Performance Optimizations
- **Parallel Optuna trials** - n_jobs=-1 for all models
- **Early stopping** - save 30-40% training time
- **GPU acceleration** - XGBoost/CatBoost GPU support (optional)
- **Reduced trial count** - Smart initialization from previous runs
- **Sparse features** - Memory optimization for 560+ lag features

### Training Flow
1. Load configs → 2. Fetch & cache data → 3. Feature engineering
4. Parallel model tuning → 5. Train base models → 6. Train meta-model
7. Calibration → 8. Evaluation → 9. Save artifacts

---

## Phase 4: Professional Backtesting (VectorBT)
**Goal**: Test 1000s of stocks in seconds, systematic screening

### File Structure
```
pquant_ultimate/backtesting/
├── strategy.py          (~400 lines) - MLStrategy with VectorBT Portfolio
├── engine.py            (~250 lines) - BacktestEngine, batch execution
├── optimizer.py         (~200 lines) - Strategy parameter grid search
├── metrics.py           (~150 lines) - Custom metrics (expectancy, SQN)
└── reporter.py          (~300 lines) - HTML reports, interactive plots

pquant_ultimate/screening/
├── screener.py          (~350 lines) - MultiTickerScreener with parallel execution
├── filters.py           (~200 lines) - FilterCriteria (expectancy, Sharpe, etc.)
├── ranker.py            (~150 lines) - Composite scoring, ranking algorithms
└── pipeline.py          (~250 lines) - End-to-end screening workflow
```

### VectorBT Strategy Logic
```python
# Pseudo-code for ML-based strategy
class MLStrategy(vbt.Portfolio):
    - Load trained model/scaler
    - Generate ML predictions for all tickers (vectorized)
    - Entry: prediction > threshold & ADR > 3.5 & max 6 positions
    - Exit: 75% at +20% gain, trailing stop 10% below peak
    - Vectorized position sizing and risk management
```

### Screening Pipeline
1. Load ticker universe (100-1000 stocks)
2. Parallel data fetch + feature engineering
3. Batch ML predictions (all tickers at once)
4. VectorBT backtest (vectorized across all tickers)
5. Filter: expectancy>10%, Sharpe>0.75, win_rate>40%, profit_factor>2
6. Rank by composite score
7. Generate HTML report with interactive charts

**Speed**: ~5-10 seconds for 100 tickers vs ~5 minutes with backtesting.py

---

## Phase 5: Deployment Architecture
**Goal**: Pi-ready, broker-API prepared, monitored

### File Structure
```
pquant_ultimate/deployment/
├── predictor.py         (~200 lines) - RealTimePredictor, model loading
├── signal_generator.py  (~250 lines) - SignalGenerator with filtering logic
├── notifier.py          (~150 lines) - Email alerts, Telegram/Discord webhooks
├── broker_connector.py  (~400 lines) - Abstract BrokerAPI + implementations
│                                       (Interactive Brokers, Alpaca, etc.)
└── scheduler/
    ├── daily_job.py     (~200 lines) - Main execution script
    ├── monitor.py       (~150 lines) - Health checks, error alerts
    └── config.yaml      - Schedule, tickers, alert settings
```

### Deployment Modes
**Mode 1: Raspberry Pi (Current)**
- Run daily_job.py via cron at market open
- Generate predictions for watchlist
- Email alerts for high-probability bottoms
- Logs to file for monitoring

**Mode 2: Broker API (Future)**
- Same prediction pipeline
- BrokerConnector places actual orders
- Position management and risk controls
- Real-time monitoring dashboard

**Mode 3: Cloud (Future)**
- AWS Lambda / Google Cloud Functions
- Scheduled triggers
- Store results in database
- Web dashboard for signals

---

## Phase 6: Utilities & Testing
```
pquant_ultimate/
├── utils/
│   ├── logging.py       - Structured logging with levels
│   ├── metrics.py       - Performance tracking, profiling
│   └── validation.py    - Data/config validation schemas
├── tests/
│   ├── test_indicators/ - Unit tests for each indicator
│   ├── test_features/   - Feature engineering tests
│   ├── test_models/     - Model training/prediction tests
│   └── test_backtesting/- Strategy and screening tests
└── notebooks/
    ├── 01_explore_data.ipynb
    ├── 02_indicator_validation.ipynb  - Compare with ta-lib
    ├── 03_model_analysis.ipynb
    └── 04_backtest_results.ipynb
```

---

## Implementation Order (Performance-Optimized)

### Sprint 1: Foundation (Week 1)
1. Project structure setup
2. Indicators module (start with trend.py, momentum.py - most used)
3. Numba optimization for hot loops
4. Unit tests + validation vs ta-lib

### Sprint 2: Data Pipeline (Week 2)
1. Data fetcher with caching
2. Feature engineering with indicator integration
3. Lag generator (vectorized)
4. Config system (YAML-based)

### Sprint 3: Model Pipeline (Week 2-3)
1. Base model interfaces
2. Ensemble manager
3. Optuna tuner with parallel execution
4. Training pipeline with proper CV

### Sprint 4: Backtesting (Week 3-4)
1. VectorBT strategy implementation
2. Backtest engine
3. Screening pipeline
4. Performance reporting

### Sprint 5: Deployment (Week 4)
1. Prediction service
2. Signal generator
3. Notifier (email/alerts)
4. Daily scheduler for Pi

### Sprint 6: Polish (Week 5)
1. Testing & validation
2. Documentation
3. Performance profiling & optimization
4. Notebook examples

---

## Key Dependencies (pyproject.toml)
```toml
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "vectorbt>=0.26",  # Professional backtesting
    "xgboost>=2.0",
    "catboost>=1.2",
    "scikit-learn>=1.4",
    "optuna>=3.5",
    "yfinance>=0.2.40",
    "numba>=0.59",     # JIT compilation
    "pyyaml>=6.0",     # Config files
    "plotly>=5.18",    # Interactive plots
    "joblib>=1.3",     # Model persistence
]
```

---

## Success Metrics
- **Speed**: 10x faster backtesting (VectorBT vs backtesting.py)
- **Scalability**: Screen 500+ stocks in <30 seconds
- **Code Quality**: All files <600 lines, 90%+ test coverage
- **Performance**: Match or exceed current model metrics
- **Maintainability**: Full type hints, documentation, modular design

---

## Current System Analysis

### Old Indicators Module (`indicators.py` - 601 lines)
**Issues:**
- Violates 600-line limit
- Heavy ta-lib dependency
- Monolithic structure
- Mixed responsibilities

**What to preserve:**
- find_pivots logic (core labeling mechanism)
- enhanced_real_time_strength_index (custom composite indicator)
- All indicator calculations (reimplement without ta-lib)

### Old Model Notebook
**What works:**
- Stacked ensemble approach (good performance)
- Feature engineering pipeline (indicators + lags)
- Pivot-based labeling strategy
- RFE feature selection

**What to improve:**
- Reproducibility (notebook → code)
- Speed (parallel tuning, caching)
- Modularity (separate concerns)
- Configuration (YAML-driven)

### Old Backtesting Notebooks
**What to replace:**
- backtesting.py → VectorBT (10x faster)
- Manual ticker filtering → Automated screening
- One-by-one testing → Batch vectorized testing
- Ad-hoc metrics → Systematic reporting

---

## Migration Strategy

### Phase 1: Indicators (No Breaking Changes)
1. Implement custom indicators in parallel
2. Validate outputs match ta-lib exactly
3. Switch imports once validated
4. Remove ta-lib dependency

### Phase 2: Model Pipeline (Gradual Migration)
1. Extract notebook code to modules
2. Add config files for reproducibility
3. Test both old/new pipelines produce same results
4. Switch to new pipeline

### Phase 3: Backtesting (New System)
1. Implement VectorBT strategy
2. Run parallel tests (backtesting.py vs VectorBT)
3. Validate results match
4. Adopt VectorBT fully

### Phase 4: Deployment (Incremental)
1. Build new prediction service
2. Test locally on Pi
3. Run in parallel with old system
4. Switch over once validated

---

## Risk Mitigation

### Technical Risks
**Risk**: Custom indicators don't match ta-lib exactly
**Mitigation**: Unit tests comparing outputs, tolerance thresholds

**Risk**: VectorBT learning curve
**Mitigation**: Start with simple strategies, documentation, examples

**Risk**: Performance regression
**Mitigation**: Benchmark at each phase, profiling, optimization

### Operational Risks
**Risk**: Pi deployment breaks
**Mitigation**: Keep old system running, gradual rollout, rollback plan

**Risk**: Model performance degrades
**Mitigation**: A/B testing, monitoring, evaluation metrics

---

## Next Steps

1. **Review & Approve Plan** - Discuss any concerns or changes
2. **Set Up Environment** - Create repo structure, install dependencies
3. **Start Sprint 1** - Begin with indicators/base.py and trend.py
4. **Iterate** - Weekly reviews, adjust plan as needed
