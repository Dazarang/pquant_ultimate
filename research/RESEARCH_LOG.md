# Research Log

## Iter 0: Baseline
XGBClassifier on all 1,336 stocks, all 220 features, threshold=0.5.
New eval system: excess return vs market, MAE path risk, multi-horizon gate, regime breakdown.
Baseline score: **-4.5912**

Key observations:
- 71,968 signals (24% of val set) -- too many, threshold needs tuning
- Excess return ~0% -- no alpha over market
- Win rate 49.7% -- coin flip
- Knife rate 19.9% -- buying into continued drops
- Bear regime slightly better than bull (+0.30% vs -0.56%)

### Iteration 1 -- REVERTED (-0.0151)
Score: -4.6063 | THRESHOLD = 0.30 

### Iteration 2 -- REVERTED (-0.0151)
Score: -4.6063 | from lightgbm import LGBMClassifier     model = LGBMClassifier(         n_estima

### Iteration 3 -- REVERTED (-0.0554)
Score: -4.6466 |         max_depth=4,         min_child_weight=10,         reg_alpha=0.5,        

### Iteration 4 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change: import numpy as np     g = df.groupby("stock_id")      # Return acceleration: 5d

### Iteration 5 -- REVERTED (-0.0010)
Score: -4.5922 |     g = df.groupby("stock_id")      # Intraday recovery: fraction of daily range

### Iteration 6 -- REVERTED (-0.0172)
Score: -4.6084 |         n_estimators=500,         learning_rate=0.03, 

### Iteration 7 -- REVERTED (-0.0384)
Score: -4.6296 | FEATURE_GROUPS = ["base", "advanced"] 

### Iteration 8 -- IMPROVED (+0.0162)
Score: -4.5750 | THRESHOLD = 0.55 
Commit: 9b6da39

### Iteration 9 -- IMPROVED (+0.0242)
Score: -4.5508 | THRESHOLD = 0.60 
Commit: 40601e5

### Iteration 10 -- IMPROVED (+0.0951)
Score: -4.4557 | THRESHOLD = 0.65 
Commit: 48ef559

### Iteration 11 -- IMPROVED (+0.1551)
Score: -4.3006 | THRESHOLD = 0.70 
Commit: 845411a
