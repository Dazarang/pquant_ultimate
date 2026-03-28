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

### Iteration 12 -- IMPROVED (+0.3282)
Score: -3.9724 | THRESHOLD = 0.75 
Commit: e133fb5

### Iteration 13 -- IMPROVED (+0.6464)
Score: -3.3260 | THRESHOLD = 0.80 
Commit: 7a2cdbf

### Iteration 14 -- IMPROVED (+0.9136)
Score: -2.4124 | THRESHOLD = 0.85 
Commit: 4e3ae9e

### Iteration 15 -- KNOWLEDGE
No code changes.

### Iteration 16 -- KNOWLEDGE
No code changes.

### Iteration 17 -- KNOWLEDGE
No code changes.

### Iteration 18 -- KNOWLEDGE
No code changes.

### Iteration 1 -- REVERTED (-0.0646)
Score: -2.4770 | THRESHOLD = 0.90 

### Iteration 2 -- IMPROVED (+0.0691)
Score: -2.3433 |     g = df.groupby("stock_id")     daily_range = df["high"] - df["low"]      # R
Commit: 867c6e7

### Iteration 3 -- REVERTED (-0.0829)
Score: -2.4262 |         gamma=1.0, 

### Iteration 4 -- IMPROVED (+0.0360)
Score: -2.3073 |     # Overnight gap: (open - prev_close) / ATR     # Captures pre-market/after-h
Commit: 037e7d6

### Iteration 5 -- REVERTED (-0.0208)
Score: -2.3281 |     # Momentum acceleration (2nd derivative of returns):     # Positive during d

### Iteration 6 -- IMPROVED (+0.1912)
Score: -2.1161 | from lightgbm import LGBMClassifier from sklearn.ensemble import VotingClassifie
Commit: 7c04390

### Iteration 7 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change: from catboost import CatBoostClassifier     cat = CatBoostClassifier(         it
