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

### Iteration 8 -- IMPROVED (+1.2562)

Score: -0.8599 | from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier     et = Ext
Commit: d72e4f1

### Iteration 9 -- KNOWLEDGE

No code changes.

### Iteration 2 -- REVERTED (-0.2314)

Score: -1.0913 |         num_leaves=63,         max_depth=-1,         subsample=0.7,         subs

### Iteration 3 -- REVERTED (-0.4830)

Score: -1.3429 | from sklearn.linear_model import LogisticRegression     lr = LogisticRegression(

### Iteration 4 -- REVERTED (-7.1806)

Score: -8.0405 | THRESHOLD = 0.90 

### Iteration 5 -- IMPROVED (+0.0823)

Score: -0.7776 |         weights=[1, 1, 2], 
Commit: 687a51a

### Iteration 6 -- REVERTED (-0.1494)

Score: -0.9270 |     # Candle morphology: lower wick ratio (3-day avg)     # Large lower wicks = 

### Iteration 7 -- IMPROVED (+0.2336)

Score: -0.5440 |     # Selling exhaustion: recent (5d) down-day volume as fraction of 20d total  
Commit: 065a819

### Iteration 8 -- REVERTED (-0.3677)

Score: -0.9117 |     # Accumulation-side volume: fraction of 5d volume on up-days     # Complemen

### Iteration 9 -- IMPROVED (+0.0005)

Score: -0.5435 |     # Variance ratio (5d/1d): mean-reversion vs trending regime     # VR < 1 = m
Commit: 3f318cb

### Iteration 10 -- IMPROVED (+0.0285)

Score: -0.5150 |     # Consecutive down days: running count of consecutive negative-return days  
Commit: 0b29a9e

### Iteration 11 -- REVERTED (-0.7004)

Score: -1.2154 |     # Drawdown z-score: how abnormal is this drawdown for THIS stock?     # Very

### Iteration 12 -- REVERTED (-0.0064)

Score: -0.5214 |         max_delta_step=1, 

### Iteration 13 -- GATE FAILED

Reason: 
Change: from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier from sklea

### Iteration 14 -- REVERTED (-0.2150)

Score: -0.7300 |     # Session decomposition: intraday (open->close) vs overnight (prev_close->op

### Iteration 15 -- REVERTED (-0.8272)

Score: -1.3422 |     # Cross-sectional ranks: where does this stock sit vs ALL peers on the same 

### Iteration 16 -- IMPROVED (+0.0616)

Score: -0.4534 |         min_child_weight=3,         gamma=0.05, 
Commit: 7d3b036

### Iteration 17 -- REVERTED (-0.1564)

Score: -0.6098 |         min_child_samples=30,         reg_lambda=1.0, 

### Iteration 18 -- REVERTED (-0.1128)

Score: -0.5662 |         min_child_weight=3,         min_split_gain=0.05, 

### Iteration 19 -- IMPROVED (+0.1055)

Score: -0.3479 |         max_depth=12,         min_samples_leaf=40, 
Commit: b60b086

### Iteration 20 -- IMPROVED (+0.1777)

Score: -0.1702 |         n_estimators=400,         n_estimators=400,         n_estimators=400, 
Commit: 00cd249

### Iteration 21 -- IMPROVED (+0.2741)

Score: 0.1039 |         n_estimators=800, 
Commit: c413349

### Iteration 22 -- REVERTED (-0.0566)

Score: 0.0473 |         n_estimators=600,         n_estimators=600, 

### Iteration 23 -- REVERTED (-0.2478)

Score: -0.1439 |         n_estimators=1200, 

### Iteration 24 -- REVERTED (-0.0129)

Score: 0.0910 |         subsample_freq=1, 

### Iteration 25 -- REVERTED (-0.4750)

Score: -0.3711 |     # Return skewness: 3rd moment captures distributional asymmetry     # Positi

### Iteration 26 -- REVERTED (-1.1221)

Score: -1.0182 |         colsample_bytree=0.6,         colsample_bytree=0.6, 

### Iteration 27 -- REVERTED (-0.9985)

Score: -0.8946 |         weights=[2, 1, 1], 

### Iteration 28 -- REVERTED (-0.1954)

Score: -0.0915 | FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interacti

### Iteration 29 -- REVERTED (-2.1689)

Score: -2.0650 | THRESHOLD = 0.80 

### Iteration 30 -- REVERTED (-0.4188)

Score: -0.3149 |         bootstrap=True, 

### Iteration 31 -- REVERTED (-3.3051)

Score: -3.2012 |     # Drawdown time structure: age and velocity of current drawdown episode     