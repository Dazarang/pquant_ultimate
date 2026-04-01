# Research Log

Walk-forward evaluation: 6 folds x 6 months (2023-03 to 2026-03), data from 2020+.

### Iteration 1 -- IMPROVED (+0.7356)
Score: -0.0160 | from lightgbm import LGBMClassifier  # noqa: E402     model = LGBMClassifier(   
Commit: cb4727f

### Iteration 2 -- REVERTED (-0.1623)
Score: -0.1783 | import numpy as np     g = df.groupby("stock_id")      candle_range = df["high"]

### Iteration 3 -- IMPROVED (+0.2438)
Score: 0.2278 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"] 
Commit: ad09516

### Iteration 4 -- REVERTED (-0.2438)
Score: -0.0160 | FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in

### Iteration 5 -- IMPROVED (+0.1281)
Score: 0.3559 | from research.model_wrappers import CatBoostWrapper  # noqa: E402     model = Ca
Commit: 352a9af

### Iteration 6 -- REVERTED (-0.0168)
Score: 0.3391 |         iterations=3000,         depth=5,         learning_rate=0.005,         l

### Iteration 7 -- REVERTED (-0.1294)
Score: 0.2265 | from sklearn.ensemble import VotingClassifier  # noqa: E402 from xgboost import 

### Iteration 8 -- REVERTED (-0.9626)
Score: -0.6067 | from research.model_wrappers import RankingXGBClassifier  # noqa: E402     model

### Iteration 9 -- REVERTED (-0.1353)
Score: 0.2206 | import numpy as np     g = df.groupby("stock_id")      # Consecutive negative-re

### Iteration 10 -- REVERTED (-0.1475)
Score: 0.2084 | from lightgbm import LGBMClassifier  # noqa: E402     model = LGBMClassifier(   

### Iteration 11 -- REVERTED (-0.3499)
Score: 0.0060 |         grow_policy="Lossguide",         max_leaves=64,         depth=8, 
