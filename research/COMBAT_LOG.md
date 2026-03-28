# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 1 -- REVERTED (-0.0151)
Score: -4.6063 vs best -4.5912
Change: THRESHOLD = 0.30 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..6277dca 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -37,7 +37,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold
-THRESHOLD = 0.5
+THRESHOLD = 0.30
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 2 -- REVERTED (-0.0151)
Score: -4.6063 vs best -4.5912
Change: from lightgbm import LGBMClassifier     model = LGBMClassifier(         n_estima
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..5d2dc3c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -11,7 +11,7 @@ import sys
 from pathlib import Path
 
 import numpy as np  # noqa: F401 -- available for researcher
-from xgboost import XGBClassifier
+from lightgbm import LGBMClassifier
 
 # Ensure project root is on path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
@@ -46,20 +46,19 @@ THRESHOLD = 0.5
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    neg = (y_train == 0).sum()
-    pos = (y_train == 1).sum()
-
-    model = XGBClassifier(
-        n_estimators=300,
+    model = LGBMClassifier(
+        n_estimators=500,
         max_depth=6,
-        learning_rate=0.05,
-        subsample=0.8,
-        colsample_bytree=0.8,
-        scale_pos_weight=neg / pos,
-        tree_method="hist",
+        learning_rate=0.03,
+        subsample=0.7,
+        colsample_bytree=0.7,
+        min_child_samples=50,
+        reg_alpha=0.1,
+        reg_lambda=1.0,
+        is_unbalance=True,
         random_state=42,
         n_jobs=-1,
-        verbosity=0,
+        verbose=-1,
     )
     return model
 
```

### Iteration 3 -- REVERTED (-0.0554)
Score: -4.6466 vs best -4.5912
Change:         max_depth=4,         min_child_weight=10,         reg_alpha=0.5,        
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..3f44e73 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -51,10 +51,13 @@ def build_model(y_train):
 
     model = XGBClassifier(
         n_estimators=300,
-        max_depth=6,
+        max_depth=4,
         learning_rate=0.05,
+        min_child_weight=10,
         subsample=0.8,
         colsample_bytree=0.8,
+        reg_alpha=0.5,
+        reg_lambda=3,
         scale_pos_weight=neg / pos,
         tree_method="hist",
         random_state=42,
```

### Iteration 5 -- REVERTED (-0.0010)
Score: -4.5922 vs best -4.5912
Change:     g = df.groupby("stock_id")      # Intraday recovery: fraction of daily range
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 68d988e..7fdb2b2 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,6 +22,37 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
+    g = df.groupby("stock_id")
+
+    # Intraday recovery: fraction of daily range recaptured from the low
+    # High value = strong buying pressure at lows (continuous hammer signal)
+    rng = df["high"] - df["low"]
+    df["intraday_recovery"] = (df["close"] - df["low"]) / rng.replace(0, float("nan"))
+    new_features.append("intraday_recovery")
+
+    # Cumulative selling pressure: sum of negative returns over 5 days
+    # Deep negative = heavy recent damage, potential capitulation zone
+    neg_ret = df["ret_1d"].clip(upper=0)
+    df["cum_sell_pressure_5d"] = neg_ret.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).sum()
+    )
+    new_features.append("cum_sell_pressure_5d")
+
+    # Volume climax: volume spike magnitude on down days only
+    # High value = capitulation selling (big volume + big drop)
+    df["volume_climax"] = (
+        df["volume_ratio"] * df["ret_1d"].abs() * (df["ret_1d"] < 0).astype(float)
+    )
+    new_features.append("volume_climax")
+
+    # Bounce from 5-day low, ATR-normalized
+    # Near zero = sitting at local low; positive = recovering from it
+    low_5d = g["low"].transform(lambda x: x.rolling(5, min_periods=1).min())
+    df["bounce_from_5d_low"] = (df["close"] - low_5d) / df["atr_14"].replace(
+        0, float("nan")
+    )
+    new_features.append("bounce_from_5d_low")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 6 -- REVERTED (-0.0172)
Score: -4.6084 vs best -4.5912
Change:         n_estimators=500,         learning_rate=0.03, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..b25f39e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -50,9 +50,9 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
 
     model = XGBClassifier(
-        n_estimators=300,
+        n_estimators=500,
         max_depth=6,
-        learning_rate=0.05,
+        learning_rate=0.03,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

### Iteration 7 -- REVERTED (-0.0384)
Score: -4.6296 vs best -4.5912
Change: FEATURE_GROUPS = ["base", "advanced"] 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..38d3b67 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -30,7 +30,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = None  # None = all
+FEATURE_GROUPS = ["base", "advanced"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 1 -- REVERTED (-0.0646)
Score: -2.4770 vs best -2.4124
Change: THRESHOLD = 0.90 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 4c92c61..0baa5ee 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -37,7 +37,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold
-THRESHOLD = 0.85
+THRESHOLD = 0.90
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 3 -- REVERTED (-0.0829)
Score: -2.4262 vs best -2.3433
Change:         gamma=1.0, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 4c92c61..377fe5f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -53,6 +53,7 @@ def build_model(y_train):
         n_estimators=300,
         max_depth=6,
         learning_rate=0.05,
+        gamma=1.0,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

### Iteration 5 -- REVERTED (-0.0208)
Score: -2.3281 vs best -2.3073
Change:     # Momentum acceleration (2nd derivative of returns):     # Positive during d
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 870bf73..1257438 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -65,6 +65,13 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).transform(lambda x: x.rolling(5, min_periods=1).mean())
     new_features.append("overnight_gap_trend_5d")
 
+    # Momentum acceleration (2nd derivative of returns):
+    # Positive during downtrend = decline decelerating = bottom forming
+    # Targets knife_rate reduction by filtering false bottoms still in freefall
+    ret_ma_5d = g["ret_1d"].transform(lambda x: x.rolling(5, min_periods=2).mean())
+    df["momentum_accel"] = ret_ma_5d - ret_ma_5d.groupby(df["stock_id"]).shift(5)
+    new_features.append("momentum_accel")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 7 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change: from catboost import CatBoostClassifier     cat = CatBoostClassifier(         it
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 81c9b10..dd13538 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -11,6 +11,7 @@ import sys
 from pathlib import Path
 
 import numpy as np  # noqa: F401 -- available for researcher
+from catboost import CatBoostClassifier
 from lightgbm import LGBMClassifier
 from sklearn.ensemble import VotingClassifier
 from xgboost import XGBClassifier
@@ -76,8 +77,19 @@ def build_model(y_train):
         verbose=-1,
     )
 
+    cat = CatBoostClassifier(
+        iterations=300,
+        depth=6,
+        learning_rate=0.05,
+        scale_pos_weight=neg / pos,
+        random_seed=44,
+        thread_count=-1,
+        verbose=0,
+        allow_writing_files=False,
+    )
+
     model = VotingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat)],
         voting="soft",
     )
     return model
```
