# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 25 -- REVERTED (-0.4750)
Score: -0.3711 vs best 0.1039
Change:     # Return skewness: 3rd moment captures distributional asymmetry     # Positi
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index c69818d..3a4796c 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -125,6 +125,19 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).cumsum()
     new_features.append("streak_depth")
 
+    # Return skewness: 3rd moment captures distributional asymmetry
+    # Positive skew during decline = large up bounces = buyers stepping in = bottom
+    # Negative skew = large drops dominate = panic selling = knife risk
+    df["return_skew_10d"] = g["ret_1d"].transform(
+        lambda x: x.rolling(10, min_periods=5).skew()
+    )
+    new_features.append("return_skew_10d")
+
+    df["return_skew_20d"] = g["ret_1d"].transform(
+        lambda x: x.rolling(20, min_periods=10).skew()
+    )
+    new_features.append("return_skew_20d")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 26 -- REVERTED (-1.1221)
Score: -1.0182 vs best 0.1039
Change:         colsample_bytree=0.6,         colsample_bytree=0.6, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..627a94c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -58,7 +58,7 @@ def build_model(y_train):
         min_child_weight=3,
         gamma=0.05,
         subsample=0.8,
-        colsample_bytree=0.8,
+        colsample_bytree=0.6,
         scale_pos_weight=neg / pos,
         tree_method="hist",
         random_state=42,
@@ -71,7 +71,7 @@ def build_model(y_train):
         max_depth=6,
         learning_rate=0.05,
         subsample=0.8,
-        colsample_bytree=0.8,
+        colsample_bytree=0.6,
         scale_pos_weight=neg / pos,
         random_state=43,
         n_jobs=-1,
```

### Iteration 27 -- REVERTED (-0.9985)
Score: -0.8946 vs best 0.1039
Change:         weights=[2, 1, 1], 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..5eed6e5 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -90,7 +90,7 @@ def build_model(y_train):
     model = VotingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et)],
         voting="soft",
-        weights=[1, 1, 2],
+        weights=[2, 1, 1],
     )
     return model
 
```

### Iteration 28 -- REVERTED (-0.1954)
Score: -0.0915 vs best 0.1039
Change: FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interacti
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..ae2b8eb 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -32,7 +32,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = None  # None = all
+FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interaction"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 29 -- REVERTED (-2.1689)
Score: -2.0650 vs best 0.1039
Change: THRESHOLD = 0.80 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..84617c4 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -39,7 +39,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold
-THRESHOLD = 0.85
+THRESHOLD = 0.80
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 30 -- REVERTED (-0.4188)
Score: -0.3149 vs best 0.1039
Change:         bootstrap=True, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..22eaa4c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -82,6 +82,7 @@ def build_model(y_train):
         n_estimators=800,
         max_depth=12,
         min_samples_leaf=40,
+        bootstrap=True,
         class_weight="balanced",
         random_state=44,
         n_jobs=-1,
```

### Iteration 31 -- REVERTED (-3.3051)
Score: -3.2012 vs best 0.1039
Change:     # Drawdown time structure: age and velocity of current drawdown episode     
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index c69818d..4c84f9f 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -125,6 +125,24 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).cumsum()
     new_features.append("streak_depth")
 
+    # Drawdown time structure: age and velocity of current drawdown episode
+    # Captures WHEN in the decline we are, not just how deep
+    # Mature slow declines = orderly selling = better bottom candidate
+    # Fresh violent drops = crash/panic = knife risk
+    is_near_high = (df["drawdown"] > -0.02).astype(int)
+    dd_episode = is_near_high.groupby(df["stock_id"]).cumsum()
+    df["drawdown_age"] = (1 - is_near_high).groupby(
+        [df["stock_id"], dd_episode]
+    ).cumsum()
+    new_features.append("drawdown_age")
+
+    # Drawdown velocity: depth per day in drawdown
+    # Near zero = slow grind down (stabilizing); very negative = rapid crash
+    df["drawdown_velocity"] = df["drawdown"] / df["drawdown_age"].replace(
+        0, float("nan")
+    )
+    new_features.append("drawdown_velocity")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 1 -- REVERTED (-1.0271)
Score: -0.9232 vs best 0.1039
Change: from research.model_wrappers import CatBoostWrapper  # noqa: E402      cb = CatB
```diff
diff --git a/research/experiment.py b/research/experiment.py
index eaa58c4..815d492 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -18,6 +18,8 @@ from lightgbm import LGBMClassifier  # noqa: E402
 from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
+from research.model_wrappers import CatBoostWrapper  # noqa: E402
+
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
 from lib.eval import tiered_eval  # noqa: E402
 from research.features_lab import add_custom_features  # noqa: E402
@@ -87,10 +89,19 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
+    cb = CatBoostWrapper(
+        iterations=400,
+        depth=6,
+        learning_rate=0.05,
+        scale_pos_weight=neg / pos,
+        verbose=0,
+        random_seed=45,
+    )
+
     model = VotingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et), ("cb", cb)],
         voting="soft",
-        weights=[1, 1, 2],
+        weights=[1, 1, 2, 1],
     )
     return model
 
```

### Iteration 2 -- REVERTED (-4.3193)
Score: -4.2154 vs best 0.1039
Change:         max_depth=8, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index eaa58c4..38d6050 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -80,7 +80,7 @@ def build_model(y_train):
 
     et = ExtraTreesClassifier(
         n_estimators=800,
-        max_depth=12,
+        max_depth=8,
         min_samples_leaf=40,
         class_weight="balanced",
         random_state=44,
```

### Iteration 3 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 142).
Change: from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # noqa: E
```diff
diff --git a/research/experiment.py b/research/experiment.py
index eaa58c4..fe66332 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -15,7 +15,7 @@ sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
 from lightgbm import LGBMClassifier  # noqa: E402
-from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier  # noqa: E402
+from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
@@ -78,7 +78,7 @@ def build_model(y_train):
         verbose=-1,
     )
 
-    et = ExtraTreesClassifier(
+    rf = RandomForestClassifier(
         n_estimators=800,
         max_depth=12,
         min_samples_leaf=40,
@@ -88,7 +88,7 @@ def build_model(y_train):
     )
 
     model = VotingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
         voting="soft",
         weights=[1, 1, 2],
     )
```
Traceback:
```
Running experiment (timeout: 1800s)...
GATE VIOLATION: Experiment crashed (exit code 142).
--- Last 30 lines of log ---
/Users/deaz/.pyenv/versions/3.14.3/lib/python3.14/multiprocessing/resource_tracker.py:396: UserWarning: resource_tracker: There appear to be 2 leaked semaphore objects to clean up at shutdown: {'/loky-95173-57zc3x6l', '/loky-95173-kqvh2jlw'}
  warnings.warn(
```

### Iteration 4 -- GATE FAILED
Reason: 
Change:         boosting_type="dart", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index eaa58c4..e6e801a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -70,6 +70,7 @@ def build_model(y_train):
         n_estimators=400,
         max_depth=6,
         learning_rate=0.05,
+        boosting_type="dart",
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```
Traceback:
```
Score: -inf
Passed tiers: False
GATE: Tiers not passed. Score=-inf
GATE_STATUS=FAILED
COMPOSITE_SCORE=-inf
```

### Iteration 5 -- REVERTED (-0.8785)
Score: -0.7746 vs best 0.1039
Change:         learning_rate=0.03,         learning_rate=0.03, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index eaa58c4..5a8fba1 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -54,7 +54,7 @@ def build_model(y_train):
     xgb = XGBClassifier(
         n_estimators=400,
         max_depth=6,
-        learning_rate=0.05,
+        learning_rate=0.03,
         min_child_weight=3,
         gamma=0.05,
         subsample=0.8,
@@ -69,7 +69,7 @@ def build_model(y_train):
     lgbm = LGBMClassifier(
         n_estimators=400,
         max_depth=6,
-        learning_rate=0.05,
+        learning_rate=0.03,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

### Iteration 8 -- REVERTED (-2.1732)
Score: -1.5352 vs best 0.6380
Change:         max_features=0.5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index b35bb99..544a147 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -83,6 +83,7 @@ def build_model(y_train):
     et = ExtraTreesClassifier(
         n_estimators=800,
         max_depth=12,
+        max_features=0.5,
         min_samples_leaf=40,
         class_weight="balanced",
         random_state=44,
```

### Iteration 9 -- REVERTED (-1.2028)
Score: -0.5648 vs best 0.6380
Change:         max_depth=18,         min_samples_leaf=50, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index b35bb99..0759948 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -82,8 +82,8 @@ def build_model(y_train):
 
     et = ExtraTreesClassifier(
         n_estimators=800,
-        max_depth=12,
-        min_samples_leaf=40,
+        max_depth=18,
+        min_samples_leaf=50,
         class_weight="balanced",
         random_state=44,
         n_jobs=-1,
```

### Iteration 10 -- REVERTED (-1.8330)
Score: -1.1950 vs best 0.6380
Change:         reg_alpha=1.0,         reg_alpha=1.0, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index b35bb99..826cb0e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -57,7 +57,7 @@ def build_model(y_train):
         learning_rate=0.05,
         min_child_weight=3,
         gamma=0.05,
-        reg_alpha=0.5,
+        reg_alpha=1.0,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
@@ -71,7 +71,7 @@ def build_model(y_train):
         n_estimators=400,
         max_depth=6,
         learning_rate=0.05,
-        reg_alpha=0.5,
+        reg_alpha=1.0,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

### Iteration 11 -- REVERTED (-0.9762)
Score: -0.3382 vs best 0.6380
Change:     # Momentum deceleration: 5d return change (this week vs last week)     # Pos
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index c69818d..fe92090 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -125,6 +125,14 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).cumsum()
     new_features.append("streak_depth")
 
+    # Momentum deceleration: 5d return change (this week vs last week)
+    # Positive = decline slowing or recovering = bottom signal
+    # Negative = decline accelerating = knife risk
+    # Explicit second-derivative that tree models can't compose from raw lag features
+    ret_5d_cur = g["close"].transform(lambda x: x.pct_change(5))
+    df["momentum_decel_5d"] = ret_5d_cur - ret_5d_cur.groupby(df["stock_id"]).shift(5)
+    new_features.append("momentum_decel_5d")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 12 -- REVERTED (-2.4597)
Score: -1.8217 vs best 0.6380
Change: THRESHOLD = 0.86 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index b35bb99..a326a60 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -39,7 +39,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold
-THRESHOLD = 0.85
+THRESHOLD = 0.86
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 13 -- REVERTED (-0.4931)
Score: 0.1449 vs best 0.6380
Change:         num_leaves=20, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index b35bb99..1c3afcc 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -70,6 +70,7 @@ def build_model(y_train):
     lgbm = LGBMClassifier(
         n_estimators=400,
         max_depth=6,
+        num_leaves=20,
         learning_rate=0.05,
         reg_alpha=0.5,
         subsample=0.8,
```

### Iteration 14 -- REVERTED (-0.9795)
Score: -0.3415 vs best 0.6380
Change:         max_depth=5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index b35bb99..f01c3c7 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -53,7 +53,7 @@ def build_model(y_train):
 
     xgb = XGBClassifier(
         n_estimators=400,
-        max_depth=6,
+        max_depth=5,
         learning_rate=0.05,
         min_child_weight=3,
         gamma=0.05,
```

### Iteration 17 -- REVERTED (-1.4174)
Score: -0.0297 vs best 1.3877
Change:     # Market dispersion: cross-sectional std of daily returns (20d smoothed)    
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 81d4bf9..2f82d0c 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -144,6 +144,16 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     )
     new_features.append("market_breadth_20d")
 
+    # Market dispersion: cross-sectional std of daily returns (20d smoothed)
+    # High = stocks moving independently (idiosyncratic risk, individual bottoms recoverable)
+    # Low = stocks moving together (correlated selling, "bottoms" more likely knives)
+    # Orthogonal to market_trend (direction) and market_breadth (participation)
+    daily_mkt_dispersion = df.groupby("date")["ret_1d"].transform("std")
+    df["market_dispersion_20d"] = daily_mkt_dispersion.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(20, min_periods=5).mean()
+    )
+    new_features.append("market_dispersion_20d")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 18 -- REVERTED (-1.6200)
Score: -0.2323 vs best 1.3877
Change:     # Relative strength vs market: per-stock excess return over market average  
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 81d4bf9..fd26c52 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -144,6 +144,16 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     )
     new_features.append("market_breadth_20d")
 
+    # Relative strength vs market: per-stock excess return over market average
+    # Positive = outperforming peers (relative recovery = genuine bottom signal)
+    # Negative = underperforming peers (relative weakness = knife risk)
+    # Orthogonal to market_trend_20d (same for all stocks) — this is stock-specific
+    excess_ret = df["ret_1d"] - daily_mkt_ret
+    df["rel_strength_10d"] = excess_ret.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(10, min_periods=3).mean()
+    )
+    new_features.append("rel_strength_10d")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 20 -- REVERTED (-2.4793)
Score: 0.1298 vs best 2.6091
Change:         weights=[1, 1, 4], 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c6399a8..66fceda 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -92,7 +92,7 @@ def build_model(y_train):
     model = VotingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et)],
         voting="soft",
-        weights=[1, 1, 3],
+        weights=[1, 1, 4],
     )
     return model
 
```

### Iteration 1 -- REVERTED (-1.5084)
Score: 1.1007 vs best 2.6091
Change:     # Volume-return correlation: rolling Pearson corr between volume_ratio and r
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 81d4bf9..9b4f161 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -144,6 +144,27 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     )
     new_features.append("market_breadth_20d")
 
+    # Volume-return correlation: rolling Pearson corr between volume_ratio and ret_1d
+    # Negative = high volume on down days (active selling) = knife risk
+    # Positive/zero = volume decoupled from direction or on up days = bottom signal
+    vol = df["volume_ratio"]
+    ret = df["ret_1d"]
+    vol_ret = vol * ret
+
+    g_vol = vol.groupby(df["stock_id"])
+    g_ret = ret.groupby(df["stock_id"])
+    g_vr = vol_ret.groupby(df["stock_id"])
+
+    mean_vol = g_vol.transform(lambda x: x.rolling(10, min_periods=5).mean())
+    mean_ret = g_ret.transform(lambda x: x.rolling(10, min_periods=5).mean())
+    mean_vr = g_vr.transform(lambda x: x.rolling(10, min_periods=5).mean())
+    std_vol = g_vol.transform(lambda x: x.rolling(10, min_periods=5).std())
+    std_ret = g_ret.transform(lambda x: x.rolling(10, min_periods=5).std())
+
+    cov_vr = mean_vr - mean_vol * mean_ret
+    df["vol_ret_corr_10d"] = cov_vr / (std_vol * std_ret).replace(0, float("nan"))
+    new_features.append("vol_ret_corr_10d")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 2 -- REVERTED (-0.8172)
Score: 1.7919 vs best 2.6091
Change:         learning_rate=0.08,         learning_rate=0.08, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c6399a8..3d243df 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -54,7 +54,7 @@ def build_model(y_train):
     xgb = XGBClassifier(
         n_estimators=400,
         max_depth=6,
-        learning_rate=0.05,
+        learning_rate=0.08,
         min_child_weight=3,
         gamma=0.05,
         reg_alpha=0.5,
@@ -70,7 +70,7 @@ def build_model(y_train):
     lgbm = LGBMClassifier(
         n_estimators=400,
         max_depth=6,
-        learning_rate=0.05,
+        learning_rate=0.08,
         reg_alpha=0.5,
         subsample=0.8,
         colsample_bytree=0.8,
```

### Iteration 3 -- REVERTED (-0.8522)
Score: 1.7569 vs best 2.6091
Change:         criterion="entropy", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c6399a8..9d6c8b3 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -84,6 +84,7 @@ def build_model(y_train):
         n_estimators=800,
         max_depth=12,
         min_samples_leaf=40,
+        criterion="entropy",
         class_weight="balanced",
         random_state=44,
         n_jobs=-1,
```

### Iteration 4 -- GATE FAILED
Reason: 
Change:     # Intraday reversal: 5d avg of (close - open) / (high - low)     # Positive 
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 81d4bf9..184db67 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -144,6 +144,16 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     )
     new_features.append("market_breadth_20d")
 
+    # Intraday reversal: 5d avg of (close - open) / (high - low)
+    # Positive = bullish candles (close > open), negative = bearish
+    # After decline, shift toward positive = buyers winning intraday = bottom signal
+    # Orthogonal to buying_pressure (close position in range, ignores open)
+    body_ratio = (df["close"] - df["open"]) / daily_range.replace(0, float("nan"))
+    df["intraday_reversal_5d"] = body_ratio.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).mean()
+    )
+    new_features.append("intraday_reversal_5d")
+
     # --- END researcher section ---
 
     return df, new_features
```
Traceback:
```
Score: -inf
Passed tiers: False
GATE: Tiers not passed. Score=-inf
GATE_STATUS=FAILED
COMPOSITE_SCORE=-inf
```

### Iteration 5 -- REVERTED (-2.4510)
Score: 0.1581 vs best 2.6091
Change:     # Loss magnitude decay: recent vs historical average loss size     # Ratio <
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 81d4bf9..5ec82a1 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -144,6 +144,20 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     )
     new_features.append("market_breadth_20d")
 
+    # Loss magnitude decay: recent vs historical average loss size
+    # Ratio < 1 = recent losses smaller than historical = selling exhaustion = bottom
+    # Ratio > 1 = recent losses larger = panic intensifying = knife
+    # Complements sell_vol_exhaustion (volume-based) with price-based selling intensity
+    neg_ret = (-df["ret_1d"]).clip(lower=0)
+    avg_loss_5d = neg_ret.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).mean()
+    )
+    avg_loss_20d = neg_ret.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(20, min_periods=5).mean()
+    )
+    df["loss_decay_ratio"] = avg_loss_5d / avg_loss_20d.replace(0, float("nan"))
+    new_features.append("loss_decay_ratio")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 8 -- GATE FAILED
Reason: 
Change:         ccp_alpha=0.0005, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 2dfeb8f..ac1c482 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -84,7 +84,7 @@ def build_model(y_train):
         n_estimators=800,
         max_depth=12,
         min_samples_leaf=40,
-        ccp_alpha=0.0003,
+        ccp_alpha=0.0005,
         class_weight="balanced",
         random_state=44,
         n_jobs=-1,
```
Traceback:
```
Score: -inf
Passed tiers: False
GATE: Tiers not passed. Score=-inf
GATE_STATUS=FAILED
COMPOSITE_SCORE=-inf
```

### Iteration 9 -- GATE FAILED
Reason: 
Change:         ccp_alpha=0.0004, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 2dfeb8f..409e403 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -84,7 +84,7 @@ def build_model(y_train):
         n_estimators=800,
         max_depth=12,
         min_samples_leaf=40,
-        ccp_alpha=0.0003,
+        ccp_alpha=0.0004,
         class_weight="balanced",
         random_state=44,
         n_jobs=-1,
```
Traceback:
```
Score: -inf
Passed tiers: False
GATE: Tiers not passed. Score=-inf
GATE_STATUS=FAILED
COMPOSITE_SCORE=-inf
```

### Iteration 10 -- REVERTED (+0.0000)
Score: 11.0198 vs best 11.0198
Change:         n_estimators=1000, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 2dfeb8f..b1fa7dd 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -81,7 +81,7 @@ def build_model(y_train):
     )
 
     et = ExtraTreesClassifier(
-        n_estimators=800,
+        n_estimators=1000,
         max_depth=12,
         min_samples_leaf=40,
         ccp_alpha=0.0003,
```
