# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 13 -- GATE FAILED
Reason: 
Change: from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier from sklea
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c8a15dc..52c09bf 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -12,7 +12,8 @@ from pathlib import Path
 
 import numpy as np  # noqa: F401 -- available for researcher
 from lightgbm import LGBMClassifier
-from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
+from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
+from sklearn.linear_model import LogisticRegression
 from xgboost import XGBClassifier
 
 # Ensure project root is on path
@@ -85,10 +86,11 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
-    model = VotingClassifier(
+    model = StackingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et)],
-        voting="soft",
-        weights=[1, 1, 2],
+        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
+        cv=3,
+        n_jobs=1,
     )
     return model
 
```
Traceback:
```
Score: -inf
Passed tiers: False
GATE: Tiers not passed. Score=-inf
GATE_STATUS=FAILED
COMPOSITE_SCORE=-inf
```

### Iteration 14 -- REVERTED (-0.2150)
Score: -0.7300 vs best -0.5150
Change:     # Session decomposition: intraday (open->close) vs overnight (prev_close->op
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index c69818d..b58fb7a 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -125,6 +125,25 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).cumsum()
     new_features.append("streak_depth")
 
+    # Session decomposition: intraday (open->close) vs overnight (prev_close->open)
+    # At bottoms, intraday returns turn positive first (trading-hours accumulation)
+    # while overnight returns lag (lingering fear); falling knives show both negative
+    intraday_ret = df["close"] / df["open"] - 1
+    overnight_ret = df["open"] / prev_close - 1
+
+    df["intraday_return_5d"] = intraday_ret.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).mean()
+    )
+    new_features.append("intraday_return_5d")
+
+    # Intraday minus overnight: positive = accumulation outpacing distribution
+    # Targets knife_rate: knives have negative divergence (no intraday buying)
+    divergence = intraday_ret - overnight_ret
+    df["session_divergence_5d"] = divergence.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).mean()
+    )
+    new_features.append("session_divergence_5d")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 15 -- REVERTED (-0.8272)
Score: -1.3422 vs best -0.5150
Change:     # Cross-sectional ranks: where does this stock sit vs ALL peers on the same 
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index c69818d..3be968d 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -125,6 +125,23 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).cumsum()
     new_features.append("streak_depth")
 
+    # Cross-sectional ranks: where does this stock sit vs ALL peers on the same day?
+    # Orthogonal to time-series features — captures relative positioning within daily universe
+    # A stock at RSI 30 when market avg RSI is 25 is NOT a bottom; RSI 30 when market is 60 IS
+    dg = df.groupby("date")
+
+    # RSI rank: near 0 = most oversold vs all peers today
+    df["rsi_day_rank"] = dg["rsi_14"].rank(pct=True)
+    new_features.append("rsi_day_rank")
+
+    # Drawdown rank: near 0 = deepest decline vs peers (most beaten down relative to market)
+    df["drawdown_day_rank"] = dg["drawdown"].rank(pct=True)
+    new_features.append("drawdown_day_rank")
+
+    # Volume rank: near 1 = highest relative volume vs peers = potential capitulation event
+    df["volume_day_rank"] = dg["volume_ratio"].rank(pct=True)
+    new_features.append("volume_day_rank")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 17 -- REVERTED (-0.1564)
Score: -0.6098 vs best -0.4534
Change:         min_child_samples=30,         reg_lambda=1.0, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 55b9e42..fb0bc33 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -70,8 +70,10 @@ def build_model(y_train):
         n_estimators=300,
         max_depth=6,
         learning_rate=0.05,
+        min_child_samples=30,
         subsample=0.8,
         colsample_bytree=0.8,
+        reg_lambda=1.0,
         scale_pos_weight=neg / pos,
         random_state=43,
         n_jobs=-1,
```

### Iteration 18 -- REVERTED (-0.1128)
Score: -0.5662 vs best -0.4534
Change:         min_child_weight=3,         min_split_gain=0.05, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 55b9e42..56fc5e0 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -70,6 +70,8 @@ def build_model(y_train):
         n_estimators=300,
         max_depth=6,
         learning_rate=0.05,
+        min_child_weight=3,
+        min_split_gain=0.05,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

### Iteration 22 -- REVERTED (-0.0566)
Score: 0.0473 vs best 0.1039
Change:         n_estimators=600,         n_estimators=600, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..774b127 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -52,7 +52,7 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
 
     xgb = XGBClassifier(
-        n_estimators=400,
+        n_estimators=600,
         max_depth=6,
         learning_rate=0.05,
         min_child_weight=3,
@@ -67,7 +67,7 @@ def build_model(y_train):
     )
 
     lgbm = LGBMClassifier(
-        n_estimators=400,
+        n_estimators=600,
         max_depth=6,
         learning_rate=0.05,
         subsample=0.8,
```

### Iteration 23 -- REVERTED (-0.2478)
Score: -0.1439 vs best 0.1039
Change:         n_estimators=1200, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..4e86428 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -79,7 +79,7 @@ def build_model(y_train):
     )
 
     et = ExtraTreesClassifier(
-        n_estimators=800,
+        n_estimators=1200,
         max_depth=12,
         min_samples_leaf=40,
         class_weight="balanced",
```

### Iteration 24 -- REVERTED (-0.0129)
Score: 0.0910 vs best 0.1039
Change:         subsample_freq=1, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 856d61c..001fb85 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -71,6 +71,7 @@ def build_model(y_train):
         max_depth=6,
         learning_rate=0.05,
         subsample=0.8,
+        subsample_freq=1,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
         random_state=43,
```

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
