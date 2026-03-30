# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 2 -- REVERTED (-0.0148)
Score: -2.3186 vs best -2.3038
Change:     g = df.groupby("stock_id")      # Candlestick microstructure     total_range
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 7ebf74c..e9afb05 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,6 +22,62 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
+    g = df.groupby("stock_id")
+
+    # Candlestick microstructure
+    total_range = (df["high"] - df["low"]).clip(lower=1e-10)
+
+    # Lower wick: buying pressure at lows (continuous version of binary hammer)
+    body_low = df[["open", "close"]].min(axis=1)
+    df["lower_wick_pct"] = (body_low - df["low"]) / total_range
+    new_features.append("lower_wick_pct")
+
+    # Body ratio: small body = indecision (doji-like reversal signal)
+    df["body_pct"] = (df["close"] - df["open"]).abs() / total_range
+    new_features.append("body_pct")
+
+    # Price deceleration: 2nd derivative of 5d returns
+    # Positive during decline = decline slowing = potential bottom
+    df["price_accel"] = g["ret_5d"].diff(5)
+    new_features.append("price_accel")
+
+    # Distance from rolling lows (% above); near 0 = at the bottom
+    rolling_min_20 = g["close"].transform(
+        lambda x: x.rolling(20, min_periods=1).min()
+    )
+    df["dist_from_low_20"] = df["close"] / rolling_min_20.clip(lower=1e-10) - 1
+    new_features.append("dist_from_low_20")
+
+    rolling_min_60 = g["close"].transform(
+        lambda x: x.rolling(60, min_periods=1).min()
+    )
+    df["dist_from_low_60"] = df["close"] / rolling_min_60.clip(lower=1e-10) - 1
+    new_features.append("dist_from_low_60")
+
+    # Down-volume concentration: selling climax indicator
+    df["_down_vol"] = df["volume"] * (df["close"] < df["open"]).astype(float)
+    down_vol_sum = g["_down_vol"].transform(
+        lambda x: x.rolling(10, min_periods=1).sum()
+    )
+    total_vol_sum = g["volume"].transform(
+        lambda x: x.rolling(10, min_periods=1).sum()
+    )
+    df["down_vol_ratio_10"] = down_vol_sum / total_vol_sum.clip(lower=1e-10)
+    df.drop(columns=["_down_vol"], inplace=True)
+    new_features.append("down_vol_ratio_10")
+
+    # Range contraction: ATR(5) / ATR(20)
+    # Low values after decline = selling exhaustion / consolidation
+    df["_daily_range"] = df["high"] - df["low"]
+    atr_5 = g["_daily_range"].transform(
+        lambda x: x.rolling(5, min_periods=1).mean()
+    )
+    atr_20 = g["_daily_range"].transform(
+        lambda x: x.rolling(20, min_periods=1).mean()
```

### Iteration 4 -- REVERTED (-0.0760)
Score: -2.1082 vs best -2.0322
Change:     g = df.groupby("stock_id")      # Intraday recovery: (close-low)/(high-low),
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 7ebf74c..6cb16e9 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,6 +22,36 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
+    g = df.groupby("stock_id")
+
+    # Intraday recovery: (close-low)/(high-low), smoothed
+    # High values during decline = buyers absorbing selling pressure at lows
+    daily_range = (df["high"] - df["low"]).clip(lower=1e-10)
+    df["_cir"] = (df["close"] - df["low"]) / daily_range
+    df["close_in_range_5d"] = g["_cir"].transform(
+        lambda x: x.rolling(5, min_periods=1).mean()
+    )
+    df["close_in_range_10d"] = g["_cir"].transform(
+        lambda x: x.rolling(10, min_periods=1).mean()
+    )
+    df.drop(columns=["_cir"], inplace=True)
+    new_features.extend(["close_in_range_5d", "close_in_range_10d"])
+
+    # Return distribution shape: rolling 20d skewness of daily returns
+    # Very negative = crash risk; rising toward 0 = selling exhaustion
+    df["return_skew_20d"] = g["ret_1d"].transform(
+        lambda x: x.rolling(20, min_periods=10).skew()
+    )
+    new_features.append("return_skew_20d")
+
+    # Decline persistence: fraction of last 20 days below SMA20
+    # High fraction = deeply entrenched downtrend (different from point-in-time price_to_sma20)
+    df["_below_sma"] = (df["close"] < df["sma_20"]).astype(float)
+    df["below_sma20_frac_20d"] = g["_below_sma"].transform(
+        lambda x: x.rolling(20, min_periods=1).mean()
+    )
+    df.drop(columns=["_below_sma"], inplace=True)
+    new_features.append("below_sma20_frac_20d")
 
     # --- END researcher section ---
 
```

### Iteration 5 -- REVERTED (-0.0036)
Score: -2.0358 vs best -2.0322
Change:     ratio = neg / pos     # Differentiated class weights: each model sees the im
```diff
diff --git a/research/experiment.py b/research/experiment.py
index e8e0929..5561c6a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -49,9 +49,16 @@ def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
     neg = (y_train == 0).sum()
     pos = (y_train == 1).sum()
-    spw = np.sqrt(neg / pos)  # moderate weight (~4.4) instead of full ratio (~19)
+    ratio = neg / pos
 
-    # Deep, heavily regularized — captures complex interactions
+    # Differentiated class weights: each model sees the imbalance differently,
+    # creating genuine ensemble diversity in precision-recall tradeoff.
+    # Top-ranked predictions require agreement across all bias levels.
+    spw_aggressive = ratio ** 0.6    # ~8.5 — finds more candidates
+    spw_moderate = ratio ** 0.5      # ~4.4 — balanced (previous default)
+    spw_conservative = ratio ** 0.35  # ~3.0 — precision-focused
+
+    # Deep, heavily regularized, aggressive on positives
     xgb = XGBClassifier(
         n_estimators=800,
         max_depth=7,
@@ -62,14 +69,14 @@ def build_model(y_train):
         reg_lambda=2.0,
         subsample=0.65,
         colsample_bytree=0.5,
-        scale_pos_weight=spw,
+        scale_pos_weight=spw_aggressive,
         tree_method="hist",
         random_state=42,
         n_jobs=-1,
         verbosity=0,
     )
 
-    # Shallow, smooth, many trees — captures gradual trends
+    # Shallow, smooth, many trees — moderate bias
     lgbm = LGBMClassifier(
         n_estimators=1200,
         max_depth=4,
@@ -79,19 +86,19 @@ def build_model(y_train):
         reg_lambda=0.5,
         subsample=0.8,
         colsample_bytree=0.7,
-        scale_pos_weight=spw,
+        scale_pos_weight=spw_moderate,
         random_state=43,
         n_jobs=-1,
         verbose=-1,
     )
 
-    # Medium depth, heavy L2 — balanced generalization
+    # Medium depth, heavy L2, conservative — precision-focused
     cat = CatBoostWrapper(
         iterations=800,
         depth=5,
         learning_rate=0.025,
         l2_leaf_reg=10.0,
-        scale_pos_weight=spw,
+        scale_pos_weight=spw_conservative,
```

### Iteration 7 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change: from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402 class MLPW
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1dec4e6..f817dcc 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -15,6 +15,7 @@ sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
 from lightgbm import LGBMClassifier  # noqa: E402
+from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
 from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
@@ -45,6 +46,36 @@ VAL_END = "2023-12-31"
 # ===========================================================================
 
 
+class MLPWrapper(BaseEstimator, ClassifierMixin):
+    """Sklearn clone()-safe wrapper for TorchMLP. Creates network at fit() time."""
+
+    def __init__(self, hidden_dims=(256, 128, 64), dropout=0.3,
+                 epochs=50, lr=1e-3, batch_size=4096, pos_weight=1.0):
+        self.hidden_dims = hidden_dims
+        self.dropout = dropout
+        self.epochs = epochs
+        self.lr = lr
+        self.batch_size = batch_size
+        self.pos_weight = pos_weight
+
+    def fit(self, X, y):
+        from research.model_wrappers import TorchClassifier, TorchMLP
+        module = TorchMLP(X.shape[1], self.hidden_dims, self.dropout)
+        self._clf = TorchClassifier(
+            module=module, epochs=self.epochs, lr=self.lr,
+            batch_size=self.batch_size, pos_weight=self.pos_weight,
+        )
+        self._clf.fit(X, y)
+        self.classes_ = np.array([0, 1])
+        return self
+
+    def predict_proba(self, X):
+        return self._clf.predict_proba(X)
+
+    def predict(self, X):
+        return self._clf.predict(X)
+
+
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
     neg = (y_train == 0).sum()
@@ -107,10 +138,20 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
+    # Neural network: smooth continuous decision boundaries, uncorrelated with trees
+    mlp = MLPWrapper(
+        hidden_dims=(256, 128, 64),
+        dropout=0.3,
+        epochs=50,
+        lr=1e-3,
+        batch_size=4096,
```
Traceback:
```
  File "/Users/deaz/Developer/03-finance/project_quant/pQuant_ultimate/.venv/lib/python3.14/site-packages/sklearn/ensemble/_base.py", line 237, in _validate_estimators
    raise ValueError(
    ...<3 lines>...
    )
ValueError: The estimator MLPWrapper should be a classifier.
```

### Iteration 9 -- REVERTED (-0.3383)
Score: -1.9019 vs best -1.5636
Change:         final_estimator=LGBMClassifier(             n_estimators=100,           
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 75b4092..dd81f3b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -16,7 +16,6 @@ sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 import numpy as np  # noqa: F401,E402 -- available for researcher
 from lightgbm import LGBMClassifier  # noqa: E402
 from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
-from sklearn.linear_model import LogisticRegression  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
 from research.model_wrappers import CatBoostWrapper  # noqa: E402
@@ -110,8 +109,16 @@ def build_model(y_train):
 
     model = StackingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
-        cv=2,
+        final_estimator=LGBMClassifier(
+            n_estimators=100,
+            max_depth=2,
+            learning_rate=0.05,
+            min_child_samples=100,
+            random_state=46,
+            n_jobs=-1,
+            verbose=-1,
+        ),
+        cv=3,
         n_jobs=1,
     )
     return model
```

### Iteration 11 -- REVERTED (-0.0355)
Score: -1.5636 vs best -1.5281
Change: FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 34f739e..69a3463 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -35,7 +35,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interaction"]
+FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "interaction"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 12 -- REVERTED (-3.2280)
Score: -4.7561 vs best -1.5281
Change:         passthrough=True, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 34f739e..20862a5 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -112,6 +112,7 @@ def build_model(y_train):
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=2,
+        passthrough=True,
         n_jobs=1,
     )
     return model
```

### Iteration 2 -- REVERTED (-0.2450)
Score: -1.6726 vs best -1.4276
Change: from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier  # noqa: E40
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..5293a82 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -15,8 +15,7 @@ sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
 from lightgbm import LGBMClassifier  # noqa: E402
-from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
-from sklearn.linear_model import LogisticRegression  # noqa: E402
+from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
 from research.model_wrappers import CatBoostWrapper  # noqa: E402
@@ -108,10 +107,9 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
-    model = StackingClassifier(
+    model = VotingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
-        cv=2,
+        voting="soft",
         n_jobs=1,
     )
     return model
```

### Iteration 3 -- REVERTED (-0.0999)
Score: -1.5275 vs best -1.4276
Change:         cv=3, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..a276d42 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -111,7 +111,7 @@ def build_model(y_train):
     model = StackingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
-        cv=2,
+        cv=3,
         n_jobs=1,
     )
     return model
```

### Iteration 4 -- REVERTED (-0.0495)
Score: -1.4771 vs best -1.4276
Change:         n_estimators=1300,         learning_rate=0.012,         n_estimators=200
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..c315bd3 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -54,9 +54,9 @@ def build_model(y_train):
 
     # Deep, heavily regularized — captures complex interactions
     xgb = XGBClassifier(
-        n_estimators=1000,
+        n_estimators=1300,
         max_depth=7,
-        learning_rate=0.016,
+        learning_rate=0.012,
         min_child_weight=20,
         gamma=0.5,
         reg_alpha=2.0,
@@ -72,9 +72,9 @@ def build_model(y_train):
 
     # Shallow, smooth, many trees — captures gradual trends
     lgbm = LGBMClassifier(
-        n_estimators=1500,
+        n_estimators=2000,
         max_depth=4,
-        learning_rate=0.008,
+        learning_rate=0.006,
         min_child_samples=80,
         reg_alpha=0.5,
         reg_lambda=0.5,
@@ -88,9 +88,9 @@ def build_model(y_train):
 
     # Medium depth, heavy L2 — balanced generalization
     cat = CatBoostWrapper(
-        iterations=1000,
+        iterations=1300,
         depth=5,
-        learning_rate=0.020,
+        learning_rate=0.015,
         l2_leaf_reg=10.0,
         scale_pos_weight=spw,
         random_seed=44,
```

### Iteration 5 -- REVERTED (-0.0068)
Score: -1.4344 vs best -1.4276
Change: from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, Stack
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..fc5393b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -15,7 +15,7 @@ sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
 from lightgbm import LGBMClassifier  # noqa: E402
-from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
+from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier  # noqa: E402
 from sklearn.linear_model import LogisticRegression  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
@@ -108,8 +108,20 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
+    # Best-split forest: stronger individual trees than ExtraTrees, complementary errors
+    rf = RandomForestClassifier(
+        n_estimators=300,
+        max_depth=12,
+        min_samples_leaf=50,
+        max_features="sqrt",
+        max_samples=0.7,
+        class_weight={0: 1, 1: spw},
+        random_state=46,
+        n_jobs=-1,
+    )
+
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra), ("rf", rf)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=2,
         n_jobs=1,
```

### Iteration 6 -- REVERTED (-0.0705)
Score: -1.4981 vs best -1.4276
Change:     # Cross-sectional z-score of 10-day return: how oversold is this stock     #
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 7ebf74c..7ed29a3 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,6 +22,15 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
+    # Cross-sectional z-score of 10-day return: how oversold is this stock
+    # relative to all stocks on the same date? Deeply negative = extreme
+    # underperformance vs peers = higher mean-reversion / bottom probability.
+    # Novel: all existing features are per-stock; this adds market-relative context.
+    g_date = df.groupby("date")["ret_10d"]
+    xmean = g_date.transform("mean")
+    xstd = g_date.transform("std").clip(lower=1e-10)
+    df["ret_xsec_zscore_10d"] = (df["ret_10d"] - xmean) / xstd
+    new_features.append("ret_xsec_zscore_10d")
 
     # --- END researcher section ---
 
```

### Iteration 7 -- GATE FAILED
Reason: GATE VIOLATION: research/model_wrappers.py was modified. Reverting.
Change:     # DART boosting: tree dropout decorrelates trees, complementary to standard 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..3238706 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -108,8 +108,27 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
+    # DART boosting: tree dropout decorrelates trees, complementary to standard GBDTs
+    lgbm_dart = LGBMClassifier(
+        boosting_type="dart",
+        n_estimators=300,
+        max_depth=6,
+        learning_rate=0.025,
+        min_child_samples=60,
+        reg_alpha=1.0,
+        reg_lambda=1.0,
+        subsample=0.7,
+        colsample_bytree=0.6,
+        drop_rate=0.1,
+        skip_drop=0.5,
+        scale_pos_weight=spw,
+        random_state=46,
+        n_jobs=-1,
+        verbose=-1,
+    )
+
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra), ("dart", lgbm_dart)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=2,
         n_jobs=1,
```
Traceback:
```
GATE VIOLATION: research/model_wrappers.py was modified. Reverting.
```

### Iteration 8 -- REVERTED (-0.2459)
Score: -1.6735 vs best -1.4276
Change:     # DART boosting: tree dropout decorrelates trees, complementary to standard 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..bccf952 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -108,8 +108,27 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
+    # DART boosting: tree dropout decorrelates trees, complementary to standard GBDTs
+    dart = LGBMClassifier(
+        boosting_type="dart",
+        n_estimators=300,
+        max_depth=6,
+        learning_rate=0.025,
+        min_child_samples=60,
+        reg_alpha=1.0,
+        reg_lambda=1.0,
+        subsample=0.7,
+        colsample_bytree=0.6,
+        drop_rate=0.1,
+        skip_drop=0.5,
+        scale_pos_weight=spw,
+        random_state=46,
+        n_jobs=-1,
+        verbose=-1,
+    )
+
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra), ("dart", dart)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=2,
         n_jobs=1,
```

### Iteration 9 -- REVERTED (-0.0673)
Score: -1.4949 vs best -1.4276
Change: from sklearn.neural_network import MLPClassifier  # noqa: E402     # Neural dive
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..1a33f1c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -17,6 +17,7 @@ import numpy as np  # noqa: F401,E402 -- available for researcher
 from lightgbm import LGBMClassifier  # noqa: E402
 from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
 from sklearn.linear_model import LogisticRegression  # noqa: E402
+from sklearn.neural_network import MLPClassifier  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
 from research.model_wrappers import CatBoostWrapper  # noqa: E402
@@ -108,8 +109,23 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
+    # Neural diversity: smooth non-linear boundaries, fundamentally different from trees
+    mlp = MLPClassifier(
+        hidden_layer_sizes=(64, 32),
+        activation='relu',
+        solver='adam',
+        alpha=1e-4,
+        batch_size=4096,
+        learning_rate='adaptive',
+        learning_rate_init=1e-3,
+        max_iter=30,
+        early_stopping=True,
+        validation_fraction=0.1,
+        random_state=46,
+    )
+
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra), ("mlp", mlp)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=2,
         n_jobs=1,
```

### Iteration 10 -- REVERTED (-0.0035)
Score: -1.4311 vs best -1.4276
Change:     spw = (neg / pos) ** 0.4  # conservative weight (~3.3) for higher precision 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c7d35b2..8d3587a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -50,7 +50,7 @@ def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
     neg = (y_train == 0).sum()
     pos = (y_train == 1).sum()
-    spw = np.sqrt(neg / pos)  # moderate weight (~4.4) instead of full ratio (~19)
+    spw = (neg / pos) ** 0.4  # conservative weight (~3.3) for higher precision at top
 
     # Deep, heavily regularized — captures complex interactions
     xgb = XGBClassifier(
```

### Iteration 12 -- REVERTED (-0.0613)
Score: -1.3484 vs best -1.2871
Change:         objective="rank:map", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..b65209f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -95,7 +95,7 @@ def build_model(y_train):
     )
 
     rank = RankingXGBClassifier(
-        objective="rank:ndcg",
+        objective="rank:map",
         group_size=200,
         n_estimators=500,
         max_depth=5,
```

### Iteration 13 -- REVERTED (-0.0101)
Score: -1.2972 vs best -1.2871
Change:         group_size=150,         n_estimators=800,         max_depth=6,         l
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..314944c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -96,14 +96,14 @@ def build_model(y_train):
 
     rank = RankingXGBClassifier(
         objective="rank:ndcg",
-        group_size=200,
-        n_estimators=500,
-        max_depth=5,
-        learning_rate=0.025,
-        subsample=0.75,
-        colsample_bytree=0.65,
-        reg_alpha=0.5,
-        reg_lambda=1.0,
+        group_size=150,
+        n_estimators=800,
+        max_depth=6,
+        learning_rate=0.015,
+        subsample=0.70,
+        colsample_bytree=0.55,
+        reg_alpha=1.5,
+        reg_lambda=2.0,
         seed=45,
         verbosity=0,
     )
```

### Iteration 14 -- REVERTED (-0.0877)
Score: -1.3748 vs best -1.2871
Change:         group_size=1000, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..d5cc894 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -96,7 +96,7 @@ def build_model(y_train):
 
     rank = RankingXGBClassifier(
         objective="rank:ndcg",
-        group_size=200,
+        group_size=1000,
         n_estimators=500,
         max_depth=5,
         learning_rate=0.025,
```

### Iteration 15 -- REVERTED (-0.0413)
Score: -1.3284 vs best -1.2871
Change:         final_estimator=LogisticRegression(C=0.1, max_iter=1000), 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..531ba69 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -110,7 +110,7 @@ def build_model(y_train):
 
     model = StackingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("rank", rank)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
+        final_estimator=LogisticRegression(C=0.1, max_iter=1000),
         cv=2,
         n_jobs=1,
     )
```

### Iteration 16 -- REVERTED (-0.0259)
Score: -1.3130 vs best -1.2871
Change:         final_estimator=LogisticRegression(C=1.0, max_iter=1000, class_weight="b
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..76eec23 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -110,7 +110,7 @@ def build_model(y_train):
 
     model = StackingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("rank", rank)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
+        final_estimator=LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
         cv=2,
         n_jobs=1,
     )
```

### Iteration 17 -- REVERTED (-0.0741)
Score: -1.3612 vs best -1.2871
Change:     g = df.groupby("stock_id")      prev_close = g["close"].shift(1)     df["ove
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index f912d4b..6a0b010 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,6 +22,14 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
+    g = df.groupby("stock_id")
+
+    prev_close = g["close"].shift(1)
+    df["overnight_gap"] = (df["open"] - prev_close) / prev_close.clip(lower=1e-10)
+    new_features.append("overnight_gap")
+
+    df["intraday_return"] = (df["close"] - df["open"]) / df["open"].clip(lower=1e-10)
+    new_features.append("intraday_return")
 
     # --- END researcher section ---
 
```

### Iteration 18 -- REVERTED (-0.3023)
Score: -1.5894 vs best -1.2871
Change: FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"] 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..fd421f6 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -35,7 +35,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interaction"]
+FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 19 -- REVERTED (-0.0163)
Score: -1.3034 vs best -1.2871
Change:         max_depth=9,         min_child_weight=50,         gamma=2.0,         reg
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..67539d7 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -54,12 +54,12 @@ def build_model(y_train):
 
     xgb = XGBClassifier(
         n_estimators=1000,
-        max_depth=7,
+        max_depth=9,
         learning_rate=0.016,
-        min_child_weight=20,
-        gamma=0.5,
-        reg_alpha=2.0,
-        reg_lambda=2.0,
+        min_child_weight=50,
+        gamma=2.0,
+        reg_alpha=5.0,
+        reg_lambda=5.0,
         subsample=0.65,
         colsample_bytree=0.5,
         scale_pos_weight=spw,
```

### Iteration 20 -- REVERTED (-0.1231)
Score: -1.4102 vs best -1.2871
Change: from sklearn.feature_selection import SelectFromModel  # noqa: E402 from sklearn
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..4429b2a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -16,7 +16,9 @@ sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 import numpy as np  # noqa: F401,E402 -- available for researcher
 from lightgbm import LGBMClassifier  # noqa: E402
 from sklearn.ensemble import StackingClassifier  # noqa: E402
+from sklearn.feature_selection import SelectFromModel  # noqa: E402
 from sklearn.linear_model import LogisticRegression  # noqa: E402
+from sklearn.pipeline import Pipeline  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
 from research.model_wrappers import CatBoostWrapper, RankingXGBClassifier  # noqa: E402
@@ -108,12 +110,26 @@ def build_model(y_train):
         verbosity=0,
     )
 
-    model = StackingClassifier(
+    stack = StackingClassifier(
         estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("rank", rank)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=2,
         n_jobs=1,
     )
+
+    selector = LGBMClassifier(
+        n_estimators=100,
+        max_depth=3,
+        scale_pos_weight=spw,
+        random_state=99,
+        n_jobs=-1,
+        verbose=-1,
+    )
+
+    model = Pipeline([
+        ("select", SelectFromModel(selector)),
+        ("stack", stack),
+    ])
     return model
 
 
```

### Iteration 21 -- REVERTED (-0.0344)
Score: -1.3215 vs best -1.2871
Change:         n_estimators=1000,         max_depth=-1,         num_leaves=31,         
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 22e2a30..bc29708 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -70,12 +70,13 @@ def build_model(y_train):
     )
 
     lgbm = LGBMClassifier(
-        n_estimators=1500,
-        max_depth=4,
-        learning_rate=0.008,
-        min_child_samples=80,
-        reg_alpha=0.5,
-        reg_lambda=0.5,
+        n_estimators=1000,
+        max_depth=-1,
+        num_leaves=31,
+        learning_rate=0.01,
+        min_child_samples=100,
+        reg_alpha=1.0,
+        reg_lambda=1.0,
         subsample=0.8,
         colsample_bytree=0.7,
         scale_pos_weight=spw,
```
