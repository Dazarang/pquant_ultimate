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
