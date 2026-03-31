# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 42 -- REVERTED (-0.0911)
Score: -1.2111 vs best -1.1200
Change:         objective="rank:map", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..3e73b2d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -69,7 +69,7 @@ def build_model(y_train):
     )
 
     rank = RankingXGBClassifier(
-        objective="rank:ndcg",
+        objective="rank:map",
         group_size=200,
         n_estimators=500,
         max_depth=5,
```

### Iteration 43 -- REVERTED (-0.0466)
Score: -1.1666 vs best -1.1200
Change: from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..4a1b622 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -14,7 +14,7 @@ from pathlib import Path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
-from sklearn.ensemble import StackingClassifier  # noqa: E402
+from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
 from sklearn.linear_model import LogisticRegression  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
@@ -82,8 +82,18 @@ def build_model(y_train):
         verbosity=0,
     )
 
+    et = ExtraTreesClassifier(
+        n_estimators=200,
+        max_depth=10,
+        min_samples_leaf=100,
+        max_features="sqrt",
+        class_weight="balanced_subsample",
+        random_state=43,
+        n_jobs=-1,
+    )
+
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("rank", rank)],
+        estimators=[("xgb", xgb), ("rank", rank), ("et", et)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=3,
         n_jobs=1,
```

### Iteration 44 -- REVERTED (-0.0401)
Score: -1.1601 vs best -1.1200
Change:         colsample_bytree=0.65, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..ad0c164 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -60,7 +60,7 @@ def build_model(y_train):
         reg_alpha=0.5,
         reg_lambda=0.5,
         subsample=0.65,
-        colsample_bytree=0.7,
+        colsample_bytree=0.65,
         scale_pos_weight=spw,
         tree_method="hist",
         random_state=42,
```

### Iteration 45 -- REVERTED (-0.1766)
Score: -1.2966 vs best -1.1200
Change:         learning_rate=0.015, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..04938d8 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -73,7 +73,7 @@ def build_model(y_train):
         group_size=200,
         n_estimators=500,
         max_depth=5,
-        learning_rate=0.02,
+        learning_rate=0.015,
         subsample=0.75,
         colsample_bytree=0.65,
         reg_alpha=0.5,
```

### Iteration 46 -- REVERTED (-0.0218)
Score: -1.1418 vs best -1.1200
Change:         reg_lambda=0.3, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..7fb80d5 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -58,7 +58,7 @@ def build_model(y_train):
         min_child_weight=3,
         gamma=0.4,
         reg_alpha=0.5,
-        reg_lambda=0.5,
+        reg_lambda=0.3,
         subsample=0.65,
         colsample_bytree=0.7,
         scale_pos_weight=spw,
```

### Iteration 47 -- REVERTED (-0.1188)
Score: -1.2388 vs best -1.1200
Change: FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..5b92bb9 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -34,7 +34,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interaction"]
+FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "interaction"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 48 -- REVERTED (-0.0223)
Score: -1.1423 vs best -1.1200
Change:         gamma=0.3, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..89d3eb4 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -56,7 +56,7 @@ def build_model(y_train):
         max_depth=7,
         learning_rate=0.016,
         min_child_weight=3,
-        gamma=0.4,
+        gamma=0.3,
         reg_alpha=0.5,
         reg_lambda=0.5,
         subsample=0.65,
```

### Iteration 49 -- REVERTED (-0.0323)
Score: -1.1523 vs best -1.1200
Change:         subsample=0.70, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..6b0f0c9 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -59,7 +59,7 @@ def build_model(y_train):
         gamma=0.4,
         reg_alpha=0.5,
         reg_lambda=0.5,
-        subsample=0.65,
+        subsample=0.70,
         colsample_bytree=0.7,
         scale_pos_weight=spw,
         tree_method="hist",
```

### Iteration 50 -- REVERTED (-0.0232)
Score: -1.1432 vs best -1.1200
Change:         n_estimators=1200, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..dba4689 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -52,7 +52,7 @@ def build_model(y_train):
     spw = np.sqrt(neg / pos)
 
     xgb = XGBClassifier(
-        n_estimators=1000,
+        n_estimators=1200,
         max_depth=7,
         learning_rate=0.016,
         min_child_weight=3,
```

### Iteration 1 -- REVERTED (-0.2479)
Score: -1.3679 vs best -1.1200
Change: import numpy as np     day_range = (df["high"] - df["low"]).replace(0, np.nan)  
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index f912d4b..37f4407 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -9,6 +9,7 @@ Rules:
   - Return the same DataFrame with new columns added.
 """
 
+import numpy as np
 import pandas as pd
 
 
@@ -21,7 +22,13 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     new_features = []
 
     # --- RESEARCHER: add features below ---
+    day_range = (df["high"] - df["low"]).replace(0, np.nan)
+    lower_shadow_frac = ((df[["open", "close"]].min(axis=1) - df["low"]) / day_range).clip(lower=0)
+    neg_price_zscore = (-df["price_zscore"]).clip(lower=0)
 
+    df["lower_shadow_frac"] = lower_shadow_frac
+    df["oversold_rejection_signal"] = lower_shadow_frac * neg_price_zscore * (1 + df["at_zscore_extreme"])
+    new_features.extend(["lower_shadow_frac", "oversold_rejection_signal"])
 
     # --- END researcher section ---
 
```

### Iteration 2 -- REVERTED (-2.4675)
Score: -3.5875 vs best -1.1200
Change:         passthrough=True, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..87ccd76 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -87,6 +87,7 @@ def build_model(y_train):
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=3,
         n_jobs=1,
+        passthrough=True,
     )
     return model
 
```

### Iteration 3 -- REVERTED (-0.2480)
Score: -1.3680 vs best -1.1200
Change:     stock_key = df["stock_id"]     g = df.groupby("stock_id", sort=False)      d
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index f912d4b..be9bf3b 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -21,6 +21,34 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     new_features = []
 
     # --- RESEARCHER: add features below ---
+    stock_key = df["stock_id"]
+    g = df.groupby("stock_id", sort=False)
+
+    df["close_percentile_20"] = g["close"].rolling(20, min_periods=10).rank(pct=True).droplevel(0).sort_index()
+    df["close_percentile_60"] = g["close"].rolling(60, min_periods=20).rank(pct=True).droplevel(0).sort_index()
+
+    prev_low = g["low"].shift(1)
+    prev_low_10 = prev_low.groupby(stock_key).rolling(10, min_periods=5).min().droplevel(0).sort_index()
+    prev_low_20 = prev_low.groupby(stock_key).rolling(20, min_periods=10).min().droplevel(0).sort_index()
+    prev_low_60 = prev_low.groupby(stock_key).rolling(60, min_periods=20).min().droplevel(0).sort_index()
+
+    prev_low_10 = prev_low_10.where(prev_low_10 != 0)
+    prev_low_20 = prev_low_20.where(prev_low_20 != 0)
+    prev_low_60 = prev_low_60.where(prev_low_60 != 0)
+
+    df["low_breach_10"] = ((prev_low_10 - df["low"]) / prev_low_10).clip(lower=0)
+    df["low_breach_20"] = ((prev_low_20 - df["low"]) / prev_low_20).clip(lower=0)
+    df["low_breach_60"] = ((prev_low_60 - df["low"]) / prev_low_60).clip(lower=0)
+
+    new_features.extend(
+        [
+            "close_percentile_20",
+            "close_percentile_60",
+            "low_breach_10",
+            "low_breach_20",
+            "low_breach_60",
+        ]
+    )
 
 
     # --- END researcher section ---
```

### Iteration 4 -- REVERTED (-0.2451)
Score: -1.3651 vs best -1.1200
Change:     stock_key = df["stock_id"]     g = df.groupby("stock_id", sort=False)      d
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index f912d4b..6859d34 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -21,6 +21,46 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     new_features = []
 
     # --- RESEARCHER: add features below ---
+    stock_key = df["stock_id"]
+    g = df.groupby("stock_id", sort=False)
+
+    daily_range = (df["high"] - df["low"]).replace(0, float("nan"))
+    recent_max_range_10d = daily_range.groupby(stock_key).transform(
+        lambda x: x.rolling(10, min_periods=3).max()
+    )
+    df["range_contraction_10d"] = daily_range / recent_max_range_10d.replace(0, float("nan"))
+
+    high_10d = g["high"].transform(lambda x: x.rolling(10, min_periods=3).max())
+    low_10d = g["low"].transform(lambda x: x.rolling(10, min_periods=3).min())
+    range_10d = (high_10d - low_10d).replace(0, float("nan"))
+    df["price_pos_10d"] = (df["close"] - low_10d) / range_10d
+
+    close_in_range = (df["close"] - df["low"]) / daily_range
+    df["buying_pressure_3d"] = close_in_range.groupby(stock_key).transform(
+        lambda x: x.rolling(3, min_periods=2).mean()
+    )
+
+    prev_close = g["close"].shift(1)
+    df["overnight_gap_atr"] = (df["open"] - prev_close) / df["atr_14"].replace(0, float("nan"))
+    df["overnight_gap_trend_5d"] = df["overnight_gap_atr"].groupby(stock_key).transform(
+        lambda x: x.rolling(5, min_periods=3).mean()
+    )
+
+    down_vol = df["volume"].where(df["ret_1d"] < 0, 0.0)
+    recent_down_vol_5d = down_vol.groupby(stock_key).transform(lambda x: x.rolling(5, min_periods=2).sum())
+    total_down_vol_20d = down_vol.groupby(stock_key).transform(lambda x: x.rolling(20, min_periods=5).sum())
+    df["sell_vol_exhaustion"] = recent_down_vol_5d / total_down_vol_20d.replace(0, float("nan"))
+
+    new_features.extend(
+        [
+            "range_contraction_10d",
+            "price_pos_10d",
+            "buying_pressure_3d",
+            "overnight_gap_atr",
+            "overnight_gap_trend_5d",
+            "sell_vol_exhaustion",
+        ]
+    )
 
 
     # --- END researcher section ---
```

### Iteration 1 -- REVERTED (-0.0963)
Score: -1.2163 vs best -1.1200
Change:     spw = (neg / pos) ** (1 / 3) 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..b10b64d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -49,7 +49,7 @@ def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
     neg = (y_train == 0).sum()
     pos = (y_train == 1).sum()
-    spw = np.sqrt(neg / pos)
+    spw = (neg / pos) ** (1 / 3)
 
     xgb = XGBClassifier(
         n_estimators=1000,
```

### Iteration 2 -- REVERTED (-0.5252)
Score: -1.6452 vs best -1.1200
Change: FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"] 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..d73f698 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -34,7 +34,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interaction"]
+FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 3 -- REVERTED (-0.0796)
Score: -1.1996 vs best -1.1200
Change:         cv=5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..2d854e0 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -85,7 +85,7 @@ def build_model(y_train):
     model = StackingClassifier(
         estimators=[("xgb", xgb), ("rank", rank)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
-        cv=3,
+        cv=5,
         n_jobs=1,
     )
     return model
```

### Iteration 4 -- REVERTED (-0.5551)
Score: -1.6751 vs best -1.1200
Change: from sklearn.ensemble import VotingClassifier  # noqa: E402     model = VotingCl
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..9a527c4 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -14,8 +14,7 @@ from pathlib import Path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
-from sklearn.ensemble import StackingClassifier  # noqa: E402
-from sklearn.linear_model import LogisticRegression  # noqa: E402
+from sklearn.ensemble import VotingClassifier  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
 from research.model_wrappers import RankingXGBClassifier  # noqa: E402
@@ -82,10 +81,9 @@ def build_model(y_train):
         verbosity=0,
     )
 
-    model = StackingClassifier(
+    model = VotingClassifier(
         estimators=[("xgb", xgb), ("rank", rank)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
-        cv=3,
+        voting="soft",
         n_jobs=1,
     )
     return model
```

### Iteration 5 -- REVERTED (-0.1549)
Score: -1.2749 vs best -1.1200
Change:         n_estimators=750, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..ff04969 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -71,7 +71,7 @@ def build_model(y_train):
     rank = RankingXGBClassifier(
         objective="rank:ndcg",
         group_size=200,
-        n_estimators=500,
+        n_estimators=750,
         max_depth=5,
         learning_rate=0.02,
         subsample=0.75,
```

### Iteration 6 -- REVERTED (-0.0719)
Score: -1.1919 vs best -1.1200
Change:         reg_lambda=0.7, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..f1ed4d6 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -77,7 +77,7 @@ def build_model(y_train):
         subsample=0.75,
         colsample_bytree=0.65,
         reg_alpha=0.5,
-        reg_lambda=1.0,
+        reg_lambda=0.7,
         seed=45,
         verbosity=0,
     )
```

### Iteration 7 -- REVERTED (+0.0000)
Score: -1.1200 vs best -1.1200
Change:         grow_policy="lossguide",         max_leaves=128, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..b2cef4f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -63,6 +63,8 @@ def build_model(y_train):
         colsample_bytree=0.7,
         scale_pos_weight=spw,
         tree_method="hist",
+        grow_policy="lossguide",
+        max_leaves=128,
         random_state=42,
         n_jobs=-1,
         verbosity=0,
```

### Iteration 8 -- REVERTED (-0.0972)
Score: -1.2172 vs best -1.1200
Change:         min_child_weight=5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..e0e4d4f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -74,6 +74,7 @@ def build_model(y_train):
         n_estimators=500,
         max_depth=5,
         learning_rate=0.02,
+        min_child_weight=5,
         subsample=0.75,
         colsample_bytree=0.65,
         reg_alpha=0.5,
```

### Iteration 9 -- REVERTED (-0.0650)
Score: -1.1850 vs best -1.1200
Change:         colsample_bynode=0.8, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..084e92e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -61,6 +61,7 @@ def build_model(y_train):
         reg_lambda=0.5,
         subsample=0.65,
         colsample_bytree=0.7,
+        colsample_bynode=0.8,
         scale_pos_weight=spw,
         tree_method="hist",
         random_state=42,
```

### Iteration 10 -- REVERTED (-0.0511)
Score: -1.1711 vs best -1.1200
Change:         learning_rate=0.02, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..f3f2e78 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -54,7 +54,7 @@ def build_model(y_train):
     xgb = XGBClassifier(
         n_estimators=1000,
         max_depth=7,
-        learning_rate=0.016,
+        learning_rate=0.02,
         min_child_weight=3,
         gamma=0.4,
         reg_alpha=0.5,
```

### Iteration 1 -- REVERTED (-0.7885)
Score: -1.9085 vs best -1.1200
Change: from research.model_wrappers import CatBoostWrapper  # noqa: E402     cat = CatB
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1ddec70..252f42d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -18,7 +18,7 @@ from sklearn.ensemble import StackingClassifier  # noqa: E402
 from sklearn.linear_model import LogisticRegression  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
-from research.model_wrappers import RankingXGBClassifier  # noqa: E402
+from research.model_wrappers import CatBoostWrapper  # noqa: E402
 
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
 from lib.eval import tiered_eval  # noqa: E402
@@ -34,7 +34,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 # CONFIGURATION -- researcher edits this section
 # ===========================================================================
 
-DATASET_PATH = "data/datasets/20260115/dataset.parquet"
+
 
 # Stock universe: None = all, "AAPL" = single, ["AAPL", "MSFT", ...] = subset
 STOCKS = None
@@ -74,22 +74,19 @@ def build_model(y_train):
         verbosity=0,
     )
 
-    rank = RankingXGBClassifier(
-        objective="rank:ndcg",
-        group_size=200,
-        n_estimators=500,
-        max_depth=5,
+    cat = CatBoostWrapper(
+        iterations=800,
+        depth=6,
         learning_rate=0.02,
-        subsample=0.75,
-        colsample_bytree=0.65,
-        reg_alpha=0.5,
-        reg_lambda=1.0,
-        seed=45,
-        verbosity=0,
+        l2_leaf_reg=3.0,
+        border_count=128,
+        auto_class_weights="SqrtBalanced",
+        random_seed=44,
+        verbose=0,
     )
 
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("rank", rank)],
+        estimators=[("xgb", xgb), ("cat", cat)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=3,
         n_jobs=1,
```

### Iteration 2 -- REVERTED (-0.2047)
Score: -1.3247 vs best -1.1200
Change:         colsample_bytree=0.75, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1ddec70..aba4e7c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -81,7 +81,7 @@ def build_model(y_train):
         max_depth=5,
         learning_rate=0.02,
         subsample=0.75,
-        colsample_bytree=0.65,
+        colsample_bytree=0.75,
         reg_alpha=0.5,
         reg_lambda=1.0,
         seed=45,
```

### Iteration 3 -- REVERTED (-0.9666)
Score: -2.0866 vs best -1.1200
Change:     model = XGBClassifier( 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1ddec70..5492e29 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -14,12 +14,8 @@ from pathlib import Path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
-from sklearn.ensemble import StackingClassifier  # noqa: E402
-from sklearn.linear_model import LogisticRegression  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
-from research.model_wrappers import RankingXGBClassifier  # noqa: E402
-
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
 from lib.eval import tiered_eval  # noqa: E402
 from research.features_lab import add_custom_features  # noqa: E402
@@ -57,7 +53,7 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    xgb = XGBClassifier(
+    model = XGBClassifier(
         n_estimators=1000,
         max_depth=7,
         learning_rate=0.016,
@@ -73,27 +69,6 @@ def build_model(y_train):
         n_jobs=-1,
         verbosity=0,
     )
-
-    rank = RankingXGBClassifier(
-        objective="rank:ndcg",
-        group_size=200,
-        n_estimators=500,
-        max_depth=5,
-        learning_rate=0.02,
-        subsample=0.75,
-        colsample_bytree=0.65,
-        reg_alpha=0.5,
-        reg_lambda=1.0,
-        seed=45,
-        verbosity=0,
-    )
-
-    model = StackingClassifier(
-        estimators=[("xgb", xgb), ("rank", rank)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
-        cv=3,
-        n_jobs=1,
-    )
     return model
 
 
```

### Iteration 5 -- REVERTED (-0.0053)
Score: 1.1911 vs best 1.1964
Change: from research.model_wrappers import CatBoostWrapper, RankingXGBClassifier  # noq
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 86077b5..e0a04d6 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -18,7 +18,7 @@ from sklearn.ensemble import StackingClassifier  # noqa: E402
 from sklearn.linear_model import LogisticRegression  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
-from research.model_wrappers import RankingXGBClassifier  # noqa: E402
+from research.model_wrappers import CatBoostWrapper, RankingXGBClassifier  # noqa: E402
 
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
 from lib.eval import tiered_eval  # noqa: E402
@@ -88,8 +88,19 @@ def build_model(y_train):
         verbosity=0,
     )
 
+    cat = CatBoostWrapper(
+        iterations=500,
+        depth=4,
+        learning_rate=0.03,
+        l2_leaf_reg=5.0,
+        border_count=128,
+        auto_class_weights="SqrtBalanced",
+        random_seed=44,
+        verbose=0,
+    )
+
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("rank", rank)],
+        estimators=[("xgb", xgb), ("rank", rank), ("cat", cat)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=3,
         n_jobs=1,
```

### Iteration 6 -- REVERTED (-0.0612)
Score: 1.1352 vs best 1.1964
Change: from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier  # n
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 86077b5..bbe0d20 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -14,8 +14,7 @@ from pathlib import Path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
-from sklearn.ensemble import StackingClassifier  # noqa: E402
-from sklearn.linear_model import LogisticRegression  # noqa: E402
+from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier  # noqa: E402
 from xgboost import XGBClassifier  # noqa: E402
 
 from research.model_wrappers import RankingXGBClassifier  # noqa: E402
@@ -90,7 +89,9 @@ def build_model(y_train):
 
     model = StackingClassifier(
         estimators=[("xgb", xgb), ("rank", rank)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
+        final_estimator=GradientBoostingClassifier(
+            n_estimators=30, max_depth=2, learning_rate=0.1, random_state=42
+        ),
         cv=3,
         n_jobs=1,
     )
```

### Iteration 8 -- REVERTED (-0.3107)
Score: 0.9218 vs best 1.2325
Change: VAL_END = "2025-03-31" 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index a61ac3d..621ad24 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -44,7 +44,7 @@ FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interacti
 
 # Temporal split boundaries
 TRAIN_END = "2024-03-31"
-VAL_END = "2024-12-31"
+VAL_END = "2025-03-31"
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 10 -- REVERTED (-1.1562)
Score: 0.1017 vs best 1.2579
Change: TRAIN_END = "2024-09-30" 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3e6b8cb..f7c5665 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -43,7 +43,7 @@ STOCKS = None
 FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interaction"]
 
 # Temporal split boundaries
-TRAIN_END = "2024-06-30"
+TRAIN_END = "2024-09-30"
 VAL_END = "2024-12-31"
 
 # ===========================================================================
```
