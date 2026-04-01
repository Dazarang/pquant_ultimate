# Combat Log

Reverted experiments with scores and diffs.

### Iteration 2 -- REVERTED (-0.1623)
Score: -0.1783 vs best -0.0160
Change: import numpy as np     g = df.groupby("stock_id")      candle_range = df["high"]
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index f912d4b..975edeb 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -9,20 +9,46 @@ Rules:
   - Return the same DataFrame with new columns added.
 """
 
+import numpy as np
 import pandas as pd
 
 
 def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
-    """Add custom features. Returns (df, new_feature_names).
-
-    The researcher edits this function to add new features.
-    Must return the df with new columns AND a list of the new column names.
-    """
     new_features = []
-
-    # --- RESEARCHER: add features below ---
-
-
-    # --- END researcher section ---
+    g = df.groupby("stock_id")
+
+    candle_range = df["high"] - df["low"]
+    safe_range = candle_range.replace(0, np.nan)
+    body_low = np.minimum(df["open"], df["close"])
+
+    # Lower shadow ratio: buying pressure at day's low (continuous hammer signal)
+    df["lower_shadow_ratio"] = (body_low - df["low"]) / safe_range
+    # Close position in day's range: 0=closed at low, 1=closed at high
+    df["close_position"] = (df["close"] - df["low"]) / safe_range
+    # Body ratio: small body = indecision/reversal candle
+    df["body_ratio"] = (df["close"] - df["open"]).abs() / safe_range
+    new_features += ["lower_shadow_ratio", "close_position", "body_ratio"]
+
+    # Momentum acceleration: positive = selling decelerating (bottoming)
+    df["price_accel"] = g["ret_1d"].diff()
+    new_features.append("price_accel")
+
+    # Range contraction: current range vs 10d avg (compression before reversal)
+    df["_rp"] = candle_range / df["close"]
+    df["range_contraction"] = df["_rp"] / g["_rp"].transform(
+        lambda x: x.rolling(10, min_periods=5).mean()
+    )
+    df.drop(columns=["_rp"], inplace=True)
+    new_features.append("range_contraction")
+
+    # Sustained lower shadow: rolling 3d mean (repeated rejection of lower prices)
+    df["lower_shadow_ma3"] = g["lower_shadow_ratio"].transform(
+        lambda x: x.rolling(3, min_periods=2).mean()
+    )
+    new_features.append("lower_shadow_ma3")
+
+    # Volume surge at depressed prices (capitulation signal)
+    df["vol_at_depressed"] = df["volume_ratio"] * np.clip(-df["drawdown"], 0, None)
+    new_features.append("vol_at_depressed")
 
```

### Iteration 4 -- REVERTED (-0.2438)
Score: -0.0160 vs best 0.2278
Change: FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1f158a4..e64a849 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -34,7 +34,7 @@ DATASET_PATH = "data/datasets/20260331/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
+FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "interaction"]
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 6 -- REVERTED (-0.0168)
Score: 0.3391 vs best 0.3559
Change:         iterations=3000,         depth=5,         learning_rate=0.005,         l
```diff
diff --git a/research/experiment.py b/research/experiment.py
index df79845..b385aef 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -48,14 +48,15 @@ def build_model(y_train):
     spw = np.sqrt(neg / pos)
 
     model = CatBoostWrapper(
-        iterations=2000,
-        depth=6,
-        learning_rate=0.01,
+        iterations=3000,
+        depth=5,
+        learning_rate=0.005,
         min_data_in_leaf=50,
         bootstrap_type="MVS",
         subsample=0.7,
         rsm=0.6,
-        l2_leaf_reg=3.0,
+        l2_leaf_reg=5.0,
+        random_strength=1.5,
         scale_pos_weight=spw,
         random_seed=42,
         verbose=0,
```

### Iteration 7 -- REVERTED (-0.1294)
Score: 0.2265 vs best 0.3559
Change: from sklearn.ensemble import VotingClassifier  # noqa: E402 from xgboost import 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index df79845..185b50e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -14,6 +14,9 @@ from pathlib import Path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
+from sklearn.ensemble import VotingClassifier  # noqa: E402
+from xgboost import XGBClassifier  # noqa: E402
+
 from research.model_wrappers import CatBoostWrapper  # noqa: E402
 
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
@@ -47,7 +50,7 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    model = CatBoostWrapper(
+    cb = CatBoostWrapper(
         iterations=2000,
         depth=6,
         learning_rate=0.01,
@@ -61,6 +64,26 @@ def build_model(y_train):
         verbose=0,
         thread_count=-1,
     )
+
+    xgb = XGBClassifier(
+        n_estimators=1500,
+        max_depth=6,
+        learning_rate=0.01,
+        min_child_weight=5,
+        subsample=0.7,
+        colsample_bytree=0.6,
+        reg_lambda=3.0,
+        scale_pos_weight=spw,
+        tree_method="hist",
+        random_state=42,
+        verbosity=0,
+        n_jobs=-1,
+    )
+
+    model = VotingClassifier(
+        estimators=[("cb", cb), ("xgb", xgb)],
+        voting="soft",
+    )
     return model
 
 
```

### Iteration 8 -- REVERTED (-0.9626)
Score: -0.6067 vs best 0.3559
Change: from research.model_wrappers import RankingXGBClassifier  # noqa: E402     model
```diff
diff --git a/research/experiment.py b/research/experiment.py
index df79845..b9af8ae 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -14,7 +14,7 @@ from pathlib import Path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
-from research.model_wrappers import CatBoostWrapper  # noqa: E402
+from research.model_wrappers import RankingXGBClassifier  # noqa: E402
 
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
 from lib.eval import tiered_eval  # noqa: E402
@@ -43,23 +43,20 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    neg = (y_train == 0).sum()
-    pos = (y_train == 1).sum()
-    spw = np.sqrt(neg / pos)
-
-    model = CatBoostWrapper(
-        iterations=2000,
-        depth=6,
+    model = RankingXGBClassifier(
+        objective="rank:ndcg",
+        n_estimators=2000,
+        max_depth=6,
         learning_rate=0.01,
-        min_data_in_leaf=50,
-        bootstrap_type="MVS",
+        min_child_weight=50,
         subsample=0.7,
-        rsm=0.6,
-        l2_leaf_reg=3.0,
-        scale_pos_weight=spw,
-        random_seed=42,
-        verbose=0,
-        thread_count=-1,
+        colsample_bytree=0.6,
+        reg_lambda=3.0,
+        tree_method="hist",
+        random_state=42,
+        verbosity=0,
+        n_jobs=-1,
+        group_size=200,
     )
     return model
 
```

### Iteration 9 -- REVERTED (-0.1353)
Score: 0.2206 vs best 0.3559
Change: import numpy as np     g = df.groupby("stock_id")      # Consecutive negative-re
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index f912d4b..b127662 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -9,20 +9,34 @@ Rules:
   - Return the same DataFrame with new columns added.
 """
 
+import numpy as np
 import pandas as pd
 
 
 def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
-    """Add custom features. Returns (df, new_feature_names).
-
-    The researcher edits this function to add new features.
-    Must return the df with new columns AND a list of the new column names.
-    """
     new_features = []
+    g = df.groupby("stock_id")
+
+    # Consecutive negative-return days (decline duration signal)
+    neg = (df["ret_1d"] < 0).astype(int)
+    df["_neg"] = neg
+
+    def _streak(s):
+        reset = s.eq(0)
+        groups = reset.cumsum()
+        return s.groupby(groups).cumsum()
 
-    # --- RESEARCHER: add features below ---
+    df["down_streak"] = g["_neg"].transform(_streak)
+    df.drop(columns=["_neg"], inplace=True)
+    new_features.append("down_streak")
 
+    # 5d return acceleration: positive = decline decelerating (bottoming)
+    df["ret_acceleration_5d"] = df["ret_5d"] - g["ret_5d"].shift(5)
+    new_features.append("ret_acceleration_5d")
 
-    # --- END researcher section ---
+    # Relative distance from 20-day low (0 = at the bottom)
+    min_20 = g["close"].transform(lambda x: x.rolling(20, min_periods=5).min())
+    df["dist_from_20d_low"] = df["close"] / min_20 - 1
+    new_features.append("dist_from_20d_low")
 
     return df, new_features
```

### Iteration 10 -- REVERTED (-0.1475)
Score: 0.2084 vs best 0.3559
Change: from lightgbm import LGBMClassifier  # noqa: E402     model = LGBMClassifier(   
```diff
diff --git a/research/experiment.py b/research/experiment.py
index df79845..82d70f9 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -14,7 +14,7 @@ from pathlib import Path
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 import numpy as np  # noqa: F401,E402 -- available for researcher
-from research.model_wrappers import CatBoostWrapper  # noqa: E402
+from lightgbm import LGBMClassifier  # noqa: E402
 
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
 from lib.eval import tiered_eval  # noqa: E402
@@ -47,19 +47,19 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    model = CatBoostWrapper(
-        iterations=2000,
-        depth=6,
+    model = LGBMClassifier(
+        n_estimators=2000,
+        num_leaves=63,
         learning_rate=0.01,
-        min_data_in_leaf=50,
-        bootstrap_type="MVS",
+        min_child_samples=50,
         subsample=0.7,
-        rsm=0.6,
-        l2_leaf_reg=3.0,
+        subsample_freq=1,
+        colsample_bytree=0.6,
+        reg_lambda=3.0,
         scale_pos_weight=spw,
-        random_seed=42,
-        verbose=0,
-        thread_count=-1,
+        random_state=42,
+        verbose=-1,
+        n_jobs=-1,
     )
     return model
 
```

### Iteration 11 -- REVERTED (-0.3499)
Score: 0.0060 vs best 0.3559
Change:         grow_policy="Lossguide",         max_leaves=64,         depth=8, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index df79845..2c3750a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -49,7 +49,9 @@ def build_model(y_train):
 
     model = CatBoostWrapper(
         iterations=2000,
-        depth=6,
+        grow_policy="Lossguide",
+        max_leaves=64,
+        depth=8,
         learning_rate=0.01,
         min_data_in_leaf=50,
         bootstrap_type="MVS",
```
