# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 14 -- REVERTED (-0.0314)
Score: -1.2180 vs best -1.1866
Change:         colsample_bylevel=0.7, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0856a47..2146739 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -61,6 +61,7 @@ def build_model(y_train):
         reg_lambda=1.0,
         subsample=0.65,
         colsample_bytree=0.7,
+        colsample_bylevel=0.7,
         scale_pos_weight=spw,
         tree_method="hist",
         random_state=42,
```

### Iteration 15 -- REVERTED (-0.0321)
Score: -1.2187 vs best -1.1866
Change:         subsample=0.80, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0856a47..dddda15 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -59,7 +59,7 @@ def build_model(y_train):
         gamma=0.5,
         reg_alpha=0.5,
         reg_lambda=1.0,
-        subsample=0.65,
+        subsample=0.80,
         colsample_bytree=0.7,
         scale_pos_weight=spw,
         tree_method="hist",
```

### Iteration 16 -- REVERTED (-0.0545)
Score: -1.2411 vs best -1.1866
Change:         min_child_weight=15, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0856a47..a9c8f0b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -74,6 +74,7 @@ def build_model(y_train):
         n_estimators=500,
         max_depth=5,
         learning_rate=0.025,
+        min_child_weight=15,
         subsample=0.75,
         colsample_bytree=0.65,
         reg_alpha=0.5,
```

### Iteration 17 -- REVERTED (-0.0424)
Score: -1.2290 vs best -1.1866
Change:         max_delta_step=1, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0856a47..53fc123 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -59,6 +59,7 @@ def build_model(y_train):
         gamma=0.5,
         reg_alpha=0.5,
         reg_lambda=1.0,
+        max_delta_step=1,
         subsample=0.65,
         colsample_bytree=0.7,
         scale_pos_weight=spw,
```

### Iteration 18 -- REVERTED (-0.0283)
Score: -1.2149 vs best -1.1866
Change:         cv=5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0856a47..219eae8 100644
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

### Iteration 20 -- REVERTED (-0.1285)
Score: -1.2693 vs best -1.1408
Change:         n_estimators=1400,         learning_rate=0.01, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..624b70e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -52,9 +52,9 @@ def build_model(y_train):
     spw = np.sqrt(neg / pos)
 
     xgb = XGBClassifier(
-        n_estimators=1000,
+        n_estimators=1400,
         max_depth=7,
-        learning_rate=0.016,
+        learning_rate=0.01,
         min_child_weight=20,
         gamma=0.5,
         reg_alpha=0.5,
```

### Iteration 21 -- REVERTED (-0.0943)
Score: -1.2351 vs best -1.1408
Change:         reg_lambda=0.5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..39c4772 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -77,7 +77,7 @@ def build_model(y_train):
         subsample=0.75,
         colsample_bytree=0.65,
         reg_alpha=0.5,
-        reg_lambda=1.0,
+        reg_lambda=0.5,
         seed=45,
         verbosity=0,
     )
```

### Iteration 22 -- REVERTED (-0.0681)
Score: -1.2089 vs best -1.1408
Change:         max_depth=6, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..fba2bd6 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -53,7 +53,7 @@ def build_model(y_train):
 
     xgb = XGBClassifier(
         n_estimators=1000,
-        max_depth=7,
+        max_depth=6,
         learning_rate=0.016,
         min_child_weight=20,
         gamma=0.5,
```

### Iteration 23 -- REVERTED (-0.1201)
Score: -1.2609 vs best -1.1408
Change:         reg_lambda=1.5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..3e63752 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -77,7 +77,7 @@ def build_model(y_train):
         subsample=0.75,
         colsample_bytree=0.65,
         reg_alpha=0.5,
-        reg_lambda=1.0,
+        reg_lambda=1.5,
         seed=45,
         verbosity=0,
     )
```

### Iteration 24 -- REVERTED (-0.0711)
Score: -1.2119 vs best -1.1408
Change:         reg_alpha=0.3, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..25c992c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -57,7 +57,7 @@ def build_model(y_train):
         learning_rate=0.016,
         min_child_weight=20,
         gamma=0.5,
-        reg_alpha=0.5,
+        reg_alpha=0.3,
         reg_lambda=0.5,
         subsample=0.65,
         colsample_bytree=0.7,
```

### Iteration 25 -- REVERTED (-0.1043)
Score: -1.2451 vs best -1.1408
Change:         gamma=0.5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..43dcf34 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -74,6 +74,7 @@ def build_model(y_train):
         n_estimators=500,
         max_depth=5,
         learning_rate=0.025,
+        gamma=0.5,
         subsample=0.75,
         colsample_bytree=0.65,
         reg_alpha=0.5,
```

### Iteration 26 -- REVERTED (-0.0286)
Score: -1.1694 vs best -1.1408
Change:         final_estimator=LogisticRegression(C=5.0, max_iter=1000), 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..df3e691 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -84,7 +84,7 @@ def build_model(y_train):
 
     model = StackingClassifier(
         estimators=[("xgb", xgb), ("rank", rank)],
-        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
+        final_estimator=LogisticRegression(C=5.0, max_iter=1000),
         cv=3,
         n_jobs=1,
     )
```

### Iteration 27 -- REVERTED (-0.0833)
Score: -1.2241 vs best -1.1408
Change:         learning_rate=0.02, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..7551036 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -54,7 +54,7 @@ def build_model(y_train):
     xgb = XGBClassifier(
         n_estimators=1000,
         max_depth=7,
-        learning_rate=0.016,
+        learning_rate=0.02,
         min_child_weight=20,
         gamma=0.5,
         reg_alpha=0.5,
```

### Iteration 28 -- REVERTED (-0.1773)
Score: -1.3181 vs best -1.1408
Change:         max_depth=4, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index ea8276f..640e0eb 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -72,7 +72,7 @@ def build_model(y_train):
         objective="rank:ndcg",
         group_size=200,
         n_estimators=500,
-        max_depth=5,
+        max_depth=4,
         learning_rate=0.025,
         subsample=0.75,
         colsample_bytree=0.65,
```

### Iteration 29 -- REVERTED (-0.2288)
Score: -1.3696 vs best -1.1408
Change:     g = df.groupby("stock_id")     avg_vol = g["volume"].transform(lambda x: x.r
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index f912d4b..220f507 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,6 +22,17 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
+    g = df.groupby("stock_id")
+    avg_vol = g["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
+    ndv = df["ret_1d"] * df["volume"] / (avg_vol + 1e-10)
+
+    df["volume_force_5d"] = ndv.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=3).sum()
+    )
+    df["volume_force_10d"] = ndv.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(10, min_periods=5).sum()
+    )
+    new_features.extend(["volume_force_5d", "volume_force_10d"])
 
     # --- END researcher section ---
 
```

### Iteration 31 -- REVERTED (-0.0432)
Score: -1.1778 vs best -1.1346
Change:         max_depth=8, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 626c2e2..2c78932 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -53,7 +53,7 @@ def build_model(y_train):
 
     xgb = XGBClassifier(
         n_estimators=1000,
-        max_depth=7,
+        max_depth=8,
         learning_rate=0.016,
         min_child_weight=20,
         gamma=0.5,
```

### Iteration 32 -- REVERTED (-0.0476)
Score: -1.1822 vs best -1.1346
Change: from lightgbm import LGBMClassifier  # noqa: E402     lgb = LGBMClassifier(     
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 626c2e2..9dd9718 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -16,7 +16,7 @@ sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 import numpy as np  # noqa: F401,E402 -- available for researcher
 from sklearn.ensemble import StackingClassifier  # noqa: E402
 from sklearn.linear_model import LogisticRegression  # noqa: E402
-from xgboost import XGBClassifier  # noqa: E402
+from lightgbm import LGBMClassifier  # noqa: E402
 
 from research.model_wrappers import RankingXGBClassifier  # noqa: E402
 
@@ -51,21 +51,21 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    xgb = XGBClassifier(
+    lgb = LGBMClassifier(
         n_estimators=1000,
-        max_depth=7,
+        num_leaves=63,
         learning_rate=0.016,
-        min_child_weight=20,
-        gamma=0.5,
+        min_child_samples=20,
+        min_split_gain=0.5,
         reg_alpha=0.5,
         reg_lambda=0.5,
         subsample=0.65,
+        subsample_freq=1,
         colsample_bytree=0.7,
         scale_pos_weight=spw,
-        tree_method="hist",
         random_state=42,
         n_jobs=-1,
-        verbosity=0,
+        verbose=-1,
     )
 
     rank = RankingXGBClassifier(
@@ -83,7 +83,7 @@ def build_model(y_train):
     )
 
     model = StackingClassifier(
-        estimators=[("xgb", xgb), ("rank", rank)],
+        estimators=[("lgb", lgb), ("rank", rank)],
         final_estimator=LogisticRegression(C=1.0, max_iter=1000),
         cv=3,
         n_jobs=1,
```

### Iteration 34 -- REVERTED (-0.0811)
Score: -1.2134 vs best -1.1323
Change:         max_depth=6, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index cf505fc..6eb95f0 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -72,7 +72,7 @@ def build_model(y_train):
         objective="rank:ndcg",
         group_size=200,
         n_estimators=500,
-        max_depth=5,
+        max_depth=6,
         learning_rate=0.02,
         subsample=0.75,
         colsample_bytree=0.65,
```

### Iteration 38 -- REVERTED (-0.0229)
Score: -1.1470 vs best -1.1241
Change:         min_child_weight=1, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index e94cf4f..7917f4d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -55,7 +55,7 @@ def build_model(y_train):
         n_estimators=1000,
         max_depth=7,
         learning_rate=0.016,
-        min_child_weight=3,
+        min_child_weight=1,
         gamma=0.5,
         reg_alpha=0.5,
         reg_lambda=0.5,
```

### Iteration 40 -- REVERTED (-0.2212)
Score: -1.3412 vs best -1.1200
Change:         group_size=150, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..ee5a6c1 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -70,7 +70,7 @@ def build_model(y_train):
 
     rank = RankingXGBClassifier(
         objective="rank:ndcg",
-        group_size=200,
+        group_size=150,
         n_estimators=500,
         max_depth=5,
         learning_rate=0.02,
```

### Iteration 41 -- REVERTED (-0.0270)
Score: -1.1470 vs best -1.1200
Change:         reg_alpha=0.4, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 960434f..8ee8fd8 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -57,7 +57,7 @@ def build_model(y_train):
         learning_rate=0.016,
         min_child_weight=3,
         gamma=0.4,
-        reg_alpha=0.5,
+        reg_alpha=0.4,
         reg_lambda=0.5,
         subsample=0.65,
         colsample_bytree=0.7,
```

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
