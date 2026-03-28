# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 1 -- REVERTED (-0.4645)
Score: 0.3611 vs best 0.8256
Change: THRESHOLD = 0.3 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 36dfea5..22936ba 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -37,7 +37,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold (default 0.5 may be too high for imbalanced data)
-THRESHOLD = 0.5
+THRESHOLD = 0.3
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 2 -- REVERTED (-0.9238)
Score: -0.0982 vs best 0.8256
Change: FEATURE_GROUPS = ["base", "advanced"] 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 36dfea5..a45db29 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -30,7 +30,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
 
 # Features: list_features("base"), list_features(["base", "advanced"]), or hand-picked
-FEATURE_GROUPS = ["base"]
+FEATURE_GROUPS = ["base", "advanced"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 3 -- REVERTED (-0.3209)
Score: 0.5047 vs best 0.8256
Change:         n_estimators=500,         max_depth=4,         learning_rate=0.03,      
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 36dfea5..6eca5ad 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -51,13 +51,15 @@ def build_model(y_train):
 
     model = XGBClassifier(
         tree_method="hist",
-        n_estimators=300,
-        max_depth=5,
-        learning_rate=0.05,
+        n_estimators=500,
+        max_depth=4,
+        learning_rate=0.03,
         subsample=0.8,
-        colsample_bytree=0.8,
+        colsample_bytree=0.7,
         scale_pos_weight=neg / pos,
-        min_child_weight=50,
+        min_child_weight=100,
+        gamma=1.0,
+        reg_lambda=3.0,
         eval_metric="logloss",
         random_state=42,
         n_jobs=-1,
```

### Iteration 6 -- REVERTED (-1.4405)
Score: 0.9706 vs best 2.4111
Change:     # Price deceleration: short-term return improving vs medium-term     # Posit
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 9ede45b..82990ae 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -43,6 +43,26 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     )
     new_features.append("volume_climax")
 
+    # Price deceleration: short-term return improving vs medium-term
+    # Positive = decline slowing or reversing (potential bottom)
+    ret_3d = g["close"].transform(lambda x: x.pct_change(3))
+    ret_10d = g["close"].transform(lambda x: x.pct_change(10))
+    df["price_deceleration"] = ret_3d * (10 / 3) - ret_10d
+    new_features.append("price_deceleration")
+
+    # Volume exhaustion: volume shrinking while price still declining
+    # High values = sellers running out of steam
+    vol_chg_5d = g["volume"].transform(lambda x: x.pct_change(5))
+    price_chg_5d = g["close"].transform(lambda x: x.pct_change(5))
+    df["volume_exhaustion"] = (-vol_chg_5d).clip(lower=0) * (price_chg_5d < 0).astype(float)
+    new_features.append("volume_exhaustion")
+
+    # Bounce from 20d low: position in recent price range (0=at low, 1=at high)
+    low_20d = g["low"].transform(lambda x: x.rolling(20, min_periods=5).min())
+    high_20d = g["high"].transform(lambda x: x.rolling(20, min_periods=5).max())
+    df["bounce_from_low"] = (df["close"] - low_20d) / (high_20d - low_20d + 1e-8)
+    new_features.append("bounce_from_low")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 7 -- REVERTED (-0.8575)
Score: 1.5536 vs best 2.4111
Change: FEATURE_GROUPS = ["base", "percentile"] 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..5a2d2f4 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -30,7 +30,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
 
 # Features: list_features("base"), list_features(["base", "advanced"]), or hand-picked
-FEATURE_GROUPS = ["base"]
+FEATURE_GROUPS = ["base", "percentile"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 8 -- REVERTED (-5.8002)
Score: -3.3891 vs best 2.4111
Change: STOCKS = None 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..0e2b919 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -27,7 +27,7 @@ from research.features_lab import add_custom_features  # noqa: E402
 DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 
 # Stock universe: None = all, "AAPL" = single, ["AAPL", "MSFT", ...] = subset
-STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
+STOCKS = None
 
 # Features: list_features("base"), list_features(["base", "advanced"]), or hand-picked
 FEATURE_GROUPS = ["base"]
```

### Iteration 9 -- REVERTED (-0.2937)
Score: 2.1174 vs best 2.4111
Change:         num_leaves=15, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..c17b7bc 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -52,7 +52,7 @@ def build_model(y_train):
     model = LGBMClassifier(
         n_estimators=300,
         max_depth=5,
-        num_leaves=31,
+        num_leaves=15,
         learning_rate=0.05,
         subsample=0.8,
         subsample_freq=1,
```

### Iteration 10 -- REVERTED (-0.1865)
Score: 2.2246 vs best 2.4111
Change: FEATURE_GROUPS = ["base", "roc"] 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..625f468 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -30,7 +30,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
 
 # Features: list_features("base"), list_features(["base", "advanced"]), or hand-picked
-FEATURE_GROUPS = ["base"]
+FEATURE_GROUPS = ["base", "roc"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 11 -- REVERTED (-0.1220)
Score: 2.2891 vs best 2.4111
Change: THRESHOLD = 0.55 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..fac2187 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -37,7 +37,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold (default 0.5 may be too high for imbalanced data)
-THRESHOLD = 0.5
+THRESHOLD = 0.55
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 12 -- REVERTED (-2.2516)
Score: 0.1595 vs best 2.4111
Change:         boosting_type="dart", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..6624cf8 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -50,6 +50,7 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
 
     model = LGBMClassifier(
+        boosting_type="dart",
         n_estimators=300,
         max_depth=5,
         num_leaves=31,
```

### Iteration 13 -- REVERTED (-0.1038)
Score: 2.3073 vs best 2.4111
Change:         reg_alpha=0.1,         reg_lambda=2.0, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..24a216a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -59,6 +59,8 @@ def build_model(y_train):
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
         min_child_samples=50,
+        reg_alpha=0.1,
+        reg_lambda=2.0,
         random_state=42,
         n_jobs=-1,
         verbose=-1,
```

### Iteration 14 -- REVERTED (-1.0826)
Score: 1.3285 vs best 2.4111
Change:         n_estimators=500,         learning_rate=0.03, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 5129954..ae6f2b5 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -50,10 +50,10 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
 
     model = LGBMClassifier(
-        n_estimators=300,
+        n_estimators=500,
         max_depth=5,
         num_leaves=31,
-        learning_rate=0.05,
+        learning_rate=0.03,
         subsample=0.8,
         subsample_freq=1,
         colsample_bytree=0.8,
```

### Iteration 17 -- REVERTED (-0.8324)
Score: 3.3651 vs best 4.1975
Change:         min_child_samples=30, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 09009d2..3f1064b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -58,7 +58,7 @@ def build_model(y_train):
         subsample_freq=1,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
-        min_child_samples=50,
+        min_child_samples=30,
         random_state=42,
         n_jobs=-1,
         verbose=-1,
```

### Iteration 18 -- REVERTED (+0.0000)
Score: 4.1975 vs best 4.1975
Change:         num_leaves=95, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 09009d2..8079f1f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -52,7 +52,7 @@ def build_model(y_train):
     model = LGBMClassifier(
         n_estimators=300,
         max_depth=7,
-        num_leaves=63,
+        num_leaves=95,
         learning_rate=0.05,
         subsample=0.8,
         subsample_freq=1,
```

### Iteration 19 -- REVERTED (-0.0060)
Score: 4.1915 vs best 4.1975
Change:         n_estimators=400, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 09009d2..85e5141 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -50,7 +50,7 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
 
     model = LGBMClassifier(
-        n_estimators=300,
+        n_estimators=400,
         max_depth=7,
         num_leaves=63,
         learning_rate=0.05,
```

### Iteration 20 -- REVERTED (-0.7323)
Score: 3.4652 vs best 4.1975
Change:         colsample_bytree=0.6, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 09009d2..4a53300 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -56,7 +56,7 @@ def build_model(y_train):
         learning_rate=0.05,
         subsample=0.8,
         subsample_freq=1,
-        colsample_bytree=0.8,
+        colsample_bytree=0.6,
         scale_pos_weight=neg / pos,
         min_child_samples=50,
         random_state=42,
```
