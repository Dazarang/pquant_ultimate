# Combat Log

Reverted experiments with scores and diffs.

### Iteration 36 -- REVERTED (-0.5221)
Score: -0.2384 vs best 0.2837
Change:                 "subsample": 0.7, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 88f9abb..2fa6452 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -81,7 +81,7 @@ def build_model(y_train):
                 "max_depth": 5,
                 "learning_rate": 0.003,
                 "min_child_weight": 5,
-                "subsample": 0.8,
+                "subsample": 0.7,
                 "colsample_bytree": 0.6,
                 "reg_alpha": 0.5,
                 "reg_lambda": 5.0,
```

### Iteration 37 -- REVERTED (-0.2829)
Score: 0.0008 vs best 0.2837
Change:                 "learning_rate": 0.002,                 num_boost_round=7000, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 88f9abb..9c2bb64 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -79,7 +79,7 @@ def build_model(y_train):
                 "eval_metric": "ndcg",
                 "tree_method": "hist",
                 "max_depth": 5,
-                "learning_rate": 0.003,
+                "learning_rate": 0.002,
                 "min_child_weight": 5,
                 "subsample": 0.8,
                 "colsample_bytree": 0.6,
@@ -90,7 +90,7 @@ def build_model(y_train):
 
             self._model = xgb.train(
                 params, dtrain,
-                num_boost_round=5000,
+                num_boost_round=7000,
                 evals=[(deval, "eval")],
                 early_stopping_rounds=100,
                 verbose_eval=200,
```

### Iteration 38 -- REVERTED (-2.2461)
Score: -1.9624 vs best 0.2837
Change:     new_features = [         "price_efficiency_10", "return_accel_10", "returns_
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 2eff964..797dfda 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,7 +22,10 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
-    new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20"]
+    new_features = [
+        "price_efficiency_10", "return_accel_10", "returns_skew_20",
+        "closing_strength", "rel_volume_5d",
+    ]
     g = df.groupby("stock_id")
 
     def _efficiency(close):
@@ -44,6 +47,12 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     df["return_accel_10"] = g["close"].transform(_accel)
     df["returns_skew_20"] = g["close"].transform(_skew)
 
+    rng = df["high"] - df["low"]
+    df["closing_strength"] = (df["close"] - df["low"]) / (rng + 1e-10)
+    df["rel_volume_5d"] = df.groupby("stock_id")["volume"].transform(
+        lambda v: v / v.rolling(5, min_periods=3).mean()
+    )
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 40 -- REVERTED (-2.5967)
Score: -1.3718 vs best 1.2249
Change:                 "colsample_bynode": 0.8, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3d462c4..eafd9b3 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -90,6 +90,7 @@ def build_model(y_train):
                 "min_child_weight": 5,
                 "subsample": 0.8,
                 "colsample_bytree": 0.6,
+                "colsample_bynode": 0.8,
                 "reg_alpha": 0.5,
                 "reg_lambda": 5.0,
                 "gamma": 0.5,
```

### Iteration 42 -- REVERTED (-3.3755)
Score: -1.8480 vs best 1.5275
Change:                 "reg_lambda": 3.0, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 17c20e6..dd32065 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -91,7 +91,7 @@ def build_model(y_train):
                 "subsample": 0.8,
                 "colsample_bytree": 0.6,
                 "reg_alpha": 0.5,
-                "reg_lambda": 5.0,
+                "reg_lambda": 3.0,
                 "gamma": 0.5,
             }
 
```

### Iteration 43 -- REVERTED (-3.5694)
Score: -2.0419 vs best 1.5275
Change:                 "colsample_bytree": 0.5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 17c20e6..5c41fd5 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -89,7 +89,7 @@ def build_model(y_train):
                 "learning_rate": 0.003,
                 "min_child_weight": 5,
                 "subsample": 0.8,
-                "colsample_bytree": 0.6,
+                "colsample_bytree": 0.5,
                 "reg_alpha": 0.5,
                 "reg_lambda": 5.0,
                 "gamma": 0.5,
```

### Iteration 44 -- REVERTED (-3.0454)
Score: -1.5179 vs best 1.5275
Change: FEATURE_GROUPS = ["base", "advanced", "lag", "roc", "percentile", "interaction"]
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 17c20e6..49c4cd2 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -33,7 +33,7 @@ DATASET_PATH = "data/datasets/20260331/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options. None = all
-FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
+FEATURE_GROUPS = ["base", "advanced", "lag", "roc", "percentile", "interaction"]
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 45 -- REVERTED (-3.1630)
Score: -1.6355 vs best 1.5275
Change:                 "objective": "rank:map",                 "eval_metric": "map", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 17c20e6..3acd5a7 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -82,8 +82,8 @@ def build_model(y_train):
             deval.set_group(self._group_sizes(qid_ev))
 
             params = {
-                "objective": "rank:ndcg",
-                "eval_metric": "ndcg",
+                "objective": "rank:map",
+                "eval_metric": "map",
                 "tree_method": "hist",
                 "max_depth": 5,
                 "learning_rate": 0.003,
```

### Iteration 46 -- REVERTED (-3.1458)
Score: -1.6183 vs best 1.5275
Change:             lr_schedule = xgb.callback.LearningRateScheduler(                 la
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 17c20e6..d901782 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -86,7 +86,6 @@ def build_model(y_train):
                 "eval_metric": "ndcg",
                 "tree_method": "hist",
                 "max_depth": 5,
-                "learning_rate": 0.003,
                 "min_child_weight": 5,
                 "subsample": 0.8,
                 "colsample_bytree": 0.6,
@@ -95,12 +94,17 @@ def build_model(y_train):
                 "gamma": 0.5,
             }
 
+            lr_schedule = xgb.callback.LearningRateScheduler(
+                lambda r: 0.01 * (0.9995 ** r)
+            )
+
             self._model = xgb.train(
                 params, dtrain,
-                num_boost_round=5000,
+                num_boost_round=8000,
                 evals=[(deval, "eval")],
                 early_stopping_rounds=200,
                 verbose_eval=200,
+                callbacks=[lr_schedule],
             )
             print(f"Best iteration: {self._model.best_iteration}")
             return self
```

### Iteration 1 -- REVERTED (-0.0283)
Score: -1.2089 vs best -1.1806
Change:             gs = 500 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 17c20e6..ac20eb2 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -62,7 +62,7 @@ def build_model(y_train):
             return sizes
 
         def fit(self, X, y):
-            gs = 1000
+            gs = 500
             n = len(y)
 
             ev_mask = np.arange(n) % 10 == 0
```

### Iteration 2 -- REVERTED (-0.3141)
Score: -1.4947 vs best -1.1806
Change:             ev_mask = np.arange(n) % 5 == 0                 "min_child_weight": 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 17c20e6..139c1bc 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -65,7 +65,7 @@ def build_model(y_train):
             gs = 1000
             n = len(y)
 
-            ev_mask = np.arange(n) % 10 == 0
+            ev_mask = np.arange(n) % 5 == 0
             tr_idx = np.where(~ev_mask)[0]
             ev_idx = np.where(ev_mask)[0]
 
@@ -87,7 +87,7 @@ def build_model(y_train):
                 "tree_method": "hist",
                 "max_depth": 5,
                 "learning_rate": 0.003,
-                "min_child_weight": 5,
+                "min_child_weight": 30,
                 "subsample": 0.8,
                 "colsample_bytree": 0.6,
                 "reg_alpha": 0.5,
```

### Iteration 3 -- REVERTED (-0.2531)
Score: -1.4337 vs best -1.1806
Change: unknown change
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 2eff964..ed1a6a8 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,28 +22,6 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
-    new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20"]
-    g = df.groupby("stock_id")
-
-    def _efficiency(close):
-        ret = close.pct_change()
-        net = ret.rolling(10, min_periods=10).sum().abs()
-        total = ret.abs().rolling(10, min_periods=10).sum()
-        return net / (total + 1e-10)
-
-    def _accel(close):
-        ret = close.pct_change()
-        recent = ret.rolling(5, min_periods=5).sum()
-        prior = ret.shift(5).rolling(5, min_periods=5).sum()
-        return recent - prior
-
-    def _skew(close):
-        return close.pct_change().rolling(20, min_periods=15).skew()
-
-    df["price_efficiency_10"] = g["close"].transform(_efficiency)
-    df["return_accel_10"] = g["close"].transform(_accel)
-    df["returns_skew_20"] = g["close"].transform(_skew)
-
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 5 -- REVERTED (-0.0853)
Score: -1.1354 vs best -1.0501
Change:                 depth=5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0b862a9..0fd52b2 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -56,7 +56,7 @@ def build_model(y_train):
 
             self._model = CatBoostClassifier(
                 iterations=3000,
-                depth=6,
+                depth=5,
                 learning_rate=0.02,
                 l2_leaf_reg=5.0,
                 random_strength=1.0,
```

### Iteration 6 -- REVERTED (-0.0696)
Score: -1.1197 vs best -1.0501
Change:                 iterations=5000,                 learning_rate=0.01, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0b862a9..bf0c571 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -55,9 +55,9 @@ def build_model(y_train):
             ev_idx = np.where(ev_mask)[0]
 
             self._model = CatBoostClassifier(
-                iterations=3000,
+                iterations=5000,
                 depth=6,
-                learning_rate=0.02,
+                learning_rate=0.01,
                 l2_leaf_reg=5.0,
                 random_strength=1.0,
                 bagging_temperature=0.8,
```

### Iteration 7 -- REVERTED (-0.0584)
Score: -1.1085 vs best -1.0501
Change:                 depth=7, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0b862a9..10fc345 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -56,7 +56,7 @@ def build_model(y_train):
 
             self._model = CatBoostClassifier(
                 iterations=3000,
-                depth=6,
+                depth=7,
                 learning_rate=0.02,
                 l2_leaf_reg=5.0,
                 random_strength=1.0,
```

### Iteration 8 -- REVERTED (-0.0392)
Score: -1.0893 vs best -1.0501
Change:     import lightgbm as lgb     class _LGBMModel:             params = {         
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 0b862a9..d91c357 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -42,9 +42,9 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    from catboost import CatBoostClassifier
+    import lightgbm as lgb
 
-    class _CatBoostModel:
+    class _LGBMModel:
         def __init__(self):
             self.classes_ = np.array([0, 1])
 
@@ -54,33 +54,44 @@ def build_model(y_train):
             tr_idx = np.where(~ev_mask)[0]
             ev_idx = np.where(ev_mask)[0]
 
-            self._model = CatBoostClassifier(
-                iterations=3000,
-                depth=6,
-                learning_rate=0.02,
-                l2_leaf_reg=5.0,
-                random_strength=1.0,
-                bagging_temperature=0.8,
-                border_count=128,
-                scale_pos_weight=5,
-                use_best_model=True,
-                early_stopping_rounds=150,
-                verbose=200,
-                task_type='CPU',
-                thread_count=-1,
+            params = {
+                'objective': 'binary',
+                'metric': 'binary_logloss',
+                'boosting_type': 'gbdt',
+                'num_leaves': 63,
+                'max_depth': -1,
+                'learning_rate': 0.02,
+                'feature_fraction': 0.7,
+                'bagging_fraction': 0.8,
+                'bagging_freq': 5,
+                'min_child_samples': 50,
+                'lambda_l2': 5.0,
+                'scale_pos_weight': 5,
+                'verbose': -1,
+                'num_threads': -1,
+            }
+
+            train_set = lgb.Dataset(X[tr_idx], y[tr_idx])
+            eval_set = lgb.Dataset(X[ev_idx], y[ev_idx], reference=train_set)
+
+            self._model = lgb.train(
+                params,
+                train_set,
+                num_boost_round=3000,
+                valid_sets=[eval_set],
+                callbacks=[
```

### Iteration 10 -- REVERTED (-0.0058)
Score: -1.0053 vs best -0.9995
Change:                     'learning_rate': 0.01,                 num_boost_round=5000,
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 182eeb6..f3a214b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -87,7 +87,7 @@ def build_model(y_train):
                     'objective': 'binary',
                     'metric': 'binary_logloss',
                     'num_leaves': 63,
-                    'learning_rate': 0.02,
+                    'learning_rate': 0.01,
                     'feature_fraction': 0.6,
                     'bagging_fraction': 0.8,
                     'bagging_freq': 1,
@@ -97,7 +97,7 @@ def build_model(y_train):
                     'verbose': -1,
                 },
                 dtrain,
-                num_boost_round=3000,
+                num_boost_round=5000,
                 valid_sets=[deval],
                 callbacks=[
                     lgb.early_stopping(150),
```

### Iteration 14 -- REVERTED (-0.0461)
Score: -0.9626 vs best -0.9165
Change:             from scipy.stats import rankdata             n = len(cat_p)         
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 8d82cd2..1cadd3c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -127,10 +127,15 @@ def build_model(y_train):
             return self
 
         def predict_proba(self, X):
+            from scipy.stats import rankdata
             cat_p = self._cat.predict_proba(X)[:, 1]
             lgb_p = self._lgb.predict(X)
             xgb_p = self._xgb.predict_proba(X)[:, 1]
-            avg = (cat_p + lgb_p + xgb_p) / 3
+            n = len(cat_p)
+            cat_r = rankdata(cat_p) / n
+            lgb_r = rankdata(lgb_p) / n
+            xgb_r = rankdata(xgb_p) / n
+            avg = (cat_r + lgb_r + xgb_r) / 3
             return np.column_stack([1 - avg, avg])
 
     return _EnsembleModel()
```

### Iteration 15 -- REVERTED (-0.0664)
Score: -0.9829 vs best -0.9165
Change:     from sklearn.ensemble import HistGradientBoostingClassifier             w = 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 8d82cd2..6d491ed 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -45,6 +45,7 @@ def build_model(y_train):
     from catboost import CatBoostClassifier
     import lightgbm as lgb
     import xgboost as xgb
+    from sklearn.ensemble import HistGradientBoostingClassifier
 
     class _EnsembleModel:
         def __init__(self):
@@ -124,13 +125,30 @@ def build_model(y_train):
             )
             print(f"XGBoost best iteration: {self._xgb.best_iteration}")
 
+            w = np.where(y[tr_idx] == 1, 5.0, 1.0)
+            self._hgb = HistGradientBoostingClassifier(
+                max_iter=2000,
+                max_leaf_nodes=63,
+                learning_rate=0.02,
+                min_samples_leaf=20,
+                l2_regularization=5.0,
+                early_stopping=True,
+                validation_fraction=0.1,
+                n_iter_no_change=150,
+                scoring='neg_log_loss',
+                verbose=1,
+            )
+            self._hgb.fit(X[tr_idx], y[tr_idx], sample_weight=w)
+            print(f"HistGBM n_iter: {self._hgb.n_iter_}")
+
             return self
 
         def predict_proba(self, X):
             cat_p = self._cat.predict_proba(X)[:, 1]
             lgb_p = self._lgb.predict(X)
             xgb_p = self._xgb.predict_proba(X)[:, 1]
-            avg = (cat_p + lgb_p + xgb_p) / 3
+            hgb_p = self._hgb.predict_proba(X)[:, 1]
+            avg = (cat_p + lgb_p + xgb_p + hgb_p) / 4
             return np.column_stack([1 - avg, avg])
 
     return _EnsembleModel()
```

### Iteration 16 -- REVERTED (-0.0077)
Score: -0.9242 vs best -0.9165
Change:                     'lambda_l1': 0.5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 8d82cd2..9cde96b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -89,6 +89,7 @@ def build_model(y_train):
                     'bagging_fraction': 0.8,
                     'bagging_freq': 1,
                     'scale_pos_weight': 5,
+                    'lambda_l1': 0.5,
                     'lambda_l2': 5.0,
                     'min_child_samples': 20,
                     'verbose': -1,
```

### Iteration 18 -- REVERTED (-0.1226)
Score: -1.0345 vs best -0.9119
Change:             split = int(n * 0.9)             tr_idx = np.arange(split)          
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 178fc56..7720f4f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -52,9 +52,9 @@ def build_model(y_train):
 
         def fit(self, X, y):
             n = len(y)
-            ev_mask = np.arange(n) % 10 == 0
-            tr_idx = np.where(~ev_mask)[0]
-            ev_idx = np.where(ev_mask)[0]
+            split = int(n * 0.9)
+            tr_idx = np.arange(split)
+            ev_idx = np.arange(split, n)
 
             self._cat = CatBoostClassifier(
                 iterations=3000,
```

### Iteration 20 -- REVERTED (-0.0215)
Score: -0.9224 vs best -0.9009
Change:                     'num_leaves': 45, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..6edfeb8 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -83,7 +83,7 @@ def build_model(y_train):
                 {
                     'objective': 'binary',
                     'metric': 'binary_logloss',
-                    'num_leaves': 63,
+                    'num_leaves': 45,
                     'learning_rate': 0.02,
                     'feature_fraction': 0.6,
                     'bagging_fraction': 0.8,
```

### Iteration 21 -- REVERTED (-0.0714)
Score: -0.9723 vs best -0.9009
Change:             self._meta = LogisticRegression(C=0.5, max_iter=300) 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..2762f8b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -129,7 +129,7 @@ def build_model(y_train):
             lgb_oof = self._lgb.predict(X[ev_idx])
             xgb_oof = self._xgb.predict_proba(X[ev_idx])[:, 1]
             meta_X = np.column_stack([cat_oof, lgb_oof, xgb_oof])
-            self._meta = LogisticRegression(C=0.1, max_iter=300)
+            self._meta = LogisticRegression(C=0.5, max_iter=300)
             self._meta.fit(meta_X, y[ev_idx])
             print(f"Meta weights: {self._meta.coef_[0]}, intercept: {self._meta.intercept_[0]:.4f}")
 
```

### Iteration 22 -- REVERTED (-0.0542)
Score: -0.9551 vs best -0.9009
Change:                 scale_pos_weight=5, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..338d37a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -64,7 +64,7 @@ def build_model(y_train):
                 random_strength=1.0,
                 bagging_temperature=0.8,
                 border_count=128,
-                scale_pos_weight=3,
+                scale_pos_weight=5,
                 use_best_model=True,
                 early_stopping_rounds=150,
                 verbose=200,
```

### Iteration 23 -- REVERTED (-0.1722)
Score: -1.0731 vs best -0.9009
Change:             from sklearn.ensemble import GradientBoostingClassifier             
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..2dee59e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -124,14 +124,20 @@ def build_model(y_train):
             )
             print(f"XGBoost best iteration: {self._xgb.best_iteration}")
 
-            from sklearn.linear_model import LogisticRegression
+            from sklearn.ensemble import GradientBoostingClassifier
             cat_oof = self._cat.predict_proba(X[ev_idx])[:, 1]
             lgb_oof = self._lgb.predict(X[ev_idx])
             xgb_oof = self._xgb.predict_proba(X[ev_idx])[:, 1]
             meta_X = np.column_stack([cat_oof, lgb_oof, xgb_oof])
-            self._meta = LogisticRegression(C=0.1, max_iter=300)
+            self._meta = GradientBoostingClassifier(
+                n_estimators=50,
+                max_depth=2,
+                learning_rate=0.1,
+                subsample=0.8,
+                min_samples_leaf=100,
+            )
             self._meta.fit(meta_X, y[ev_idx])
-            print(f"Meta weights: {self._meta.coef_[0]}, intercept: {self._meta.intercept_[0]:.4f}")
+            print(f"Meta feature importances: {self._meta.feature_importances_}")
 
             return self
 
```

### Iteration 24 -- REVERTED (-0.0268)
Score: -0.9277 vs best -0.9009
Change:             idx = np.arange(n)             ev_es_mask = idx % 10 == 0           
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..d5455ce 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -52,9 +52,12 @@ def build_model(y_train):
 
         def fit(self, X, y):
             n = len(y)
-            ev_mask = np.arange(n) % 10 == 0
-            tr_idx = np.where(~ev_mask)[0]
-            ev_idx = np.where(ev_mask)[0]
+            idx = np.arange(n)
+            ev_es_mask = idx % 10 == 0
+            ev_meta_mask = idx % 10 == 1
+            tr_idx = np.where(~ev_es_mask & ~ev_meta_mask)[0]
+            ev_es_idx = np.where(ev_es_mask)[0]
+            ev_meta_idx = np.where(ev_meta_mask)[0]
 
             self._cat = CatBoostClassifier(
                 iterations=3000,
@@ -73,12 +76,12 @@ def build_model(y_train):
             )
             self._cat.fit(
                 X[tr_idx], y[tr_idx],
-                eval_set=(X[ev_idx], y[ev_idx]),
+                eval_set=(X[ev_es_idx], y[ev_es_idx]),
             )
             print(f"CatBoost best iteration: {self._cat.best_iteration_}")
 
             dtrain = lgb.Dataset(X[tr_idx], y[tr_idx])
-            deval = lgb.Dataset(X[ev_idx], y[ev_idx], reference=dtrain)
+            deval = lgb.Dataset(X[ev_es_idx], y[ev_es_idx], reference=dtrain)
             self._lgb = lgb.train(
                 {
                     'objective': 'binary',
@@ -119,18 +122,18 @@ def build_model(y_train):
             )
             self._xgb.fit(
                 X[tr_idx], y[tr_idx],
-                eval_set=[(X[ev_idx], y[ev_idx])],
+                eval_set=[(X[ev_es_idx], y[ev_es_idx])],
                 verbose=200,
             )
             print(f"XGBoost best iteration: {self._xgb.best_iteration}")
 
             from sklearn.linear_model import LogisticRegression
-            cat_oof = self._cat.predict_proba(X[ev_idx])[:, 1]
-            lgb_oof = self._lgb.predict(X[ev_idx])
-            xgb_oof = self._xgb.predict_proba(X[ev_idx])[:, 1]
+            cat_oof = self._cat.predict_proba(X[ev_meta_idx])[:, 1]
+            lgb_oof = self._lgb.predict(X[ev_meta_idx])
+            xgb_oof = self._xgb.predict_proba(X[ev_meta_idx])[:, 1]
             meta_X = np.column_stack([cat_oof, lgb_oof, xgb_oof])
             self._meta = LogisticRegression(C=0.1, max_iter=300)
-            self._meta.fit(meta_X, y[ev_idx])
+            self._meta.fit(meta_X, y[ev_meta_idx])
             print(f"Meta weights: {self._meta.coef_[0]}, intercept: {self._meta.intercept_[0]:.4f}")
 
             return self
```

### Iteration 25 -- REVERTED (-0.1029)
Score: -1.0038 vs best -0.9009
Change:             from scipy.special import expit             gs = 1000             n_
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..89719b9 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -103,31 +103,40 @@ def build_model(y_train):
             )
             print(f"LightGBM best iteration: {self._lgb.best_iteration}")
 
-            self._xgb = xgb.XGBClassifier(
-                n_estimators=3000,
-                max_depth=6,
-                learning_rate=0.02,
-                subsample=0.8,
-                colsample_bytree=0.6,
-                scale_pos_weight=8,
-                reg_lambda=5.0,
-                min_child_weight=5,
-                tree_method='hist',
+            from scipy.special import expit
+            gs = 1000
+            n_tr, n_ev = len(tr_idx), len(ev_idx)
+            qid_tr = np.repeat(np.arange(n_tr // gs + 1), gs)[:n_tr]
+            qid_ev = np.repeat(np.arange(n_ev // gs + 1), gs)[:n_ev]
+            dtrain_r = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
+            dtrain_r.set_group(np.bincount(qid_tr).tolist())
+            deval_r = xgb.DMatrix(X[ev_idx], label=y[ev_idx])
+            deval_r.set_group(np.bincount(qid_ev).tolist())
+            self._xgb = xgb.train(
+                {
+                    "objective": "rank:ndcg",
+                    "eval_metric": "ndcg",
+                    "tree_method": "hist",
+                    "max_depth": 5,
+                    "learning_rate": 0.02,
+                    "min_child_weight": 5,
+                    "subsample": 0.8,
+                    "colsample_bytree": 0.6,
+                    "reg_lambda": 5.0,
+                    "gamma": 0.5,
+                },
+                dtrain_r,
+                num_boost_round=3000,
+                evals=[(deval_r, "eval")],
                 early_stopping_rounds=150,
-                eval_metric='logloss',
-                verbosity=1,
-            )
-            self._xgb.fit(
-                X[tr_idx], y[tr_idx],
-                eval_set=[(X[ev_idx], y[ev_idx])],
-                verbose=200,
+                verbose_eval=200,
             )
-            print(f"XGBoost best iteration: {self._xgb.best_iteration}")
+            print(f"XGBoost ranking best iteration: {self._xgb.best_iteration}")
 
             from sklearn.linear_model import LogisticRegression
             cat_oof = self._cat.predict_proba(X[ev_idx])[:, 1]
             lgb_oof = self._lgb.predict(X[ev_idx])
-            xgb_oof = self._xgb.predict_proba(X[ev_idx])[:, 1]
```

### Iteration 26 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 137).
Change:             from research.model_wrappers import FocalTorchClassifier, TorchMLP  
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..b33466d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -44,7 +44,6 @@ def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
     from catboost import CatBoostClassifier
     import lightgbm as lgb
-    import xgboost as xgb
 
     class _EnsembleModel:
         def __init__(self):
@@ -103,32 +102,25 @@ def build_model(y_train):
             )
             print(f"LightGBM best iteration: {self._lgb.best_iteration}")
 
-            self._xgb = xgb.XGBClassifier(
-                n_estimators=3000,
-                max_depth=6,
-                learning_rate=0.02,
-                subsample=0.8,
-                colsample_bytree=0.6,
-                scale_pos_weight=8,
-                reg_lambda=5.0,
-                min_child_weight=5,
-                tree_method='hist',
-                early_stopping_rounds=150,
-                eval_metric='logloss',
-                verbosity=1,
-            )
-            self._xgb.fit(
-                X[tr_idx], y[tr_idx],
-                eval_set=[(X[ev_idx], y[ev_idx])],
-                verbose=200,
+            from research.model_wrappers import FocalTorchClassifier, TorchMLP
+            input_dim = X.shape[1]
+            module = TorchMLP(input_dim, hidden_dims=(256, 128, 64), dropout=0.3)
+            self._focal = FocalTorchClassifier(
+                module=module,
+                epochs=30,
+                lr=1e-3,
+                batch_size=1024,
+                alpha=0.75,
+                focal_gamma=2.0,
             )
-            print(f"XGBoost best iteration: {self._xgb.best_iteration}")
+            self._focal.fit(X[tr_idx], y[tr_idx])
+            print("FocalMLP training complete")
 
             from sklearn.linear_model import LogisticRegression
             cat_oof = self._cat.predict_proba(X[ev_idx])[:, 1]
             lgb_oof = self._lgb.predict(X[ev_idx])
-            xgb_oof = self._xgb.predict_proba(X[ev_idx])[:, 1]
-            meta_X = np.column_stack([cat_oof, lgb_oof, xgb_oof])
+            focal_oof = self._focal.predict_proba(X[ev_idx])[:, 1]
+            meta_X = np.column_stack([cat_oof, lgb_oof, focal_oof])
             self._meta = LogisticRegression(C=0.1, max_iter=300)
             self._meta.fit(meta_X, y[ev_idx])
             print(f"Meta weights: {self._meta.coef_[0]}, intercept: {self._meta.intercept_[0]:.4f}")
@@ -138,8 +130,8 @@ def build_model(y_train):
```
Traceback:
```
Immutability check: PASSED
Running experiment (timeout: 2700s)...
research/gate.sh: line 61:  8543 Killed: 9                  perl -e "alarm $TIMEOUT_SECONDS; exec @ARGV" uv run python research/experiment.py > "$LOG_FILE" 2>&1
GATE VIOLATION: Experiment crashed (exit code 137).
--- Last 30 lines of log ---
```

### Iteration 27 -- REVERTED (-0.5914)
Score: -1.4923 vs best -0.9009
Change:             # --- Feature selection: quick LGB to find top features by gain --- 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..99531bb 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -56,6 +56,35 @@ def build_model(y_train):
             tr_idx = np.where(~ev_mask)[0]
             ev_idx = np.where(ev_mask)[0]
 
+            # --- Feature selection: quick LGB to find top features by gain ---
+            _TOP_K = 120
+            dtrain_fs = lgb.Dataset(X[tr_idx], y[tr_idx])
+            deval_fs = lgb.Dataset(X[ev_idx], y[ev_idx], reference=dtrain_fs)
+            _fs_model = lgb.train(
+                {
+                    'objective': 'binary',
+                    'metric': 'binary_logloss',
+                    'num_leaves': 31,
+                    'learning_rate': 0.05,
+                    'feature_fraction': 0.7,
+                    'bagging_fraction': 0.8,
+                    'bagging_freq': 1,
+                    'scale_pos_weight': 5,
+                    'verbose': -1,
+                },
+                dtrain_fs,
+                num_boost_round=500,
+                valid_sets=[deval_fs],
+                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
+            )
+            importances = _fs_model.feature_importance(importance_type='gain')
+            k = min(_TOP_K, X.shape[1])
+            self._sel_idx = np.argsort(importances)[-k:]
+            self._sel_idx.sort()
+            print(f"Feature selection: kept {k}/{X.shape[1]} features (top by gain)")
+            X = X[:, self._sel_idx]
+            # --- End feature selection ---
+
             self._cat = CatBoostClassifier(
                 iterations=3000,
                 depth=6,
@@ -136,6 +165,7 @@ def build_model(y_train):
             return self
 
         def predict_proba(self, X):
+            X = X[:, self._sel_idx]
             cat_p = self._cat.predict_proba(X)[:, 1]
             lgb_p = self._lgb.predict(X)
             xgb_p = self._xgb.predict_proba(X)[:, 1]
```

### Iteration 28 -- REVERTED (-0.6249)
Score: -1.5258 vs best -0.9009
Change:             # Time-decay sample weights: upweight recent data (rows are chronolo
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 792379c..dbfbee0 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -56,6 +56,10 @@ def build_model(y_train):
             tr_idx = np.where(~ev_mask)[0]
             ev_idx = np.where(ev_mask)[0]
 
+            # Time-decay sample weights: upweight recent data (rows are chronological)
+            # Oldest training samples get 0.3x, newest get 1.0x
+            w = np.linspace(0.3, 1.0, n)
+
             self._cat = CatBoostClassifier(
                 iterations=3000,
                 depth=6,
@@ -74,11 +78,12 @@ def build_model(y_train):
             self._cat.fit(
                 X[tr_idx], y[tr_idx],
                 eval_set=(X[ev_idx], y[ev_idx]),
+                sample_weight=w[tr_idx],
             )
             print(f"CatBoost best iteration: {self._cat.best_iteration_}")
 
-            dtrain = lgb.Dataset(X[tr_idx], y[tr_idx])
-            deval = lgb.Dataset(X[ev_idx], y[ev_idx], reference=dtrain)
+            dtrain = lgb.Dataset(X[tr_idx], y[tr_idx], weight=w[tr_idx])
+            deval = lgb.Dataset(X[ev_idx], y[ev_idx], weight=w[ev_idx], reference=dtrain)
             self._lgb = lgb.train(
                 {
                     'objective': 'binary',
@@ -120,6 +125,7 @@ def build_model(y_train):
             self._xgb.fit(
                 X[tr_idx], y[tr_idx],
                 eval_set=[(X[ev_idx], y[ev_idx])],
+                sample_weight=w[tr_idx],
                 verbose=200,
             )
             print(f"XGBoost best iteration: {self._xgb.best_iteration}")
@@ -130,7 +136,7 @@ def build_model(y_train):
             xgb_oof = self._xgb.predict_proba(X[ev_idx])[:, 1]
             meta_X = np.column_stack([cat_oof, lgb_oof, xgb_oof])
             self._meta = LogisticRegression(C=0.1, max_iter=300)
-            self._meta.fit(meta_X, y[ev_idx])
+            self._meta.fit(meta_X, y[ev_idx], sample_weight=w[ev_idx])
             print(f"Meta weights: {self._meta.coef_[0]}, intercept: {self._meta.intercept_[0]:.4f}")
 
             return self
```
