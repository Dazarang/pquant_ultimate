# Combat Log

Reverted experiments with scores and diffs.

### Iteration 13 -- REVERTED (-2.2175)
Score: -2.2103 vs best 0.0072
Change:     from catboost import CatBoostClassifier, Pool     class _CatBoostPivot:     
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..fdf906d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -42,71 +42,42 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    import xgboost as xgb
-    from scipy.special import expit
+    from catboost import CatBoostClassifier, Pool
 
-    class _AdaptiveRanking:
+    class _CatBoostPivot:
         def __init__(self):
             self.classes_ = np.array([0, 1])
 
-        @staticmethod
-        def _group_sizes(qid):
-            sizes, cur, cnt = [], qid[0], 1
-            for i in range(1, len(qid)):
-                if qid[i] == cur:
-                    cnt += 1
-                else:
-                    sizes.append(cnt)
-                    cur, cnt = qid[i], 1
-            sizes.append(cnt)
-            return sizes
-
         def fit(self, X, y):
-            gs = 1000
             n = len(y)
             split = int(n * 0.9)
 
-            qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
-            qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
-
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
-            dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
-
-            params = {
-                "objective": "rank:ndcg",
-                "eval_metric": "ndcg",
-                "tree_method": "hist",
-                "max_depth": 4,
-                "learning_rate": 0.005,
-                "min_child_weight": 5,
-                "subsample": 0.8,
-                "colsample_bytree": 0.6,
-                "reg_alpha": 0.5,
-                "reg_lambda": 5.0,
-                "gamma": 0.5,
-            }
-
-            self._model = xgb.train(
-                params, dtrain,
-                num_boost_round=5000,
-                evals=[(deval, "eval")],
```

### Iteration 14 -- REVERTED (-1.6823)
Score: -1.6751 vs best 0.0072
Change:         def _date_groups(d):             if len(d) <= 1:                 return 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..8c7ff4e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -50,29 +50,27 @@ def build_model(y_train):
             self.classes_ = np.array([0, 1])
 
         @staticmethod
-        def _group_sizes(qid):
-            sizes, cur, cnt = [], qid[0], 1
-            for i in range(1, len(qid)):
-                if qid[i] == cur:
-                    cnt += 1
-                else:
-                    sizes.append(cnt)
-                    cur, cnt = qid[i], 1
-            sizes.append(cnt)
-            return sizes
+        def _date_groups(d):
+            if len(d) <= 1:
+                return [len(d)]
+            breaks = np.where(np.abs(np.diff(d)) > 1e-6)[0] + 1
+            return np.diff(np.concatenate([[0], breaks, [len(d)]])).tolist()
 
         def fit(self, X, y):
-            gs = 1000
+            date_col = X[:, -1]
+            X_feat = X[:, :-1]
+
             n = len(y)
             split = int(n * 0.9)
 
-            qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
-            qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
+            dtrain = xgb.DMatrix(X_feat[:split], label=y[:split])
+            dtrain.set_group(self._date_groups(date_col[:split]))
+            deval = xgb.DMatrix(X_feat[split:], label=y[split:])
+            deval.set_group(self._date_groups(date_col[split:]))
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
-            dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
+            tr_groups = len(self._date_groups(date_col[:split]))
+            ev_groups = len(self._date_groups(date_col[split:]))
+            print(f"Date-aligned groups: train={tr_groups}, eval={ev_groups}")
 
             params = {
                 "objective": "rank:ndcg",
@@ -99,7 +97,7 @@ def build_model(y_train):
             return self
 
         def predict_proba(self, X):
-            dtest = xgb.DMatrix(X)
+            dtest = xgb.DMatrix(X[:, :-1])
             scores = self._model.predict(
                 dtest, iteration_range=(0, self._model.best_iteration + 1)
             )
diff --git a/research/features_lab.py b/research/features_lab.py
index 2eff964..22c7527 100644
```

### Iteration 15 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             dtrain = xgb.DMatrix(X[:split], label=y[:split])             dtrain.
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..26021a8 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -69,11 +69,6 @@ def build_model(y_train):
             qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
             qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
-            dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
-
             params = {
                 "objective": "rank:ndcg",
                 "eval_metric": "ndcg",
@@ -88,8 +83,29 @@ def build_model(y_train):
                 "gamma": 0.5,
             }
 
-            self._model = xgb.train(
+            dtrain = xgb.DMatrix(X[:split], label=y[:split])
+            dtrain.set_group(self._group_sizes(qid_tr))
+            deval = xgb.DMatrix(X[split:], label=y[split:])
+            deval.set_group(self._group_sizes(qid_ev))
+
+            scout = xgb.train(
                 params, dtrain,
+                num_boost_round=2000,
+                evals=[(deval, "eval")],
+                early_stopping_rounds=50,
+                verbose_eval=False,
+            )
+            scores = scout.predict(dtrain, iteration_range=(0, scout.best_iteration + 1))
+            hard_neg = (scores >= np.percentile(scores, 98)) & (y[:split] == 0)
+            print(f"Scout iter: {scout.best_iteration}, hard negatives: {hard_neg.sum()}")
+
+            weights = np.ones(split, dtype=np.float32)
+            weights[hard_neg] = 3.0
+            dtrain_w = xgb.DMatrix(X[:split], label=y[:split], weight=weights)
+            dtrain_w.set_group(self._group_sizes(qid_tr))
+
+            self._model = xgb.train(
+                params, dtrain_w,
                 num_boost_round=5000,
                 evals=[(deval, "eval")],
                 early_stopping_rounds=100,
```
Traceback:
```
  [bt] (4) 5   libffi.dylib                        0x0000000195646050 ffi_call_SYSV + 80
  [bt] (5) 6   libffi.dylib                        0x000000019564f5b8 ffi_call_int + 1220
  [bt] (6) 7   _ctypes.cpython-314-darwin.so       0x0000000103544828 _ctypes_callproc + 1304
  [bt] (7) 8   _ctypes.cpython-314-darwin.so       0x000000010354125c PyCFuncPtr_call + 800
  [bt] (8) 9   libpython3.14.dylib                 0x00000001035c9d58 _PyObject_MakeTpCall + 304
```

### Iteration 16 -- REVERTED (-2.2635)
Score: -2.2563 vs best 0.0072
Change:     import lightgbm as lgb     class _LGBMLambdaRank:         def _make_groups(t
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..ec2a8c1 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -42,71 +42,68 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    import xgboost as xgb
+    import lightgbm as lgb
     from scipy.special import expit
 
-    class _AdaptiveRanking:
+    class _LGBMLambdaRank:
         def __init__(self):
             self.classes_ = np.array([0, 1])
 
         @staticmethod
-        def _group_sizes(qid):
-            sizes, cur, cnt = [], qid[0], 1
-            for i in range(1, len(qid)):
-                if qid[i] == cur:
-                    cnt += 1
-                else:
-                    sizes.append(cnt)
-                    cur, cnt = qid[i], 1
-            sizes.append(cnt)
-            return sizes
+        def _make_groups(total, gs):
+            groups = []
+            remaining = total
+            while remaining > 0:
+                groups.append(min(gs, remaining))
+                remaining -= gs
+            return groups
 
         def fit(self, X, y):
             gs = 1000
             n = len(y)
             split = int(n * 0.9)
 
-            qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
-            qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
+            group_tr = self._make_groups(split, gs)
+            group_ev = self._make_groups(n - split, gs)
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
-            dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
+            dtrain = lgb.Dataset(X[:split], label=y[:split], group=group_tr)
+            deval = lgb.Dataset(X[split:], label=y[split:], group=group_ev, reference=dtrain)
 
             params = {
-                "objective": "rank:ndcg",
-                "eval_metric": "ndcg",
-                "tree_method": "hist",
-                "max_depth": 4,
+                "objective": "lambdarank",
+                "metric": "ndcg",
```

### Iteration 17 -- REVERTED (-2.2358)
Score: -2.2286 vs best 0.0072
Change:             from xgboost import XGBClassifier              neg, pos = int((y == 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..412cd06 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -62,16 +62,34 @@ def build_model(y_train):
             return sizes
 
         def fit(self, X, y):
+            from xgboost import XGBClassifier
+
+            neg, pos = int((y == 0).sum()), int((y == 1).sum())
+            scout = XGBClassifier(
+                n_estimators=200, max_depth=2, learning_rate=0.1,
+                scale_pos_weight=neg / pos, tree_method="hist",
+                verbosity=0,
+            )
+            scout.fit(X, y)
+            proba = scout.predict_proba(X)[:, 1]
+
+            neg_mask = y == 0
+            neg_threshold = np.percentile(proba[neg_mask], 50)
+            mask = (y == 1) | (proba >= neg_threshold)
+
+            X_f, y_f = X[mask], y[mask]
+            print(f"Hard-neg filter: {len(X)} -> {len(X_f)} ({100*y_f.mean():.1f}% pos)")
+
             gs = 1000
-            n = len(y)
+            n = len(y_f)
             split = int(n * 0.9)
 
             qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
             qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
+            dtrain = xgb.DMatrix(X_f[:split], label=y_f[:split])
             dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
+            deval = xgb.DMatrix(X_f[split:], label=y_f[split:])
             deval.set_group(self._group_sizes(qid_ev))
 
             params = {
```

### Iteration 18 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             weights = np.linspace(0.5, 1.0, split)             dtrain = xgb.DMat
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..09ce3d4 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -69,7 +69,8 @@ def build_model(y_train):
             qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
             qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
+            weights = np.linspace(0.5, 1.0, split)
+            dtrain = xgb.DMatrix(X[:split], label=y[:split], weight=weights)
             dtrain.set_group(self._group_sizes(qid_tr))
             deval = xgb.DMatrix(X[split:], label=y[split:])
             deval.set_group(self._group_sizes(qid_ev))
```
Traceback:
```
  [bt] (4) 5   libffi.dylib                        0x0000000195646050 ffi_call_SYSV + 80
  [bt] (5) 6   libffi.dylib                        0x000000019564f5b8 ffi_call_int + 1220
  [bt] (6) 7   _ctypes.cpython-314-darwin.so       0x000000010993c828 _ctypes_callproc + 1304
  [bt] (7) 8   _ctypes.cpython-314-darwin.so       0x000000010993925c PyCFuncPtr_call + 800
  [bt] (8) 9   libpython3.14.dylib                 0x0000000104b59d58 _PyObject_MakeTpCall + 304
```

### Iteration 19 -- REVERTED (-1.2034)
Score: -1.1962 vs best 0.0072
Change:             base_params = {             configs = [                 {"max_depth"
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..310cfeb 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -74,36 +74,49 @@ def build_model(y_train):
             deval = xgb.DMatrix(X[split:], label=y[split:])
             deval.set_group(self._group_sizes(qid_ev))
 
-            params = {
+            base_params = {
                 "objective": "rank:ndcg",
                 "eval_metric": "ndcg",
                 "tree_method": "hist",
-                "max_depth": 4,
-                "learning_rate": 0.005,
                 "min_child_weight": 5,
-                "subsample": 0.8,
-                "colsample_bytree": 0.6,
                 "reg_alpha": 0.5,
                 "reg_lambda": 5.0,
                 "gamma": 0.5,
             }
 
-            self._model = xgb.train(
-                params, dtrain,
-                num_boost_round=5000,
-                evals=[(deval, "eval")],
-                early_stopping_rounds=100,
-                verbose_eval=200,
-            )
-            print(f"Best iteration: {self._model.best_iteration}")
+            configs = [
+                {"max_depth": 4, "learning_rate": 0.005, "subsample": 0.8, "colsample_bytree": 0.6},
+                {"max_depth": 3, "learning_rate": 0.01, "subsample": 0.6, "colsample_bytree": 0.9},
+                {"max_depth": 6, "learning_rate": 0.003, "subsample": 0.9, "colsample_bytree": 0.3},
+            ]
+
+            self._models = []
+            for i, cfg in enumerate(configs):
+                params = {**base_params, **cfg}
+                model = xgb.train(
+                    params, dtrain,
+                    num_boost_round=2000,
+                    evals=[(deval, "eval")],
+                    early_stopping_rounds=50,
+                    verbose_eval=200,
+                )
+                print(f"Sub-model {i}: best_iteration={model.best_iteration}")
+                self._models.append(model)
+
             return self
 
         def predict_proba(self, X):
             dtest = xgb.DMatrix(X)
-            scores = self._model.predict(
-                dtest, iteration_range=(0, self._model.best_iteration + 1)
-            )
-            proba_1 = expit(scores)
+            all_scores = []
+            for model in self._models:
```

### Iteration 20 -- REVERTED (-2.2725)
Score: -2.2653 vs best 0.0072
Change:     class _GlobalClassifier:                 "objective": "binary:logistic",    
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..0a39ba0 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -43,40 +43,20 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
     import xgboost as xgb
-    from scipy.special import expit
-
-    class _AdaptiveRanking:
+    class _GlobalClassifier:
         def __init__(self):
             self.classes_ = np.array([0, 1])
 
-        @staticmethod
-        def _group_sizes(qid):
-            sizes, cur, cnt = [], qid[0], 1
-            for i in range(1, len(qid)):
-                if qid[i] == cur:
-                    cnt += 1
-                else:
-                    sizes.append(cnt)
-                    cur, cnt = qid[i], 1
-            sizes.append(cnt)
-            return sizes
-
         def fit(self, X, y):
-            gs = 1000
             n = len(y)
             split = int(n * 0.9)
 
-            qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
-            qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
-
             dtrain = xgb.DMatrix(X[:split], label=y[:split])
-            dtrain.set_group(self._group_sizes(qid_tr))
             deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
 
             params = {
-                "objective": "rank:ndcg",
-                "eval_metric": "ndcg",
+                "objective": "binary:logistic",
+                "eval_metric": "aucpr",
                 "tree_method": "hist",
                 "max_depth": 4,
                 "learning_rate": 0.005,
@@ -100,13 +80,12 @@ def build_model(y_train):
 
         def predict_proba(self, X):
             dtest = xgb.DMatrix(X)
-            scores = self._model.predict(
+            proba_1 = self._model.predict(
                 dtest, iteration_range=(0, self._model.best_iteration + 1)
             )
-            proba_1 = expit(scores)
             return np.column_stack([1 - proba_1, proba_1])
 
-    return _AdaptiveRanking()
```

### Iteration 21 -- REVERTED (-1.8306)
Score: -1.8234 vs best 0.0072
Change:     new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20",
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 2eff964..44d37ed 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,7 +22,8 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
-    new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20"]
+    new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20",
+                    "xs_ret5_rank", "xs_ret20_rank"]
     g = df.groupby("stock_id")
 
     def _efficiency(close):
@@ -44,6 +45,11 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     df["return_accel_10"] = g["close"].transform(_accel)
     df["returns_skew_20"] = g["close"].transform(_skew)
 
+    ret5 = g["close"].pct_change(5)
+    ret20 = g["close"].pct_change(20)
+    df["xs_ret5_rank"] = ret5.groupby(df["date"]).rank(pct=True)
+    df["xs_ret20_rank"] = ret20.groupby(df["date"]).rank(pct=True)
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 22 -- REVERTED (-2.0291)
Score: -2.0219 vs best 0.0072
Change:     from research.model_wrappers import FocalTorchClassifier, TorchMLP     class
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..024213a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -42,71 +42,29 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    import xgboost as xgb
-    from scipy.special import expit
+    from research.model_wrappers import FocalTorchClassifier, TorchMLP
 
-    class _AdaptiveRanking:
+    class _FocalMLP:
         def __init__(self):
             self.classes_ = np.array([0, 1])
 
-        @staticmethod
-        def _group_sizes(qid):
-            sizes, cur, cnt = [], qid[0], 1
-            for i in range(1, len(qid)):
-                if qid[i] == cur:
-                    cnt += 1
-                else:
-                    sizes.append(cnt)
-                    cur, cnt = qid[i], 1
-            sizes.append(cnt)
-            return sizes
-
         def fit(self, X, y):
-            gs = 1000
-            n = len(y)
-            split = int(n * 0.9)
-
-            qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
-            qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
-
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
-            dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
-
-            params = {
-                "objective": "rank:ndcg",
-                "eval_metric": "ndcg",
-                "tree_method": "hist",
-                "max_depth": 4,
-                "learning_rate": 0.005,
-                "min_child_weight": 5,
-                "subsample": 0.8,
-                "colsample_bytree": 0.6,
-                "reg_alpha": 0.5,
-                "reg_lambda": 5.0,
-                "gamma": 0.5,
-            }
-
-            self._model = xgb.train(
-                params, dtrain,
-                num_boost_round=5000,
-                evals=[(deval, "eval")],
```

### Iteration 23 -- REVERTED (-1.3781)
Score: -1.3709 vs best 0.0072
Change:                 "grow_policy": "lossguide",                 "max_leaves": 31,   
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..5e1d54b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -78,7 +78,9 @@ def build_model(y_train):
                 "objective": "rank:ndcg",
                 "eval_metric": "ndcg",
                 "tree_method": "hist",
-                "max_depth": 4,
+                "grow_policy": "lossguide",
+                "max_leaves": 31,
+                "max_depth": 0,
                 "learning_rate": 0.005,
                 "min_child_weight": 5,
                 "subsample": 0.8,
```

### Iteration 24 -- REVERTED (-1.4864)
Score: -1.4792 vs best 0.0072
Change:                 "objective": "rank:pairwise", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..9070b8a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -75,7 +75,7 @@ def build_model(y_train):
             deval.set_group(self._group_sizes(qid_ev))
 
             params = {
-                "objective": "rank:ndcg",
+                "objective": "rank:pairwise",
                 "eval_metric": "ndcg",
                 "tree_method": "hist",
                 "max_depth": 4,
```

### Iteration 25 -- REVERTED (-0.4254)
Score: -0.4182 vs best 0.0072
Change:             deval.set_group([n - split]) 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..8c567f3 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -67,12 +67,11 @@ def build_model(y_train):
             split = int(n * 0.9)
 
             qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
-            qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
 
             dtrain = xgb.DMatrix(X[:split], label=y[:split])
             dtrain.set_group(self._group_sizes(qid_tr))
             deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
+            deval.set_group([n - split])
 
             params = {
                 "objective": "rank:ndcg",
```

### Iteration 26 -- REVERTED (-2.2535)
Score: -2.2463 vs best 0.0072
Change:     from sklearn.ensemble import ExtraTreesClassifier      return ExtraTreesClas
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..e3da10e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -42,71 +42,16 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    import xgboost as xgb
-    from scipy.special import expit
-
-    class _AdaptiveRanking:
-        def __init__(self):
-            self.classes_ = np.array([0, 1])
-
-        @staticmethod
-        def _group_sizes(qid):
-            sizes, cur, cnt = [], qid[0], 1
-            for i in range(1, len(qid)):
-                if qid[i] == cur:
-                    cnt += 1
-                else:
-                    sizes.append(cnt)
-                    cur, cnt = qid[i], 1
-            sizes.append(cnt)
-            return sizes
-
-        def fit(self, X, y):
-            gs = 1000
-            n = len(y)
-            split = int(n * 0.9)
-
-            qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
-            qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
-
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
-            dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
-            deval.set_group(self._group_sizes(qid_ev))
-
-            params = {
-                "objective": "rank:ndcg",
-                "eval_metric": "ndcg",
-                "tree_method": "hist",
-                "max_depth": 4,
-                "learning_rate": 0.005,
-                "min_child_weight": 5,
-                "subsample": 0.8,
-                "colsample_bytree": 0.6,
-                "reg_alpha": 0.5,
-                "reg_lambda": 5.0,
-                "gamma": 0.5,
-            }
-
-            self._model = xgb.train(
-                params, dtrain,
-                num_boost_round=5000,
-                evals=[(deval, "eval")],
-                early_stopping_rounds=100,
-                verbose_eval=200,
```

### Iteration 27 -- REVERTED (+0.0000)
Score: 0.0072 vs best 0.0072
Change:             base_params = {             scout = xgb.train(                 {**ba
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..44cdb28 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -74,12 +74,11 @@ def build_model(y_train):
             deval = xgb.DMatrix(X[split:], label=y[split:])
             deval.set_group(self._group_sizes(qid_ev))
 
-            params = {
+            base_params = {
                 "objective": "rank:ndcg",
                 "eval_metric": "ndcg",
                 "tree_method": "hist",
                 "max_depth": 4,
-                "learning_rate": 0.005,
                 "min_child_weight": 5,
                 "subsample": 0.8,
                 "colsample_bytree": 0.6,
@@ -88,10 +87,37 @@ def build_model(y_train):
                 "gamma": 0.5,
             }
 
+            scout = xgb.train(
+                {**base_params, "learning_rate": 0.02},
+                dtrain,
+                num_boost_round=500,
+                evals=[(deval, "eval")],
+                early_stopping_rounds=30,
+                verbose_eval=False,
+            )
+
+            importance = scout.get_score(importance_type="gain")
+            n_feat = X.shape[1]
+            gains = np.zeros(n_feat)
+            for k, v in importance.items():
+                gains[int(k[1:])] = v
+
+            max_gain = gains.max()
+            self._feat_idx = np.where(gains >= max_gain * 0.1)[0]
+            if len(self._feat_idx) < 20:
+                self._feat_idx = np.arange(n_feat)
+            print(f"Feature selection: {n_feat} -> {len(self._feat_idx)}")
+
+            dtrain2 = xgb.DMatrix(X[:split][:, self._feat_idx], label=y[:split])
+            dtrain2.set_group(self._group_sizes(qid_tr))
+            deval2 = xgb.DMatrix(X[split:][:, self._feat_idx], label=y[split:])
+            deval2.set_group(self._group_sizes(qid_ev))
+
             self._model = xgb.train(
-                params, dtrain,
+                {**base_params, "learning_rate": 0.005},
+                dtrain2,
                 num_boost_round=5000,
-                evals=[(deval, "eval")],
+                evals=[(deval2, "eval")],
                 early_stopping_rounds=100,
                 verbose_eval=200,
             )
@@ -99,7 +125,7 @@ def build_model(y_train):
             return self
```

### Iteration 28 -- REVERTED (-2.4795)
Score: -2.4723 vs best 0.0072
Change:             gs = 5000 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..7601ad6 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -62,7 +62,7 @@ def build_model(y_train):
             return sizes
 
         def fit(self, X, y):
-            gs = 1000
+            gs = 5000
             n = len(y)
             split = int(n * 0.9)
 
```

### Iteration 29 -- REVERTED (-2.1638)
Score: -2.1566 vs best 0.0072
Change: FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction", "rolli
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..5ed3a7d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -33,7 +33,7 @@ DATASET_PATH = "data/datasets/20260331/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options. None = all
-FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
+FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction", "rolling"]
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 30 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             from scipy.stats import rankdata              # Extract grading sign
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..96aa7a1 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -62,16 +62,29 @@ def build_model(y_train):
             return sizes
 
         def fit(self, X, y):
+            from scipy.stats import rankdata
+
+            # Extract grading signal (last col = _pivot_grade) and strip from features
+            grade = X[:, -1].copy()
+            X = X[:, :-1]
+
+            # Graded relevance: pivots scored [1.0, 1.5] by drawdown depth rank
+            y_graded = y.copy().astype(float)
+            pos_mask = y == 1
+            if pos_mask.sum() > 1:
+                pos_ranks = rankdata(grade[pos_mask]) / pos_mask.sum()
+                y_graded[pos_mask] = 1.0 + pos_ranks * 0.5
+
             gs = 1000
-            n = len(y)
+            n = len(y_graded)
             split = int(n * 0.9)
 
             qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
             qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
+            dtrain = xgb.DMatrix(X[:split], label=y_graded[:split])
             dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
+            deval = xgb.DMatrix(X[split:], label=y_graded[split:])
             deval.set_group(self._group_sizes(qid_ev))
 
             params = {
@@ -99,6 +112,7 @@ def build_model(y_train):
             return self
 
         def predict_proba(self, X):
+            X = X[:, :-1]  # Strip _pivot_grade column
             dtest = xgb.DMatrix(X)
             scores = self._model.predict(
                 dtest, iteration_range=(0, self._model.best_iteration + 1)
diff --git a/research/features_lab.py b/research/features_lab.py
index 2eff964..1aaa086 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -44,6 +44,15 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     df["return_accel_10"] = g["close"].transform(_accel)
     df["returns_skew_20"] = g["close"].transform(_skew)
 
+    # Drawdown depth from rolling 20-day max — used for graded relevance labels.
+    # Deeper drawdown at pivot = higher quality bottom. Backward-looking only.
+    def _drawdown_depth(close):
+        rolling_max = close.rolling(20, min_periods=10).max()
+        return (rolling_max - close) / (rolling_max + 1e-10)
+
+    df["_pivot_grade"] = g["close"].transform(_drawdown_depth)
```
Traceback:
```
  [bt] (4) 5   libxgboost.dylib                    0x0000000121b11ccc xgboost::obj::FitIntercept::InitEstimation(xgboost::MetaInfo const&, xgboost::linalg::Tensor<float, 1>*) const + 440
  [bt] (5) 6   libxgboost.dylib                    0x0000000121aa56f0 xgboost::LearnerImpl::UpdateOneIter(int, std::__1::shared_ptr<xgboost::DMatrix>) + 644
  [bt] (6) 7   libxgboost.dylib                    0x00000001218594f8 XGBoosterUpdateOneIter + 144
  [bt] (7) 8   libffi.dylib                        0x0000000195646050 ffi_call_SYSV + 80
  [bt] (8) 9   libffi.dylib                        0x000000019564f5b8 ffi_call_int + 1220
```

### Iteration 31 -- REVERTED (-1.7807)
Score: -1.7735 vs best 0.0072
Change:         @staticmethod         def _grade_labels(X, y):             from sklearn.
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..a488d64 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -61,17 +61,36 @@ def build_model(y_train):
             sizes.append(cnt)
             return sizes
 
+        @staticmethod
+        def _grade_labels(X, y):
+            from sklearn.linear_model import LogisticRegression
+            graded = y.copy().astype(np.int32)
+            pos_mask = y == 1
+            n_pos = int(pos_mask.sum())
+            if n_pos < 20:
+                return graded
+            lr = LogisticRegression(max_iter=300, C=0.1, solver='lbfgs')
+            lr.fit(X, y)
+            probs = lr.predict_proba(X)[:, 1]
+            threshold = np.median(probs[pos_mask])
+            upgrade_mask = pos_mask & (probs >= threshold)
+            graded[upgrade_mask] = 2
+            print(f"Graded labels: {n_pos} pos -> {int(upgrade_mask.sum())} grade-2, {n_pos - int(upgrade_mask.sum())} grade-1")
+            return graded
+
         def fit(self, X, y):
             gs = 1000
             n = len(y)
             split = int(n * 0.9)
 
+            graded = self._grade_labels(X, y)
+
             qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
             qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
+            dtrain = xgb.DMatrix(X[:split], label=graded[:split])
             dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
+            deval = xgb.DMatrix(X[split:], label=graded[split:])
             deval.set_group(self._group_sizes(qid_ev))
 
             params = {
```

### Iteration 32 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             w_tr = np.where(y[:split] == 1, 3.0, 1.0)             w_ev = np.wher
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 3fa0f52..681aeb0 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -69,9 +69,12 @@ def build_model(y_train):
             qid_tr = np.repeat(np.arange(split // gs + 1), gs)[:split]
             qid_ev = np.repeat(np.arange((n - split) // gs + 1), gs)[:(n - split)]
 
-            dtrain = xgb.DMatrix(X[:split], label=y[:split])
+            w_tr = np.where(y[:split] == 1, 3.0, 1.0)
+            w_ev = np.where(y[split:] == 1, 3.0, 1.0)
+
+            dtrain = xgb.DMatrix(X[:split], label=y[:split], weight=w_tr)
             dtrain.set_group(self._group_sizes(qid_tr))
-            deval = xgb.DMatrix(X[split:], label=y[split:])
+            deval = xgb.DMatrix(X[split:], label=y[split:], weight=w_ev)
             deval.set_group(self._group_sizes(qid_ev))
 
             params = {
```
Traceback:
```
  [bt] (4) 5   libffi.dylib                        0x0000000195646050 ffi_call_SYSV + 80
  [bt] (5) 6   libffi.dylib                        0x000000019564f5b8 ffi_call_int + 1220
  [bt] (6) 7   _ctypes.cpython-314-darwin.so       0x000000010581c828 _ctypes_callproc + 1304
  [bt] (7) 8   _ctypes.cpython-314-darwin.so       0x000000010581925c PyCFuncPtr_call + 800
  [bt] (8) 9   libpython3.14.dylib                 0x00000001058cdd58 _PyObject_MakeTpCall + 304
```

### Iteration 34 -- REVERTED (-1.3479)
Score: -1.2720 vs best 0.0759
Change:                 "max_depth": 6, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index e27207f..6edec16 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -78,7 +78,7 @@ def build_model(y_train):
                 "objective": "rank:ndcg",
                 "eval_metric": "ndcg",
                 "tree_method": "hist",
-                "max_depth": 5,
+                "max_depth": 6,
                 "learning_rate": 0.005,
                 "min_child_weight": 5,
                 "subsample": 0.8,
```

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
