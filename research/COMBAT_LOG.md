# Combat Log

Reverted experiments with scores and diffs.

### Iteration 4 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 142).
Change: class _EnsembleCB:     """Bayesian model averaging: multiple CatBoost posterior 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..81b57fd 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -60,14 +60,41 @@ class _EarlyStopCB:
         return self._model.predict_proba(X)
 
 
+class _EnsembleCB:
+    """Bayesian model averaging: multiple CatBoost posterior samples, averaged."""
+
+    def __init__(self, val_frac, seeds, **kwargs):
+        self._val_frac = val_frac
+        self._seeds = seeds
+        self._kwargs = kwargs
+
+    def fit(self, X, y):
+        from catboost import CatBoostClassifier, Pool
+        n = len(X)
+        cut = int(n * (1 - self._val_frac))
+        tp = Pool(X[:cut], y[:cut])
+        ep = Pool(X[cut:], y[cut:])
+        self._models = []
+        for seed in self._seeds:
+            m = CatBoostClassifier(**{**self._kwargs, 'random_seed': seed})
+            m.fit(tp, eval_set=ep)
+            self._models.append(m)
+        self.classes_ = self._models[0].classes_
+        return self
+
+    def predict_proba(self, X):
+        return np.mean([m.predict_proba(X) for m in self._models], axis=0)
+
+
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
     neg = (y_train == 0).sum()
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    return _EarlyStopCB(
+    return _EnsembleCB(
         val_frac=0.1,
+        seeds=[42, 137],
         iterations=3000,
         depth=7,
         learning_rate=0.01,
@@ -77,7 +104,6 @@ def build_model(y_train):
         l2_leaf_reg=3.0,
         posterior_sampling=True,
         scale_pos_weight=spw,
-        random_seed=42,
         verbose=0,
         thread_count=-1,
         od_type="Iter",
```
Traceback:
```
Immutability check: PASSED
Running experiment (timeout: 1800s)...
GATE VIOLATION: Experiment crashed (exit code 142).
--- Last 30 lines of log ---
```

### Iteration 5 -- REVERTED (-0.0546)
Score: 0.6984 vs best 0.7530
Change: class _EarlyStopXGB:     """XGBoost with internal early stopping on a temporal h
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..cc32e34 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -40,19 +40,23 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 # ===========================================================================
 
 
-class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
+class _EarlyStopXGB:
+    """XGBoost with internal early stopping on a temporal holdout."""
 
     def __init__(self, val_frac, **kwargs):
         self._val_frac = val_frac
         self._kwargs = kwargs
 
     def fit(self, X, y):
-        from catboost import CatBoostClassifier, Pool
+        from xgboost import XGBClassifier
         n = len(X)
         cut = int(n * (1 - self._val_frac))
-        self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
+        self._model = XGBClassifier(**self._kwargs)
+        self._model.fit(
+            X[:cut], y[:cut],
+            eval_set=[(X[cut:], y[cut:])],
+            verbose=False,
+        )
         self.classes_ = self._model.classes_
         return self
 
@@ -66,23 +70,23 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    return _EarlyStopCB(
+    return _EarlyStopXGB(
         val_frac=0.1,
-        iterations=3000,
-        depth=7,
+        n_estimators=3000,
+        max_depth=7,
         learning_rate=0.01,
-        min_data_in_leaf=50,
-        boosting_type="Ordered",
-        rsm=0.6,
-        l2_leaf_reg=3.0,
-        posterior_sampling=True,
+        min_child_weight=50,
+        tree_method="hist",
+        subsample=0.8,
+        colsample_bytree=0.6,
+        reg_lambda=3.0,
+        gamma=0.1,
         scale_pos_weight=spw,
-        random_seed=42,
-        verbose=0,
-        thread_count=-1,
```

### Iteration 6 -- REVERTED (-1.9465)
Score: -1.1935 vs best 0.7530
Change: class _FocalMLP:     """Focal-loss MLP with BatchNorm and temporal early stoppin
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..fe25002 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -40,49 +40,115 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 # ===========================================================================
 
 
-class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
-
-    def __init__(self, val_frac, **kwargs):
-        self._val_frac = val_frac
-        self._kwargs = kwargs
+class _FocalMLP:
+    """Focal-loss MLP with BatchNorm and temporal early stopping."""
+
+    def __init__(self, hidden=(256, 128, 64), dropout=0.3, alpha=0.8,
+                 gamma=2.0, lr=1e-3, weight_decay=1e-4, batch_size=4096,
+                 epochs=100, patience=10, val_frac=0.1):
+        self._hidden = hidden
+        self._dropout = dropout
+        self._alpha = alpha
+        self._gamma = gamma
+        self._lr = lr
+        self._wd = weight_decay
+        self._bs = batch_size
+        self._epochs = epochs
+        self._patience = patience
+        self._vf = val_frac
+
+    def _make_net(self, d_in):
+        from torch import nn
+        layers = []
+        prev = d_in
+        for h in self._hidden:
+            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(self._dropout)]
+            prev = h
+        layers.append(nn.Linear(prev, 1))
+        return nn.Sequential(*layers)
+
+    def _focal(self, logits, targets):
+        import torch
+        import torch.nn.functional as F
+        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
+        p = torch.sigmoid(logits)
+        pt = targets * p + (1 - targets) * (1 - p)
+        at = targets * self._alpha + (1 - targets) * (1 - self._alpha)
+        return (at * (1 - pt) ** self._gamma * bce).mean()
 
     def fit(self, X, y):
-        from catboost import CatBoostClassifier, Pool
-        n = len(X)
-        cut = int(n * (1 - self._val_frac))
-        self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
-        self.classes_ = self._model.classes_
+        import torch
+        from torch.utils.data import DataLoader, TensorDataset
+
```

### Iteration 7 -- REVERTED (-1.4130)
Score: -0.6600 vs best 0.7530
Change: from research.model_wrappers import RankingXGBClassifier  # noqa: E402     retur
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..45699ce 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -18,6 +18,7 @@ import numpy as np  # noqa: F401,E402 -- available for researcher
 from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
 from lib.eval import tiered_eval  # noqa: E402
 from research.features_lab import add_custom_features  # noqa: E402
+from research.model_wrappers import RankingXGBClassifier  # noqa: E402
 
 # ===========================================================================
 # DATA -- fixed, do not edit
@@ -62,27 +63,16 @@ class _EarlyStopCB:
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    neg = (y_train == 0).sum()
-    pos = (y_train == 1).sum()
-    spw = np.sqrt(neg / pos)
-
-    return _EarlyStopCB(
-        val_frac=0.1,
-        iterations=3000,
-        depth=7,
+    return RankingXGBClassifier(
+        objective="rank:ndcg",
+        group_size=200,
+        n_estimators=1500,
+        max_depth=7,
         learning_rate=0.01,
-        min_data_in_leaf=50,
-        boosting_type="Ordered",
-        rsm=0.6,
-        l2_leaf_reg=3.0,
-        posterior_sampling=True,
-        scale_pos_weight=spw,
-        random_seed=42,
-        verbose=0,
-        thread_count=-1,
-        od_type="Iter",
-        od_wait=200,
-        use_best_model=True,
+        min_child_weight=50,
+        subsample=0.8,
+        colsample_bytree=0.6,
+        reg_lambda=3.0,
     )
 
 
```

### Iteration 8 -- REVERTED (-0.2378)
Score: 0.5152 vs best 0.7530
Change:         # Temporal decay: down-weight old data, up-weight recent data.         #
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..520a7a7 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -51,8 +51,14 @@ class _EarlyStopCB:
         from catboost import CatBoostClassifier, Pool
         n = len(X)
         cut = int(n * (1 - self._val_frac))
+        # Temporal decay: down-weight old data, up-weight recent data.
+        # Market regimes shift; 2021 patterns are less predictive for 2024-2026.
+        train_weights = np.linspace(0.3, 1.0, cut)
         self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
+        self._model.fit(
+            Pool(X[:cut], y[:cut], weight=train_weights),
+            eval_set=Pool(X[cut:], y[cut:]),
+        )
         self.classes_ = self._model.classes_
         return self
 
```

### Iteration 9 -- REVERTED (-0.0553)
Score: 0.6977 vs best 0.7530
Change:     """CatBoost with feature importance pruning and early stopping.     Two-stag
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..7d0d35c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -41,23 +41,46 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 
 class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
+    """CatBoost with feature importance pruning and early stopping.
 
-    def __init__(self, val_frac, **kwargs):
+    Two-stage training: a quick scout model identifies the most important
+    features, then the full model trains on only those features.
+    """
+
+    def __init__(self, val_frac, top_k_pct=0.5, **kwargs):
         self._val_frac = val_frac
+        self._top_k_pct = top_k_pct
         self._kwargs = kwargs
+        self._feature_mask = None
 
     def fit(self, X, y):
         from catboost import CatBoostClassifier, Pool
         n = len(X)
         cut = int(n * (1 - self._val_frac))
+        tr_pool = Pool(X[:cut], y[:cut])
+        val_pool = Pool(X[cut:], y[cut:])
+
+        # Stage 1: quick scout to rank features by importance
+        scout_params = {**self._kwargs, "iterations": 500, "od_wait": 50}
+        scout = CatBoostClassifier(**scout_params)
+        scout.fit(tr_pool, eval_set=val_pool)
+
+        importances = scout.get_feature_importance()
+        k = max(10, int(X.shape[1] * self._top_k_pct))
+        self._feature_mask = np.sort(np.argsort(importances)[-k:])
+        print(f"  Feature pruning: {X.shape[1]} -> {k} features (top {self._top_k_pct:.0%})")
+
+        # Stage 2: full train on selected features only
         self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
+        self._model.fit(
+            Pool(X[:cut][:, self._feature_mask], y[:cut]),
+            eval_set=Pool(X[cut:][:, self._feature_mask], y[cut:]),
+        )
         self.classes_ = self._model.classes_
         return self
 
     def predict_proba(self, X):
-        return self._model.predict_proba(X)
+        return self._model.predict_proba(X[:, self._feature_mask])
 
 
 def build_model(y_train):
@@ -68,6 +91,7 @@ def build_model(y_train):
 
     return _EarlyStopCB(
         val_frac=0.1,
+        top_k_pct=0.5,
```

### Iteration 10 -- REVERTED (-0.2207)
Score: 0.5323 vs best 0.7530
Change:         boosting_type="Plain",         bootstrap_type="MVS",         subsample=0
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..4925b1a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -72,10 +72,11 @@ def build_model(y_train):
         depth=7,
         learning_rate=0.01,
         min_data_in_leaf=50,
-        boosting_type="Ordered",
+        boosting_type="Plain",
+        bootstrap_type="MVS",
+        subsample=0.8,
         rsm=0.6,
         l2_leaf_reg=3.0,
-        posterior_sampling=True,
         scale_pos_weight=spw,
         random_seed=42,
         verbose=0,
```

### Iteration 1 -- REVERTED (-0.3426)
Score: 0.4104 vs best 0.7530
Change: class _DownsampleCB:     """CatBoost with negative downsampling and temporal hol
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..b72fa54 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -40,19 +40,40 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 # ===========================================================================
 
 
-class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
+class _DownsampleCB:
+    """CatBoost with negative downsampling and temporal holdout early stopping."""
 
-    def __init__(self, val_frac, **kwargs):
+    def __init__(self, neg_ratio, val_frac, **kwargs):
+        self._neg_ratio = neg_ratio
         self._val_frac = val_frac
         self._kwargs = kwargs
 
     def fit(self, X, y):
         from catboost import CatBoostClassifier, Pool
+
         n = len(X)
         cut = int(n * (1 - self._val_frac))
-        self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
+        X_val, y_val = X[cut:], y[cut:]
+        X_tr, y_tr = X[:cut], y[:cut]
+
+        pos_idx = np.where(y_tr == 1)[0]
+        neg_idx = np.where(y_tr == 0)[0]
+        n_neg_keep = int(len(pos_idx) * self._neg_ratio)
+
+        rng = np.random.RandomState(42)
+        if n_neg_keep < len(neg_idx):
+            neg_sub = rng.choice(neg_idx, n_neg_keep, replace=False)
+            idx = np.sort(np.concatenate([pos_idx, neg_sub]))
+        else:
+            idx = np.arange(len(y_tr))
+
+        X_ds, y_ds = X_tr[idx], y_tr[idx]
+        spw = np.sqrt((y_ds == 0).sum() / (y_ds == 1).sum())
+
+        kwargs = dict(self._kwargs)
+        kwargs["scale_pos_weight"] = spw
+        self._model = CatBoostClassifier(**kwargs)
+        self._model.fit(Pool(X_ds, y_ds), eval_set=Pool(X_val, y_val))
         self.classes_ = self._model.classes_
         return self
 
@@ -62,11 +83,8 @@ class _EarlyStopCB:
 
 def build_model(y_train):
     """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
-    neg = (y_train == 0).sum()
-    pos = (y_train == 1).sum()
-    spw = np.sqrt(neg / pos)
-
-    return _EarlyStopCB(
+    return _DownsampleCB(
```

### Iteration 2 -- REVERTED (-0.3852)
Score: 0.3678 vs best 0.7530
Change:     """CatBoost with internal early stopping on a random holdout."""         n_v
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..94cb15c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -41,7 +41,7 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 
 
 class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
+    """CatBoost with internal early stopping on a random holdout."""
 
     def __init__(self, val_frac, **kwargs):
         self._val_frac = val_frac
@@ -50,9 +50,13 @@ class _EarlyStopCB:
     def fit(self, X, y):
         from catboost import CatBoostClassifier, Pool
         n = len(X)
-        cut = int(n * (1 - self._val_frac))
+        n_val = int(n * self._val_frac)
+        rng = np.random.RandomState(42)
+        val_idx = rng.choice(n, size=n_val, replace=False)
+        train_mask = np.ones(n, dtype=bool)
+        train_mask[val_idx] = False
         self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
+        self._model.fit(Pool(X[train_mask], y[train_mask]), eval_set=Pool(X[val_idx], y[val_idx]))
         self.classes_ = self._model.classes_
         return self
 
```

### Iteration 3 -- REVERTED (-0.0320)
Score: 0.7210 vs best 0.7530
Change:     g = df.groupby("stock_id")     ret_5d = g["close"].pct_change(5)     vol_cha
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 4010461..e9b54ba 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -52,10 +52,18 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     prev_high = df.groupby("stock_id")["high"].shift(1)
     df["close_vs_prev_high"] = (df["close"] - prev_high) / prev_high.clip(lower=1e-10)
 
+    g = df.groupby("stock_id")
+    ret_5d = g["close"].pct_change(5)
+    vol_change_5d = g["volume"].pct_change(5)
+    df["vol_price_agreement_5d"] = ret_5d * vol_change_5d
+
+    df["knife_risk"] = (-ret_5d).clip(lower=0) * (1 - df["close_position"])
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
+        "vol_price_agreement_5d", "knife_risk",
     ]
 
     # --- END researcher section ---
```

### Iteration 4 -- REVERTED (-0.1186)
Score: 0.6344 vs best 0.7530
Change:     cp_lag5 = df.groupby("stock_id")["close_position"].shift(5)     df["close_po
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 4010461..c4ef38f 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -52,10 +52,14 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     prev_high = df.groupby("stock_id")["high"].shift(1)
     df["close_vs_prev_high"] = (df["close"] - prev_high) / prev_high.clip(lower=1e-10)
 
+    cp_lag5 = df.groupby("stock_id")["close_position"].shift(5)
+    df["close_position_change_5d"] = df["close_position"] - cp_lag5
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
+        "close_position_change_5d",
     ]
 
     # --- END researcher section ---
```

### Iteration 5 -- REVERTED (-0.0107)
Score: 0.7423 vs best 0.7530
Change:     # Indicator-trajectory features: rate of change for key bottom indicators   
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 4010461..d7a0e16 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -52,10 +52,20 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     prev_high = df.groupby("stock_id")["high"].shift(1)
     df["close_vs_prev_high"] = (df["close"] - prev_high) / prev_high.clip(lower=1e-10)
 
+    # Indicator-trajectory features: rate of change for key bottom indicators
+    # not covered by the roc group (which has rsi/volume/atr/macd changes only).
+    # These capture the DYNAMICS leading into a pivot low -- drawdown decelerating,
+    # stochastic recovering, BB position climbing from extreme lows.
+    g = df.groupby("stock_id")
+    df["drawdown_velocity_5d"] = g["drawdown"].diff(5)
+    df["stoch_recovery_5d"] = g["stoch_k"].diff(5)
+    df["bb_position_trend_5d"] = g["bb_position"].diff(5)
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
+        "drawdown_velocity_5d", "stoch_recovery_5d", "bb_position_trend_5d",
     ]
 
     # --- END researcher section ---
```

### Iteration 6 -- REVERTED (-0.6194)
Score: 0.1336 vs best 0.7530
Change: class _EarlyStopLGBM:     """LightGBM with internal early stopping on a temporal
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..0d3762b 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -40,24 +40,32 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 # ===========================================================================
 
 
-class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
+class _EarlyStopLGBM:
+    """LightGBM with internal early stopping on a temporal holdout."""
 
-    def __init__(self, val_frac, **kwargs):
+    def __init__(self, val_frac, **params):
         self._val_frac = val_frac
-        self._kwargs = kwargs
+        self._params = params
 
     def fit(self, X, y):
-        from catboost import CatBoostClassifier, Pool
+        import lightgbm as lgb
         n = len(X)
         cut = int(n * (1 - self._val_frac))
-        self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
-        self.classes_ = self._model.classes_
+        train_data = lgb.Dataset(X[:cut], y[:cut])
+        val_data = lgb.Dataset(X[cut:], y[cut:], reference=train_data)
+        self._model = lgb.train(
+            self._params,
+            train_data,
+            num_boost_round=3000,
+            valid_sets=[val_data],
+            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
+        )
+        self.classes_ = np.array([0, 1])
         return self
 
     def predict_proba(self, X):
-        return self._model.predict_proba(X)
+        preds = self._model.predict(X)
+        return np.column_stack([1 - preds, preds])
 
 
 def build_model(y_train):
@@ -66,23 +74,19 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    return _EarlyStopCB(
+    return _EarlyStopLGBM(
         val_frac=0.1,
-        iterations=3000,
-        depth=7,
+        objective="binary",
+        metric="binary_logloss",
+        num_leaves=127,
         learning_rate=0.01,
         min_data_in_leaf=50,
```

### Iteration 7 -- REVERTED (-0.0574)
Score: 0.6956 vs best 0.7530
Change:     # --- Multi-day sequential features for reversal quality ---      # Short-te
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 4010461..a2c4e27 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -52,10 +52,31 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     prev_high = df.groupby("stock_id")["high"].shift(1)
     df["close_vs_prev_high"] = (df["close"] - prev_high) / prev_high.clip(lower=1e-10)
 
+    # --- Multi-day sequential features for reversal quality ---
+
+    # Short-term drawdown from 20-day high (complements existing 252d drawdown)
+    rolling_high_20d = df.groupby("stock_id")["close"].transform(
+        lambda s: s.rolling(20, min_periods=5).max()
+    ).clip(lower=1e-10)
+    df["drawdown_20d"] = (df["close"] - rolling_high_20d) / rolling_high_20d
+
+    # Return momentum shift: 5d return minus 10d return
+    # Positive = decline decelerating (reversal signal), negative = accelerating (knife)
+    g = df.groupby("stock_id")["close"]
+    ret_5d = g.transform(lambda s: s.pct_change(5))
+    ret_10d = g.transform(lambda s: s.pct_change(10))
+    df["return_momentum_shift"] = ret_5d - ret_10d
+
+    # Volume skewness over 10 days: positive skew = capitulation spikes present
+    df["volume_skew_10d"] = df.groupby("stock_id")["volume"].transform(
+        lambda s: s.rolling(10, min_periods=5).skew()
+    )
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
+        "drawdown_20d", "return_momentum_shift", "volume_skew_10d",
     ]
 
     # --- END researcher section ---
```

### Iteration 8 -- REVERTED (-0.1922)
Score: 0.5608 vs best 0.7530
Change:         eval_metric="PRAUC", 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..cd3b293 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -77,6 +77,7 @@ def build_model(y_train):
         l2_leaf_reg=3.0,
         posterior_sampling=True,
         scale_pos_weight=spw,
+        eval_metric="PRAUC",
         random_seed=42,
         verbose=0,
         thread_count=-1,
```

### Iteration 9 -- REVERTED (-0.2993)
Score: 0.4537 vs best 0.7530
Change:     # Pivot-point dynamics: 2nd-order trend features capturing turning-point sig
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 4010461..4b2f119 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -52,10 +52,20 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     prev_high = df.groupby("stock_id")["high"].shift(1)
     df["close_vs_prev_high"] = (df["close"] - prev_high) / prev_high.clip(lower=1e-10)
 
+    # Pivot-point dynamics: 2nd-order trend features capturing turning-point signals
+    ret_1d = df.groupby("stock_id")["close"].pct_change(1)
+    df["return_accel"] = ret_1d.groupby(df["stock_id"]).diff(1)
+
+    df["close_position_delta_3d"] = df.groupby("stock_id")["close_position"].diff(3)
+    df["wick_ratio_delta_3d"] = df.groupby("stock_id")["lower_wick_ratio"].diff(3)
+    df["vol_ratio_delta_3d"] = df.groupby("stock_id")["volume_sma_ratio"].diff(3)
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
+        "return_accel", "close_position_delta_3d",
+        "wick_ratio_delta_3d", "vol_ratio_delta_3d",
     ]
 
     # --- END researcher section ---
```

### Iteration 12 -- REVERTED (-0.0202)
Score: 0.7657 vs best 0.7859
Change:     cp_5d = g["close_position"].transform(         lambda s: s.rolling(5, min_pe
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 3176c3f..0e2af74 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -60,11 +60,23 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
             df[name] = g[col].shift(lag)
             lag_cols.append(name)
 
+    cp_5d = g["close_position"].transform(
+        lambda s: s.rolling(5, min_periods=2).mean()
+    )
+    df["close_position_5d_mean"] = cp_5d
+    df["close_position_vs_5d"] = df["close_position"] - cp_5d
+
+    df["lower_wick_5d_mean"] = g["lower_wick_ratio"].transform(
+        lambda s: s.rolling(5, min_periods=2).mean()
+    )
+
+    rolling_feats = ["close_position_5d_mean", "close_position_vs_5d", "lower_wick_5d_mean"]
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
-    ] + lag_cols
+    ] + lag_cols + rolling_feats
 
     # --- END researcher section ---
 
```

### Iteration 13 -- REVERTED (-0.2192)
Score: 0.5667 vs best 0.7859
Change:     for col in ["rsi_14", "drawdown", "close_position", "lower_wick_ratio",     
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 3176c3f..ec1afeb 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -54,7 +54,8 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     g = df.groupby("stock_id")
     lag_cols = []
-    for col in ["rsi_14", "drawdown", "close_position", "lower_wick_ratio"]:
+    for col in ["rsi_14", "drawdown", "close_position", "lower_wick_ratio",
+                 "volume_sma_ratio", "signed_candle"]:
         for lag in [1, 2, 3]:
             name = f"{col}_lag{lag}"
             df[name] = g[col].shift(lag)
```

### Iteration 14 -- REVERTED (-0.0567)
Score: 0.7292 vs best 0.7859
Change:         depth=6, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..5042f2f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -69,7 +69,7 @@ def build_model(y_train):
     return _EarlyStopCB(
         val_frac=0.1,
         iterations=3000,
-        depth=7,
+        depth=6,
         learning_rate=0.01,
         min_data_in_leaf=50,
         boosting_type="Ordered",
```

### Iteration 15 -- REVERTED (-0.1021)
Score: 0.6838 vs best 0.7859
Change: def _days_since_rolling_high(s, window):     rolling_max = s.rolling(window, min
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 3176c3f..b0ab8e9 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -12,6 +12,13 @@ Rules:
 import pandas as pd
 
 
+def _days_since_rolling_high(s, window):
+    rolling_max = s.rolling(window, min_periods=1).max()
+    is_at_max = s >= rolling_max
+    groups = is_at_max.cumsum()
+    return groups.groupby(groups).cumcount()
+
+
 def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     """Add custom features. Returns (df, new_feature_names).
 
@@ -60,11 +67,17 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
             df[name] = g[col].shift(lag)
             lag_cols.append(name)
 
+    decline_cols = []
+    for window in [20, 60]:
+        name = f"days_since_high_{window}"
+        df[name] = g["high"].transform(lambda s, w=window: _days_since_rolling_high(s, w))
+        decline_cols.append(name)
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
-    ] + lag_cols
+    ] + lag_cols + decline_cols
 
     # --- END researcher section ---
 
```

### Iteration 16 -- REVERTED (-0.5929)
Score: 0.1930 vs best 0.7859
Change: class _EarlyStopLGBM:     """LightGBM with early stopping on a temporal holdout.
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..303f915 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -40,24 +40,33 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 # ===========================================================================
 
 
-class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
+class _EarlyStopLGBM:
+    """LightGBM with early stopping on a temporal holdout."""
 
-    def __init__(self, val_frac, **kwargs):
+    def __init__(self, val_frac, **params):
         self._val_frac = val_frac
-        self._kwargs = kwargs
+        self._params = params
 
     def fit(self, X, y):
-        from catboost import CatBoostClassifier, Pool
+        import lightgbm as lgb
         n = len(X)
         cut = int(n * (1 - self._val_frac))
-        self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
-        self.classes_ = self._model.classes_
+        train_ds = lgb.Dataset(X[:cut], y[:cut], free_raw_data=False)
+        val_ds = lgb.Dataset(X[cut:], y[cut:], reference=train_ds, free_raw_data=False)
+        self._booster = lgb.train(
+            self._params,
+            train_ds,
+            num_boost_round=3000,
+            valid_sets=[val_ds],
+            valid_names=["val"],
+            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
+        )
+        self.classes_ = np.array([0, 1])
         return self
 
     def predict_proba(self, X):
-        return self._model.predict_proba(X)
+        p1 = self._booster.predict(X)
+        return np.column_stack([1 - p1, p1])
 
 
 def build_model(y_train):
@@ -66,23 +75,23 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    return _EarlyStopCB(
+    return _EarlyStopLGBM(
         val_frac=0.1,
-        iterations=3000,
-        depth=7,
+        objective="binary",
+        boosting_type="gbdt",
+        max_depth=-1,
+        num_leaves=63,
```

### Iteration 17 -- REVERTED (-0.1115)
Score: 0.6744 vs best 0.7859
Change:     df["rsi_min_10d"] = g["rsi_14"].transform(lambda s: s.rolling(10, min_period
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 3176c3f..9812a56 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -60,11 +60,18 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
             df[name] = g[col].shift(lag)
             lag_cols.append(name)
 
+    df["rsi_min_10d"] = g["rsi_14"].transform(lambda s: s.rolling(10, min_periods=3).min())
+    df["drawdown_min_10d"] = g["drawdown"].transform(lambda s: s.rolling(10, min_periods=3).min())
+    df["volume_z_max_10d"] = g["volume_z"].transform(lambda s: s.rolling(10, min_periods=3).max())
+    df["ret_min_5d"] = g["ret_1d"].transform(lambda s: s.rolling(5, min_periods=2).min())
+
+    extreme_cols = ["rsi_min_10d", "drawdown_min_10d", "volume_z_max_10d", "ret_min_5d"]
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
-    ] + lag_cols
+    ] + lag_cols + extreme_cols
 
     # --- END researcher section ---
 
```

### Iteration 18 -- REVERTED (-0.4033)
Score: 0.3826 vs best 0.7859
Change: FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction", "rolli
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..3119bd8 100644
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

### Iteration 19 -- REVERTED (-0.1682)
Score: 0.6177 vs best 0.7859
Change:     df["_abs_daily_ret"] = g["close"].diff().abs()     sum_abs_ret = df.groupby(
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 3176c3f..9694924 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -60,10 +60,18 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
             df[name] = g[col].shift(lag)
             lag_cols.append(name)
 
+    df["_abs_daily_ret"] = g["close"].diff().abs()
+    sum_abs_ret = df.groupby("stock_id")["_abs_daily_ret"].transform(
+        lambda s: s.rolling(10, min_periods=5).sum()
+    ).clip(lower=1e-10)
+    df["trend_efficiency_10d"] = g["close"].diff(10) / sum_abs_ret
+    df.drop(columns=["_abs_daily_ret"], inplace=True)
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
+        "trend_efficiency_10d",
     ] + lag_cols
 
     # --- END researcher section ---
```

### Iteration 20 -- REVERTED (-0.2891)
Score: 0.4968 vs best 0.7859
Change:     xrank_cols = []     for col in ["rsi_14", "drawdown", "close_position"]:    
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 3176c3f..da80857 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -60,11 +60,21 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
             df[name] = g[col].shift(lag)
             lag_cols.append(name)
 
+    xrank_cols = []
+    for col in ["rsi_14", "drawdown", "close_position"]:
+        rank_name = f"{col}_xrank"
+        df[rank_name] = df.groupby("date")[col].rank(pct=True)
+        xrank_cols.append(rank_name)
+
+    df["market_stress"] = df.groupby("date")["rsi_14"].transform(
+        lambda x: (x < 30).mean()
+    )
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
-    ] + lag_cols
+    ] + lag_cols + xrank_cols + ["market_stress"]
 
     # --- END researcher section ---
 
```

### Iteration 21 -- REVERTED (-0.1148)
Score: 0.6711 vs best 0.7859
Change:         depth=8,         l2_leaf_reg=7.0, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..b64879d 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -69,12 +69,12 @@ def build_model(y_train):
     return _EarlyStopCB(
         val_frac=0.1,
         iterations=3000,
-        depth=7,
+        depth=8,
         learning_rate=0.01,
         min_data_in_leaf=50,
         boosting_type="Ordered",
         rsm=0.6,
-        l2_leaf_reg=3.0,
+        l2_leaf_reg=7.0,
         posterior_sampling=True,
         scale_pos_weight=spw,
         random_seed=42,
```

### Iteration 22 -- REVERTED (-0.3623)
Score: 0.4236 vs best 0.7859
Change:         iterations=4000,         depth=5,         min_data_in_leaf=80,         r
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..d147557 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -68,20 +68,21 @@ def build_model(y_train):
 
     return _EarlyStopCB(
         val_frac=0.1,
-        iterations=3000,
-        depth=7,
+        iterations=4000,
+        depth=5,
         learning_rate=0.01,
-        min_data_in_leaf=50,
+        min_data_in_leaf=80,
         boosting_type="Ordered",
-        rsm=0.6,
-        l2_leaf_reg=3.0,
+        rsm=0.4,
+        l2_leaf_reg=5.0,
+        random_strength=2.0,
         posterior_sampling=True,
         scale_pos_weight=spw,
         random_seed=42,
         verbose=0,
         thread_count=-1,
         od_type="Iter",
-        od_wait=200,
+        od_wait=250,
         use_best_model=True,
     )
 
```

### Iteration 23 -- REVERTED (-0.1399)
Score: 0.6460 vs best 0.7859
Change:     ret = g["close"].pct_change()     is_down = (ret < 0).astype(float)      df[
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 3176c3f..1e9cb85 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -60,10 +60,29 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
             df[name] = g[col].shift(lag)
             lag_cols.append(name)
 
+    ret = g["close"].pct_change()
+    is_down = (ret < 0).astype(float)
+
+    df["down_days_5d"] = is_down.groupby(df["stock_id"]).transform(
+        lambda s: s.rolling(5, min_periods=1).sum()
+    )
+
+    not_down = (is_down == 0)
+    group_id = not_down.groupby(df["stock_id"]).cumsum()
+    df["down_streak"] = is_down.groupby([df["stock_id"], group_id]).cumsum()
+
+    past_down = is_down.groupby(df["stock_id"]).shift(1)
+    past_down_count = past_down.groupby(df["stock_id"]).transform(
+        lambda s: s.rolling(3, min_periods=1).sum()
+    )
+    today_up = (df["close"] > df["open"]).astype(float)
+    df["reversal_candle"] = today_up * (past_down_count / 3)
+
     new_features = [
         "close_position", "lower_wick_ratio", "gap_return",
         "body_ratio", "volume_sma_ratio", "high_vol_reversal",
         "signed_candle", "range_ratio_10d", "close_vs_prev_high",
+        "down_days_5d", "down_streak", "reversal_candle",
     ] + lag_cols
 
     # --- END researcher section ---
```

### Iteration 24 -- REVERTED (-0.1075)
Score: 0.6784 vs best 0.7859
Change: class _EnsembleCB:     """CatBoost ensemble: multiple seeds with averaged predic
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..3a16d2c 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -40,24 +40,33 @@ FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]
 # ===========================================================================
 
 
-class _EarlyStopCB:
-    """CatBoost with internal early stopping on a temporal holdout."""
+class _EnsembleCB:
+    """CatBoost ensemble: multiple seeds with averaged predictions.
 
-    def __init__(self, val_frac, **kwargs):
+    With posterior_sampling=True, different seeds yield genuinely diverse models.
+    Averaging reduces prediction variance and smooths the probability surface,
+    improving ranking quality at top-fraction budget cutoffs.
+    """
+
+    def __init__(self, seeds, val_frac, **kwargs):
+        self._seeds = seeds
         self._val_frac = val_frac
         self._kwargs = kwargs
 
     def fit(self, X, y):
         from catboost import CatBoostClassifier, Pool
-        n = len(X)
-        cut = int(n * (1 - self._val_frac))
-        self._model = CatBoostClassifier(**self._kwargs)
-        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
-        self.classes_ = self._model.classes_
+        cut = int(len(X) * (1 - self._val_frac))
+        tr, vl = Pool(X[:cut], y[:cut]), Pool(X[cut:], y[cut:])
+        self._models = []
+        for s in self._seeds:
+            m = CatBoostClassifier(**{**self._kwargs, "random_seed": s})
+            m.fit(tr, eval_set=vl)
+            self._models.append(m)
+        self.classes_ = self._models[0].classes_
         return self
 
     def predict_proba(self, X):
-        return self._model.predict_proba(X)
+        return np.mean([m.predict_proba(X) for m in self._models], axis=0)
 
 
 def build_model(y_train):
@@ -66,9 +75,10 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
     spw = np.sqrt(neg / pos)
 
-    return _EarlyStopCB(
+    return _EnsembleCB(
+        seeds=[42, 123, 271],
         val_frac=0.1,
-        iterations=3000,
+        iterations=1000,
         depth=7,
         learning_rate=0.01,
         min_data_in_leaf=50,
```

### Iteration 1 -- REVERTED (-0.1664)
Score: 0.6195 vs best 0.7859
Change:         iterations=5000,         learning_rate=0.005, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 1a54d1f..166316a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -68,9 +68,9 @@ def build_model(y_train):
 
     return _EarlyStopCB(
         val_frac=0.1,
-        iterations=3000,
+        iterations=5000,
         depth=7,
-        learning_rate=0.01,
+        learning_rate=0.005,
         min_data_in_leaf=50,
         boosting_type="Ordered",
         rsm=0.6,
```
