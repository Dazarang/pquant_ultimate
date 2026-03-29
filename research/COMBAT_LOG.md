# Combat Log

What worked, what failed, and why. Read this BEFORE starting a new iteration.

### Iteration 3 -- REVERTED (-0.0554)
Score: -4.6466 vs best -4.5912
Change:         max_depth=4,         min_child_weight=10,         reg_alpha=0.5,        
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..3f44e73 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -51,10 +51,13 @@ def build_model(y_train):
 
     model = XGBClassifier(
         n_estimators=300,
-        max_depth=6,
+        max_depth=4,
         learning_rate=0.05,
+        min_child_weight=10,
         subsample=0.8,
         colsample_bytree=0.8,
+        reg_alpha=0.5,
+        reg_lambda=3,
         scale_pos_weight=neg / pos,
         tree_method="hist",
         random_state=42,
```

### Iteration 5 -- REVERTED (-0.0010)
Score: -4.5922 vs best -4.5912
Change:     g = df.groupby("stock_id")      # Intraday recovery: fraction of daily range
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 68d988e..7fdb2b2 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -22,6 +22,37 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
 
     # --- RESEARCHER: add features below ---
 
+    g = df.groupby("stock_id")
+
+    # Intraday recovery: fraction of daily range recaptured from the low
+    # High value = strong buying pressure at lows (continuous hammer signal)
+    rng = df["high"] - df["low"]
+    df["intraday_recovery"] = (df["close"] - df["low"]) / rng.replace(0, float("nan"))
+    new_features.append("intraday_recovery")
+
+    # Cumulative selling pressure: sum of negative returns over 5 days
+    # Deep negative = heavy recent damage, potential capitulation zone
+    neg_ret = df["ret_1d"].clip(upper=0)
+    df["cum_sell_pressure_5d"] = neg_ret.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).sum()
+    )
+    new_features.append("cum_sell_pressure_5d")
+
+    # Volume climax: volume spike magnitude on down days only
+    # High value = capitulation selling (big volume + big drop)
+    df["volume_climax"] = (
+        df["volume_ratio"] * df["ret_1d"].abs() * (df["ret_1d"] < 0).astype(float)
+    )
+    new_features.append("volume_climax")
+
+    # Bounce from 5-day low, ATR-normalized
+    # Near zero = sitting at local low; positive = recovering from it
+    low_5d = g["low"].transform(lambda x: x.rolling(5, min_periods=1).min())
+    df["bounce_from_5d_low"] = (df["close"] - low_5d) / df["atr_14"].replace(
+        0, float("nan")
+    )
+    new_features.append("bounce_from_5d_low")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 6 -- REVERTED (-0.0172)
Score: -4.6084 vs best -4.5912
Change:         n_estimators=500,         learning_rate=0.03, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..b25f39e 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -50,9 +50,9 @@ def build_model(y_train):
     pos = (y_train == 1).sum()
 
     model = XGBClassifier(
-        n_estimators=300,
+        n_estimators=500,
         max_depth=6,
-        learning_rate=0.05,
+        learning_rate=0.03,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

### Iteration 7 -- REVERTED (-0.0384)
Score: -4.6296 vs best -4.5912
Change: FEATURE_GROUPS = ["base", "advanced"] 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index af3e4f4..38d3b67 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -30,7 +30,7 @@ DATASET_PATH = "data/datasets/20260115/dataset.parquet"
 STOCKS = None
 
 # Feature groups: see list_features() for options
-FEATURE_GROUPS = None  # None = all
+FEATURE_GROUPS = ["base", "advanced"]
 
 # Temporal split boundaries
 TRAIN_END = "2022-12-31"
```

### Iteration 1 -- REVERTED (-0.0646)
Score: -2.4770 vs best -2.4124
Change: THRESHOLD = 0.90 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 4c92c61..0baa5ee 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -37,7 +37,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold
-THRESHOLD = 0.85
+THRESHOLD = 0.90
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 3 -- REVERTED (-0.0829)
Score: -2.4262 vs best -2.3433
Change:         gamma=1.0, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 4c92c61..377fe5f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -53,6 +53,7 @@ def build_model(y_train):
         n_estimators=300,
         max_depth=6,
         learning_rate=0.05,
+        gamma=1.0,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

### Iteration 5 -- REVERTED (-0.0208)
Score: -2.3281 vs best -2.3073
Change:     # Momentum acceleration (2nd derivative of returns):     # Positive during d
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 870bf73..1257438 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -65,6 +65,13 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).transform(lambda x: x.rolling(5, min_periods=1).mean())
     new_features.append("overnight_gap_trend_5d")
 
+    # Momentum acceleration (2nd derivative of returns):
+    # Positive during downtrend = decline decelerating = bottom forming
+    # Targets knife_rate reduction by filtering false bottoms still in freefall
+    ret_ma_5d = g["ret_1d"].transform(lambda x: x.rolling(5, min_periods=2).mean())
+    df["momentum_accel"] = ret_ma_5d - ret_ma_5d.groupby(df["stock_id"]).shift(5)
+    new_features.append("momentum_accel")
+
     # --- END researcher section ---
 
     return df, new_features
```


### Iteration 2 -- REVERTED (-0.2314)
Score: -1.0913 vs best -0.8599
Change:         num_leaves=63,         max_depth=-1,         subsample=0.7,         subs
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 031991e..849cf9f 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -66,10 +66,13 @@ def build_model(y_train):
 
     lgbm = LGBMClassifier(
         n_estimators=300,
-        max_depth=6,
+        num_leaves=63,
+        max_depth=-1,
         learning_rate=0.05,
-        subsample=0.8,
-        colsample_bytree=0.8,
+        subsample=0.7,
+        subsample_freq=1,
+        colsample_bytree=0.6,
+        min_child_samples=50,
         scale_pos_weight=neg / pos,
         random_state=43,
         n_jobs=-1,
```

### Iteration 3 -- REVERTED (-0.4830)
Score: -1.3429 vs best -0.8599
Change: from sklearn.linear_model import LogisticRegression     lr = LogisticRegression(
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 031991e..a5f899a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -13,6 +13,7 @@ from pathlib import Path
 import numpy as np  # noqa: F401 -- available for researcher
 from lightgbm import LGBMClassifier
 from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
+from sklearn.linear_model import LogisticRegression
 from xgboost import XGBClassifier
 
 # Ensure project root is on path
@@ -85,8 +86,15 @@ def build_model(y_train):
         n_jobs=-1,
     )
 
+    lr = LogisticRegression(
+        C=0.1,
+        class_weight="balanced",
+        max_iter=1000,
+        random_state=45,
+    )
+
     model = VotingClassifier(
-        estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et)],
+        estimators=[("xgb", xgb), ("lgbm", lgbm), ("et", et), ("lr", lr)],
         voting="soft",
     )
     return model
```

### Iteration 4 -- REVERTED (-7.1806)
Score: -8.0405 vs best -0.8599
Change: THRESHOLD = 0.90 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index 031991e..bad462a 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -39,7 +39,7 @@ TRAIN_END = "2022-12-31"
 VAL_END = "2023-12-31"
 
 # Prediction threshold
-THRESHOLD = 0.85
+THRESHOLD = 0.90
 
 # ===========================================================================
 # MODEL -- researcher edits this section
```

### Iteration 6 -- REVERTED (-0.1494)
Score: -0.9270 vs best -0.7776
Change:     # Candle morphology: lower wick ratio (3-day avg)     # Large lower wicks = 
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 870bf73..62f8e28 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -65,6 +65,33 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).transform(lambda x: x.rolling(5, min_periods=1).mean())
     new_features.append("overnight_gap_trend_5d")
 
+    # Candle morphology: lower wick ratio (3-day avg)
+    # Large lower wicks = buying at intraday lows = hammer pattern = bottom signal
+    body_low = df[["open", "close"]].min(axis=1)
+    total_range = (df["high"] - df["low"]).replace(0, float("nan"))
+    lower_wick = (body_low - df["low"]) / total_range
+    df["lower_wick_ratio_3d"] = lower_wick.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(3, min_periods=1).mean()
+    )
+    new_features.append("lower_wick_ratio_3d")
+
+    # Candle morphology: body-to-range ratio (3-day avg)
+    # Small bodies relative to range = indecision/turning point
+    # At bottoms: indecision candles precede reversal
+    body_size = (df["close"] - df["open"]).abs() / total_range
+    df["candle_body_pct_3d"] = body_size.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(3, min_periods=1).mean()
+    )
+    new_features.append("candle_body_pct_3d")
+
+    # Bullish candle fraction over 5 days
+    # Green candles appearing during decline = buyer re-entry
+    bullish = (df["close"] > df["open"]).astype(float)
+    df["bullish_candle_pct_5d"] = bullish.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).mean()
+    )
+    new_features.append("bullish_candle_pct_5d")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 8 -- REVERTED (-0.3677)
Score: -0.9117 vs best -0.5440
Change:     # Accumulation-side volume: fraction of 5d volume on up-days     # Complemen
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index 529a1a5..35343fc 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -89,6 +89,24 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     )
     new_features.append("vol_wtd_momentum_10d")
 
+    # Accumulation-side volume: fraction of 5d volume on up-days
+    # Complements sell_vol_exhaustion: true bottoms show buyers stepping in
+    up_vol = df["volume"] * (df["ret_1d"] >= 0).astype(float)
+    total_vol_5d = df["volume"].groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).sum()
+    )
+    up_vol_5d = up_vol.groupby(df["stock_id"]).transform(
+        lambda x: x.rolling(5, min_periods=1).sum()
+    )
+    df["buy_vol_ratio_5d"] = up_vol_5d / total_vol_5d.replace(0, float("nan"))
+    new_features.append("buy_vol_ratio_5d")
+
+    # Distribution-to-accumulation transition: change in buy_vol_ratio over 5 days
+    # Positive = buying fraction increasing = accumulation starting
+    shifted = df["buy_vol_ratio_5d"].groupby(df["stock_id"]).shift(5)
+    df["buy_vol_accel"] = df["buy_vol_ratio_5d"] - shifted
+    new_features.append("buy_vol_accel")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 11 -- REVERTED (-0.7004)
Score: -1.2154 vs best -0.5150
Change:     # Drawdown z-score: how abnormal is this drawdown for THIS stock?     # Very
```diff
diff --git a/research/features_lab.py b/research/features_lab.py
index c69818d..b890f1c 100644
--- a/research/features_lab.py
+++ b/research/features_lab.py
@@ -125,6 +125,23 @@ def add_custom_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
     ).cumsum()
     new_features.append("streak_depth")
 
+    # Drawdown z-score: how abnormal is this drawdown for THIS stock?
+    # Very negative = unusually deep decline = potential overreaction = bottom
+    # Normalizes cross-stock: -10% in a stable utility is extreme, normal for volatile tech
+    # Targets worst_decile and knife_rate by filtering stocks whose decline is within normal range
+    dd_mean = g["drawdown"].transform(lambda x: x.rolling(252, min_periods=60).mean())
+    dd_std = g["drawdown"].transform(lambda x: x.rolling(252, min_periods=60).std())
+    df["drawdown_zscore"] = (df["drawdown"] - dd_mean) / dd_std.replace(0, float("nan"))
+    new_features.append("drawdown_zscore")
+
+    # Volatility regime ratio: 5-day vs 20-day return volatility
+    # < 1 = recent volatility below normal (stabilizing after sell-off = bottom signal)
+    # > 1 = recent volatility above normal (crisis intensifying = knife risk)
+    # Targets knife_rate: knives have expanding volatility, true bottoms have contracting
+    vol_5d = g["ret_1d"].transform(lambda x: x.rolling(5, min_periods=2).std())
+    df["vol_regime_ratio"] = vol_5d / df["volatility_20d"].replace(0, float("nan"))
+    new_features.append("vol_regime_ratio")
+
     # --- END researcher section ---
 
     return df, new_features
```

### Iteration 12 -- REVERTED (-0.0064)
Score: -0.5214 vs best -0.5150
Change:         max_delta_step=1, 
```diff
diff --git a/research/experiment.py b/research/experiment.py
index c8a15dc..9c0cbbe 100644
--- a/research/experiment.py
+++ b/research/experiment.py
@@ -55,6 +55,7 @@ def build_model(y_train):
         n_estimators=300,
         max_depth=6,
         learning_rate=0.05,
+        max_delta_step=1,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=neg / pos,
```

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
