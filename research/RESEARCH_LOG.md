# Research Log

Walk-forward evaluation: 4 folds x 6 months (2024-03 to 2026-03), data from 2020+.

### Iteration 1 -- IMPROVED (+0.7356)
Score: -0.0160 | from lightgbm import LGBMClassifier  # noqa: E402     model = LGBMClassifier(   
Commit: cb4727f

### Iteration 2 -- REVERTED (-0.1623)
Score: -0.1783 | import numpy as np     g = df.groupby("stock_id")      candle_range = df["high"]

### Iteration 3 -- IMPROVED (+0.2438)
Score: 0.2278 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"] 
Commit: ad09516

### Iteration 4 -- REVERTED (-0.2438)
Score: -0.0160 | FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in

### Iteration 5 -- IMPROVED (+0.1281)
Score: 0.3559 | from research.model_wrappers import CatBoostWrapper  # noqa: E402     model = Ca
Commit: 352a9af

### Iteration 6 -- REVERTED (-0.0168)
Score: 0.3391 |         iterations=3000,         depth=5,         learning_rate=0.005,         l

### Iteration 7 -- REVERTED (-0.1294)
Score: 0.2265 | from sklearn.ensemble import VotingClassifier  # noqa: E402 from xgboost import 

### Iteration 8 -- REVERTED (-0.9626)
Score: -0.6067 | from research.model_wrappers import RankingXGBClassifier  # noqa: E402     model

### Iteration 9 -- REVERTED (-0.1353)
Score: 0.2206 | import numpy as np     g = df.groupby("stock_id")      # Consecutive negative-re

### Iteration 10 -- REVERTED (-0.1475)
Score: 0.2084 | from lightgbm import LGBMClassifier  # noqa: E402     model = LGBMClassifier(   

### Iteration 11 -- REVERTED (-0.3499)
Score: 0.0060 |         grow_policy="Lossguide",         max_leaves=64,         depth=8, 

### Iteration 1 -- IMPROVED (+0.0039)
Score: 0.3598 |         posterior_sampling=True, 
Commit: 104b0f7

### Iteration 2 -- REVERTED (-0.0373)
Score: 0.3225 |         rsm=0.5, 

### Iteration 3 -- REVERTED (-0.0663)
Score: 0.2935 |         sampling_frequency="PerTreeLevel", 

### Iteration 1 -- REVERTED (+0.0000)
Score: 0.3598 |         min_data_in_leaf=30, 

### Iteration 2 -- REVERTED (-0.0781)
Score: 0.2817 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile"] 

### Iteration 3 -- REVERTED (-0.0349)
Score: 0.3249 |         l2_leaf_reg=1.0, 

### Iteration 4 -- REVERTED (-0.5289)
Score: -0.1691 | FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interacti

### Iteration 5 -- REVERTED (-0.6028)
Score: -0.2430 | class _LazyFocal:     """Defers TorchMLP construction to fit() to get input_dim 

### Iteration 1 -- KNOWLEDGE
No code changes.

### Iteration 1 -- REVERTED (-0.1153)
Score: 0.2445 |         bootstrap_type="Bayesian",         bagging_temperature=1.0, 

### Iteration 2 -- REVERTED (-0.0079)
Score: 0.3519 |         depth=7, 

### Iteration 3 -- REVERTED (-0.1009)
Score: 0.2589 |         iterations=3000, 

### Iteration 1 -- IMPROVED (+0.0722)
Score: 0.4320 | class _EarlyStopCB:     """CatBoost with internal early stopping on a temporal h
Commit: 3c8bc00

### Iteration 2 -- KNOWLEDGE
No code changes.

### Iteration 1 -- REVERTED (-0.0556)
Score: 0.3764 |         iterations=6000,         learning_rate=0.005, 

### Iteration 2 -- REVERTED (-0.1289)
Score: 0.3031 |         eval_metric="PRAUC", 

### Iteration 3 -- REVERTED (+0.0000)
Score: 0.4320 |         od_wait=300, 

### Iteration 4 -- REVERTED (-0.1010)
Score: 0.3310 |         weights = np.linspace(0.5, 1.5, cut)         self._model.fit(Pool(X[:cut

### Iteration 5 -- REVERTED (-0.1854)
Score: 0.2466 | FEATURE_GROUPS = ["base", "advanced", "lag", "roc", "percentile", "interaction"]

### Iteration 6 -- REVERTED (-0.1220)
Score: 0.3100 | class _Ensemble:     def __init__(self, *models):         self._models = models 

### Iteration 7 -- REVERTED (-0.1098)
Score: 0.3222 |     """CatBoost with importance-based feature selection and early stopping."""  

### Iteration 8 -- REVERTED (-0.0946)
Score: 0.3374 | class _EarlyStopXGB:     """XGBoost with internal early stopping on a temporal h

### Iteration 9 -- REVERTED (-0.2789)
Score: 0.1531 |     def __init__(self, val_frac, neg_ratio=5, **kwargs):         self._neg_ratio

### Iteration 10 -- IMPROVED (+0.0056)
Score: 0.4376 |         boosting_type="Ordered", 
Commit: 783df28

### Iteration 11 -- REVERTED (-0.2919)
Score: 0.1457 |         val_frac=0.15, 

### Iteration 12 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:         langevin=True,         diffusion_temperature=1000, 

### Iteration 13 -- REVERTED (+0.0000)
Score: 0.4376 |         leaf_estimation_iterations=10, 

### Iteration 14 -- REVERTED (-0.3195)
Score: 0.1181 |         iterations=5000,         depth=4, 

### Iteration 15 -- IMPROVED (+0.0901)
Score: 0.5277 |     rng = (df["high"] - df["low"]).clip(lower=1e-10)      df["close_position"] =
Commit: b6c3672

### Iteration 16 -- IMPROVED (+0.0401)
Score: 0.5678 |     df["body_ratio"] = (df["close"] - df["open"]).abs() / rng      vol_sma20 = d
Commit: a9096a1

### Iteration 17 -- IMPROVED (+0.0645)
Score: 0.6323 |     df["signed_candle"] = (df["close"] - df["open"]) / rng      df["_range"] = d
Commit: 58908ed

### Iteration 18 -- REVERTED (-0.1274)
Score: 0.5049 |     # --- Multi-day rolling aggregations of candle features ---     g = df.group

### Iteration 19 -- REVERTED (-0.1702)
Score: 0.4621 |     rolling_5d_low = df.groupby("stock_id")["low"].transform(         lambda s: 

### Iteration 20 -- REVERTED (-0.0818)
Score: 0.5505 |     df["_cv"] = df["close"] * df["volume"]     g2 = df.groupby("stock_id")     s

### Iteration 21 -- REVERTED (-0.0382)
Score: 0.5941 |     g = df.groupby("stock_id")     df["close_position_lag1"] = g["close_position

### Iteration 22 -- REVERTED (-0.0029)
Score: 0.6294 |         langevin=True,         diffusion_temperature=10000, 

### Iteration 23 -- REVERTED (-0.1460)
Score: 0.4863 |     prev_prev_close = df.groupby("stock_id")["close"].shift(2)     ret_t = (df["

### Iteration 24 -- REVERTED (-0.1164)
Score: 0.5159 |     upper_wick = df["high"] - df[["open", "close"]].max(axis=1)     df["upper_wi

### Iteration 25 -- REVERTED (-0.4423)
Score: 0.1900 | class _EarlyStopLGBM:     """LightGBM with internal early stopping on a temporal

### Iteration 26 -- REVERTED (-0.0951)
Score: 0.5372 |     df["_ret_1d"] = df.groupby("stock_id")["close"].pct_change()     df["daily_r

### Iteration 27 -- IMPROVED (+0.1207)
Score: 0.7530 |         depth=7, 
Commit: 5708622

### Iteration 28 -- REVERTED (-0.0878)
Score: 0.6652 |         depth=8, 

### Iteration 29 -- REVERTED (-0.1100)
Score: 0.6430 |         l2_leaf_reg=5.0, 

### Iteration 30 -- REVERTED (-0.0304)
Score: 0.7226 |         rsm=0.5, 

### Iteration 31 -- REVERTED (-0.1156)
Score: 0.6374 |     ret = df.groupby("stock_id")["close"].pct_change().fillna(0)     down = (ret

### Iteration 32 -- REVERTED (-0.0657)
Score: 0.6873 |         learning_rate=0.02, 

### Iteration 33 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 142).
Change: FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interacti

### Iteration 34 -- REVERTED (-0.4033)
Score: 0.3497 |         auto_class_weights="Balanced", 

### Iteration 35 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:         grow_policy="Depthwise", 

### Iteration 36 -- REVERTED (-0.0266)
Score: 0.7264 |         rsm=0.7, 

### Iteration 37 -- REVERTED (+0.0000)
Score: 0.7530 |         bagging_temperature=0.5, 

### Iteration 38 -- REVERTED (-0.0410)
Score: 0.7120 |         l2_leaf_reg=2.0, 

### Iteration 39 -- REVERTED (-0.0906)
Score: 0.6624 |     spw = (neg / pos) ** 0.4 

### Iteration 1 -- REVERTED (-0.4581)
Score: 0.2949 |         min_data_in_leaf=30,         eval_metric="AUC", 

### Iteration 2 -- REVERTED (-1.2111)
Score: -0.4581 | FEATURE_GROUPS = ["base", "roc", "percentile", "interaction"] 

### Iteration 1 -- REVERTED (-0.1098)
Score: 0.6432 | from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402 class _WeightedE

### Iteration 2 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 142).
Change: FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction", "lag"]

### Iteration 3 -- REVERTED (-0.3066)
Score: 0.4464 | FEATURE_GROUPS = ["base", "advanced", "percentile", "interaction"] 

### Iteration 4 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 142).
Change: class _EnsembleCB:     """Bayesian model averaging: multiple CatBoost posterior 

### Iteration 5 -- REVERTED (-0.0546)
Score: 0.6984 | class _EarlyStopXGB:     """XGBoost with internal early stopping on a temporal h

### Iteration 6 -- REVERTED (-1.9465)
Score: -1.1935 | class _FocalMLP:     """Focal-loss MLP with BatchNorm and temporal early stoppin

### Iteration 7 -- REVERTED (-1.4130)
Score: -0.6600 | from research.model_wrappers import RankingXGBClassifier  # noqa: E402     retur

### Iteration 8 -- REVERTED (-0.2378)
Score: 0.5152 |         # Temporal decay: down-weight old data, up-weight recent data.         #

### Iteration 9 -- REVERTED (-0.0553)
Score: 0.6977 |     """CatBoost with feature importance pruning and early stopping.     Two-stag

### Iteration 10 -- REVERTED (-0.2207)
Score: 0.5323 |         boosting_type="Plain",         bootstrap_type="MVS",         subsample=0

### Iteration 1 -- REVERTED (-0.3426)
Score: 0.4104 | class _DownsampleCB:     """CatBoost with negative downsampling and temporal hol

### Iteration 2 -- REVERTED (-0.3852)
Score: 0.3678 |     """CatBoost with internal early stopping on a random holdout."""         n_v

### Iteration 3 -- REVERTED (-0.0320)
Score: 0.7210 |     g = df.groupby("stock_id")     ret_5d = g["close"].pct_change(5)     vol_cha

### Iteration 4 -- REVERTED (-0.1186)
Score: 0.6344 |     cp_lag5 = df.groupby("stock_id")["close_position"].shift(5)     df["close_po

### Iteration 5 -- REVERTED (-0.0107)
Score: 0.7423 |     # Indicator-trajectory features: rate of change for key bottom indicators   

### Iteration 6 -- REVERTED (-0.6194)
Score: 0.1336 | class _EarlyStopLGBM:     """LightGBM with internal early stopping on a temporal

### Iteration 7 -- REVERTED (-0.0574)
Score: 0.6956 |     # --- Multi-day sequential features for reversal quality ---      # Short-te

### Iteration 8 -- REVERTED (-0.1922)
Score: 0.5608 |         eval_metric="PRAUC", 

### Iteration 9 -- REVERTED (-0.2993)
Score: 0.4537 |     # Pivot-point dynamics: 2nd-order trend features capturing turning-point sig

### Iteration 10 -- IMPROVED (+0.0068)
Score: 0.7598 |     g = df.groupby("stock_id")     lag_cols = []     for col in ["rsi_14", "draw
Commit: 4aefdbf

### Iteration 11 -- IMPROVED (+0.0261)
Score: 0.7859 |     for col in ["rsi_14", "drawdown", "close_position", "lower_wick_ratio"]: 
Commit: ecc34d5

### Iteration 12 -- REVERTED (-0.0202)
Score: 0.7657 |     cp_5d = g["close_position"].transform(         lambda s: s.rolling(5, min_pe

### Iteration 13 -- REVERTED (-0.2192)
Score: 0.5667 |     for col in ["rsi_14", "drawdown", "close_position", "lower_wick_ratio",     

### Iteration 14 -- REVERTED (-0.0567)
Score: 0.7292 |         depth=6, 

### Iteration 15 -- REVERTED (-0.1021)
Score: 0.6838 | def _days_since_rolling_high(s, window):     rolling_max = s.rolling(window, min

### Iteration 16 -- REVERTED (-0.5929)
Score: 0.1930 | class _EarlyStopLGBM:     """LightGBM with early stopping on a temporal holdout.

### Iteration 17 -- REVERTED (-0.1115)
Score: 0.6744 |     df["rsi_min_10d"] = g["rsi_14"].transform(lambda s: s.rolling(10, min_period

### Iteration 18 -- REVERTED (-0.4033)
Score: 0.3826 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction", "rolli

### Iteration 19 -- REVERTED (-0.1682)
Score: 0.6177 |     df["_abs_daily_ret"] = g["close"].diff().abs()     sum_abs_ret = df.groupby(

### Iteration 20 -- REVERTED (-0.2891)
Score: 0.4968 |     xrank_cols = []     for col in ["rsi_14", "drawdown", "close_position"]:    

### Iteration 21 -- REVERTED (-0.1148)
Score: 0.6711 |         depth=8,         l2_leaf_reg=7.0, 

### Iteration 22 -- REVERTED (-0.3623)
Score: 0.4236 |         iterations=4000,         depth=5,         min_data_in_leaf=80,         r

### Iteration 23 -- REVERTED (-0.1399)
Score: 0.6460 |     ret = g["close"].pct_change()     is_down = (ret < 0).astype(float)      df[

### Iteration 24 -- REVERTED (-0.1075)
Score: 0.6784 | class _EnsembleCB:     """CatBoost ensemble: multiple seeds with averaged predic

### Iteration 1 -- KNOWLEDGE
No code changes.

### Iteration 1 -- REVERTED (-0.1664)
Score: 0.6195 |         iterations=5000,         learning_rate=0.005, 
