# Research Log

Walk-forward evaluation: 4 folds x 6 months (2024-03 to 2026-03), data from 2020+.

### Iteration 1 -- IMPROVED (+inf)
Score: -2.0918 |         n_estimators=500,         max_depth=4,         learning_rate=0.02,      
Commit: 7b167ca

### Iteration 2 -- REVERTED (-0.1354)
Score: -2.2272 |     eps = 1e-10     hl_range = df["high"] - df["low"]  eps      df["intraday_rec

### Iteration 3 -- IMPROVED (+0.0657)
Score: -2.0261 |         n_estimators=800,         learning_rate=0.012, 
Commit: e3a59bf

### Iteration 4 -- IMPROVED (+0.0086)
Score: -2.0175 |         n_estimators=1200,         learning_rate=0.008, 
Commit: 31d6e40

### Iteration 5 -- IMPROVED (+0.0237)
Score: -1.9938 |         n_estimators=1800,         learning_rate=0.005, 
Commit: 59e9b5e

### Iteration 6 -- REVERTED (-0.0667)
Score: -2.0605 |         n_estimators=2500,         learning_rate=0.003, 

### Iteration 7 -- REVERTED (-0.0548)
Score: -2.0486 |         max_depth=5, 

### Iteration 8 -- REVERTED (-0.0357)
Score: -2.0295 |         min_child_weight=6, 

### Iteration 9 -- REVERTED (-0.1835)
Score: -2.1773 |     from lightgbm import LGBMClassifier     return LGBMClassifier(         n_est

### Iteration 10 -- IMPROVED (+0.3573)
Score: -1.6365 |     from research.model_wrappers import RankingXGBClassifier     return RankingX
Commit: 0c6b89c

### Iteration 11 -- REVERTED (-0.0516)
Score: -1.6881 |         colsample_bytree=0.4, 

### Iteration 12 -- REVERTED (-0.1447)
Score: -1.7812 |         group_size=50, 

### Iteration 13 -- REVERTED (-0.2018)
Score: -1.8383 |         objective="rank:pairwise", 

### Iteration 14 -- REVERTED (-0.3257)
Score: -1.9622 |     from sklearn.ensemble import VotingClassifier     from research.model_wrappe

### Iteration 15 -- REVERTED (-0.1166)
Score: -1.7531 |     from research.model_wrappers import FocalTorchClassifier, TorchMLP      n_fe

### Iteration 16 -- IMPROVED (+0.1035)
Score: -1.5330 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"] 
Commit: 00da3ac

### Iteration 17 -- REVERTED (-0.0624)
Score: -1.5954 |         n_estimators=2400, 

### Iteration 18 -- REVERTED (-0.0086)
Score: -1.5416 |         max_depth=3, 

### Iteration 19 -- REVERTED (-0.0155)
Score: -1.5485 |         reg_lambda=3.0, 

### Iteration 20 -- REVERTED (-0.2056)
Score: -1.7386 |     g = df.groupby("stock_id")      def _down_streak(close):         ret = close

### Iteration 21 -- REVERTED (-0.0852)
Score: -1.6182 | FEATURE_GROUPS = ["base", "advanced", "roc", "rolling"] 

### Iteration 22 -- IMPROVED (+0.0334)
Score: -1.4996 |         group_size=1000, 
Commit: 1b03659

### Iteration 23 -- REVERTED (-0.5028)
Score: -2.0024 |         group_size=2000, 

### Iteration 24 -- REVERTED (-0.1489)
Score: -1.6485 |         learning_rate=0.008, 

### Iteration 25 -- REVERTED (-0.1435)
Score: -1.6431 |         colsample_bytree=0.7, 

### Iteration 26 -- REVERTED (-0.1426)
Score: -1.6422 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction", "lag"]

### Iteration 27 -- IMPROVED (+0.3642)
Score: -1.1354 |     new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20"]
Commit: 95e3ae8

### Iteration 28 -- REVERTED (-0.7704)
Score: -1.9058 |     new_features = [         "price_efficiency_10", "return_accel_10", "returns_

### Iteration 29 -- REVERTED (-0.4869)
Score: -1.6223 |     new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20",

### Iteration 30 -- REVERTED (-0.0719)
Score: -1.2073 |         subsample=0.7, 

### Iteration 1 -- REVERTED (-0.4828)
Score: -1.6182 |         objective="rank:map", 

### Iteration 2 -- REVERTED (-0.4996)
Score: -1.6350 |     new_features = ["price_efficiency_10", "return_accel_10", "days_since_20d_lo

### Iteration 3 -- REVERTED (-1.1389)
Score: -2.2743 |     from xgboost import XGBClassifier     neg = (y_train == 0).sum()     pos = (

### Iteration 4 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 142).
Change:         booster="dart",         n_estimators=1000,         learning_rate=0.008, 

### Iteration 5 -- REVERTED (-1.1702)
Score: -2.3056 | FEATURE_GROUPS = None         n_estimators=3000,         max_depth=3,         le

### Iteration 6 -- IMPROVED (+0.8745)
Score: -0.2609 |     import xgboost as xgb     from scipy.special import expit      class _Adapti
Commit: 7577c54

### Iteration 7 -- REVERTED (-0.0600)
Score: -0.3209 |                 "eval_metric": "ndcg@10", 

### Iteration 8 -- REVERTED (-0.0323)
Score: -0.2932 |                 "gamma": 0.2, 

### Iteration 9 -- IMPROVED (+0.2681)
Score: 0.0072 |                 "min_child_weight": 5, 
Commit: 0fc9c4d

### Iteration 10 -- REVERTED (-0.2758)
Score: -0.2686 |                 "min_child_weight": 3, 

### Iteration 11 -- REVERTED (-0.3623)
Score: -0.3551 |                 "reg_lambda": 10.0, 

### Iteration 12 -- REVERTED (-0.1808)
Score: -0.1736 |                 "reg_alpha": 0.1, 

### Iteration 13 -- REVERTED (-2.2175)
Score: -2.2103 |     from catboost import CatBoostClassifier, Pool     class _CatBoostPivot:     

### Iteration 14 -- REVERTED (-1.6823)
Score: -1.6751 |         def _date_groups(d):             if len(d) <= 1:                 return 

### Iteration 15 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             dtrain = xgb.DMatrix(X[:split], label=y[:split])             dtrain.

### Iteration 16 -- REVERTED (-2.2635)
Score: -2.2563 |     import lightgbm as lgb     class _LGBMLambdaRank:         def _make_groups(t

### Iteration 17 -- REVERTED (-2.2358)
Score: -2.2286 |             from xgboost import XGBClassifier              neg, pos = int((y == 

### Iteration 18 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             weights = np.linspace(0.5, 1.0, split)             dtrain = xgb.DMat

### Iteration 19 -- REVERTED (-1.2034)
Score: -1.1962 |             base_params = {             configs = [                 {"max_depth"

### Iteration 20 -- REVERTED (-2.2725)
Score: -2.2653 |     class _GlobalClassifier:                 "objective": "binary:logistic",    

### Iteration 21 -- REVERTED (-1.8306)
Score: -1.8234 |     new_features = ["price_efficiency_10", "return_accel_10", "returns_skew_20",

### Iteration 22 -- REVERTED (-2.0291)
Score: -2.0219 |     from research.model_wrappers import FocalTorchClassifier, TorchMLP     class

### Iteration 23 -- REVERTED (-1.3781)
Score: -1.3709 |                 "grow_policy": "lossguide",                 "max_leaves": 31,   

### Iteration 24 -- REVERTED (-1.4864)
Score: -1.4792 |                 "objective": "rank:pairwise", 

### Iteration 25 -- REVERTED (-0.4254)
Score: -0.4182 |             deval.set_group([n - split]) 

### Iteration 26 -- REVERTED (-2.2535)
Score: -2.2463 |     from sklearn.ensemble import ExtraTreesClassifier      return ExtraTreesClas

### Iteration 27 -- REVERTED (+0.0000)
Score: 0.0072 |             base_params = {             scout = xgb.train(                 {**ba

### Iteration 28 -- REVERTED (-2.4795)
Score: -2.4723 |             gs = 5000 

### Iteration 29 -- REVERTED (-2.1638)
Score: -2.1566 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction", "rolli

### Iteration 30 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             from scipy.stats import rankdata              # Extract grading sign

### Iteration 31 -- REVERTED (-1.7807)
Score: -1.7735 |         @staticmethod         def _grade_labels(X, y):             from sklearn.

### Iteration 32 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change:             w_tr = np.where(y[:split] == 1, 3.0, 1.0)             w_ev = np.wher

### Iteration 33 -- IMPROVED (+0.0687)
Score: 0.0759 |                 "max_depth": 5, 
Commit: 2b87e88

### Iteration 34 -- REVERTED (-1.3479)
Score: -1.2720 |                 "max_depth": 6, 

### Iteration 35 -- IMPROVED (+0.2078)
Score: 0.2837 |                 "learning_rate": 0.003, 
Commit: e8c59d5

### Iteration 36 -- REVERTED (-0.5221)
Score: -0.2384 |                 "subsample": 0.7, 

### Iteration 37 -- REVERTED (-0.2829)
Score: 0.0008 |                 "learning_rate": 0.002,                 num_boost_round=7000, 

### Iteration 38 -- REVERTED (-2.2461)
Score: -1.9624 |     new_features = [         "price_efficiency_10", "return_accel_10", "returns_

### Iteration 39 -- IMPROVED (+0.9412)
Score: 1.2249 |             ev_mask = np.arange(n) % 10 == 0             tr_idx = np.where(~ev_m
Commit: f20be74

### Iteration 40 -- REVERTED (-2.5967)
Score: -1.3718 |                 "colsample_bynode": 0.8, 

### Iteration 41 -- IMPROVED (+0.3026)
Score: 1.5275 |                 early_stopping_rounds=200, 
Commit: 7e3eb5f

### Iteration 42 -- REVERTED (-3.3755)
Score: -1.8480 |                 "reg_lambda": 3.0, 

### Iteration 43 -- REVERTED (-3.5694)
Score: -2.0419 |                 "colsample_bytree": 0.5, 

### Iteration 44 -- REVERTED (-3.0454)
Score: -1.5179 | FEATURE_GROUPS = ["base", "advanced", "lag", "roc", "percentile", "interaction"]

### Iteration 45 -- REVERTED (-3.1630)
Score: -1.6355 |                 "objective": "rank:map",                 "eval_metric": "map", 

### Iteration 46 -- REVERTED (-3.1458)
Score: -1.6183 |             lr_schedule = xgb.callback.LearningRateScheduler(                 la

### Iteration 47 -- KNOWLEDGE
No code changes.

### Iteration 48 -- EVAL RESET (dataset + eval changes)
Score: -1.1806 | Rebuilt dataset: dropped 142 penny stocks (close < $1), 1210 stocks remain. Eval changes: winsorize returns at 1st/99th pctile, formula weights 0.40/0.20/0.10/0.10/0.05/0.15, W=n/(n+50), 5 budgets (dropped 0.05%), missing cells excluded. Previous best 1.5275 was inflated by WYLD.ST +41900% outlier. New baseline: -1.1806.

### Iteration 1 -- REVERTED (-0.0283)
Score: -1.2089 |             gs = 500 

### Iteration 2 -- REVERTED (-0.3141)
Score: -1.4947 |             ev_mask = np.arange(n) % 5 == 0                 "min_child_weight": 

### Iteration 3 -- REVERTED (-0.2531)
Score: -1.4337 | unknown change

### Iteration 4 -- IMPROVED (+0.1305)
Score: -1.0501 |     from catboost import CatBoostClassifier     class _CatBoostModel:           
Commit: b45f253

### Iteration 5 -- REVERTED (-0.0853)
Score: -1.1354 |                 depth=5, 

### Iteration 6 -- REVERTED (-0.0696)
Score: -1.1197 |                 iterations=5000,                 learning_rate=0.01, 

### Iteration 7 -- REVERTED (-0.0584)
Score: -1.1085 |                 depth=7, 

### Iteration 8 -- REVERTED (-0.0392)
Score: -1.0893 |     import lightgbm as lgb     class _LGBMModel:             params = {         

### Iteration 9 -- IMPROVED (+0.0506)
Score: -0.9995 |     import lightgbm as lgb      class _EnsembleModel:         """CatBoost  Light
Commit: 4f83f4e

### Iteration 10 -- REVERTED (-0.0058)
Score: -1.0053 |                     'learning_rate': 0.01,                 num_boost_round=5000,

### Iteration 11 -- IMPROVED (+0.0029)
Score: -0.9966 |             avg = np.sqrt(cat_p * lgb_p) 
Commit: dea0d8c

### Iteration 12 -- IMPROVED (+0.0590)
Score: -0.9376 |     import xgboost as xgb             self._xgb = xgb.XGBClassifier(            
Commit: bcdf0a0

### Iteration 13 -- IMPROVED (+0.0211)
Score: -0.9165 |             avg = (cat_p  lgb_p  xgb_p) / 3 
Commit: fa73436

### Iteration 14 -- REVERTED (-0.0461)
Score: -0.9626 |             from scipy.stats import rankdata             n = len(cat_p)         

### Iteration 15 -- REVERTED (-0.0664)
Score: -0.9829 |     from sklearn.ensemble import HistGradientBoostingClassifier             w = 

### Iteration 16 -- REVERTED (-0.0077)
Score: -0.9242 |                     'lambda_l1': 0.5, 

### Iteration 17 -- IMPROVED (+0.0046)
Score: -0.9119 |             from sklearn.linear_model import LogisticRegression             cat_
Commit: fe46062

### Iteration 18 -- REVERTED (-0.1226)
Score: -1.0345 |             split = int(n * 0.9)             tr_idx = np.arange(split)          

### Iteration 19 -- IMPROVED (+0.0110)
Score: -0.9009 |                 scale_pos_weight=3,                 scale_pos_weight=8, 
Commit: f46b25c

### Iteration 20 -- REVERTED (-0.0215)
Score: -0.9224 |                     'num_leaves': 45, 

### Iteration 21 -- REVERTED (-0.0714)
Score: -0.9723 |             self._meta = LogisticRegression(C=0.5, max_iter=300) 

### Iteration 22 -- REVERTED (-0.0542)
Score: -0.9551 |                 scale_pos_weight=5, 

### Iteration 23 -- REVERTED (-0.1722)
Score: -1.0731 |             from sklearn.ensemble import GradientBoostingClassifier             

### Iteration 24 -- REVERTED (-0.0268)
Score: -0.9277 |             idx = np.arange(n)             ev_es_mask = idx % 10 == 0           

### Iteration 25 -- REVERTED (-0.1029)
Score: -1.0038 |             from scipy.special import expit             gs = 1000             n_

### Iteration 26 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 137).
Change:             from research.model_wrappers import FocalTorchClassifier, TorchMLP  

### Iteration 27 -- REVERTED (-0.5914)
Score: -1.4923 |             # --- Feature selection: quick LGB to find top features by gain --- 

### Iteration 28 -- REVERTED (-0.6249)
Score: -1.5258 |             # Time-decay sample weights: upweight recent data (rows are chronolo

### Iteration 29 -- KNOWLEDGE
No code changes.

### Iteration 30 -- EVAL RESET (dataset rebuild + embargo fix)
Score: -1.6805 | Dataset rebuilt: added adr_z, adr_change_20d, adr lags to lib/features.py; fixed inf in adr_change_20d. embargo_sessions 13->14 (label leakage fix for window_variations). 1210 stocks, 2,787,712 rows. Previous best -0.9009. New baseline: -1.6805.
