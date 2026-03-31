# Research Log

## Iter 0: Baseline (Multi-Budget Metric)

XGBClassifier on all 1,336 stocks, all 220 features. Single model, no threshold (multi-budget eval with fractional tie weights).
Score: -2.4916

### Iteration 1 -- IMPROVED (+0.1878)
Score: -2.3038 |     spw = np.sqrt(neg / pos)  # moderate weight (~4.4) instead of full ratio (~1
Commit: aeb2d44

### Iteration 2 -- REVERTED (-0.0148)
Score: -2.3186 |     g = df.groupby("stock_id")      # Candlestick microstructure     total_range

### Iteration 3 -- IMPROVED (+0.2716)
Score: -2.0322 |     # Deep, heavily regularized — captures complex interactions         n_estima
Commit: c6be7b0

### Iteration 4 -- REVERTED (-0.0760)
Score: -2.1082 |     g = df.groupby("stock_id")      # Intraday recovery: (close-low)/(high-low),

### Iteration 5 -- REVERTED (-0.0036)
Score: -2.0358 |     ratio = neg / pos     # Differentiated class weights: each model sees the im

### Iteration 6 -- IMPROVED (+0.3627)
Score: -1.6695 | from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier  # noqa: E40
Commit: eaa24e2

### Iteration 7 -- GATE FAILED
Reason: GATE VIOLATION: Experiment crashed (exit code 1).
Change: from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402 class MLPW

### Iteration 8 -- IMPROVED (+0.1059)
Score: -1.5636 | from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E
Commit: b6cb135

### Iteration 9 -- REVERTED (-0.3383)
Score: -1.9019 |         final_estimator=LGBMClassifier(             n_estimators=100,           

### Iteration 10 -- IMPROVED (+0.0355)
Score: -1.5281 | FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interacti
Commit: 1ec5f0f

### Iteration 11 -- REVERTED (-0.0355)
Score: -1.5636 | FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in

### Iteration 12 -- REVERTED (-3.2280)
Score: -4.7561 |         passthrough=True, 

### Iteration 1 -- IMPROVED (+0.1005)
Score: -1.4276 |         n_estimators=1000,         learning_rate=0.016,         n_estimators=150
Commit: 52fd57f

### Iteration 2 -- REVERTED (-0.2450)
Score: -1.6726 | from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier  # noqa: E40

### Iteration 3 -- REVERTED (-0.0999)
Score: -1.5275 |         cv=3, 

### Iteration 4 -- REVERTED (-0.0495)
Score: -1.4771 |         n_estimators=1300,         learning_rate=0.012,         n_estimators=200

### Iteration 5 -- REVERTED (-0.0068)
Score: -1.4344 | from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, Stack

### Iteration 6 -- REVERTED (-0.0705)
Score: -1.4981 |     # Cross-sectional z-score of 10-day return: how oversold is this stock     #

### Iteration 7 -- GATE FAILED
Reason: GATE VIOLATION: research/model_wrappers.py was modified. Reverting.
Change:     # DART boosting: tree dropout decorrelates trees, complementary to standard 

### Iteration 8 -- REVERTED (-0.2459)
Score: -1.6735 |     # DART boosting: tree dropout decorrelates trees, complementary to standard 

### Iteration 9 -- REVERTED (-0.0673)
Score: -1.4949 | from sklearn.neural_network import MLPClassifier  # noqa: E402     # Neural dive

### Iteration 10 -- REVERTED (-0.0035)
Score: -1.4311 |     spw = (neg / pos) ** 0.4  # conservative weight (~3.3) for higher precision 

### Iteration 11 -- IMPROVED (+0.1405)
Score: -1.2871 | from sklearn.ensemble import StackingClassifier  # noqa: E402 from research.mode
Commit: b6d5dbf

### Iteration 12 -- REVERTED (-0.0613)
Score: -1.3484 |         objective="rank:map", 

### Iteration 13 -- REVERTED (-0.0101)
Score: -1.2972 |         group_size=150,         n_estimators=800,         max_depth=6,         l

### Iteration 14 -- REVERTED (-0.0877)
Score: -1.3748 |         group_size=1000, 

### Iteration 15 -- REVERTED (-0.0413)
Score: -1.3284 |         final_estimator=LogisticRegression(C=0.1, max_iter=1000), 

### Iteration 16 -- REVERTED (-0.0259)
Score: -1.3130 |         final_estimator=LogisticRegression(C=1.0, max_iter=1000, class_weight="b

### Iteration 17 -- REVERTED (-0.0741)
Score: -1.3612 |     g = df.groupby("stock_id")      prev_close = g["close"].shift(1)     df["ove

### Iteration 18 -- REVERTED (-0.3023)
Score: -1.5894 | FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"] 

### Iteration 19 -- REVERTED (-0.0163)
Score: -1.3034 |         max_depth=9,         min_child_weight=50,         gamma=2.0,         reg

### Iteration 20 -- REVERTED (-0.1231)
Score: -1.4102 | from sklearn.feature_selection import SelectFromModel  # noqa: E402 from sklearn

### Iteration 21 -- REVERTED (-0.0344)
Score: -1.3215 |         n_estimators=1000,         max_depth=-1,         num_leaves=31,         

### Iteration 1 -- REVERTED (-1.9362)
Score: -3.2063 |     from research.model_wrappers import FocalTorchClassifier, TorchMLP     neg =

### Iteration 2 -- REVERTED (-0.8576)
Score: -2.1277 |     model = XGBClassifier( 

### Iteration 3 -- REVERTED (-0.0147)
Score: -1.2848 | from research.model_wrappers import CatBoostWrapper, RankingXGBClassifier  # noq

### Iteration 4 -- REVERTED (-0.8576)
Score: -2.1277 |     model = XGBClassifier( 

### Iteration 5 -- REVERTED (-0.0403)
Score: -1.3104 | from research.model_wrappers import CatBoostWrapper, RankingXGBClassifier  # noq

### Iteration 6 -- KNOWLEDGE
No code changes.

### Iteration 1 -- KNOWLEDGE
No code changes.

### Iteration 1 -- IMPROVED (+0.0195)
Score: -1.2506 |         reg_alpha=0.5,         colsample_bytree=0.7, 
Commit: 88a4cef

### Iteration 2 -- REVERTED (-0.0024)
Score: -1.2530 |         max_depth=0,         max_leaves=128,         grow_policy="lossguide", 

### Iteration 3 -- REVERTED (-0.0064)
Score: -1.2570 |         min_child_weight=15,         gamma=0.3, 

### Iteration 4 -- REVERTED (-0.1288)
Score: -1.3794 |     g_dd = df.groupby("stock_id")["drawdown"]     df["drawdown_velocity_5d"] = d

### Iteration 5 -- REVERTED (-0.0425)
Score: -1.2931 |     spw = (neg / pos) ** 0.6 

### Iteration 6 -- REVERTED (-0.1773)
Score: -1.4279 |     gr = df.groupby("stock_id")["ret_1d"]     vol_5 = gr.transform(lambda x: x.r

### Iteration 7 -- REVERTED (-0.1232)
Score: -1.3738 | FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in

### Iteration 8 -- REVERTED (-3.1663)
Score: -4.4169 |         passthrough=True, 

### Iteration 9 -- REVERTED (-0.4414)
Score: -1.6920 | from sklearn.ensemble import VotingClassifier  # noqa: E402     model = VotingCl

### Iteration 10 -- REVERTED (-0.1762)
Score: -1.4268 |         n_estimators=800, 

### Iteration 11 -- IMPROVED (+0.0343)
Score: -1.2163 |         cv=3, 
Commit: 92461be

### Iteration 12 -- IMPROVED (+0.0297)
Score: -1.1866 |         reg_lambda=1.0, 
Commit: 69232c6

### Iteration 13 -- REVERTED (-0.0060)
Score: -1.1926 |         gamma=0.2, 

### Iteration 14 -- REVERTED (-0.0314)
Score: -1.2180 |         colsample_bylevel=0.7, 

### Iteration 15 -- REVERTED (-0.0321)
Score: -1.2187 |         subsample=0.80, 

### Iteration 16 -- REVERTED (-0.0545)
Score: -1.2411 |         min_child_weight=15, 

### Iteration 17 -- REVERTED (-0.0424)
Score: -1.2290 |         max_delta_step=1, 

### Iteration 18 -- REVERTED (-0.0283)
Score: -1.2149 |         cv=5, 

### Iteration 19 -- IMPROVED (+0.0458)
Score: -1.1408 |         reg_lambda=0.5, 
Commit: dd6b99b

### Iteration 20 -- REVERTED (-0.1285)
Score: -1.2693 |         n_estimators=1400,         learning_rate=0.01, 

### Iteration 21 -- REVERTED (-0.0943)
Score: -1.2351 |         reg_lambda=0.5, 

### Iteration 22 -- REVERTED (-0.0681)
Score: -1.2089 |         max_depth=6, 

### Iteration 23 -- REVERTED (-0.1201)
Score: -1.2609 |         reg_lambda=1.5, 

### Iteration 24 -- REVERTED (-0.0711)
Score: -1.2119 |         reg_alpha=0.3, 

### Iteration 25 -- REVERTED (-0.1043)
Score: -1.2451 |         gamma=0.5, 

### Iteration 26 -- REVERTED (-0.0286)
Score: -1.1694 |         final_estimator=LogisticRegression(C=5.0, max_iter=1000), 

### Iteration 27 -- REVERTED (-0.0833)
Score: -1.2241 |         learning_rate=0.02, 

### Iteration 28 -- REVERTED (-0.1773)
Score: -1.3181 |         max_depth=4, 

### Iteration 29 -- REVERTED (-0.2288)
Score: -1.3696 |     g = df.groupby("stock_id")     avg_vol = g["volume"].transform(lambda x: x.r

### Iteration 30 -- IMPROVED (+0.0062)
Score: -1.1346 |         learning_rate=0.02, 
Commit: fe6fb24

### Iteration 31 -- REVERTED (-0.0432)
Score: -1.1778 |         max_depth=8, 

### Iteration 32 -- REVERTED (-0.0476)
Score: -1.1822 | from lightgbm import LGBMClassifier  # noqa: E402     lgb = LGBMClassifier(     

### Iteration 33 -- IMPROVED (+0.0023)
Score: -1.1323 |         min_child_weight=12, 
Commit: e637072

### Iteration 34 -- REVERTED (-0.0811)
Score: -1.2134 |         max_depth=6, 

### Iteration 35 -- IMPROVED (+0.0040)
Score: -1.1283 |         min_child_weight=8, 
Commit: a882931

### Iteration 36 -- IMPROVED (+0.0011)
Score: -1.1272 |         min_child_weight=5, 
Commit: dace9c5

### Iteration 37 -- IMPROVED (+0.0031)
Score: -1.1241 |         min_child_weight=3, 
Commit: abb10dd

### Iteration 38 -- REVERTED (-0.0229)
Score: -1.1470 |         min_child_weight=1, 

### Iteration 39 -- IMPROVED (+0.0041)
Score: -1.1200 |         gamma=0.4, 
Commit: f3ab845

### Iteration 40 -- REVERTED (-0.2212)
Score: -1.3412 |         group_size=150, 

### Iteration 41 -- REVERTED (-0.0270)
Score: -1.1470 |         reg_alpha=0.4, 

### Iteration 42 -- REVERTED (-0.0911)
Score: -1.2111 |         objective="rank:map", 

### Iteration 43 -- REVERTED (-0.0466)
Score: -1.1666 | from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E

### Iteration 44 -- REVERTED (-0.0401)
Score: -1.1601 |         colsample_bytree=0.65, 

### Iteration 45 -- REVERTED (-0.1766)
Score: -1.2966 |         learning_rate=0.015, 

### Iteration 46 -- REVERTED (-0.0218)
Score: -1.1418 |         reg_lambda=0.3, 

### Iteration 47 -- REVERTED (-0.1188)
Score: -1.2388 | FEATURE_GROUPS = ["base", "advanced", "lag", "rolling", "roc", "percentile", "in

### Iteration 48 -- REVERTED (-0.0223)
Score: -1.1423 |         gamma=0.3, 

### Iteration 49 -- REVERTED (-0.0323)
Score: -1.1523 |         subsample=0.70, 

### Iteration 50 -- REVERTED (-0.0232)
Score: -1.1432 |         n_estimators=1200, 
