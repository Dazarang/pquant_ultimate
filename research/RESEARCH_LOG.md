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
