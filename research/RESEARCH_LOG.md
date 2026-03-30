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
