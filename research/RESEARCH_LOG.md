# Research Log

### Iteration 1 -- REVERTED (-0.4645)
Score: 0.3611 | THRESHOLD = 0.3 

### Iteration 2 -- REVERTED (-0.9238)
Score: -0.0982 | FEATURE_GROUPS = ["base", "advanced"] 

### Iteration 3 -- REVERTED (-0.3209)
Score: 0.5047 |         n_estimators=500,         max_depth=4,         learning_rate=0.03,      

### Iteration 4 -- IMPROVED (+0.9187)
Score: 1.7443 | from lightgbm import LGBMClassifier     model = LGBMClassifier(         num_leav
Commit: e347248

### Iteration 5 -- IMPROVED (+0.6668)
Score: 2.4111 |     g = df.groupby("stock_id")      # Consecutive down days: count of consecutiv
Commit: dedaeb3

### Iteration 6 -- REVERTED (-1.4405)
Score: 0.9706 |     # Price deceleration: short-term return improving vs medium-term     # Posit

### Iteration 7 -- REVERTED (-0.8575)
Score: 1.5536 | FEATURE_GROUPS = ["base", "percentile"] 

### Iteration 8 -- REVERTED (-5.8002)
Score: -3.3891 | STOCKS = None 

### Iteration 9 -- REVERTED (-0.2937)
Score: 2.1174 |         num_leaves=15, 

### Iteration 10 -- REVERTED (-0.1865)
Score: 2.2246 | FEATURE_GROUPS = ["base", "roc"] 

### Iteration 11 -- REVERTED (-0.1220)
Score: 2.2891 | THRESHOLD = 0.55 

### Iteration 12 -- REVERTED (-2.2516)
Score: 0.1595 |         boosting_type="dart", 

### Iteration 13 -- REVERTED (-0.1038)
Score: 2.3073 |         reg_alpha=0.1,         reg_lambda=2.0, 

### Iteration 14 -- REVERTED (-1.0826)
Score: 1.3285 |         n_estimators=500,         learning_rate=0.03, 

### Iteration 15 -- IMPROVED (+0.2741)
Score: 2.6852 |         max_depth=7, 
Commit: 84dc027

### Iteration 16 -- IMPROVED (+1.5123)
Score: 4.1975 |         num_leaves=63, 
Commit: ce0f6c2

### Iteration 17 -- REVERTED (-0.8324)
Score: 3.3651 |         min_child_samples=30, 

### Iteration 18 -- REVERTED (+0.0000)
Score: 4.1975 |         num_leaves=95, 

### Iteration 19 -- REVERTED (-0.0060)
Score: 4.1915 |         n_estimators=400, 

### Iteration 20 -- REVERTED (-0.7323)
Score: 3.4652 |         colsample_bytree=0.6, 
