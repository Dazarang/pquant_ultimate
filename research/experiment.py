#!/usr/bin/env python3
"""Autoresearch experiment -- the ONLY file (along with features_lab.py) the researcher edits.

The researcher can change anything in the CONFIGURATION and MODEL sections.
The PIPELINE section is fixed -- do not edit.

Prints "COMPOSITE_SCORE=X.XX" as the last line for gate.sh to extract.
"""

import sys
from pathlib import Path

# Ensure project root is on path (must precede all local imports)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: F401,E402 -- available for researcher

from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
from lib.eval import tiered_eval  # noqa: E402
from research.features_lab import add_custom_features  # noqa: E402

# ===========================================================================
# DATA -- fixed, do not edit
# ===========================================================================

DATASET_PATH = "data/datasets/20260331/dataset.parquet"

# ===========================================================================
# CONFIGURATION -- researcher edits this section
# ===========================================================================

# Stock universe: None = all, "AAPL" = single, ["AAPL", "MSFT", ...] = subset
STOCKS = None

# Feature groups: see list_features() for options. None = all
FEATURE_GROUPS = ["base", "advanced", "roc", "percentile", "interaction"]

# ===========================================================================
# MODEL -- researcher edits this section
# ===========================================================================


def build_model(y_train):
    """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    import xgboost as xgb

    class _EnsembleModel:
        def __init__(self):
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            n = len(y)
            ev_mask = np.arange(n) % 10 == 0
            tr_idx = np.where(~ev_mask)[0]
            ev_idx = np.where(ev_mask)[0]

            self._cat = CatBoostClassifier(
                iterations=3000,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=5.0,
                random_strength=1.0,
                bagging_temperature=0.8,
                border_count=128,
                scale_pos_weight=5,
                use_best_model=True,
                early_stopping_rounds=150,
                verbose=200,
                task_type='CPU',
                thread_count=-1,
            )
            self._cat.fit(
                X[tr_idx], y[tr_idx],
                eval_set=(X[ev_idx], y[ev_idx]),
            )
            print(f"CatBoost best iteration: {self._cat.best_iteration_}")

            dtrain = lgb.Dataset(X[tr_idx], y[tr_idx])
            deval = lgb.Dataset(X[ev_idx], y[ev_idx], reference=dtrain)
            self._lgb = lgb.train(
                {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': 63,
                    'learning_rate': 0.02,
                    'feature_fraction': 0.6,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'scale_pos_weight': 5,
                    'lambda_l2': 5.0,
                    'min_child_samples': 20,
                    'verbose': -1,
                },
                dtrain,
                num_boost_round=3000,
                valid_sets=[deval],
                callbacks=[
                    lgb.early_stopping(150),
                    lgb.log_evaluation(200),
                ],
            )
            print(f"LightGBM best iteration: {self._lgb.best_iteration}")

            self._xgb = xgb.XGBClassifier(
                n_estimators=3000,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.6,
                scale_pos_weight=5,
                reg_lambda=5.0,
                min_child_weight=5,
                tree_method='hist',
                early_stopping_rounds=150,
                eval_metric='logloss',
                verbosity=1,
            )
            self._xgb.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[ev_idx], y[ev_idx])],
                verbose=200,
            )
            print(f"XGBoost best iteration: {self._xgb.best_iteration}")

            return self

        def predict_proba(self, X):
            cat_p = self._cat.predict_proba(X)[:, 1]
            lgb_p = self._lgb.predict(X)
            xgb_p = self._xgb.predict_proba(X)[:, 1]
            avg = (cat_p + lgb_p + xgb_p) / 3
            return np.column_stack([1 - avg, avg])

    return _EnsembleModel()


# ===========================================================================
# PIPELINE -- do not edit below this line
# ===========================================================================

# Walk-forward folds: (train_end, val_end). Non-overlapping 6-month val windows.
# 4 equal 6-month folds, aligned backwards from data end. 2 years OOS coverage.
_WF_FOLDS = [
    ("2024-03-15", "2024-09-15"),
    ("2024-09-15", "2025-03-15"),
    ("2025-03-15", "2025-09-15"),
    ("2025-09-15", "2026-03-15"),
]


def run():
    features = list_features(FEATURE_GROUPS)
    df, feature_cols = load_dataset(DATASET_PATH, stocks=STOCKS, features=features)

    df, new_features = add_custom_features(df)
    feature_cols = feature_cols + new_features

    df[new_features] = df[new_features].replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    df = df[df["date"] >= "2021-01-01"].reset_index(drop=True)

    fold_scores = []
    for fold_idx, (train_end, val_end) in enumerate(_WF_FOLDS, 1):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx}/{len(_WF_FOLDS)}: train_end={train_end}, val_end={val_end}")
        print(f"{'=' * 60}")

        train, val, test = temporal_split(df, train_end=train_end, val_end=val_end, include_test=False)

        if val.empty or train.empty:
            print(f"  FOLD {fold_idx}: skipped (empty split)")
            fold_scores.append(float("-inf"))
            continue

        train_s, val_s, _, _ = scale(train, val, test, feature_cols)

        X_train = train_s[feature_cols].values
        y_train = train_s[LABEL_COL].values
        X_val = val_s[feature_cols].values
        y_val = val_s[LABEL_COL].values

        print(f"Training on {X_train.shape[0]:,} rows, {X_train.shape[1]} features...")
        model = build_model(y_train)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        print(f"Predictions: proba range [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")

        results = tiered_eval(val, y_val, y_pred_proba)
        fold_score = results.get("tier3", float("-inf"))
        fold_scores.append(fold_score)
        print(f"FOLD_{fold_idx}_SCORE={fold_score:.4f}")

    score = np.mean(fold_scores)
    passed = all(s != float("-inf") for s in fold_scores)

    print(f"\nFold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"\nPASSED={passed}")
    print(f"COMPOSITE_SCORE={score:.4f}")
    return {"fold_scores": fold_scores, "score": score, "passed": passed}


if __name__ == "__main__":
    run()
