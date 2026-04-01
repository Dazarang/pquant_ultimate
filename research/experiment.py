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


class _EarlyStopCB:
    """CatBoost with internal early stopping on a temporal holdout."""

    def __init__(self, val_frac, **kwargs):
        self._val_frac = val_frac
        self._kwargs = kwargs

    def fit(self, X, y):
        from catboost import CatBoostClassifier, Pool
        n = len(X)
        cut = int(n * (1 - self._val_frac))
        self._model = CatBoostClassifier(**self._kwargs)
        self._model.fit(Pool(X[:cut], y[:cut]), eval_set=Pool(X[cut:], y[cut:]))
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def build_model(y_train):
    """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = np.sqrt(neg / pos)

    return _EarlyStopCB(
        val_frac=0.1,
        iterations=3000,
        depth=6,
        learning_rate=0.01,
        min_data_in_leaf=50,
        boosting_type="Ordered",
        rsm=0.6,
        l2_leaf_reg=3.0,
        posterior_sampling=True,
        scale_pos_weight=spw,
        random_seed=42,
        verbose=0,
        thread_count=-1,
        od_type="Iter",
        od_wait=200,
        use_best_model=True,
    )


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
