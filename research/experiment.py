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
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from research.model_wrappers import CatBoostWrapper  # noqa: E402

from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split  # noqa: E402
from lib.eval import tiered_eval  # noqa: E402
from research.features_lab import add_custom_features  # noqa: E402

# ===========================================================================
# CONFIGURATION -- researcher edits this section
# ===========================================================================

DATASET_PATH = "data/datasets/20260115/dataset.parquet"

# Stock universe: None = all, "AAPL" = single, ["AAPL", "MSFT", ...] = subset
STOCKS = None

# Feature groups: see list_features() for options
FEATURE_GROUPS = ["base", "advanced", "rolling", "roc", "percentile", "interaction"]

# Temporal split boundaries
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

# ===========================================================================
# MODEL -- researcher edits this section
# ===========================================================================


def build_model(y_train):
    """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = np.sqrt(neg / pos)  # moderate weight (~4.4) instead of full ratio (~19)

    # Deep, heavily regularized — captures complex interactions
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.02,
        min_child_weight=20,
        gamma=0.5,
        reg_alpha=2.0,
        reg_lambda=2.0,
        subsample=0.65,
        colsample_bytree=0.5,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # Shallow, smooth, many trees — captures gradual trends
    lgbm = LGBMClassifier(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.01,
        min_child_samples=80,
        reg_alpha=0.5,
        reg_lambda=0.5,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=spw,
        random_state=43,
        n_jobs=-1,
        verbose=-1,
    )

    # Medium depth, heavy L2 — balanced generalization
    cat = CatBoostWrapper(
        iterations=800,
        depth=5,
        learning_rate=0.025,
        l2_leaf_reg=10.0,
        scale_pos_weight=spw,
        random_seed=44,
        verbose=0,
    )

    # Bagging-based diversity: random splits + bootstrap = uncorrelated errors
    extra = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight={0: 1, 1: spw},
        random_state=45,
        n_jobs=-1,
    )

    model = StackingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("extra", extra)],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=2,
        n_jobs=1,
    )
    return model


# ===========================================================================
# PIPELINE -- do not edit below this line
# ===========================================================================


def run():
    # 1. Load dataset
    features = list_features(FEATURE_GROUPS)
    df, feature_cols = load_dataset(DATASET_PATH, stocks=STOCKS, features=features)

    # 2. Apply custom features from lab
    df, new_features = add_custom_features(df)
    feature_cols = feature_cols + new_features

    # 3. Clean any new NaN/inf from custom features
    df[new_features] = df[new_features].replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # 4. Split
    train, val, test = temporal_split(df, train_end=TRAIN_END, val_end=VAL_END)

    # 5. Scale
    train_s, val_s, test_s, scaler = scale(train, val, test, feature_cols)

    # 6. Extract X/y
    X_train = train_s[feature_cols].values
    y_train = train_s[LABEL_COL].values
    X_val = val_s[feature_cols].values
    y_val = val_s[LABEL_COL].values

    # 7. Train
    print(f"\nTraining on {X_train.shape[0]:,} rows, {X_train.shape[1]} features...")
    model = build_model(y_train)
    model.fit(X_train, y_train)

    # 8. Predict on val
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    print(f"Predictions: proba range [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")

    # 9. Tiered evaluation (multi-budget, threshold-free)
    results = tiered_eval(val, y_val, y_pred_proba)

    # 10. Output final metric for gate.sh
    score = results.get("tier3", float("-inf"))
    passed = results.get("passed", False)
    print(f"\nPASSED={passed}")
    print(f"COMPOSITE_SCORE={score:.4f}")
    return results


if __name__ == "__main__":
    run()
