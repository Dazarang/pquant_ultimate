#!/usr/bin/env python3
"""Autoresearch experiment -- the ONLY file (along with features_lab.py) the researcher edits.

The researcher can change anything in the CONFIGURATION and MODEL sections.
The PIPELINE section is fixed -- do not edit.

Prints "COMPOSITE_SCORE=X.XX" as the last line for gate.sh to extract.
"""

import sys
from pathlib import Path

import numpy as np  # noqa: F401 -- available for researcher
from xgboost import XGBClassifier

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
FEATURE_GROUPS = None  # None = all

# Temporal split boundaries
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

# Prediction threshold
THRESHOLD = 0.85

# ===========================================================================
# MODEL -- researcher edits this section
# ===========================================================================


def build_model(y_train):
    """Return a fitted-ready model. Researcher chooses model type and hyperparams."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
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

    # 8. Predict on val (unscaled val for forward returns)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > THRESHOLD).astype(int)
    print(f"Predictions: {y_pred.sum()} signals / {len(y_pred)} total")

    # 9. Tiered evaluation
    results = tiered_eval(val, y_val, y_pred, y_pred_proba)

    # 10. Output final metric for gate.sh
    score = results.get("tier3", float("-inf"))
    passed = results.get("passed", False)
    print(f"\nPASSED={passed}")
    print(f"COMPOSITE_SCORE={score:.4f}")
    return results


if __name__ == "__main__":
    run()
