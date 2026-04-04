"""Fast smoke tests for the eval stack on a small real-data sample."""

import numpy as np

from lib.data import LABEL_COL, list_features, load_dataset, temporal_split
from lib.eval import backtest_quick, benchmark_random_entry, select_top_frac, tiered_eval

DS_PATH = "data/datasets/20260115/dataset.parquet"
STOCKS = ["AAPL", "MSFT", "TSLA"]
FEATURES = list_features(["base", "advanced", "roc", "percentile", "interaction"])


def test_eval_stack_smoke_on_real_sample():
    """Run the main eval components on a small parquet-backed split."""
    df, _ = load_dataset(DS_PATH, stocks=STOCKS, features=FEATURES)
    _, val, _ = temporal_split(df)

    y_true = val[LABEL_COL].to_numpy()
    rng = np.random.default_rng(42)
    y_proba = rng.random(len(val)) * 0.05
    y_proba[y_true == 1] += rng.random((y_true == 1).sum()) * 0.6 + 0.2

    results = tiered_eval(val, y_true, y_proba, min_ap=0.01)
    assert "tier1" in results
    assert "tier2" in results
    assert "tier3" in results

    y_pred = select_top_frac(y_proba, 0.05)

    benchmark = benchmark_random_entry(val, y_pred, horizon=10, n_simulations=20)
    assert "model_mean" in benchmark
    assert benchmark["n_signals"] >= 1

    val_bt = val.copy()
    val_bt["prediction"] = y_pred
    trades = backtest_quick(val_bt, max_hold_days=10)
    assert trades is not None
