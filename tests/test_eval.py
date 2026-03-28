"""Tests for lib/eval.py -- 3-tier evaluation framework."""

import numpy as np
import pandas as pd
import pytest

from lib.data import LABEL_COL, load_dataset, scale, temporal_split
from lib.eval import (
    backtest_quick,
    benchmark_random_entry,
    composite_score,
    evaluate,
    forward_returns,
    plot_confusion,
    print_report,
    tiered_eval,
)

DS_PATH = "data/datasets/20260115/dataset.parquet"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def val_split():
    """Load 3-stock dataset, split, return unscaled val + scaled val."""
    df, fc = load_dataset(DS_PATH, stocks=["AAPL", "MSFT", "TSLA"])
    train, val, test = temporal_split(df)
    train_s, val_s, test_s, _ = scale(train, val, test, fc)
    return val, val_s, fc


@pytest.fixture(scope="module")
def simulated_preds(val_split):
    """Simulate imperfect model predictions on val set."""
    val, val_s, fc = val_split
    y_val = val[LABEL_COL].values

    np.random.seed(42)
    y_proba = np.random.rand(len(y_val)) * 0.1
    y_proba[y_val == 1] += np.random.rand((y_val == 1).sum()) * 0.7
    y_pred = (y_proba > 0.3).astype(int)

    return y_val, y_pred, y_proba


# ---------------------------------------------------------------------------
# Tier 1: Classification metrics
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_required_keys(self, simulated_preds):
        y_true, y_pred, y_proba = simulated_preds
        result = evaluate(y_true, y_pred, y_proba)
        for key in ["precision", "recall", "f1", "roc_auc", "avg_precision"]:
            assert key in result

    def test_without_proba(self, simulated_preds):
        y_true, y_pred, _ = simulated_preds
        result = evaluate(y_true, y_pred)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "roc_auc" not in result

    def test_values_in_range(self, simulated_preds):
        y_true, y_pred, y_proba = simulated_preds
        result = evaluate(y_true, y_pred, y_proba)
        for v in result.values():
            assert -1e-9 <= v <= 1.0 + 1e-9

    def test_all_zeros_pred(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        result = evaluate(y_true, y_pred)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_perfect_pred(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = evaluate(y_true, y_pred)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0


class TestPrintReport:
    def test_runs_without_error(self, simulated_preds, capsys):
        y_true, y_pred, _ = simulated_preds
        print_report(y_true, y_pred)
        output = capsys.readouterr().out
        assert "precision" in output
        assert "recall" in output


class TestPlotConfusion:
    def test_returns_figure(self, simulated_preds):
        import matplotlib

        matplotlib.use("Agg")
        y_true, y_pred, _ = simulated_preds
        fig = plot_confusion(y_true, y_pred)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# Tier 2: Forward returns
# ---------------------------------------------------------------------------


class TestForwardReturns:
    def test_returns_dict(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        result = forward_returns(val, y_pred)
        assert isinstance(result, dict)

    def test_default_horizons(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        result = forward_returns(val, y_pred)
        for h in [5, 10, 20]:
            assert f"mean_{h}d" in result
            assert f"win_rate_{h}d" in result
            assert f"profit_factor_{h}d" in result
            assert f"n_signals_{h}d" in result

    def test_custom_horizons(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        result = forward_returns(val, y_pred, horizons=[3, 7])
        assert "mean_3d" in result
        assert "mean_7d" in result
        assert "mean_10d" not in result

    def test_no_signals_returns_empty(self, val_split):
        val, _, _ = val_split
        y_pred = np.zeros(len(val), dtype=int)
        result = forward_returns(val, y_pred)
        assert result == {}

    def test_win_rate_bounded(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        result = forward_returns(val, y_pred)
        for k, v in result.items():
            if "win_rate" in k:
                assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Tier 3: Composite score
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_returns_float(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        score = composite_score(val, y_pred)
        assert isinstance(score, float)

    def test_no_signals_returns_neg_inf(self, val_split):
        val, _, _ = val_split
        y_pred = np.zeros(len(val), dtype=int)
        score = composite_score(val, y_pred)
        assert score == float("-inf")

    def test_custom_horizon(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        s5 = composite_score(val, y_pred, horizon=5)
        s20 = composite_score(val, y_pred, horizon=20)
        # Just verify both compute without error; values may differ
        assert isinstance(s5, float)
        assert isinstance(s20, float)


# ---------------------------------------------------------------------------
# Tiered eval (full funnel)
# ---------------------------------------------------------------------------


class TestTieredEval:
    def test_full_pass(self, val_split, simulated_preds):
        val, _, _ = val_split
        y_true, y_pred, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_pred, y_proba)
        assert "tier1" in result
        # tier2/tier3 depend on simulated data quality

    def test_early_stop_low_ap(self, val_split):
        val, _, _ = val_split
        y_true = val[LABEL_COL].values
        # Random noise predictions -- AP should be very low
        np.random.seed(99)
        y_pred = (np.random.rand(len(y_true)) > 0.97).astype(int)
        y_proba = np.random.rand(len(y_true)) * 0.05
        result = tiered_eval(val, y_true, y_pred, y_proba, min_ap=0.5)
        assert result["passed"] is False
        assert "tier1" in result
        assert "tier2" not in result

    def test_result_structure(self, val_split, simulated_preds):
        val, _, _ = val_split
        y_true, y_pred, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_pred, y_proba)
        assert "passed" in result
        assert isinstance(result["passed"], bool)
        assert isinstance(result["tier1"], dict)


# ---------------------------------------------------------------------------
# Backtest quick
# ---------------------------------------------------------------------------


class TestBacktestQuick:
    def test_returns_dataframe(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        df = val.copy()
        df["prediction"] = y_pred
        trades = backtest_quick(df)
        assert isinstance(trades, pd.DataFrame)

    def test_trade_columns(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        df = val.copy()
        df["prediction"] = y_pred
        trades = backtest_quick(df)
        if len(trades) > 0:
            expected = ["stock_id", "entry_date", "exit_date", "entry_price",
                        "exit_price", "return_pct", "exit_reason"]
            for c in expected:
                assert c in trades.columns

    def test_exit_reasons_valid(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        df = val.copy()
        df["prediction"] = y_pred
        trades = backtest_quick(df)
        if len(trades) > 0:
            assert set(trades["exit_reason"].unique()).issubset({"target", "stop", "max_hold"})

    def test_no_signals_empty(self, val_split):
        val, _, _ = val_split
        df = val.copy()
        df["prediction"] = 0
        trades = backtest_quick(df)
        assert len(trades) == 0

    def test_custom_params(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        df = val.copy()
        df["prediction"] = y_pred
        trades = backtest_quick(df, target_pct=0.05, stop_pct=0.03, max_hold_days=10)
        assert isinstance(trades, pd.DataFrame)


# ---------------------------------------------------------------------------
# Benchmark: random entry
# ---------------------------------------------------------------------------


class TestBenchmarkRandomEntry:
    def test_returns_dict(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        result = benchmark_random_entry(val, y_pred, n_simulations=50)
        assert isinstance(result, dict)

    def test_required_keys(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        result = benchmark_random_entry(val, y_pred, n_simulations=50)
        for key in ["model_mean", "random_mean", "excess_return", "z_score", "p_value", "significant"]:
            assert key in result

    def test_p_value_bounded(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, y_pred, _ = simulated_preds
        result = benchmark_random_entry(val, y_pred, n_simulations=50)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_no_signals_returns_error(self, val_split):
        val, _, _ = val_split
        y_pred = np.zeros(len(val), dtype=int)
        result = benchmark_random_entry(val, y_pred, n_simulations=10)
        assert "error" in result

    def test_actual_labels_significant(self, val_split):
        """Actual PivotLow labels should beat random entry."""
        val, _, _ = val_split
        y_pred = val[LABEL_COL].values
        if y_pred.sum() < 5:
            pytest.skip("Not enough signals in val")
        result = benchmark_random_entry(val, y_pred, n_simulations=100)
        assert result["excess_return"] > 0
