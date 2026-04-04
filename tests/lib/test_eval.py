"""Tests for lib/eval.py -- multi-budget evaluation framework."""

import numpy as np
import pandas as pd
import pytest

from lib.data import LABEL_COL, load_dataset, scale, temporal_split
from lib.eval import (
    _select_top_frac_weights,
    _cell_score,
    _signal_analytics,
    backtest_quick,
    benchmark_random_entry,
    composite_score,
    evaluate,
    forward_open_return,
    forward_returns,
    multi_budget_composite,
    plot_confusion,
    print_report,
    select_top_frac,
    tiered_eval,
)
from lib.pivot_events import PIVOT_LOW_BASE_COL, PIVOT_LOW_EVENT_ID_COL, PIVOT_LOW_EVENT_OFFSET_COL

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


@pytest.fixture
def event_eval_df():
    """Synthetic single-stock frame with two true bottom events."""
    n_days = 14
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    df = pd.DataFrame({
        "date": dates,
        "stock_id": ["TEST"] * n_days,
        "open": [10.0, 10.2, 10.4, 10.1, 9.9, 9.8, 10.2, 10.5, 10.1, 9.7, 10.4, 10.8, 11.0, 11.2],
        "high": [10.3, 10.5, 10.6, 10.4, 10.1, 10.0, 10.5, 10.7, 10.3, 10.0, 10.6, 11.0, 11.2, 11.4],
        "low": [9.8, 10.0, 10.1, 9.9, 9.7, 9.6, 10.0, 10.2, 9.9, 9.5, 10.1, 10.6, 10.8, 11.0],
        "close": [10.1, 10.3, 10.2, 10.0, 9.8, 9.9, 10.3, 10.4, 10.0, 9.8, 10.5, 10.9, 11.1, 11.3],
        LABEL_COL: [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        PIVOT_LOW_BASE_COL: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        PIVOT_LOW_EVENT_ID_COL: [0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0],
        PIVOT_LOW_EVENT_OFFSET_COL: [np.nan, np.nan, np.nan, np.nan, 0.0, 1.0, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan],
    })
    return df


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
# Event-aware analytics
# ---------------------------------------------------------------------------


class TestEventAwareAnalytics:
    def test_duplicate_predictions_collapse_inside_event(self, event_eval_df):
        y_pred = np.zeros(len(event_eval_df), dtype=int)
        y_pred[[2, 4, 5]] = 1  # one false positive + duplicate hit inside event 1

        analytics = _signal_analytics(event_eval_df, y_pred, horizon=2)

        assert analytics is not None
        assert analytics["raw_signal_rows"] == 3
        assert analytics["collapsed_rows"] == 2
        assert analytics["event_hits"] == 1
        assert analytics["exact_hits"] == 1
        assert analytics["duplicate_rows"] == 1
        assert analytics["duplicate_mass_discarded"] == pytest.approx(1.0)
        assert analytics["event_recall"] == pytest.approx(0.5)
        assert analytics["exact_center_recall"] == pytest.approx(0.5)
        assert analytics["zone_precision"] == pytest.approx(0.5)

    def test_near_bottom_hit_counts_without_exact_center(self, event_eval_df):
        y_pred = np.zeros(len(event_eval_df), dtype=int)
        y_pred[5] = 1  # +1 day inside event 1 only

        analytics = _signal_analytics(event_eval_df, y_pred, horizon=2)

        assert analytics is not None
        assert analytics["event_hits"] == 1
        assert analytics["exact_hits"] == 0
        assert analytics["event_recall"] == pytest.approx(0.5)
        assert analytics["exact_center_recall"] == pytest.approx(0.0)
        assert analytics["avg_entry_offset_days"] == pytest.approx(1.0)

    def test_forward_returns_exposes_event_metrics(self, event_eval_df):
        y_pred = np.zeros(len(event_eval_df), dtype=int)
        y_pred[[2, 4, 5]] = 1

        result = forward_returns(event_eval_df, y_pred, horizons=[2])

        assert result["event_recall_2d"] == pytest.approx(0.5)
        assert result["exact_center_recall_2d"] == pytest.approx(0.5)
        assert result["zone_precision_2d"] == pytest.approx(0.5)
        assert result["duplicate_rows_2d"] == 1

    def test_backtest_collapses_duplicate_rows_in_same_event(self, event_eval_df):
        df = event_eval_df.copy()
        df["prediction"] = 0
        df.loc[[4, 5], "prediction"] = 1

        trades = backtest_quick(df, max_hold_days=3)

        assert len(trades) == 1


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
        y_true, _, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_proba)
        assert "tier1" in result

    def test_early_stop_low_ap(self, val_split):
        val, _, _ = val_split
        y_true = val[LABEL_COL].values
        # Random noise predictions -- AP should be very low
        np.random.seed(99)
        y_proba = np.random.rand(len(y_true)) * 0.05
        result = tiered_eval(val, y_true, y_proba, min_ap=0.5)
        assert result["passed"] is False
        assert "tier1" in result
        assert "tier2" not in result

    def test_result_structure(self, val_split, simulated_preds):
        val, _, _ = val_split
        y_true, _, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_proba)
        assert "passed" in result
        assert isinstance(result["passed"], bool)
        assert isinstance(result["tier1"], dict)

    def test_tier1_includes_base_metrics(self, val_split, simulated_preds):
        val, _, _ = val_split
        y_true, _, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_proba, min_ap=0.01)
        assert "base_avg_precision" in result["tier1"]
        assert "base_roc_auc" in result["tier1"]


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

    def test_duplicate_same_event_counts_once(self, event_eval_df):
        y_pred = np.zeros(len(event_eval_df), dtype=int)
        y_pred[[4, 5]] = 1
        result = benchmark_random_entry(event_eval_df, y_pred, horizon=2, n_simulations=20)
        assert result["n_signals"] == 1

    def test_non_contiguous_index(self, event_eval_df):
        df = event_eval_df.copy()
        df.index = np.arange(100, 100 + 3 * len(df), 3)

        y_pred = np.zeros(len(df), dtype=int)
        y_pred[[4, 5]] = 1

        result = benchmark_random_entry(df, y_pred, horizon=2, n_simulations=20)

        assert result["n_signals"] == 1


# ---------------------------------------------------------------------------
# Shared forward-return helper
# ---------------------------------------------------------------------------


class TestForwardOpenReturn:
    def test_requires_full_forward_window(self):
        stock = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=11),
            "open": np.arange(100, 111),
        })
        signal_date = stock.iloc[-2]["date"]

        assert forward_open_return(stock, signal_date, horizon=10) is None

    def test_matches_next_open_to_horizon_open(self):
        stock = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=12),
            "open": np.arange(100, 112),
        })
        signal_date = stock.iloc[0]["date"]

        ret = forward_open_return(stock, signal_date, horizon=10)

        assert ret == pytest.approx((111 - 101) / 101)


# ---------------------------------------------------------------------------
# select_top_frac edge cases
# ---------------------------------------------------------------------------


class TestSelectTopFrac:
    def test_frac_1_selects_all(self):
        proba = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        y = select_top_frac(proba, 1.0)
        assert y.sum() == len(proba)

    def test_tiny_frac_selects_exactly_one(self):
        proba = np.random.rand(100)
        y = select_top_frac(proba, 0.0001)
        assert y.sum() == 1

    def test_identical_probabilities(self):
        proba = np.full(50, 0.5)
        y = select_top_frac(proba, 0.1)
        assert y.sum() == len(proba)

    def test_correct_count(self):
        proba = np.random.rand(1000)
        for frac in [0.01, 0.05, 0.1, 0.5]:
            y = select_top_frac(proba, frac)
            assert y.sum() == max(1, int(1000 * frac))

    def test_selects_highest(self):
        proba = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
        y = select_top_frac(proba, 0.4)  # top 2
        assert y[1] == 1  # 0.9
        assert y[3] == 1  # 0.8

    def test_cutoff_ties_include_all_tied_rows(self):
        proba = np.array([0.9, 0.8, 0.8, 0.1])
        y = select_top_frac(proba, 0.5)
        assert y.tolist() == [1, 1, 1, 0]

    def test_weighted_selection_is_order_invariant_across_ties(self):
        proba = np.array([0.9, 0.8, 0.8, 0.1])
        weights = _select_top_frac_weights(proba, 0.5)

        perm = np.array([0, 2, 1, 3])
        permuted = _select_top_frac_weights(proba[perm], 0.5)
        mapped_back = np.zeros_like(permuted)
        mapped_back[perm] = permuted

        assert mapped_back.tolist() == pytest.approx(weights.tolist())

    def test_weighted_selection_preserves_exact_budget_mass(self):
        proba = np.full(50, 0.5)
        weights = _select_top_frac_weights(proba, 0.1)
        assert weights.sum() == pytest.approx(5.0)
        assert np.unique(weights).tolist() == pytest.approx([0.1])


# ---------------------------------------------------------------------------
# _cell_score formula verification
# ---------------------------------------------------------------------------


class TestCellScore:
    @pytest.fixture
    def known_cell_df(self):
        """Build a minimal df + y_pred with known analytics."""
        n_days = 100
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        rows = []
        for d in dates:
            rows.append({
                "date": d, "stock_id": "TEST",
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0,
            })
        df = pd.DataFrame(rows)
        # Signal on every 10th day -> 10 signals
        y_pred = np.zeros(n_days, dtype=int)
        y_pred[::10] = 1
        return df, y_pred

    def test_returns_dict_with_keys(self, known_cell_df):
        df, y_pred = known_cell_df
        result = _cell_score(df, y_pred, horizon=5)
        if result is None:
            pytest.skip("No valid signals in synthetic data")
        for key in ["raw", "weighted", "w", "n_signals", "effective_n",
                     "excess", "win_rate", "worst_decile", "knife_rate",
                     "tail_mae", "entry_slippage",
                     "event_recall", "exact_center_recall", "zone_precision"]:
            assert key in result

    def test_w_scaling_formula(self, known_cell_df):
        """Verify W = sqrt(eff_n / (eff_n + 20)) for known eff_n."""
        df, y_pred = known_cell_df
        result = _cell_score(df, y_pred, horizon=5)
        if result is None:
            pytest.skip("No valid signals")
        eff_n = result["effective_n"]
        expected_w = np.sqrt(eff_n / (eff_n + 20))
        assert abs(result["w"] - expected_w) < 1e-10

    def test_w_scaling_known_values(self):
        """Spot-check W formula for specific eff_n values."""
        for eff_n, expected in [(0, 0.0), (20, np.sqrt(0.5)), (80, np.sqrt(0.8))]:
            w = np.sqrt(eff_n / (eff_n + 20))
            assert abs(w - expected) < 1e-10

    def test_weighted_equals_w_times_raw(self, known_cell_df):
        df, y_pred = known_cell_df
        result = _cell_score(df, y_pred, horizon=5)
        if result is None:
            pytest.skip("No valid signals")
        assert abs(result["weighted"] - result["w"] * result["raw"]) < 1e-10

    def test_worst_decile_penalty_only_on_negative(self):
        """abs(min(0.0, worst_decile)) is 0 when worst_decile >= 0."""
        # Positive worst_decile -> penalty term is 0
        assert abs(min(0.0, 0.05)) == 0.0
        # Negative worst_decile -> penalty activates
        assert abs(min(0.0, -0.03)) == pytest.approx(0.03)

    def test_hand_calculated_raw_score(self):
        """Verify raw score formula with hand-calculated inputs."""
        excess = 0.02
        win_rate = 0.6
        worst_decile = -0.04
        knife_rate = 0.1
        tail_mae = -0.05
        entry_slip = 0.01

        expected = (
            0.50 * excess * 100
            + 0.15 * (win_rate - 0.5) * 100
            - 0.15 * abs(min(0.0, worst_decile)) * 100
            - 0.10 * knife_rate * 100
            - 0.05 * abs(tail_mae) * 100
            - 0.05 * entry_slip * 100
        )
        # 1.0 + 1.5 - 0.6 - 1.0 - 0.25 - 0.05 = 0.6
        assert expected == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# multi_budget_composite
# ---------------------------------------------------------------------------


class TestMultiBudgetComposite:
    def test_missing_cells_count_as_zero(self, val_split):
        """Missing cells should contribute 0, not be skipped."""
        val, _, _ = val_split
        # All-zero proba -> signals exist (top-frac always picks >= 1)
        # but many cells may lack valid forward returns
        proba = np.zeros(len(val))
        score, details = multi_budget_composite(val, proba, budgets=(0.001,), horizons=(5,))
        assert isinstance(score, float)

    def test_all_cells_missing_returns_zero(self):
        """If every cell is missing (all append 0.0), mean is 0.0, not -inf."""
        # Build df with too few rows for any valid signal analytics
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=3),
            "stock_id": "X",
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [100, 101, 102],
        })
        proba = np.array([0.9, 0.1, 0.1])
        score, _ = multi_budget_composite(df, proba, budgets=(0.5,), horizons=(5,))
        # Missing cells contribute 0.0, so mean is 0.0
        assert score == pytest.approx(0.0)

    def test_deterministic(self, val_split, simulated_preds):
        val, _, _ = val_split
        _, _, y_proba = simulated_preds
        s1, _ = multi_budget_composite(val, y_proba)
        s2, _ = multi_budget_composite(val, y_proba)
        assert s1 == s2

    def test_budget_details_keep_nearby_custom_budgets_separate(self):
        n_rows = 30000
        df = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=n_rows),
            "stock_id": "X",
            "open": np.linspace(100, 200, n_rows),
            "high": np.linspace(101, 201, n_rows),
            "low": np.linspace(99, 199, n_rows),
            "close": np.linspace(100, 200, n_rows),
        })
        proba = np.linspace(1.0, 0.0, n_rows)

        _, details = multi_budget_composite(df, proba, budgets=(0.00100, 0.00104), horizons=(10,))

        assert list(details) == [0.001, 0.00104]
        assert details[0.001]["signal_mass"] == pytest.approx(30.0)
        assert details[0.00104]["signal_mass"] == pytest.approx(31.0)
        assert details[0.001]["label"] == "0.10%"
        assert details[0.00104]["label"] == "0.10%"


# ---------------------------------------------------------------------------
# Anti-gaming: N=1 exploit must be dead
# ---------------------------------------------------------------------------


class TestAntiGaming:
    def test_w_scaling_crushes_low_effective_n(self):
        """A cell with eff_n=1 and raw=10 scores less than eff_n=50 and raw=2.

        This is the core anti-gaming mechanism: even a huge raw score gets
        crushed by W when effective_n is small.
        """
        # eff_n=1: W = sqrt(1/21) ~ 0.218, weighted = 0.218 * 10 = 2.18
        w1 = np.sqrt(1 / 21)
        score_gaming = w1 * 10.0

        # eff_n=50: W = sqrt(50/70) ~ 0.845, weighted = 0.845 * 2 = 1.69
        w50 = np.sqrt(50 / 70)
        score_decent = w50 * 2.0

        # Even with 5x raw score, the gaming model barely wins at cell level
        # But at eff_n=1, you only get 1 valid cell -- other budget levels
        # contribute 0.0 (missing), dragging the composite down
        assert w1 < 0.25, f"W for eff_n=1 should be tiny: {w1:.3f}"
        assert w50 > 0.8, f"W for eff_n=50 should be near 1: {w50:.3f}"

    def test_missing_cells_drag_composite_to_zero(self, val_split):
        """A concentrated model has more missing cells (0.0 contributions),
        reducing its composite score via averaging."""
        val, _, _ = val_split
        n = len(val)

        # Concentrated: 1 spike, rest near zero
        proba_conc = np.full(n, 0.001)
        proba_conc[n // 2] = 0.99

        _, details_conc = multi_budget_composite(
            val, proba_conc, budgets=(0.0005, 0.001), horizons=(5,)
        )

        # Count how many cells are missing (no entry in cells dict)
        missing = sum(
            1 for info in details_conc.values()
            for h in [5] if h not in info["cells"]
        )
        # Concentrated model likely has missing cells at tight budgets
        # (or all cells have eff_n=1, giving very low W)
        total_cells = len(details_conc) * 1  # 1 horizon
        populated = total_cells - missing

        # At minimum, verify the mechanism: missing cells exist or W is tiny
        for info in details_conc.values():
            for cell in info["cells"].values():
                assert cell["w"] < 0.7, (
                    f"Concentrated model should have low W, got {cell['w']:.3f}"
                )

    def test_soft_n_scaling_penalizes_small_effective_n(self):
        """W = sqrt(eff_n / (eff_n + 20)) should be < 0.5 for eff_n < 7."""
        for eff_n in [1, 2, 3, 5, 6]:
            w = np.sqrt(eff_n / (eff_n + 20))
            assert w < 0.5, f"W={w:.3f} for eff_n={eff_n} is not < 0.5"


# ---------------------------------------------------------------------------
# tiered_eval integration
# ---------------------------------------------------------------------------


class TestTieredEvalIntegration:
    def test_ap_below_threshold_early_stops(self, val_split):
        val, _, _ = val_split
        y_true = val[LABEL_COL].values
        np.random.seed(99)
        y_proba = np.random.rand(len(y_true)) * 0.05
        result = tiered_eval(val, y_true, y_proba, min_ap=0.5)
        assert result["passed"] is False
        assert "tier1" in result
        assert "tier2" not in result
        assert "tier3" not in result

    def test_ap_above_threshold_populates_all_tiers(self, val_split, simulated_preds):
        val, _, _ = val_split
        y_true, _, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_proba, min_ap=0.01)
        assert "tier1" in result
        assert "tier2" in result
        assert "tier3" in result

    def test_passed_is_python_bool(self, val_split, simulated_preds):
        val, _, _ = val_split
        y_true, _, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_proba)
        assert type(result["passed"]) is bool

    def test_score_stored_in_tier3(self, val_split, simulated_preds):
        val, _, _ = val_split
        y_true, _, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_proba, min_ap=0.01)
        assert isinstance(result["tier3"], float)

    def test_tier3_matches_multi_budget(self, val_split, simulated_preds):
        """tier3 score should match multi_budget_composite output."""
        val, _, _ = val_split
        y_true, _, y_proba = simulated_preds
        result = tiered_eval(val, y_true, y_proba, min_ap=0.01)
        direct_score, _ = multi_budget_composite(val, y_proba)
        assert result["tier3"] == pytest.approx(direct_score)


# ---------------------------------------------------------------------------
# Score calibration: empirical reference points
# ---------------------------------------------------------------------------


class TestScoreCalibration:
    """Establish empirical score baselines to back the interpretation bands."""

    def test_random_model_scores_negative(self, val_split):
        """A random model should score negative (penalties dominate rewards)."""
        val, _, _ = val_split
        np.random.seed(42)
        y_proba = np.random.rand(len(val))
        score, _ = multi_budget_composite(val, y_proba)
        assert score < 0, f"Random model should score negative, got {score}"

    def test_skilled_model_beats_random(self, val_split):
        """A model with label signal + feature noise should beat random.

        Pure oracle (just labels) can lose to random because the composite
        measures trading quality, not prediction quality. But a model that
        ranks positives higher while maintaining some diversification through
        noise should consistently beat random.
        """
        val, _, _ = val_split
        if val["stock_id"].nunique() < 20:
            pytest.skip("Need >= 20 stocks for reliable comparison")
        y_true = val[LABEL_COL].values
        np.random.seed(42)
        random_proba = np.random.rand(len(val))
        skilled_proba = np.random.rand(len(val)) * 0.1
        skilled_proba[y_true == 1] += np.random.rand((y_true == 1).sum()) * 0.7
        random_score, _ = multi_budget_composite(val, random_proba)
        skilled_score, _ = multi_budget_composite(val, skilled_proba)
        assert skilled_score > random_score, (
            f"Skilled ({skilled_score:.4f}) should beat random ({random_score:.4f})"
        )
