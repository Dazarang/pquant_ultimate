"""Tests for research/utils/diagnostics.py.

Covers feature importance extraction across all model types in utils/model_wrappers.py
and eval detail TSV generation. Uses mock objects to avoid heavy training.
"""

import numpy as np
import pytest
from pathlib import Path

from research.utils.diagnostics import extract_importances, write_feat_importance, write_eval_detail, _fmt_budget

FEATURES = ["rsi_14", "macd_hist", "drawdown_pct", "atr_14", "obv_slope_20"]
N_FEAT = len(FEATURES)


# ---------------------------------------------------------------------------
# Mock models simulating each library's importance interface
# ---------------------------------------------------------------------------


class MockSklearnModel:
    """Simulates XGBClassifier, RandomForest, ExtraTrees, GBM, AdaBoost, etc."""
    def __init__(self, n):
        self.feature_importances_ = np.random.rand(n)


class MockCatBoost:
    """Simulates CatBoostClassifier."""
    def __init__(self, n):
        self._imp = np.random.rand(n)

    def get_feature_importance(self):
        return self._imp


class MockLGBMBooster:
    """Simulates lightgbm.Booster from lgb.train()."""
    def __init__(self, n):
        self._imp = np.random.rand(n)

    def feature_importance(self, importance_type="gain"):
        return self._imp


class MockXGBBooster:
    """Simulates xgboost.Booster from xgb.train() with f0/f1/... keys."""
    def __init__(self, n):
        self._scores = {f"f{i}": float(np.random.rand()) for i in range(n)}

    def get_score(self, importance_type="gain"):
        return self._scores


class MockXGBBoosterNamedFeatures:
    """Simulates xgboost.Booster with actual feature names as keys."""
    def __init__(self, feature_cols):
        self._scores = {f: float(np.random.rand()) for f in feature_cols}

    def get_score(self, importance_type="gain"):
        return self._scores


class MockXGBBoosterSparse:
    """Simulates xgboost.Booster where some features have zero importance (missing from dict)."""
    def __init__(self, feature_cols):
        self._scores = {f: float(np.random.rand()) for f in feature_cols[:3]}

    def get_score(self, importance_type="gain"):
        return self._scores


# --- Neural net mocks: all have fit/predict but no importances ---

class MockTorchClassifier:
    """Simulates TorchClassifier -- no importances."""
    def __init__(self):
        self.module = object()
        self.epochs = 50

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockFocalTorchClassifier:
    """Simulates FocalTorchClassifier -- no importances."""
    def __init__(self):
        self.module = object()
        self.alpha = 0.25
        self.focal_gamma = 2.0

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockSequenceClassifier:
    """Simulates SequenceClassifier -- no importances."""
    def __init__(self):
        self.module = object()
        self.window = 10

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockFocalSequenceClassifier:
    """Simulates FocalSequenceClassifier -- no importances."""
    def __init__(self):
        self.module = object()
        self.window = 10
        self.alpha = 0.25
        self.focal_gamma = 2.0

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockDirectUtility:
    """Simulates DirectUtilityClassifier -- no importances."""
    def __init__(self):
        self.module = object()
        self.pos_weight = 1.0

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockPolicyGradient:
    """Simulates PolicyGradientClassifier -- no importances."""
    def __init__(self):
        self.module = object()
        self.entropy_coef = 0.01
        self.baseline = True

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


# --- Wrapper/ensemble mocks ---

class MockCatBoostWrapper:
    """Simulates CatBoostWrapper -- importance in ._model."""
    def __init__(self, n):
        self._model = MockCatBoost(n)
        self._cb_kwargs = {}

    def fit(self, X, y): return self
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockRankingXGBClassifier:
    """Simulates RankingXGBClassifier -- xgb.Booster in ._model."""
    def __init__(self, n):
        self._model = MockXGBBooster(n)
        self.objective = "rank:map"
        self.group_size = 200

    def fit(self, X, y): return self
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockCurrentEnsemble:
    """Simulates the current _EnsembleModel in experiment.py.
    Has ._cat (CatBoost), ._lgb (LGB Booster), ._xgb (XGBClassifier), ._meta (LogisticRegression).
    """
    def __init__(self, n):
        self._cat = MockCatBoost(n)
        self._lgb = MockLGBMBooster(n)
        self._xgb = MockSklearnModel(n)
        self._meta = _MockLogReg()
        self.classes_ = np.array([0, 1])

    def fit(self, X, y): return self
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class _MockLogReg:
    """LogisticRegression meta-learner -- has coef_ but NOT feature_importances_."""
    def __init__(self):
        self.coef_ = np.array([[0.3, 0.4, 0.3]])
        self.intercept_ = np.array([0.1])

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))


class MockVotingEnsemble:
    """Simulates sklearn VotingClassifier with estimators_ list."""
    def __init__(self, n):
        self.estimators_ = [MockSklearnModel(n), MockSklearnModel(n)]

    def fit(self, X, y): return self
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockStackingEnsemble:
    """Simulates sklearn StackingClassifier."""
    def __init__(self, n):
        self.estimators_ = [MockSklearnModel(n), MockCatBoost(n)]
        self.final_estimator_ = _MockLogReg()

    def fit(self, X, y): return self
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


# --- Edge case mocks ---

class MockCompletelyCustom:
    """A model with no standard importance interface and no fit/predict."""
    def __init__(self):
        self.some_param = 42
        self.another_param = "hello"


class MockNestedDeep:
    """Model nested 4+ levels deep -- should stop at depth 3."""
    def __init__(self, n):
        self._inner = type("L1", (), {"_inner": type("L2", (), {"_inner": type("L3", (), {"_inner": MockSklearnModel(n)})()})()})()


class MockCircularRef:
    """Model with circular reference -- should not infinite loop."""
    def __init__(self):
        self._self_ref = self


class MockMixedEnsemble:
    """Ensemble mixing tree models (have importances) and neural (no importances)."""
    def __init__(self, n):
        self._tree = MockSklearnModel(n)
        self._neural = MockTorchClassifier()

    def fit(self, X, y): return self
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class MockImportanceWrongSize:
    """Model returning wrong number of importances."""
    def __init__(self):
        self.feature_importances_ = np.random.rand(999)


class MockCatBoostThrows:
    """CatBoost that throws on get_feature_importance."""
    def get_feature_importance(self):
        raise RuntimeError("model not fitted")


class MockPureNeuralModel:
    """A pure neural net model (no sub-models with importances)."""
    def __init__(self):
        self.module = object()
        self.lr = 1e-3
        self.batch_size = 512

    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


# ---------------------------------------------------------------------------
# Feature importance extraction tests
# ---------------------------------------------------------------------------


class TestExtractImportances:

    def test_sklearn_standard(self):
        model = MockSklearnModel(N_FEAT)
        extracted, skipped = extract_importances(model, FEATURES)
        assert len(extracted) == 1
        name, imp = extracted[0]
        assert name == "MockSklearnModel"
        assert set(imp.keys()) == set(FEATURES)
        assert all(isinstance(v, float) for v in imp.values())
        assert skipped == []

    def test_catboost(self):
        extracted, skipped = extract_importances(MockCatBoost(N_FEAT), FEATURES)
        assert len(extracted) == 1
        assert extracted[0][0] == "MockCatBoost"
        assert set(extracted[0][1].keys()) == set(FEATURES)
        assert skipped == []

    def test_lgbm_booster(self):
        extracted, skipped = extract_importances(MockLGBMBooster(N_FEAT), FEATURES)
        assert len(extracted) == 1
        assert extracted[0][0] == "MockLGBMBooster"
        assert skipped == []

    def test_xgb_booster_f_indices(self):
        extracted, skipped = extract_importances(MockXGBBooster(N_FEAT), FEATURES)
        assert len(extracted) == 1
        assert extracted[0][0] == "MockXGBBooster"
        assert set(extracted[0][1].keys()) == set(FEATURES)
        assert skipped == []

    def test_xgb_booster_named_features(self):
        extracted, skipped = extract_importances(MockXGBBoosterNamedFeatures(FEATURES), FEATURES)
        assert len(extracted) == 1
        assert set(extracted[0][1].keys()) == set(FEATURES)

    def test_xgb_booster_sparse(self):
        """XGB Booster may omit zero-importance features from get_score."""
        extracted, skipped = extract_importances(MockXGBBoosterSparse(FEATURES), FEATURES)
        assert len(extracted) == 1
        assert set(extracted[0][1].keys()) == set(FEATURES[:3])

    # --- Neural models: no importances, should appear in skipped ---

    def test_torch_classifier_skipped(self):
        extracted, skipped = extract_importances(MockTorchClassifier(), FEATURES)
        assert extracted == []
        assert "MockTorchClassifier" in skipped

    def test_focal_torch_classifier_skipped(self):
        extracted, skipped = extract_importances(MockFocalTorchClassifier(), FEATURES)
        assert extracted == []
        assert "MockFocalTorchClassifier" in skipped

    def test_sequence_classifier_skipped(self):
        extracted, skipped = extract_importances(MockSequenceClassifier(), FEATURES)
        assert extracted == []
        assert "MockSequenceClassifier" in skipped

    def test_focal_sequence_classifier_skipped(self):
        extracted, skipped = extract_importances(MockFocalSequenceClassifier(), FEATURES)
        assert extracted == []
        assert "MockFocalSequenceClassifier" in skipped

    def test_direct_utility_skipped(self):
        extracted, skipped = extract_importances(MockDirectUtility(), FEATURES)
        assert extracted == []
        assert "MockDirectUtility" in skipped

    def test_policy_gradient_skipped(self):
        extracted, skipped = extract_importances(MockPolicyGradient(), FEATURES)
        assert extracted == []
        assert "MockPolicyGradient" in skipped

    def test_pure_neural_skipped(self):
        extracted, skipped = extract_importances(MockPureNeuralModel(), FEATURES)
        assert extracted == []
        assert "MockPureNeuralModel" in skipped

    # --- Wrappers: should recurse into sub-models ---

    def test_catboost_wrapper(self):
        extracted, skipped = extract_importances(MockCatBoostWrapper(N_FEAT), FEATURES)
        assert len(extracted) == 1
        assert extracted[0][0] == "MockCatBoost"

    def test_ranking_xgb(self):
        extracted, skipped = extract_importances(MockRankingXGBClassifier(N_FEAT), FEATURES)
        assert len(extracted) == 1
        assert extracted[0][0] == "MockXGBBooster"

    # --- Ensembles ---

    def test_current_ensemble(self):
        """The actual ensemble in experiment.py: CatBoost + LGB + XGB + meta."""
        extracted, skipped = extract_importances(MockCurrentEnsemble(N_FEAT), FEATURES)
        assert len(extracted) == 3
        names = {r[0] for r in extracted}
        assert "MockCatBoost" in names
        assert "MockLGBMBooster" in names
        assert "MockSklearnModel" in names

    def test_voting_ensemble(self):
        extracted, skipped = extract_importances(MockVotingEnsemble(N_FEAT), FEATURES)
        assert len(extracted) == 2
        assert all(r[0] == "MockSklearnModel" for r in extracted)

    def test_stacking_ensemble(self):
        extracted, skipped = extract_importances(MockStackingEnsemble(N_FEAT), FEATURES)
        names = [r[0] for r in extracted]
        assert "MockSklearnModel" in names
        assert "MockCatBoost" in names

    def test_mixed_ensemble_extracted_and_skipped(self):
        """Ensemble with trees (have importances) + neural (no importances)."""
        extracted, skipped = extract_importances(MockMixedEnsemble(N_FEAT), FEATURES)
        assert len(extracted) == 1
        assert extracted[0][0] == "MockSklearnModel"
        assert "MockTorchClassifier" in skipped

    # --- Edge cases ---

    def test_completely_custom_returns_empty(self):
        extracted, skipped = extract_importances(MockCompletelyCustom(), FEATURES)
        assert extracted == []
        assert skipped == []  # no fit/predict, so not a model

    def test_depth_limit(self):
        extracted, skipped = extract_importances(MockNestedDeep(N_FEAT), FEATURES)
        assert extracted == []

    def test_circular_reference(self):
        extracted, skipped = extract_importances(MockCircularRef(), FEATURES)
        assert extracted == []

    def test_wrong_size_importances_skipped(self):
        extracted, skipped = extract_importances(MockImportanceWrongSize(), FEATURES)
        assert extracted == []

    def test_catboost_throws_skipped(self):
        extracted, skipped = extract_importances(MockCatBoostThrows(), FEATURES)
        assert extracted == []

    def test_empty_feature_cols(self):
        extracted, skipped = extract_importances(MockSklearnModel(0), [])
        assert len(extracted) == 1
        assert extracted[0][1] == {}


# ---------------------------------------------------------------------------
# Feature importance TSV writing tests
# ---------------------------------------------------------------------------


class TestWriteFeatImportance:

    def _read(self, path):
        return Path(path).read_text()

    def test_basic_output(self, tmp_path):
        path = tmp_path / "feat.tsv"
        imp = [("XGB", dict(zip(FEATURES, [0.5, 0.3, 0.1, 0.05, 0.05])))]
        write_feat_importance([imp], [[]], FEATURES, path)

        text = self._read(path)
        assert "## MODELS" in text
        assert "extracted: XGB" in text
        assert "## TOP 20 BY GAIN" in text
        assert "## BOTTOM 20 BY GAIN (averaged across folds)" in text
        assert "## SUMMARY" in text
        assert "total_features: 5" in text

    def test_models_section_with_skipped(self, tmp_path):
        path = tmp_path / "feat.tsv"
        imp = [("CatBoost", dict(zip(FEATURES, [0.5, 0.3, 0.1, 0.05, 0.05])))]
        write_feat_importance([imp], [["TorchMLP"]], FEATURES, path)

        text = self._read(path)
        assert "extracted: CatBoost" in text
        assert "no_importance: TorchMLP" in text

    def test_pure_neural_writes_models_only(self, tmp_path):
        """When no model provides importances, file still written with models section."""
        path = tmp_path / "feat.tsv"
        write_feat_importance([[]], [["TorchClassifier", "FocalTorchClassifier"]], FEATURES, path)

        text = self._read(path)
        assert "## MODELS" in text
        assert "no_importance: FocalTorchClassifier, TorchClassifier" in text
        assert "## TOP 10" not in text  # no importance data

    def test_top10_ordering(self, tmp_path):
        path = tmp_path / "feat.tsv"
        vals = {f: float(i) for i, f in enumerate(FEATURES)}
        imp = [("XGB", vals)]
        write_feat_importance([imp], [[]], FEATURES, path)

        lines = self._read(path).split("\n")
        data_lines = [l for l in lines if l and l[0].isdigit() and "\t" in l]
        assert "obv_slope_20" in data_lines[0]

    def test_multi_fold_averaging(self, tmp_path):
        path = tmp_path / "feat.tsv"
        imp1 = [("XGB", dict(zip(FEATURES, [1.0, 0, 0, 0, 0])))]
        imp2 = [("XGB", dict(zip(FEATURES, [0, 0, 0, 0, 1.0])))]
        write_feat_importance([imp1, imp2], [[], []], FEATURES, path)

        text = self._read(path)
        assert "rsi_14" in text
        assert "obv_slope_20" in text

    def test_multi_model_per_fold(self, tmp_path):
        path = tmp_path / "feat.tsv"
        imp = [
            ("CatBoost", dict(zip(FEATURES, [0.5, 0.3, 0.1, 0.05, 0.05]))),
            ("LGB", dict(zip(FEATURES, [0.1, 0.1, 0.5, 0.2, 0.1]))),
        ]
        write_feat_importance([imp], [[]], FEATURES, path)
        text = self._read(path)
        assert "extracted: CatBoost, LGB" in text
        assert "total_features: 5" in text

    def test_empty_importances_no_skipped(self, tmp_path):
        """No importances and no skipped models -- don't write file."""
        path = tmp_path / "feat.tsv"
        write_feat_importance([], [], FEATURES, path)
        assert not path.exists()

    def test_cumulative_share(self, tmp_path):
        path = tmp_path / "feat.tsv"
        imp = [("XGB", dict(zip(FEATURES, [100.0, 0, 0, 0, 0])))]
        write_feat_importance([imp], [[]], FEATURES, path)
        text = self._read(path)
        assert "top20_cumulative_share: 1.00" in text

    def test_tsv_tab_separated(self, tmp_path):
        path = tmp_path / "feat.tsv"
        imp = [("XGB", dict(zip(FEATURES, [0.5, 0.3, 0.1, 0.05, 0.05])))]
        write_feat_importance([imp], [[]], FEATURES, path)

        for line in self._read(path).split("\n"):
            if line and line[0].isdigit() and "\t" in line:
                parts = line.split("\t")
                assert len(parts) == 4, f"Expected 4 columns: {line}"


# ---------------------------------------------------------------------------
# Eval detail TSV writing tests
# ---------------------------------------------------------------------------


def _make_cell(excess=0.01, win_rate=0.55, worst_decile=-0.05, knife_rate=0.1,
               tail_mae=-0.03, entry_slippage=0.005, weighted=-0.5,
               effective_n=50.0, **extra):
    return {
        "weighted": weighted, "effective_n": effective_n,
        "excess": excess, "win_rate": win_rate,
        "worst_decile": worst_decile, "knife_rate": knife_rate,
        "tail_mae": tail_mae, "entry_slippage": entry_slippage,
        "raw": -1.0, "w": 0.5, "n_signals": 10.0,
        "event_recall": 0.1, "exact_center_recall": 0.05,
        "zone_precision": 0.2, "event_hits": 5, "n_true_events": 50,
        "exact_hits": 2, "n_true_bases": 40, "n_rows": 10, "tied_rows": 0,
        "duplicate_rows": 0, "duplicate_mass_discarded": 0.0,
        "avg_entry_offset_days": 0.5, "avg_entry_slippage": 0.005,
        **extra,
    }


def _make_fold_result(score, ap=0.08, auc=0.62, budgets=(0.001, 0.0025), horizons=(5, 10)):
    cells = {}
    for h in horizons:
        cells[h] = _make_cell(weighted=score / len(budgets))
    details = {}
    for b in budgets:
        details[b] = {"budget": b, "label": f"{b*100:.2f}%", "signal_mass": 10.0, "n_rows": 10, "tied_rows": 0, "cells": dict(cells)}
    details["_meta"] = {"valid_cells": len(budgets) * len(horizons), "total_cells": len(budgets) * len(horizons)}
    return {
        "tier1": {"avg_precision": ap, "roc_auc": auc, "base_avg_precision": ap * 0.8, "base_roc_auc": auc * 0.95},
        "tier2": details,
        "tier3": score,
        "passed": True,
    }


class TestWriteEvalDetail:

    def _read(self, path):
        return Path(path).read_text()

    def test_basic_structure(self, tmp_path):
        path = tmp_path / "eval.tsv"
        scores = [-1.5, -1.2]
        results = [_make_fold_result(s) for s in scores]
        write_eval_detail(scores, results, path)

        text = self._read(path)
        assert "## FOLDS" in text
        assert "## CELLS" in text
        assert "fold\tscore\tap\tauc" in text
        assert "mean\t" in text
        assert "std\t" in text

    def test_fold_scores_correct(self, tmp_path):
        path = tmp_path / "eval.tsv"
        scores = [-2.0, -1.0, -0.5, -1.5]
        results = [_make_fold_result(s) for s in scores]
        write_eval_detail(scores, results, path)

        text = self._read(path)
        assert "-2.0000" in text
        assert "-1.0000" in text
        assert "-0.5000" in text
        assert "-1.5000" in text
        assert "-1.2500" in text  # mean

    def test_cell_metrics_present(self, tmp_path):
        path = tmp_path / "eval.tsv"
        write_eval_detail([-1.0], [_make_fold_result(-1.0)], path)

        text = self._read(path)
        for header in ["budget", "horizon", "weighted", "eff_n", "excess", "win_rate", "worst_dec", "knife", "tail_mae", "slip"]:
            assert header in text

    def test_budget_formatting(self):
        assert _fmt_budget(0.001) == "0.10%"
        assert _fmt_budget(0.0025) == "0.25%"
        assert _fmt_budget(0.005) == "0.50%"
        assert _fmt_budget(0.01) == "1%"
        assert _fmt_budget(0.02) == "2%"

    def test_four_folds(self, tmp_path):
        path = tmp_path / "eval.tsv"
        scores = [-1.8, -1.1, -1.5, -1.4]
        results = [_make_fold_result(s) for s in scores]
        write_eval_detail(scores, results, path)

        text = self._read(path)
        fold_lines = [l for l in text.split("\n") if l and l.split("\t")[0] in ("1", "2", "3", "4")]
        assert len(fold_lines) == 4

    def test_missing_cells(self, tmp_path):
        """Folds with missing tier2 cells should not crash."""
        path = tmp_path / "eval.tsv"
        r1 = _make_fold_result(-1.0, budgets=(0.001,), horizons=(5,))
        r2 = {"tier1": {"avg_precision": 0.06, "roc_auc": 0.55}, "tier2": {}, "tier3": float("-inf"), "passed": False}
        write_eval_detail([-1.0, float("-inf")], [r1, r2], path)

        text = self._read(path)
        assert "## FOLDS" in text
        assert "## CELLS" in text

    def test_empty_tier2(self, tmp_path):
        path = tmp_path / "eval.tsv"
        r = {"tier1": {"avg_precision": 0.06, "roc_auc": 0.55}, "tier2": {}, "tier3": -2.0, "passed": True}
        write_eval_detail([-2.0], [r], path)
        text = self._read(path)
        assert "## FOLDS" in text

    def test_nan_in_cell_metrics(self, tmp_path):
        """NaN values in cells should not crash."""
        path = tmp_path / "eval.tsv"
        r = _make_fold_result(-1.0)
        # Inject NaN into a cell
        for budget_info in r["tier2"].values():
            if isinstance(budget_info, dict) and "cells" in budget_info:
                for cell in budget_info["cells"].values():
                    cell["excess"] = float("nan")
                    break
                break
        write_eval_detail([-1.0], [r], path)
        text = self._read(path)
        assert "## CELLS" in text

    def test_inf_fold_score(self, tmp_path):
        """Fold with -inf score should not crash."""
        path = tmp_path / "eval.tsv"
        r = {"tier1": {}, "tier2": {}, "tier3": float("-inf"), "passed": False}
        write_eval_detail([float("-inf")], [r], path)
        text = self._read(path)
        assert "-inf" in text

    def test_tsv_tab_separated(self, tmp_path):
        path = tmp_path / "eval.tsv"
        write_eval_detail([-1.0], [_make_fold_result(-1.0)], path)

        for line in self._read(path).split("\n"):
            if line and not line.startswith("##") and not line.startswith("fold\t") and not line.startswith("budget\t"):
                if "\t" in line:
                    parts = line.split("\t")
                    assert len(parts) >= 3, f"Too few columns: {line}"

    def test_single_fold(self, tmp_path):
        path = tmp_path / "eval.tsv"
        write_eval_detail([-1.5], [_make_fold_result(-1.5)], path)
        text = self._read(path)
        assert "std\t0.0000" in text

    def test_all_budgets_and_horizons(self, tmp_path):
        """Full 5-budget x 3-horizon grid."""
        path = tmp_path / "eval.tsv"
        budgets = (0.001, 0.0025, 0.005, 0.01, 0.02)
        horizons = (5, 10, 20)
        r = _make_fold_result(-1.0, budgets=budgets, horizons=horizons)
        write_eval_detail([-1.0], [r], path)

        text = self._read(path)
        cell_lines = [l for l in text.split("\n") if "%" in l and "d\t" in l]
        assert len(cell_lines) == 15
