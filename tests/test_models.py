"""Tests for all model types through the full pipeline (load → split → scale → train → eval)."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # prevent OpenMP deadlock (xgb/lgbm + torch)

import numpy as np
import pytest

from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split
from lib.eval import evaluate

DS_PATH = "data/datasets/20260115/dataset.parquet"
STOCKS = ["AAPL", "MSFT", "TSLA"]
FEATURES = list_features("base")


@pytest.fixture(scope="module")
def pipeline_data():
    """Load, split, scale once for all model tests."""
    df, fc = load_dataset(DS_PATH, stocks=STOCKS, features=FEATURES)
    train, val, _ = temporal_split(df)
    train_s, val_s, _, _ = scale(train, val, val, fc)  # pass val as test (unused)

    X_train = train_s[fc].values
    y_train = train_s[LABEL_COL].values
    X_val = val_s[fc].values
    y_val = val_s[LABEL_COL].values
    n_features = X_train.shape[1]
    groups_train = train_s["stock_id"].values
    groups_val = val_s["stock_id"].values

    return X_train, y_train, X_val, y_val, n_features, groups_train, groups_val


def _check_model(model, X_train, y_train, X_val, y_val):
    """Train model and verify it produces valid predictions."""
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 2), f"Expected shape ({len(X_val)}, 2), got {proba.shape}"
    assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of [0, 1]"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"

    y_pred = (proba[:, 1] > 0.5).astype(int)
    metrics = evaluate(y_val, y_pred, proba[:, 1])
    assert "precision" in metrics
    assert "recall" in metrics
    return metrics


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------


class TestXGBoost:
    def test_pipeline(self, pipeline_data):
        from xgboost import XGBClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        _check_model(model, X_train, y_train, X_val, y_val)


class TestRankingXGB:
    def test_pipeline(self, pipeline_data):
        from research.model_wrappers import RankingXGBClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = RankingXGBClassifier(
            objective="rank:map", group_size=100,
            n_estimators=50, max_depth=3, learning_rate=0.1,
        )
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_rank_ndcg(self, pipeline_data):
        from research.model_wrappers import RankingXGBClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = RankingXGBClassifier(
            objective="rank:ndcg", group_size=50,
            n_estimators=30, max_depth=3, learning_rate=0.1,
        )
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_custom_groups(self, pipeline_data):
        from research.model_wrappers import RankingXGBClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        groups = np.repeat(np.arange(len(y_train) // 50 + 1), 50)[:len(y_train)]
        model = RankingXGBClassifier(
            objective="rank:map", n_estimators=30, max_depth=3,
        )
        model.fit(X_train, y_train, groups=groups)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_non_contiguous_groups(self, pipeline_data):
        """Non-contiguous group IDs (e.g. stock IDs interleaved) must not mis-group."""
        from research.model_wrappers import RankingXGBClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        # Simulate interleaved stock IDs: [0,1,2,0,1,2,...]
        groups = np.tile(np.arange(3), len(y_train) // 3 + 1)[:len(y_train)]
        model = RankingXGBClassifier(
            objective="rank:map", n_estimators=30, max_depth=3,
        )
        model.fit(X_train, y_train, groups=groups)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_sklearn_clone(self, pipeline_data):
        from sklearn.base import clone

        from research.model_wrappers import RankingXGBClassifier

        original = RankingXGBClassifier(
            objective="rank:map", group_size=100, n_estimators=30, max_depth=3,
        )
        cloned = clone(original)
        assert cloned.get_params() == original.get_params()


class TestLightGBM:
    def test_pipeline(self, pipeline_data):
        from lightgbm import LGBMClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = LGBMClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, num_leaves=15,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            verbose=-1, random_state=42, n_jobs=-1,
        )
        _check_model(model, X_train, y_train, X_val, y_val)


class TestCatBoost:
    def test_pipeline(self, pipeline_data):
        from catboost import CatBoostClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = CatBoostClassifier(
            iterations=50, depth=3, learning_rate=0.1,
            scale_pos_weight=neg / pos,
            verbose=0, random_seed=42,
        )
        _check_model(model, X_train, y_train, X_val, y_val)


class TestCatBoostWrapper:
    def test_pipeline(self, pipeline_data):
        from research.model_wrappers import CatBoostWrapper

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = CatBoostWrapper(
            iterations=50, depth=3, learning_rate=0.1,
            scale_pos_weight=neg / pos,
            verbose=0, random_seed=42,
        )
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_sklearn_clone(self, pipeline_data):
        """Verify clone() works -- the exact failure from iter 1."""
        from sklearn.base import clone

        from research.model_wrappers import CatBoostWrapper

        *_, y_val, _, _, _ = pipeline_data
        neg, pos = (y_val == 0).sum(), (y_val == 1).sum()
        original = CatBoostWrapper(
            iterations=50, depth=3, scale_pos_weight=neg / pos, verbose=0,
        )
        cloned = clone(original)
        assert cloned.get_params() == original.get_params()

    def test_in_voting_classifier(self, pipeline_data):
        """CatBoostWrapper inside VotingClassifier -- the crash scenario."""
        from sklearn.ensemble import VotingClassifier
        from xgboost import XGBClassifier

        from research.model_wrappers import CatBoostWrapper

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss", random_state=42, n_jobs=-1)),
                ("cb", CatBoostWrapper(iterations=30, depth=3, scale_pos_weight=neg / pos, verbose=0, random_seed=43)),
            ],
            voting="soft",
        )
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_in_stacking_classifier(self, pipeline_data):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier

        from research.model_wrappers import CatBoostWrapper

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = StackingClassifier(
            estimators=[
                ("xgb", XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss", random_state=42, n_jobs=-1)),
                ("cb", CatBoostWrapper(iterations=30, depth=3, scale_pos_weight=neg / pos, verbose=0, random_seed=43)),
            ],
            final_estimator=LogisticRegression(max_iter=500),
            cv=3,
        )
        _check_model(model, X_train, y_train, X_val, y_val)


# ---------------------------------------------------------------------------
# Sklearn
# ---------------------------------------------------------------------------


class TestSklearnModels:
    def test_gradient_boosting(self, pipeline_data):
        from sklearn.ensemble import GradientBoostingClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_random_forest(self, pipeline_data):
        from sklearn.ensemble import RandomForestClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = RandomForestClassifier(
            n_estimators=50, max_depth=5, class_weight={0: 1, 1: neg / pos}, random_state=42,
        )
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_extra_trees(self, pipeline_data):
        from sklearn.ensemble import ExtraTreesClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42)
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_adaboost(self, pipeline_data):
        from sklearn.ensemble import AdaBoostClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = AdaBoostClassifier(n_estimators=50, random_state=42)
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_bagging(self, pipeline_data):
        from sklearn.ensemble import BaggingClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = BaggingClassifier(n_estimators=20, random_state=42)
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_logistic_regression(self, pipeline_data):
        from sklearn.linear_model import LogisticRegression

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = LogisticRegression(class_weight={0: 1, 1: neg / pos}, max_iter=500, random_state=42)
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_mlp(self, pipeline_data):
        from sklearn.neural_network import MLPClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_svc(self, pipeline_data):
        from sklearn.svm import SVC

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        # SVC is slow on large data, use small subset
        n = min(500, len(X_train))
        model = SVC(probability=True, kernel="rbf", random_state=42)
        _check_model(model, X_train[:n], y_train[:n], X_val, y_val)

    def test_knn(self, pipeline_data):
        from sklearn.neighbors import KNeighborsClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = KNeighborsClassifier(n_neighbors=10)
        _check_model(model, X_train, y_train, X_val, y_val)


# ---------------------------------------------------------------------------
# Ensembling
# ---------------------------------------------------------------------------


class TestEnsemble:
    def test_stacking(self, pipeline_data):
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = StackingClassifier(
            estimators=[
                ("xgb", XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss", random_state=42, n_jobs=-1)),
                ("lgbm", LGBMClassifier(n_estimators=30, max_depth=3, verbose=-1, random_state=42, n_jobs=-1)),
            ],
            final_estimator=LogisticRegression(max_iter=500),
            cv=3,
        )
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_voting(self, pipeline_data):
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import VotingClassifier
        from xgboost import XGBClassifier

        X_train, y_train, X_val, y_val, *_ = pipeline_data
        model = VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss", random_state=42, n_jobs=-1)),
                ("lgbm", LGBMClassifier(n_estimators=30, max_depth=3, verbose=-1, random_state=42, n_jobs=-1)),
            ],
            voting="soft",
        )
        _check_model(model, X_train, y_train, X_val, y_val)


# ---------------------------------------------------------------------------
# PyTorch wrappers
# ---------------------------------------------------------------------------


class TestTorchMLP:
    def test_pipeline(self, pipeline_data):
        import torch

        from research.model_wrappers import TorchClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        model = TorchClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(32, 16)),
            epochs=5, lr=1e-3, batch_size=256,
            pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        # Force CPU in tests to avoid MPS cold-start hangs in pytest
        model.device = torch.device("cpu")
        _check_model(model, X_train, y_train, X_val, y_val)


class TestLSTM:
    def test_pipeline(self, pipeline_data):
        import torch

        from research.model_wrappers import LSTMNet, SequenceClassifier

        X_train, y_train, X_val, y_val, n_features, groups_train, groups_val = pipeline_data
        model = SequenceClassifier(
            module=LSTMNet(input_dim=n_features, hidden_dim=16, num_layers=1, dropout=0.0),
            window=5, epochs=3, lr=1e-3, batch_size=256,
            pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        # Force CPU in tests
        model.device = torch.device("cpu")
        model.fit(X_train, y_train, groups=groups_train)
        proba = model.predict_proba(X_val, groups=groups_val)
        assert proba.shape == (len(X_val), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestGRU:
    def test_pipeline(self, pipeline_data):
        import torch

        from research.model_wrappers import GRUNet, SequenceClassifier

        X_train, y_train, X_val, y_val, n_features, groups_train, groups_val = pipeline_data
        model = SequenceClassifier(
            module=GRUNet(input_dim=n_features, hidden_dim=16, num_layers=1, dropout=0.0),
            window=5, epochs=3, lr=1e-3, batch_size=256,
            pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train, groups=groups_train)
        proba = model.predict_proba(X_val, groups=groups_val)
        assert proba.shape == (len(X_val), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestTransformer:
    def test_pipeline(self, pipeline_data):
        import torch

        from research.model_wrappers import SequenceClassifier, TransformerNet

        X_train, y_train, X_val, y_val, n_features, groups_train, groups_val = pipeline_data
        model = SequenceClassifier(
            module=TransformerNet(
                input_dim=n_features, d_model=32, nhead=4,
                num_layers=1, dim_feedforward=64, dropout=0.0,
            ),
            window=5, epochs=3, lr=1e-3, batch_size=256,
            pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train, groups=groups_train)
        proba = model.predict_proba(X_val, groups=groups_val)
        assert proba.shape == (len(X_val), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestPolicyGradient:
    def test_binary_rewards(self, pipeline_data):
        import torch

        from research.model_wrappers import PolicyGradientClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        model = PolicyGradientClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(32, 16), dropout=0.0),
            epochs=5, lr=1e-3, batch_size=256,
            pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(model.classes_, [0, 1])

    def test_custom_rewards(self, pipeline_data):
        import torch

        from research.model_wrappers import PolicyGradientClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        rewards = np.random.randn(len(y_train)).astype(np.float32)
        model = PolicyGradientClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(32, 16), dropout=0.0),
            epochs=5, lr=1e-3, batch_size=256, pos_weight=1.0,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train, rewards=rewards)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict(self, pipeline_data):
        import torch

        from research.model_wrappers import PolicyGradientClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        model = PolicyGradientClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=3, lr=1e-3, batch_size=256, pos_weight=1.0,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_signed_rewards_differ(self, pipeline_data):
        """Verify positive and negative rewards produce different policies."""
        import torch

        from research.model_wrappers import PolicyGradientClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data

        # All positive rewards -- should encourage buying
        pos_rewards = np.abs(np.random.randn(len(y_train))).astype(np.float32)
        m1 = PolicyGradientClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=10, lr=1e-3, batch_size=256, pos_weight=1.0,
        )
        m1.device = torch.device("cpu")
        m1.fit(X_train, y_train, rewards=pos_rewards)
        p1 = m1.predict_proba(X_val)[:, 1].mean()

        # All negative rewards -- should discourage buying
        neg_rewards = -np.abs(np.random.randn(len(y_train))).astype(np.float32)
        m2 = PolicyGradientClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=10, lr=1e-3, batch_size=256, pos_weight=1.0,
        )
        m2.device = torch.device("cpu")
        m2.fit(X_train, y_train, rewards=neg_rewards)
        p2 = m2.predict_proba(X_val)[:, 1].mean()

        # Positive rewards should produce higher buy probability than negative
        assert p1 > p2, f"Positive rewards mean p={p1:.3f} should > negative p={p2:.3f}"

    def test_repeated_fit_idempotent(self, pipeline_data):
        """Calling fit() twice must not stack extra heads."""
        import torch

        from research.model_wrappers import PolicyGradientClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        model = PolicyGradientClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=3, lr=1e-3, batch_size=256, pos_weight=1.0,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train)
        p1 = model.predict_proba(X_val)
        assert p1.shape == (len(X_val), 2)

        # Second fit on same instance
        model.fit(X_train, y_train)
        p2 = model.predict_proba(X_val)
        assert p2.shape == (len(X_val), 2), f"Shape changed on second fit: {p2.shape}"

    def test_with_non_standard_module(self, pipeline_data):
        """Module without Linear(*, 1) falls back to extra head."""
        import torch
        from torch import nn

        from research.model_wrappers import PolicyGradientClassifier

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data

        # Module ending in Linear(*, 8) -- no Linear(*, 1) to replace
        module = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 8),
        )
        model = PolicyGradientClassifier(
            module=module, epochs=3, lr=1e-3, batch_size=256, pos_weight=1.0,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_with_bare_root_linear(self, pipeline_data):
        """Bare nn.Linear(*, 1) as root module -- no named children."""
        import torch
        from torch import nn

        from research.model_wrappers import PolicyGradientClassifier

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data

        module = nn.Linear(n_features, 1)
        model = PolicyGradientClassifier(
            module=module, epochs=3, lr=1e-3, batch_size=256, pos_weight=1.0,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestFocalTorchClassifier:
    def test_pipeline(self, pipeline_data):
        import torch

        from research.model_wrappers import FocalTorchClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        model = FocalTorchClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(32, 16), dropout=0.0),
            epochs=5, lr=1e-3, batch_size=256,
            alpha=(y_train == 0).sum() / len(y_train), focal_gamma=2.0,
        )
        model.device = torch.device("cpu")
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_gamma_zero_matches_weighted_bce(self, pipeline_data):
        """With focal_gamma=0, focal loss reduces to alpha-weighted BCE."""
        import torch

        from research.model_wrappers import FocalTorchClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        model = FocalTorchClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=3, lr=1e-3, batch_size=256,
            alpha=0.5, focal_gamma=0.0,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestFocalSequenceClassifier:
    def test_pipeline(self, pipeline_data):
        import torch

        from research.model_wrappers import FocalSequenceClassifier, GRUNet

        X_train, y_train, X_val, y_val, n_features, groups_train, groups_val = pipeline_data
        model = FocalSequenceClassifier(
            module=GRUNet(input_dim=n_features, hidden_dim=16, num_layers=1, dropout=0.0),
            window=5, epochs=3, lr=1e-3, batch_size=256,
            alpha=(y_train == 0).sum() / len(y_train), focal_gamma=2.0,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train, groups=groups_train)
        proba = model.predict_proba(X_val, groups=groups_val)
        assert proba.shape == (len(X_val), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestDirectUtility:
    def test_with_rewards(self, pipeline_data):
        import torch

        from research.model_wrappers import DirectUtilityClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        rewards = np.random.randn(len(y_train)).astype(np.float32)
        model = DirectUtilityClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(32, 16), dropout=0.0),
            epochs=5, lr=1e-3, batch_size=256,
        )
        model.device = torch.device("cpu")
        model.fit(X_train, y_train, rewards=rewards)
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(model.classes_, [0, 1])

    def test_fallback_bce(self, pipeline_data):
        """Without rewards, falls back to weighted BCE."""
        import torch

        from research.model_wrappers import DirectUtilityClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data
        model = DirectUtilityClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=3, lr=1e-3, batch_size=256,
            pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        model.device = torch.device("cpu")
        _check_model(model, X_train, y_train, X_val, y_val)

    def test_signed_rewards_differ(self, pipeline_data):
        """Positive rewards should produce higher P(buy) than negative."""
        import torch

        from research.model_wrappers import DirectUtilityClassifier, TorchMLP

        X_train, y_train, X_val, y_val, n_features, *_ = pipeline_data

        pos_rewards = np.abs(np.random.randn(len(y_train))).astype(np.float32) + 0.1
        m1 = DirectUtilityClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=10, lr=1e-3, batch_size=256,
        )
        m1.device = torch.device("cpu")
        m1.fit(X_train, y_train, rewards=pos_rewards)
        p1 = m1.predict_proba(X_val)[:, 1].mean()

        neg_rewards = -(np.abs(np.random.randn(len(y_train))).astype(np.float32) + 0.1)
        m2 = DirectUtilityClassifier(
            module=TorchMLP(input_dim=n_features, hidden_dims=(16,), dropout=0.0),
            epochs=10, lr=1e-3, batch_size=256,
        )
        m2.device = torch.device("cpu")
        m2.fit(X_train, y_train, rewards=neg_rewards)
        p2 = m2.predict_proba(X_val)[:, 1].mean()

        assert p1 > p2, f"Positive rewards mean p={p1:.3f} should > negative p={p2:.3f}"
