"""Ready-to-use model wrappers for non-sklearn models.

All wrappers expose sklearn-compatible .fit(X, y) and .predict_proba(X) methods.
Import and use in experiment.py's build_model().

This file is IMMUTABLE -- do not edit during research.

NOTE: torch is lazy-loaded — importing CatBoostWrapper does NOT trigger a torch
import, avoiding the OpenMP thread-pool conflict between torch and XGB/LGBM.
"""

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# CatBoost (sklearn clone()-compatible)
# ---------------------------------------------------------------------------


class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible wrapper for CatBoostClassifier.

    CatBoost's constructor mutates certain params (e.g. scale_pos_weight),
    which breaks sklearn.base.clone() used by VotingClassifier/StackingClassifier.
    This wrapper stores params cleanly and delegates to a fresh CatBoostClassifier
    at fit() time.

    Usage in experiment.py:
        from research.model_wrappers import CatBoostWrapper

        def build_model(y_train):
            neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
            cb = CatBoostWrapper(
                iterations=400, depth=6, learning_rate=0.05,
                scale_pos_weight=neg / pos, verbose=0, random_seed=43,
            )
            ...  # use in VotingClassifier, StackingClassifier, or standalone
    """

    def __init__(self, **kwargs):
        # Store every kwarg as an instance attribute so sklearn's
        # get_params()/set_params()/clone() work correctly.
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._cb_kwargs = kwargs

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self._cb_kwargs}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            self._cb_kwargs[key] = value
        return self

    def fit(self, X, y):
        self._model = CatBoostClassifier(**self._cb_kwargs)
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(X).astype(int).ravel()

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# ---------------------------------------------------------------------------
# XGBoost ranking wrapper
# ---------------------------------------------------------------------------


class RankingXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost with ranking objective (rank:map or rank:ndcg).

    XGBoost with a ranking objective (rank:map / rank:ndcg).

    Ranking requires query groups (samples competing against each other).
    Pass groups to fit(), or the wrapper creates groups of `group_size`
    from consecutive rows (with time-sorted data, this approximates
    same-period comparisons).

    Outputs pseudo-probabilities via sigmoid(ranking_score) for predict_proba.

    Usage in experiment.py:
        from research.model_wrappers import RankingXGBClassifier

        def build_model(y_train):
            return RankingXGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                objective="rank:map", group_size=200,
            )
    """

    def __init__(self, objective="rank:map", group_size=200, **kwargs):
        self.objective = objective
        self.group_size = group_size
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._xgb_kwargs = kwargs

    def get_params(self, deep=True):
        params = {"objective": self.objective, "group_size": self.group_size}
        params.update({k: getattr(self, k) for k in self._xgb_kwargs})
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            if key not in ("objective", "group_size"):
                self._xgb_kwargs[key] = value
        return self

    @staticmethod
    def _qid_to_group_sizes(qid):
        """Convert per-row qid array to group sizes (consecutive runs)."""
        if len(qid) == 0:
            return []
        sizes = []
        current, count = qid[0], 1
        for i in range(1, len(qid)):
            if qid[i] == current:
                count += 1
            else:
                sizes.append(count)
                current, count = qid[i], 1
        sizes.append(count)
        return sizes

    def fit(self, X, y, groups=None):
        import xgboost as xgb

        if groups is not None:
            qid = np.asarray(groups)
            # XGBoost ranking requires data sorted by group -- sort if needed
            order = np.argsort(qid, kind="stable")
            X, y, qid = np.asarray(X)[order], np.asarray(y)[order], qid[order]
        else:
            gs = self.group_size
            qid = np.repeat(np.arange(len(y) // gs + 1), gs)[:len(y)]

        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(self._qid_to_group_sizes(qid))

        params = {
            "objective": self.objective,
            "eval_metric": "map" if "map" in self.objective else "ndcg",
            "tree_method": "hist",
            **{k: v for k, v in self._xgb_kwargs.items()
               if k not in ("n_estimators",)},
        }
        n_rounds = self._xgb_kwargs.get("n_estimators", 500)
        self._model = xgb.train(params, dtrain, num_boost_round=n_rounds)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import xgboost as xgb

        dtest = xgb.DMatrix(X)
        scores = self._model.predict(dtest)
        proba_1 = expit(scores)
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# PyTorch wrappers (lazy-loaded to avoid OpenMP conflicts)
# ---------------------------------------------------------------------------


class _BaseTorchClassifier:

    def __init__(self, module, epochs, lr, batch_size, pos_weight):
        import torch

        self.module = module
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _train(self, X_t, y_t):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        if len(X_t) == 0:
            return

        self.module.to(self.device)
        self.module.train()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        pw = torch.tensor([self.pos_weight], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.module(xb), yb).backward()
                optimizer.step()

    def _batched_logits(self, X):
        """Run batched inference on numpy array, return flat logits."""
        import torch

        if len(X) == 0:
            return np.array([], dtype=np.float32)

        self.module.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = torch.tensor(X[i:i + self.batch_size], dtype=torch.float32).to(self.device)
                all_logits.append(self.module(batch).cpu().numpy().flatten())
        return np.concatenate(all_logits)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class TorchClassifier(_BaseTorchClassifier):
    """Sklearn-compatible wrapper for any PyTorch binary classifier.

    Usage in experiment.py:
        from research.model_wrappers import TorchClassifier, TorchMLP

        def build_model(y_train):
            return TorchClassifier(
                module=TorchMLP(input_dim=54, hidden_dims=(128, 64)),
                epochs=50, lr=1e-3, batch_size=512,
                pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            )
    """

    def __init__(self, module, epochs=50, lr=1e-3, batch_size=512, pos_weight=1.0):
        super().__init__(module, epochs, lr, batch_size, pos_weight)

    def fit(self, X, y):
        import torch

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self._train(X_t, y_t)
        return self

    def predict_proba(self, X):
        logits = self._batched_logits(X)
        proba_1 = expit(logits)
        return np.column_stack([1 - proba_1, proba_1])


def _focal_loss(logits, targets, alpha, gamma):
    """Numerically stable focal loss (Lin et al. 2017).

    Uses F.binary_cross_entropy_with_logits internally for stability.
    """
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = targets * p + (1 - targets) * (1 - p)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    focal_weight = (1 - p_t) ** gamma
    return (alpha_t * focal_weight * bce).mean()


def _train_focal_loop(module, loader, optimizer, device, alpha, gamma, epochs):
    """Shared focal loss training loop for FocalTorchClassifier and FocalSequenceClassifier."""
    for _ in range(epochs):
        module.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = _focal_loss(module(xb), yb, alpha, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class FocalTorchClassifier(_BaseTorchClassifier):
    """TorchClassifier variant using focal loss.

    Focal loss: -alpha * (1-p)^gamma * log(p). Reduces gradient contribution
    from high-confidence predictions, shifting training focus toward harder
    examples.

    Compatible with TorchClassifier interface.

    Usage in experiment.py:
        from research.model_wrappers import FocalTorchClassifier, TorchMLP

        def build_model(y_train):
            return FocalTorchClassifier(
                module=TorchMLP(input_dim=54, hidden_dims=(128, 64)),
                epochs=50, lr=1e-3, batch_size=512,
                alpha=(y_train == 0).sum() / len(y_train),  # class balance
                focal_gamma=2.0,  # focusing strength
            )
    """

    def __init__(self, module, epochs=50, lr=1e-3, batch_size=512,
                 alpha=0.25, focal_gamma=2.0, pos_weight=1.0):
        super().__init__(module, epochs, lr, batch_size, pos_weight)
        self.alpha = alpha
        self.focal_gamma = focal_gamma

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if len(X) == 0:
            return self
        self.module.to(self.device)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        _train_focal_loop(self.module, loader, optimizer, self.device,
                          self.alpha, self.focal_gamma, self.epochs)
        return self

    def predict_proba(self, X):
        logits = self._batched_logits(X)
        proba_1 = expit(logits)
        return np.column_stack([1 - proba_1, proba_1])


class SequenceClassifier(_BaseTorchClassifier):
    """Sklearn-compatible wrapper for sequence models (LSTM, GRU, Transformer).

    Reshapes flat (samples, features) into (samples, window, features) via
    sliding windows. Pass groups= to fit/predict_proba to prevent windows
    from crossing stock boundaries.

    Usage in experiment.py:
        from research.model_wrappers import SequenceClassifier, LSTMNet

        def build_model(y_train):
            return SequenceClassifier(
                module=LSTMNet(input_dim=54, hidden_dim=64),
                window=10, epochs=30, lr=1e-3, batch_size=256,
                pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            )
    """

    def __init__(self, module, window=10, epochs=30, lr=1e-3, batch_size=256, pos_weight=1.0):
        super().__init__(module, epochs, lr, batch_size, pos_weight)
        self.window = window

    def _build_sequences(self, X, y=None, groups=None):
        """Build sliding windows. When groups provided, windows don't cross group boundaries."""
        w = self.window
        n = X.shape[0]
        sequences = []
        labels = [] if y is not None else None
        valid = []

        if groups is None:
            for i in range(w, n):
                sequences.append(X[i - w: i])
                valid.append(i)
                if y is not None:
                    labels.append(y[i])
        else:
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                X_g = X[idx]
                for j in range(w, len(X_g)):
                    sequences.append(X_g[j - w: j])
                    valid.append(idx[j])
                    if y is not None:
                        labels.append(y[idx[j]])

        sequences = np.array(sequences)
        valid = np.array(valid)
        if y is not None:
            return sequences, np.array(labels), valid
        return sequences, valid

    def fit(self, X, y, groups=None):
        import torch

        X_seq, y_seq, _ = self._build_sequences(X, y, groups)
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
        self._train(X_t, y_t)
        return self

    def predict_proba(self, X, groups=None):
        X_seq, valid = self._build_sequences(X, groups=groups)
        proba_1 = np.full(len(X), 0.5)
        if len(valid) > 0:
            logits = self._batched_logits(X_seq)
            proba_1[valid] = expit(logits)
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X, groups=None):
        return (self.predict_proba(X, groups)[:, 1] > 0.5).astype(int)


class FocalSequenceClassifier(SequenceClassifier):
    """SequenceClassifier variant using focal loss.

    Usage in experiment.py:
        from research.model_wrappers import FocalSequenceClassifier, GRUNet

        def build_model(y_train):
            return FocalSequenceClassifier(
                module=GRUNet(input_dim=54, hidden_dim=64),
                window=10, epochs=30, lr=1e-3, batch_size=256,
                alpha=(y_train == 0).sum() / len(y_train), focal_gamma=2.0,
            )
    """

    def __init__(self, module, window=10, epochs=30, lr=1e-3, batch_size=256,
                 alpha=0.25, focal_gamma=2.0, pos_weight=1.0):
        super().__init__(module, window, epochs, lr, batch_size, pos_weight)
        self.alpha = alpha
        self.focal_gamma = focal_gamma

    def _train(self, X_t, y_t):
        """Override base _train with focal loss."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if len(X_t) == 0:
            return
        self.module.to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        _train_focal_loop(self.module, loader, optimizer, self.device,
                          self.alpha, self.focal_gamma, self.epochs)


# ---------------------------------------------------------------------------
# Direct Utility optimization
# ---------------------------------------------------------------------------


class DirectUtilityClassifier(_BaseTorchClassifier):
    """Optimizes expected reward directly without REINFORCE sampling.

    Instead of sampling actions and using log-probability gradients, directly
    maximizes: E[reward] = P(buy|x) * r_buy + P(skip|x) * r_skip.

    With r_skip=0: loss = -mean(sigmoid(logit) * reward).
    Positive rewards increase P(buy), negative rewards decrease it.
    Falls back to weighted BCE when rewards=None.

    Usage in experiment.py:
        from research.model_wrappers import DirectUtilityClassifier, TorchMLP

        def build_model(y_train):
            return DirectUtilityClassifier(
                module=TorchMLP(input_dim=54, hidden_dims=(128, 64)),
                epochs=50, lr=1e-3, batch_size=512,
            )

        # With custom rewards:
        model.fit(X_train, y_train, rewards=forward_returns_5d)
    """

    def __init__(self, module, epochs=50, lr=1e-3, batch_size=512, pos_weight=1.0):
        super().__init__(module, epochs, lr, batch_size, pos_weight)

    def fit(self, X, y, rewards=None):
        """Train by maximizing expected utility.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels.
            rewards: Optional signed reward per sample. Positive = buying was
                     good. When provided, optimizes expected reward directly.
                     When None, falls back to pos_weight-weighted BCE.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if len(X) == 0:
            self.classes_ = np.array([0, 1])
            return self

        self.module.to(self.device)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        if rewards is not None:
            r_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            loader = DataLoader(TensorDataset(X_t, y_t, r_t),
                                batch_size=self.batch_size, shuffle=True)

            for _ in range(self.epochs):
                self.module.train()
                for xb, yb, rb in loader:
                    xb = xb.to(self.device)
                    rb = rb.to(self.device)
                    logits = self.module(xb)
                    p_buy = torch.sigmoid(logits)
                    # Expected utility: P(buy)*r_buy + P(skip)*r_skip
                    # With r_skip = 0: utility = P(buy) * reward
                    utility = p_buy * rb
                    loss = -utility.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        else:
            # No rewards -- fall back to weighted BCE
            self._train(X_t, y_t)

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        logits = self._batched_logits(X)
        proba_1 = expit(logits)
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Policy Gradient (REINFORCE) classifier
# ---------------------------------------------------------------------------


class PolicyGradientClassifier(ClassifierMixin, _BaseTorchClassifier):
    """Sklearn-compatible REINFORCE classifier for reward-based training.

    Instead of cross-entropy on binary labels, trains a policy network using
    REINFORCE with optional custom rewards (e.g. forward returns).

    The network outputs action logits for 2 classes. During training, actions
    are sampled from the policy; during inference, softmax gives probabilities.

    Usage in experiment.py:
        from research.model_wrappers import PolicyGradientClassifier, TorchMLP

        def build_model(y_train):
            return PolicyGradientClassifier(
                module=TorchMLP(input_dim=54, hidden_dims=(128, 64)),
                epochs=50, lr=1e-3, batch_size=512,
                entropy_coef=0.01, baseline=True,
            )

        # In pipeline, with custom rewards:
        model.fit(X_train, y_train, rewards=forward_returns_5d)
    """

    def __init__(self, module, epochs=50, lr=1e-3, batch_size=512,
                 entropy_coef=0.01, baseline=True, pos_weight=1.0):
        super().__init__(module, epochs, lr, batch_size, pos_weight)
        self.entropy_coef = entropy_coef
        self.baseline = baseline
        self._has_extra_head = False
        self._policy_head_built = False

    def _build_policy_head(self, module):
        """Replace single-output head with 2-class logits for policy gradient.

        Finds the last Linear(*, 1) layer and swaps it for Linear(*, 2).
        If no such layer exists, appends a Linear(last_out, 2) head.
        Idempotent: safe to call on repeated fit().
        """
        import torch.nn as nn

        # Already converted on a previous fit() call
        if self._policy_head_built:
            return module

        last_linear = None
        last_name = None
        for name, layer in module.named_modules():
            if isinstance(layer, nn.Linear) and layer.out_features == 1:
                last_linear = layer
                last_name = name

        if last_linear is not None and last_name:
            # Named child -- replace in-place
            new_layer = nn.Linear(last_linear.in_features, 2)
            parts = last_name.split(".")
            parent = module
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            if parts[-1].isdigit():
                parent[int(parts[-1])] = new_layer
            else:
                setattr(parent, parts[-1], new_layer)
            self._policy_head_built = True
        elif last_linear is not None:
            # Root module IS the Linear(*, 1) -- module outputs (batch, 1)
            self._policy_head = nn.Linear(last_linear.out_features, 2)
            self._has_extra_head = True
            self._policy_head_built = True
        else:
            # No Linear(*, 1) -- find any last Linear and append a 2-class head
            last_any = None
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    last_any = layer
            if last_any is not None:
                self._policy_head = nn.Linear(last_any.out_features, 2)
                self._has_extra_head = True
                self._policy_head_built = True
            else:
                raise ValueError("Module has no Linear layers; cannot build policy head")

        return module

    def _policy_forward(self, x):
        """Forward pass returning 2-class logits."""
        out = self.module(x)
        if self._has_extra_head:
            out = self._policy_head(out)
        return out

    def fit(self, X, y, rewards=None):
        """Train with REINFORCE.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (used as actions for reward assignment).
            rewards: Optional signed reward per sample (e.g. forward returns).
                     Positive = buying was good, negative = buying was bad.
                     Action 1 (buy) receives rb; action 0 (skip) receives -rb
                     (opportunity cost: skipping a good trade is penalized).
                     If None, uses +pos_weight for y=1 and -1 for y=0.
        """
        import torch
        from torch.distributions import Categorical
        from torch.utils.data import DataLoader, TensorDataset

        if len(X) == 0:
            self.classes_ = np.array([0, 1])
            return self

        self.module = self._build_policy_head(self.module)
        self.module.to(self.device)
        if self._has_extra_head:
            self._policy_head.to(self.device)
        all_params = list(self.module.parameters())
        if self._has_extra_head:
            all_params += list(self._policy_head.parameters())
        optimizer = torch.optim.Adam(all_params, lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        if rewards is not None:
            r_t = torch.tensor(rewards, dtype=torch.float32)
        else:
            r_t = None

        loader = DataLoader(TensorDataset(X_t, y_t, r_t if r_t is not None else y_t),
                            batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            self.module.train()
            for batch in loader:
                xb = batch[0].to(self.device)
                yb = batch[1].to(self.device)
                rb = batch[2].to(self.device) if rewards is not None else None

                logits = self._policy_forward(xb)
                dist = Categorical(logits=logits)
                actions = dist.sample()

                if rb is not None:
                    # Custom rewards: action=buy gets rb, action=skip gets -rb.
                    # Opportunity cost: positive rb reinforces buy/penalizes skip,
                    # negative rb penalizes buy/reinforces skip.
                    batch_rewards = torch.where(actions == 1, rb, -rb)
                else:
                    # Default: asymmetric binary reward from labels
                    correct = (actions == yb).float()
                    weight = torch.where(yb == 1, self.pos_weight, 1.0)
                    batch_rewards = correct * weight - (1 - correct) * weight

                if self.baseline:
                    batch_rewards = batch_rewards - batch_rewards.mean()

                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                policy_loss = -(log_probs * batch_rewards).mean()
                loss = policy_loss - self.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        if len(X) == 0:
            return np.empty((0, 2), dtype=np.float32)

        self.module.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = torch.tensor(X[i:i + self.batch_size], dtype=torch.float32).to(self.device)
                logits = self._policy_forward(batch)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Lazy-loaded nn.Module subclasses (torch imported only when accessed)
# ---------------------------------------------------------------------------


def _build_torch_modules():
    """Define nn.Module subclasses — called once on first access."""
    import torch
    from torch import nn

    class TorchMLP(nn.Module):
        """Configurable MLP for tabular classification."""

        def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.3):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class LSTMNet(nn.Module):
        """LSTM for sequential tabular data."""

        def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    class GRUNet(nn.Module):
        """GRU for sequential tabular data."""

        def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    class TransformerNet(nn.Module):
        """Transformer encoder for sequential tabular data.

        Projects input features to d_model, adds learned positional encoding,
        runs through TransformerEncoder layers, and pools the final timestep.
        """

        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                     dim_feedforward=128, dropout=0.1):
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            self.pos_enc = nn.Parameter(torch.zeros(1, 512, d_model, dtype=torch.float32))
            nn.init.trunc_normal_(self.pos_enc, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            seq_len = x.size(1)
            x = self.proj(x) + self.pos_enc[:, :seq_len, :]
            x = self.encoder(x)
            return self.head(x[:, -1, :])

    globals()["TorchMLP"] = TorchMLP
    globals()["LSTMNet"] = LSTMNet
    globals()["GRUNet"] = GRUNet
    globals()["TransformerNet"] = TransformerNet


_LAZY_MODULES = ("TorchMLP", "LSTMNet", "GRUNet", "TransformerNet")


def __getattr__(name):
    if name in _LAZY_MODULES:
        _build_torch_modules()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
