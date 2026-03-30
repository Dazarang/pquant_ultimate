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
        logits = self._batched_logits(X_seq)
        proba_1 = np.full(len(X), 0.5)
        proba_1[valid] = expit(logits)
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X, groups=None):
        return (self.predict_proba(X, groups)[:, 1] > 0.5).astype(int)


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
                epochs=50, lr=1e-3, batch_size=512, gamma=0.99,
                entropy_coef=0.01, baseline=True,
            )

        # In pipeline, with custom rewards:
        model.fit(X_train, y_train, rewards=forward_returns_5d)
    """

    def __init__(self, module, epochs=50, lr=1e-3, batch_size=512,
                 gamma=0.99, entropy_coef=0.01, baseline=True, pos_weight=1.0):
        super().__init__(module, epochs, lr, batch_size, pos_weight)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.baseline = baseline

    def _build_policy_head(self, module):
        """Replace single-output head with 2-class logits for policy gradient."""
        import torch.nn as nn

        # Find the last Linear layer and replace with 2-output version
        last_linear = None
        last_name = None
        for name, layer in module.named_modules():
            if isinstance(layer, nn.Linear) and layer.out_features == 1:
                last_linear = layer
                last_name = name

        if last_linear is not None:
            new_layer = nn.Linear(last_linear.in_features, 2)
            # Navigate to parent and replace
            parts = last_name.split(".")
            parent = module
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            if parts[-1].isdigit():
                parent[int(parts[-1])] = new_layer
            else:
                setattr(parent, parts[-1], new_layer)

        return module

    def fit(self, X, y, rewards=None):
        """Train with REINFORCE.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (used as actions for reward assignment).
            rewards: Optional custom reward per sample. If None, uses
                     +pos_weight for correct minority (y=1), +1 for correct
                     majority (y=0), -1 for incorrect.
        """
        import torch
        from torch.distributions import Categorical
        from torch.utils.data import DataLoader, TensorDataset

        self.module = self._build_policy_head(self.module)
        self.module.to(self.device)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)

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

                logits = self.module(xb)
                dist = Categorical(logits=logits)
                actions = dist.sample()

                if rb is not None:
                    # Custom rewards: reward if action matches label, scaled by reward magnitude
                    correct = (actions == yb).float()
                    batch_rewards = correct * rb.abs() - (1 - correct) * rb.abs()
                else:
                    # Default: asymmetric binary reward
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

        self.module.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = torch.tensor(X[i:i + self.batch_size], dtype=torch.float32).to(self.device)
                logits = self.module(batch)
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
        """GRU for sequential tabular data. Lighter alternative to LSTMNet."""

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
            self.pos_enc = nn.Parameter(torch.zeros(1, 512, d_model))
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
