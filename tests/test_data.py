"""Tests for lib/data.py -- loading, filtering, splitting, scaling."""

import numpy as np
import pandas as pd
import pytest

from lib.data import LABEL_COL, META_COLS, load_dataset, scale, temporal_split

DS_PATH = "data/datasets/20260115/dataset.parquet"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_dataset():
    """Load full dataset once for the module."""
    return load_dataset(DS_PATH)


@pytest.fixture(scope="module")
def small_dataset():
    """Load 3-stock subset for faster tests."""
    return load_dataset(DS_PATH, stocks=["AAPL", "MSFT", "TSLA"])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_structure(self, full_dataset):
        df, feature_cols = full_dataset
        for c in META_COLS:
            assert c in df.columns
        assert LABEL_COL in df.columns
        assert "PivotHigh" not in df.columns

    def test_label_binary(self, full_dataset):
        df, _ = full_dataset
        assert set(df[LABEL_COL].unique()).issubset({0, 1})

    def test_no_feature_leakage(self, full_dataset):
        _, feature_cols = full_dataset
        for c in feature_cols:
            assert c not in META_COLS
            assert c != LABEL_COL

    def test_no_inf_nan_in_features(self, full_dataset):
        df, feature_cols = full_dataset
        assert np.isinf(df[feature_cols].values).sum() == 0
        assert df[feature_cols].isna().sum().sum() == 0

    def test_no_duplicate_rows(self, full_dataset):
        df, _ = full_dataset
        assert df.duplicated(subset=["date", "stock_id"]).sum() == 0

    def test_date_is_datetime(self, full_dataset):
        df, _ = full_dataset
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_column_count(self, full_dataset):
        df, feature_cols = full_dataset
        assert len(df.columns) == len(META_COLS) + len(feature_cols) + 1


# ---------------------------------------------------------------------------
# Stock filtering
# ---------------------------------------------------------------------------


class TestStockFiltering:
    def test_single_stock(self):
        df, _ = load_dataset(DS_PATH, stocks="AAPL")
        assert df["stock_id"].nunique() == 1
        assert (df["stock_id"] == "AAPL").all()

    def test_stock_subset(self):
        tickers = ["AAPL", "MSFT", "TSLA"]
        df, _ = load_dataset(DS_PATH, stocks=tickers)
        assert df["stock_id"].nunique() == 3
        assert set(df["stock_id"].unique()) == set(tickers)

    def test_missing_stock_warns(self, capsys):
        load_dataset(DS_PATH, stocks=["AAPL", "DOESNTEXIST"])
        assert "DOESNTEXIST" in capsys.readouterr().out

    def test_feature_cols_consistent(self):
        _, fc1 = load_dataset(DS_PATH, stocks="AAPL")
        _, fc3 = load_dataset(DS_PATH, stocks=["AAPL", "MSFT", "TSLA"])
        assert fc1 == fc3


# ---------------------------------------------------------------------------
# Feature filtering
# ---------------------------------------------------------------------------


class TestFeatureFiltering:
    def test_five_features(self):
        feats = ["rsi_14", "macd", "volume_z", "drawdown", "ret_1d"]
        df, fc = load_dataset(DS_PATH, stocks="AAPL", features=feats)
        assert len(fc) == 5
        assert set(fc) == set(feats)
        assert len(df.columns) == len(META_COLS) + 5 + 1

    def test_ten_features(self):
        feats = ["rsi_14", "macd", "volume_z", "drawdown", "ret_1d",
                 "adx", "bb_position", "atr_14", "stoch_k", "ema_20"]
        df, fc = load_dataset(DS_PATH, stocks="AAPL", features=feats)
        assert len(fc) == 10
        assert len(df.columns) == len(META_COLS) + 10 + 1

    def test_all_features_default(self, full_dataset):
        _, fc = full_dataset
        assert len(fc) == 231

    def test_bad_feature_ignored(self, capsys):
        _, fc = load_dataset(DS_PATH, stocks="AAPL", features=["rsi_14", "FAKE"])
        assert len(fc) == 1
        assert "FAKE" in capsys.readouterr().out

    def test_combined_stock_and_feature_filter(self):
        feats = ["rsi_14", "macd"]
        df, fc = load_dataset(DS_PATH, stocks=["AAPL", "MSFT"], features=feats)
        assert df["stock_id"].nunique() == 2
        assert len(fc) == 2


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_no_temporal_overlap(self, small_dataset):
        df, _ = small_dataset
        train, val, test = temporal_split(df)
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

    def test_embargo_gap(self, small_dataset):
        df, _ = small_dataset
        train, val, test = temporal_split(df)
        dates = sorted(df["date"].unique())
        gap1 = dates.index(val["date"].min()) - dates.index(train["date"].max())
        gap2 = dates.index(test["date"].min()) - dates.index(val["date"].max())
        assert gap1 >= 14
        assert gap2 >= 14

    def test_rows_accounted_for(self, small_dataset):
        df, _ = small_dataset
        train, val, test = temporal_split(df)
        embargo = df[
            ((df["date"] > train["date"].max()) & (df["date"] < val["date"].min()))
            | ((df["date"] > val["date"].max()) & (df["date"] < test["date"].min()))
        ]
        assert len(train) + len(val) + len(test) + len(embargo) == len(df)

    def test_label_in_all_splits(self, small_dataset):
        df, _ = small_dataset
        train, val, test = temporal_split(df)
        for split in [train, val, test]:
            assert LABEL_COL in split.columns
            assert set(split[LABEL_COL].unique()).issubset({0, 1})

    def test_all_splits_nonempty(self, small_dataset):
        df, _ = small_dataset
        train, val, test = temporal_split(df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_single_stock_split(self):
        df, _ = load_dataset(DS_PATH, stocks="AAPL")
        train, val, test = temporal_split(df)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


class TestScale:
    def test_train_mean_zero(self, small_dataset):
        df, fc = small_dataset
        train, val, test = temporal_split(df)
        train_s, _, _, _ = scale(train, val, test, fc)
        assert train_s[fc].mean().abs().mean() < 0.01

    def test_train_std_one(self, small_dataset):
        df, fc = small_dataset
        train, val, test = temporal_split(df)
        train_s, _, _, _ = scale(train, val, test, fc)
        assert abs(train_s[fc].std().mean() - 1.0) < 0.05

    def test_scaler_feature_count(self, small_dataset):
        df, fc = small_dataset
        train, val, test = temporal_split(df)
        _, _, _, scaler = scale(train, val, test, fc)
        assert scaler.n_features_in_ == len(fc)

    def test_originals_unchanged(self, small_dataset):
        df, fc = small_dataset
        train, val, test = temporal_split(df)
        train_orig = train[fc].iloc[0].values.copy()
        train_s, _, _, _ = scale(train, val, test, fc)
        np.testing.assert_array_equal(train[fc].iloc[0].values, train_orig)
        assert not np.allclose(train_s[fc].iloc[0].values, train_orig)

    def test_no_nan_after_scale(self, small_dataset):
        df, fc = small_dataset
        train, val, test = temporal_split(df)
        train_s, val_s, test_s, _ = scale(train, val, test, fc)
        for split in [train_s, val_s, test_s]:
            assert split[fc].isna().sum().sum() == 0

    def test_meta_label_unchanged(self, small_dataset):
        df, fc = small_dataset
        train, val, test = temporal_split(df)
        train_s, _, _, _ = scale(train, val, test, fc)
        for c in META_COLS + [LABEL_COL]:
            if c in train.columns:
                pd.testing.assert_series_equal(train_s[c], train[c])

    def test_works_with_feature_subset(self):
        feats = ["rsi_14", "macd", "volume_z"]
        df, fc = load_dataset(DS_PATH, stocks="AAPL", features=feats)
        train, val, test = temporal_split(df)
        train_s, val_s, test_s, scaler = scale(train, val, test, fc)
        assert scaler.n_features_in_ == 3
        assert train_s[fc].isna().sum().sum() == 0
