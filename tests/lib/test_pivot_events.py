"""Tests for pivot-low event metadata and label semantics."""

import numpy as np
import pandas as pd

from lib.pivot_events import (
    PIVOT_LOW_BASE_COL,
    PIVOT_LOW_EVENT_ID_COL,
    PIVOT_LOW_EVENT_OFFSET_COL,
    annotate_pivot_low_events,
    ensure_pivot_low_event_columns,
)


class TestAnnotatePivotLowEvents:
    def test_adjacent_day_only_expands_within_tolerance(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=7),
            "stock_id": ["AAA"] * 7,
            "open": [10.0, 9.0, 8.1, 8.0, 8.3, 9.0, 10.0],
            "high": [10.2, 9.2, 8.3, 8.1, 8.5, 9.2, 10.2],
            "low": [9.8, 8.8, 7.9, 7.8, 8.1, 8.8, 9.8],
            "close": [10.0, 9.0, 8.07, 8.0, 8.25, 9.0, 10.0],
            "volume": [100] * 7,
        })

        result = annotate_pivot_low_events(df, lb=2, rb=2)

        assert result["PivotLow"].tolist() == [0, 0, 1, 1, 0, 0, 0]
        assert result[PIVOT_LOW_BASE_COL].tolist() == [0, 0, 0, 1, 0, 0, 0]
        assert result[PIVOT_LOW_EVENT_ID_COL].tolist() == [0, 0, 1, 1, 0, 0, 0]
        assert np.isnan(result[PIVOT_LOW_EVENT_OFFSET_COL].iloc[0])
        assert result[PIVOT_LOW_EVENT_OFFSET_COL].iloc[2] == -1
        assert result[PIVOT_LOW_EVENT_OFFSET_COL].iloc[3] == 0
        assert np.isnan(result[PIVOT_LOW_EVENT_OFFSET_COL].iloc[4])

    def test_multiple_base_pivots_get_distinct_event_ids(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=12),
            "stock_id": ["AAA"] * 12,
            "open": [10, 9, 8, 9, 10, 11, 10, 9, 8, 9, 10, 11],
            "high": [10.2, 9.2, 8.2, 9.2, 10.2, 11.2, 10.2, 9.2, 8.2, 9.2, 10.2, 11.2],
            "low": [9.8, 8.8, 7.8, 8.8, 9.8, 10.8, 9.8, 8.8, 7.8, 8.8, 9.8, 10.8],
            "close": [10, 9, 8, 9, 10, 11, 10, 9, 8, 9, 10, 11],
            "volume": [100] * 12,
        })

        result = annotate_pivot_low_events(df, lb=1, rb=1)

        event_rows = result[result["PivotLow"] == 1]
        assert event_rows[PIVOT_LOW_EVENT_ID_COL].nunique() == 2
        assert result[PIVOT_LOW_BASE_COL].sum() == 2

    def test_missing_event_columns_can_be_reconstructed_from_ohlcv(self):
        close = np.array(
            [130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102,
             100, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127],
            dtype=float,
        )
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=len(close)),
            "stock_id": ["AAA"] * len(close),
            "open": close + 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [100] * len(close),
        })

        expected = annotate_pivot_low_events(df)
        reconstructed = ensure_pivot_low_event_columns(df)

        assert reconstructed["PivotLow"].tolist() == expected["PivotLow"].tolist()
        assert reconstructed[PIVOT_LOW_BASE_COL].tolist() == expected[PIVOT_LOW_BASE_COL].tolist()
        assert reconstructed[PIVOT_LOW_EVENT_ID_COL].tolist() == expected[PIVOT_LOW_EVENT_ID_COL].tolist()
