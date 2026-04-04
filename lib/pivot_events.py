"""Shared pivot-low label semantics and event metadata.

This module is the single source of truth for the ML target:
- the exact base pivot center (`PivotLow_base`)
- the expanded buyable zone label (`PivotLow`)
- a unique bottom event id (`PivotLow_event_id`)
- the signed row offset from the base pivot (`PivotLow_event_offset`)

The expansion logic intentionally mirrors the production label definition:
adjacent days are included only when their close is within the configured
price tolerance of the base pivot close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.pattern import find_pivots

PIVOT_LABEL_LB = 8
PIVOT_LABEL_RB = 13
PIVOT_LABEL_WINDOW_VARIATIONS = (-1, 1)
PIVOT_LABEL_PRICE_TOLERANCE = 0.01

LABEL_COL = "PivotLow"
PIVOT_LOW_BASE_COL = "PivotLow_base"
PIVOT_LOW_EVENT_ID_COL = "PivotLow_event_id"
PIVOT_LOW_EVENT_OFFSET_COL = "PivotLow_event_offset"

PIVOT_LOW_EVENT_COLS = [
    PIVOT_LOW_BASE_COL,
    PIVOT_LOW_EVENT_ID_COL,
    PIVOT_LOW_EVENT_OFFSET_COL,
]

NON_FEATURE_COLS = [LABEL_COL, *PIVOT_LOW_EVENT_COLS]


def _candidate_is_better(
    current_distance: int,
    current_base_close: float,
    current_base_idx: int,
    candidate_distance: int,
    candidate_base_close: float,
    candidate_base_idx: int,
) -> bool:
    """Resolve rare overlapping-event assignments deterministically."""
    if candidate_distance != current_distance:
        return candidate_distance < current_distance
    if not np.isclose(candidate_base_close, current_base_close):
        return candidate_base_close < current_base_close
    return candidate_base_idx < current_base_idx


def annotate_pivot_low_events(
    stock_df: pd.DataFrame,
    lb: int = PIVOT_LABEL_LB,
    rb: int = PIVOT_LABEL_RB,
    window_variations: tuple[int, ...] = PIVOT_LABEL_WINDOW_VARIATIONS,
    price_tolerance: float = PIVOT_LABEL_PRICE_TOLERANCE,
) -> pd.DataFrame:
    """Return `stock_df` plus PivotLow base/event metadata.

    The input must contain a single stock ordered by date.
    """
    if stock_df.empty:
        result = stock_df.copy()
        result[LABEL_COL] = pd.Series(dtype=np.int8)
        result[PIVOT_LOW_BASE_COL] = pd.Series(dtype=np.int8)
        result[PIVOT_LOW_EVENT_ID_COL] = pd.Series(dtype=np.int32)
        result[PIVOT_LOW_EVENT_OFFSET_COL] = pd.Series(dtype=np.float32)
        return result

    result = stock_df.copy()
    if "close" not in result.columns:
        raise ValueError("annotate_pivot_low_events requires a 'close' column")

    result = result.sort_values("date").copy()

    _, base_low = find_pivots(
        result,
        lb=lb,
        rb=rb,
        return_boolean=True,
        window_variations=None,
        price_tolerance=price_tolerance,
    )
    _, expanded_low = find_pivots(
        result,
        lb=lb,
        rb=rb,
        return_boolean=True,
        window_variations=list(window_variations),
        price_tolerance=price_tolerance,
    )

    close = result["close"].to_numpy(dtype=float)
    n_rows = len(result)
    event_id = np.zeros(n_rows, dtype=np.int32)
    event_offset = np.full(n_rows, np.nan, dtype=np.float32)
    assigned_distance = np.full(n_rows, np.inf, dtype=np.float64)
    assigned_base_close = np.full(n_rows, np.inf, dtype=np.float64)
    assigned_base_idx = np.full(n_rows, np.iinfo(np.int32).max, dtype=np.int32)

    base_mask = base_low.to_numpy(dtype=bool)
    expanded_mask = expanded_low.to_numpy(dtype=bool)

    event_num = 0
    for base_idx in np.flatnonzero(base_mask):
        event_num += 1
        base_price = close[base_idx]

        event_id[base_idx] = event_num
        event_offset[base_idx] = 0
        assigned_distance[base_idx] = 0
        assigned_base_close[base_idx] = base_price
        assigned_base_idx[base_idx] = base_idx

        for offset in window_variations:
            adj_idx = base_idx + offset
            if not (0 <= adj_idx < n_rows):
                continue

            price_diff = abs(close[adj_idx] - base_price) / base_price
            if price_diff > price_tolerance:
                continue

            candidate_distance = abs(offset)
            if event_id[adj_idx] == 0 or _candidate_is_better(
                int(assigned_distance[adj_idx]),
                float(assigned_base_close[adj_idx]),
                int(assigned_base_idx[adj_idx]),
                candidate_distance,
                float(base_price),
                int(base_idx),
            ):
                event_id[adj_idx] = event_num
                event_offset[adj_idx] = offset
                assigned_distance[adj_idx] = candidate_distance
                assigned_base_close[adj_idx] = base_price
                assigned_base_idx[adj_idx] = base_idx

    result[LABEL_COL] = expanded_mask.astype(np.int8)
    result[PIVOT_LOW_BASE_COL] = base_mask.astype(np.int8)
    result[PIVOT_LOW_EVENT_ID_COL] = event_id
    result[PIVOT_LOW_EVENT_OFFSET_COL] = event_offset.astype(np.float32)

    return result


def ensure_pivot_low_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with PivotLow event columns available.

    Existing columns are preserved. Missing metadata is reconstructed from OHLCV
    using the same labeling mechanics as dataset generation.
    """
    required = [LABEL_COL, *PIVOT_LOW_EVENT_COLS]
    if all(col in df.columns for col in required):
        return df

    if "stock_id" not in df.columns or "date" not in df.columns:
        raise ValueError("PivotLow event reconstruction requires 'stock_id' and 'date'")

    parts = []
    for _, stock_df in df.groupby("stock_id", sort=False):
        stock_df = stock_df.sort_values("date").copy()
        annotated = annotate_pivot_low_events(stock_df)

        merged = stock_df.copy()
        for col in required:
            if col not in merged.columns:
                merged[col] = annotated[col].values
        parts.append(merged)

    return pd.concat(parts).sort_index()
