"""Model evaluation: multi-budget ranking framework (threshold-free)."""

import numba
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RETURN_WINSORIZE_QUANTILE = 0.01  # Clip returns at 1st/99th percentile

from lib.pivot_events import (
    LABEL_COL,
    PIVOT_LOW_BASE_COL,
    PIVOT_LOW_EVENT_ID_COL,
    PIVOT_LOW_EVENT_OFFSET_COL,
    ensure_pivot_low_event_columns,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None) -> dict:
    """Calculate classification metrics. Returns dict of metric name -> value."""
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["avg_precision"] = average_precision_score(y_true, y_pred_proba)

    return metrics


def print_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """Plot confusion matrix. Returns matplotlib figure."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    fig.tight_layout()
    return fig


@numba.njit(cache=True, fastmath=True)
def _backtest_single_trade(highs, lows, closes, entry_price, stop_pct, max_hold, start_idx):
    """Find exit with decaying trailing stop from new highs.

    Trailing stop starts at stop_pct, halves every 5 days.
    exit_type: -1=stop, 0=hold/end.
    """
    end_idx = min(start_idx + max_hold, len(highs))
    peak = entry_price

    for i in range(start_idx, end_idx):
        if highs[i] > peak:
            peak = highs[i]
        days_held = i - start_idx
        current_stop = stop_pct / (2 ** (days_held // 5))
        stop_price = peak * (1 - current_stop)
        if lows[i] <= stop_price:
            return i, stop_price, -1

    last_idx = end_idx - 1
    if last_idx < start_idx:
        last_idx = start_idx
    return last_idx, closes[last_idx], 0


_EXIT_REASONS = {-1: "stop", 0: "hold/end"}


def backtest_quick(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    stop_pct: float = 0.07,
    max_hold_days: int = 30,
) -> pd.DataFrame:
    """Backtest predicted pivot bottoms with decaying trailing stop.

    Trailing stop starts at stop_pct, halves every 5 days.
    Operates per stock.
    """
    trades = []
    df = ensure_pivot_low_event_columns(df).copy()

    for stock_id, stock_df in df.groupby("stock_id", sort=False):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)

        highs = stock_df["high"].values.astype(np.float64)
        lows = stock_df["low"].values.astype(np.float64)
        closes = stock_df["close"].values.astype(np.float64)
        opens = stock_df["open"].values.astype(np.float64)
        dates = stock_df["date"].values
        n = len(stock_df)
        valid_entry = pd.Series(np.arange(n) + 1 < n, index=stock_df.index)
        collapsed = _collapsed_binary_prediction_rows(stock_df, pred_col, valid_mask=valid_entry)
        signals = collapsed["date"].tolist() if not collapsed.empty else []
        if not signals:
            continue

        date_to_idx = {d: i for i, d in enumerate(dates)}

        for signal_date in signals:
            idx = date_to_idx.get(signal_date)
            if idx is None:
                continue
            entry_idx = idx + 1
            if entry_idx >= n:
                continue

            entry_price = opens[entry_idx]
            exit_idx, exit_price, exit_type = _backtest_single_trade(
                highs, lows, closes, entry_price, stop_pct, max_hold_days, entry_idx,
            )

            trades.append({
                "stock_id": stock_id,
                "entry_date": dates[entry_idx],
                "exit_date": dates[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": (exit_price - entry_price) / entry_price,
                "exit_reason": _EXIT_REASONS[exit_type],
            })

    if not trades:
        cols = ["stock_id", "entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "exit_reason"]
        return pd.DataFrame(columns=cols)

    result = pd.DataFrame(trades)
    _print_backtest_summary(result)
    return result


def _print_backtest_summary(trades: pd.DataFrame) -> None:
    """Print summary stats for backtest trades."""
    n = len(trades)
    pos = trades[trades["return_pct"] > 0]
    neg = trades[trades["return_pct"] <= 0]
    pos_avg = pos["return_pct"].mean() if len(pos) > 0 else 0
    neg_avg = neg["return_pct"].mean() if len(neg) > 0 else 0
    print(f"\nBacktest: {n} trades")
    print(f"  Win rate:    {len(pos) / n:.1%}")
    print(f"  Avg return:  {trades['return_pct'].mean():.2%}")
    print(f"  Med return:  {trades['return_pct'].median():.2%}")
    print(f"  Positive: {len(pos)} (avg {pos_avg:+.2%})  |  Negative: {len(neg)} (avg {neg_avg:+.2%})")


# ---------------------------------------------------------------------------
# Shared: per-signal analytics
# ---------------------------------------------------------------------------


def _valid_event_panel(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build a per-row panel with forward-return inputs and event metadata."""
    df = ensure_pivot_low_event_columns(df).copy()
    df = df.sort_values(["stock_id", "date"]).copy()

    entry = df.groupby("stock_id")["open"].shift(-1)
    exit_ = df.groupby("stock_id")["open"].shift(-(horizon + 1))
    fwd = exit_ / entry - 1

    min_low = df.groupby("stock_id")["low"].transform(
        lambda x: x.rolling(horizon).min().shift(-horizon)
    )
    mae = min_low / entry - 1

    # Market benchmark from RAW returns (before winsorization)
    market_by_date = fwd.groupby(df["date"]).mean()
    market = df["date"].map(market_by_date)

    # Winsorize forward returns and MAE at 1st/99th percentile
    fwd_valid = fwd.dropna()
    if len(fwd_valid) > 0:
        p_lo = fwd_valid.quantile(RETURN_WINSORIZE_QUANTILE)
        p_hi = fwd_valid.quantile(1 - RETURN_WINSORIZE_QUANTILE)
        fwd = fwd.clip(lower=p_lo, upper=p_hi)

    mae_valid = mae.dropna()
    if len(mae_valid) > 0:
        mae_p_lo = mae_valid.quantile(RETURN_WINSORIZE_QUANTILE)
        mae = mae.clip(lower=mae_p_lo)
    valid = fwd.notna() & mae.notna() & market.notna()

    panel = df[[
        "stock_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        LABEL_COL,
        PIVOT_LOW_BASE_COL,
        PIVOT_LOW_EVENT_ID_COL,
        PIVOT_LOW_EVENT_OFFSET_COL,
    ]].copy()
    panel["_entry"] = entry
    panel["_exit"] = exit_
    panel["_return"] = fwd
    panel["_mae"] = mae
    panel["_market"] = market
    panel["_valid"] = valid

    event_rows = panel[(panel[PIVOT_LOW_EVENT_ID_COL] > 0) & panel["_valid"]].copy()
    if event_rows.empty:
        panel["_best_zone_entry"] = np.nan
        return panel

    best_zone_entry = event_rows.groupby(["stock_id", PIVOT_LOW_EVENT_ID_COL])["_entry"].min()
    panel["_best_zone_entry"] = list(
        zip(panel["stock_id"], panel[PIVOT_LOW_EVENT_ID_COL])
    )
    panel["_best_zone_entry"] = panel["_best_zone_entry"].map(best_zone_entry)
    return panel


def _collapse_selected_signals(selected: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse duplicate predictions inside the same true bottom event."""
    if selected.empty:
        return selected.copy(), {
            "duplicate_rows": 0,
            "duplicate_mass_discarded": 0.0,
        }

    selected = selected.sort_values(["stock_id", "date"]).copy()
    hit_mask = selected[PIVOT_LOW_EVENT_ID_COL] > 0

    misses = selected.loc[~hit_mask].copy()
    misses["_event_exact_hit"] = False
    misses["_duplicate_rows"] = 0
    misses["_duplicate_mass_discarded"] = 0.0

    reps = []
    duplicate_rows = 0
    duplicate_mass_discarded = 0.0
    hit_signals = selected.loc[hit_mask].sort_values(
        ["stock_id", PIVOT_LOW_EVENT_ID_COL, "date"]
    )
    for _, grp in hit_signals.groupby(["stock_id", PIVOT_LOW_EVENT_ID_COL], sort=False):
        rep = grp.iloc[0].copy()
        total_weight = float(grp["_weight"].sum())
        rep["_weight"] = min(1.0, total_weight)
        rep["_event_exact_hit"] = bool((grp[PIVOT_LOW_BASE_COL] == 1).any())
        rep["_duplicate_rows"] = len(grp) - 1
        rep["_duplicate_mass_discarded"] = max(0.0, total_weight - rep["_weight"])

        duplicate_rows += int(rep["_duplicate_rows"])
        duplicate_mass_discarded += float(rep["_duplicate_mass_discarded"])
        reps.append(rep)

    hit_reps = pd.DataFrame(reps) if reps else selected.iloc[0:0].copy()
    collapsed = pd.concat([misses, hit_reps], ignore_index=False)
    collapsed = collapsed.sort_values(["stock_id", "date"]).copy()

    return collapsed, {
        "duplicate_rows": duplicate_rows,
        "duplicate_mass_discarded": duplicate_mass_discarded,
    }


def _collapsed_binary_prediction_rows(
    df: pd.DataFrame,
    pred_col: str,
    valid_mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Return selected rows after event-aware duplicate collapsing."""
    annotated = ensure_pivot_low_event_columns(df).copy()

    if valid_mask is None:
        mask = pd.Series(True, index=annotated.index)
    else:
        if isinstance(valid_mask, pd.Series):
            mask = valid_mask.reindex(annotated.index, fill_value=False).astype(bool)
        else:
            values = np.asarray(valid_mask, dtype=bool)
            if len(values) != len(annotated):
                raise ValueError("valid_mask must align 1:1 with df rows")
            mask = pd.Series(values, index=annotated.index)

    annotated = annotated.sort_values(["stock_id", "date"]).copy()
    mask = mask.reindex(annotated.index, fill_value=False)

    selected = annotated[(annotated[pred_col] == 1) & mask].copy()
    if selected.empty:
        return selected

    selected["_weight"] = 1.0
    collapsed, _ = _collapse_selected_signals(selected)
    return collapsed


def _signal_analytics(df: pd.DataFrame, y_pred: np.ndarray, horizon: int) -> dict | None:
    """Compute forward returns, MAE, and market benchmark per evaluated entry.

    Duplicate predictions inside the same true bottom event are collapsed to the
    earliest valid tradable row and their cumulative credit is capped at 1.
    Returns None if no valid evaluated entries exist for the given horizon.
    """
    signal_weights = np.asarray(y_pred, dtype=float)
    if len(signal_weights) != len(df):
        raise ValueError("signal weights must align 1:1 with df rows")

    panel = _valid_event_panel(df, horizon)
    panel["_weight"] = signal_weights

    valid_selected = panel[(panel["_weight"] > 0) & panel["_valid"]].copy()
    if valid_selected.empty:
        return None

    collapsed, duplicate_stats = _collapse_selected_signals(valid_selected)
    if collapsed.empty:
        return None

    valid_true_events = panel[(panel[PIVOT_LOW_EVENT_ID_COL] > 0) & panel["_valid"]]
    n_true_events = int(
        valid_true_events[["stock_id", PIVOT_LOW_EVENT_ID_COL]].drop_duplicates().shape[0]
    )
    n_true_bases = int(panel[(panel[PIVOT_LOW_BASE_COL] == 1) & panel["_valid"]].shape[0])

    hit_entries = collapsed[collapsed[PIVOT_LOW_EVENT_ID_COL] > 0].copy()
    event_hits = int(hit_entries[["stock_id", PIVOT_LOW_EVENT_ID_COL]].drop_duplicates().shape[0])
    exact_hits = int(hit_entries["_event_exact_hit"].sum()) if not hit_entries.empty else 0
    total_mass = float(collapsed["_weight"].sum())
    hit_mass = float(hit_entries["_weight"].sum()) if not hit_entries.empty else 0.0

    avg_entry_offset = np.nan
    avg_entry_slippage = np.nan
    if not hit_entries.empty:
        avg_entry_offset = float(
            np.average(
                hit_entries[PIVOT_LOW_EVENT_OFFSET_COL].to_numpy(dtype=float),
                weights=hit_entries["_weight"].to_numpy(dtype=float),
            )
        )
        slippage = hit_entries["_entry"] / hit_entries["_best_zone_entry"] - 1
        avg_entry_slippage = float(
            np.average(slippage.to_numpy(dtype=float), weights=hit_entries["_weight"].to_numpy(dtype=float))
        )

    return {
        "returns": collapsed["_return"],
        "mae": collapsed["_mae"],
        "market": collapsed["_market"],
        "dates": collapsed["date"],
        "weights": collapsed["_weight"],
        "raw_signal_rows": int(len(valid_selected)),
        "raw_signal_mass": float(valid_selected["_weight"].sum()),
        "collapsed_rows": int(len(collapsed)),
        "collapsed_signal_mass": total_mass,
        "duplicate_rows": int(duplicate_stats["duplicate_rows"]),
        "duplicate_mass_discarded": float(duplicate_stats["duplicate_mass_discarded"]),
        "false_positive_rows": int((collapsed[PIVOT_LOW_EVENT_ID_COL] == 0).sum()),
        "event_hits": event_hits,
        "exact_hits": exact_hits,
        "n_true_events": n_true_events,
        "n_true_bases": n_true_bases,
        "event_recall": event_hits / n_true_events if n_true_events else 0.0,
        "exact_center_recall": exact_hits / n_true_bases if n_true_bases else 0.0,
        "zone_precision": hit_mass / total_mass if total_mass > 0 else 0.0,
        "avg_entry_offset_days": avg_entry_offset,
        "avg_entry_slippage": avg_entry_slippage,
    }


# ---------------------------------------------------------------------------
# Tier 2: Forward Returns
# ---------------------------------------------------------------------------


def forward_returns(df: pd.DataFrame, y_pred: np.ndarray, horizons: list[int] | None = None) -> dict:
    """Measure actual returns at fixed horizons for evaluated event-aware entries.

    Entry at next-day open, exit at open N days later. Duplicate predictions
    inside a true bottom event collapse to one earliest tradable entry.
    """
    if horizons is None:
        horizons = [5, 10, 20]

    results = {}
    for h in horizons:
        analytics = _signal_analytics(df, y_pred, h)
        if analytics is None:
            continue

        fwd = analytics["returns"]
        mae = analytics["mae"]
        market = analytics["market"]
        dates = analytics["dates"]
        weights = analytics["weights"]
        excess = fwd - market

        wins = fwd[fwd > 0]
        losses = fwd[fwd < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")

        results[f"mean_{h}d"] = _weighted_mean(fwd, weights)
        results[f"excess_{h}d"] = _weighted_mean(excess, weights)
        results[f"win_rate_{h}d"] = _weighted_mean(fwd > 0, weights)
        results[f"avg_win_{h}d"] = _weighted_mean(wins, weights.loc[wins.index]) if len(wins) else 0.0
        results[f"avg_loss_{h}d"] = _weighted_mean(losses, weights.loc[losses.index]) if len(losses) else 0.0
        results[f"profit_factor_{h}d"] = pf
        results[f"mae_{h}d"] = _weighted_mean(mae, weights)
        results[f"worst_mae_{h}d"] = mae.min()
        results[f"n_signals_{h}d"] = len(fwd)
        results[f"effective_n_{h}d"] = _effective_signal_days(dates, weights)
        results[f"event_recall_{h}d"] = analytics["event_recall"]
        results[f"exact_center_recall_{h}d"] = analytics["exact_center_recall"]
        results[f"zone_precision_{h}d"] = analytics["zone_precision"]
        results[f"event_hits_{h}d"] = analytics["event_hits"]
        results[f"true_events_{h}d"] = analytics["n_true_events"]
        results[f"duplicate_rows_{h}d"] = analytics["duplicate_rows"]
        results[f"duplicate_mass_{h}d"] = analytics["duplicate_mass_discarded"]
        results[f"avg_entry_offset_{h}d"] = analytics["avg_entry_offset_days"]
        results[f"avg_entry_slippage_{h}d"] = analytics["avg_entry_slippage"]

    _print_forward_returns(results, horizons)
    return results


def _print_forward_returns(results: dict, horizons: list[int]) -> None:
    print("\nForward Returns:")
    for h in horizons:
        if f"mean_{h}d" not in results:
            continue
        n = int(results[f"n_signals_{h}d"])
        eff_n = int(results[f"effective_n_{h}d"])
        print(
            f"  {h:>2}d: mean={results[f'mean_{h}d']:+.2%}  "
            f"excess={results[f'excess_{h}d']:+.2%}  "
            f"win={results[f'win_rate_{h}d']:.1%}  "
            f"PF={results[f'profit_factor_{h}d']:.2f}  "
            f"MAE={results[f'mae_{h}d']:+.2%}  "
            f"event_recall={results[f'event_recall_{h}d']:.1%}  "
            f"zone_precision={results[f'zone_precision_{h}d']:.1%}  "
            f"n={n} (eff={eff_n})"
        )


# ---------------------------------------------------------------------------
# Tier 3: Composite Score
# ---------------------------------------------------------------------------


def composite_score(df: pd.DataFrame, y_pred: np.ndarray, horizon: int = 10) -> float:
    """Risk-adjusted composite score with proper scaling and path risk.

    Uses excess return (vs equal-weight market), win-rate edge (centered on 50%),
    tail-risk penalty, falling-knife penalty, and MAE penalty. Higher is better.
    """
    analytics = _signal_analytics(df, y_pred, horizon)
    if analytics is None:
        return float("-inf")

    fwd = analytics["returns"]
    market = analytics["market"]
    mae = analytics["mae"]
    dates = analytics["dates"]
    weights = analytics["weights"]

    excess_return = _weighted_mean(fwd - market, weights)
    mean_return = _weighted_mean(fwd, weights)
    market_mean = _weighted_mean(market, weights)
    win_rate = _weighted_mean(fwd > 0, weights)
    worst_decile = _weighted_quantile(fwd, weights, 0.1)
    knife_rate = _weighted_mean(fwd < -0.05, weights)
    mean_mae = _weighted_mean(mae, weights)
    effective_n = _effective_signal_days(dates, weights)

    score = (
        0.30 * excess_return * 100
        + 0.25 * (win_rate - 0.5) * 100
        - 0.20 * abs(min(0.0, worst_decile)) * 100
        - 0.10 * knife_rate * 100
        - 0.15 * abs(mean_mae) * 100
    )

    print(f"\nComposite Score ({horizon}d): {score:.2f}")
    print(f"  Excess return: {excess_return:+.2%} (raw: {mean_return:+.2%}, mkt: {market_mean:+.2%})")
    print(f"  Win rate:      {win_rate:.1%} (edge: {win_rate - 0.5:+.1%})")
    print(f"  Worst 10%:     {worst_decile:+.2%}")
    print(f"  Knife rate:    {knife_rate:.1%} (>{5}% loss)")
    print(f"  Mean MAE:      {mean_mae:+.2%}")
    print(f"  Signals:       {len(fwd)} (effective: {_format_count(effective_n)})")
    print(
        f"  Event recall:  {analytics['event_recall']:.1%} "
        f"({analytics['event_hits']}/{analytics['n_true_events']})"
    )
    print(
        f"  Exact recall:  {analytics['exact_center_recall']:.1%} "
        f"({analytics['exact_hits']}/{analytics['n_true_bases']})"
    )
    print(
        f"  Zone precision:{analytics['zone_precision']:.1%}  "
        f"dup_rows={analytics['duplicate_rows']}  "
        f"dup_mass={analytics['duplicate_mass_discarded']:.2f}"
    )

    return score


# ---------------------------------------------------------------------------
# Regime breakdown (informational)
# ---------------------------------------------------------------------------


def regime_breakdown(df: pd.DataFrame, y_pred: np.ndarray, horizon: int = 10) -> dict:
    """Signal quality per market regime. Uses trailing 20d equal-weight return."""
    analytics = _signal_analytics(df, y_pred, horizon)
    if analytics is None:
        print("  No valid signals")
        return {}

    fwd = analytics["returns"]
    mae = analytics["mae"]
    dates = analytics["dates"]

    trailing = df.groupby("stock_id")["close"].transform(lambda x: x.pct_change(20))
    trailing_by_date = trailing.groupby(df["date"]).mean()
    signal_trailing = dates.map(trailing_by_date)

    regime = pd.Series("unknown", index=fwd.index)
    regime[signal_trailing > 0] = "bull"
    regime[signal_trailing <= 0] = "bear"

    result = {}
    for r in ["bull", "bear"]:
        mask = regime == r
        if not mask.any():
            continue
        r_fwd = fwd[mask]
        r_mae = mae[mask]
        result[r] = {
            "mean_return": r_fwd.mean(),
            "win_rate": (r_fwd > 0).mean(),
            "knife_rate": (r_fwd < -0.05).mean(),
            "mean_mae": r_mae.mean(),
            "n_signals": len(r_fwd),
        }
        print(
            f"  {r:>4}: mean={r_fwd.mean():+.2%}  "
            f"win={(r_fwd > 0).mean():.1%}  "
            f"MAE={r_mae.mean():+.2%}  "
            f"knife={(r_fwd < -0.05).mean():.1%}  "
            f"n={len(r_fwd)}"
        )

    return result


# ---------------------------------------------------------------------------
# Multi-budget evaluation (threshold-free)
# ---------------------------------------------------------------------------

BUDGET_FRACS = (0.001, 0.0025, 0.005, 0.01, 0.02)

# _cell_score formula weights (must sum to 1.0)
CELL_SCORE_WEIGHTS = {
    "excess": 0.40,
    "win_rate": 0.20,
    "worst_decile": 0.10,
    "knife_rate": 0.10,
    "tail_mae": 0.05,
    "entry_slippage": 0.15,
}


def _budget_signal_count(n_rows: int, frac: float) -> int:
    """Return the target number of rows for a budget fraction."""
    if n_rows <= 0:
        return 0
    return max(1, int(n_rows * frac))


def _select_top_frac_weights(y_pred_proba: np.ndarray, frac: float) -> np.ndarray:
    """Return exact-budget weights, splitting cutoff ties evenly."""
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    n_rows = len(y_pred_proba)
    if n_rows == 0:
        return np.zeros(0, dtype=float)

    k = _budget_signal_count(n_rows, frac)
    if k >= n_rows:
        return np.ones(n_rows, dtype=float)

    cutoff = np.partition(y_pred_proba, n_rows - k)[n_rows - k]
    above = y_pred_proba > cutoff
    tied = y_pred_proba == cutoff

    weights = above.astype(float)
    remaining = k - int(above.sum())
    if remaining <= 0:
        return weights

    tie_count = int(tied.sum())
    if tie_count == 0:
        return weights

    weights[tied] = remaining / tie_count
    return weights


def select_top_frac(y_pred_proba: np.ndarray, frac: float) -> np.ndarray:
    """Select top fraction of predictions as binary signals.

    If probabilities tie at the cutoff, all tied rows are included rather than
    picking an arbitrary subset. The composite scorer uses fractional weights
    internally to preserve the exact budget mass.
    """
    weights = _select_top_frac_weights(y_pred_proba, frac)
    return (weights > 0).astype(int)


def _weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    """Compute a weighted mean for aligned values and weights."""
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _weighted_quantile(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, q: float) -> float:
    """Compute a weighted quantile for aligned values and weights."""
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    order = np.argsort(values_arr)
    sorted_values = values_arr[order]
    sorted_weights = weights_arr[order]
    cdf = np.cumsum(sorted_weights) / sorted_weights.sum()
    idx = min(np.searchsorted(cdf, q, side="left"), len(sorted_values) - 1)
    return float(sorted_values[idx])


def _effective_signal_days(dates: pd.Series, weights: pd.Series | np.ndarray) -> float:
    """Approximate independent bet-days, allowing fractional cutoff ties."""
    weight_series = pd.Series(np.asarray(weights, dtype=float), index=dates.index)
    per_date = weight_series.groupby(dates).sum()
    return float(np.minimum(per_date.to_numpy(), 1.0).sum())


def _format_budget_label(frac: float) -> str:
    """Format a budget fraction as a percentage label."""
    return f"{frac * 100:.2f}%"


def _format_count(value: float) -> str:
    """Format counts cleanly while preserving fractional values."""
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.2f}"


def forward_open_return(stock_df: pd.DataFrame, signal_date, horizon: int) -> float | None:
    """Return next-open to horizon-open return, or None if the full window is missing."""
    future_opens = stock_df.loc[stock_df["date"] > signal_date, "open"].head(horizon + 1)
    if len(future_opens) < horizon + 1:
        return None

    entry = future_opens.iloc[0]
    exit_ = future_opens.iloc[horizon]
    return float((exit_ - entry) / entry)


def _cell_score(df: pd.DataFrame, y_pred: np.ndarray, horizon: int) -> dict | None:
    """Score one (budget, horizon) cell. Returns metrics dict or None."""
    analytics = _signal_analytics(df, y_pred, horizon)
    if analytics is None:
        return None

    fwd = analytics["returns"]
    market = analytics["market"]
    mae = analytics["mae"]
    dates = analytics["dates"]
    weights = analytics["weights"]

    effective_n = _effective_signal_days(dates, weights)
    excess = _weighted_mean(fwd - market, weights)
    win_rate = _weighted_mean(fwd > 0, weights)
    worst_decile = _weighted_quantile(fwd, weights, 0.1)
    knife_rate = _weighted_mean(fwd < -0.05, weights)
    tail_mae = _weighted_quantile(mae, weights, 0.25)
    slip = analytics["avg_entry_slippage"]
    entry_slip = abs(slip) if not np.isnan(slip) else 0.0

    cw = CELL_SCORE_WEIGHTS
    raw = (
        cw["excess"] * excess * 100
        + cw["win_rate"] * (win_rate - 0.5) * 100
        - cw["worst_decile"] * abs(min(0.0, worst_decile)) * 100
        - cw["knife_rate"] * knife_rate * 100
        - cw["tail_mae"] * abs(tail_mae) * 100
        - cw["entry_slippage"] * entry_slip * 100
    )

    w = effective_n / (effective_n + 50)

    return {
        "raw": raw, "weighted": w * raw, "w": w,
        "n_signals": float(weights.sum()),
        "n_rows": int((weights > 0).sum()),
        "tied_rows": int(((weights > 0) & (weights < 1)).sum()),
        "effective_n": effective_n,
        "excess": excess, "win_rate": win_rate,
        "worst_decile": worst_decile, "knife_rate": knife_rate,
        "tail_mae": tail_mae, "entry_slippage": entry_slip,
        "event_recall": analytics["event_recall"],
        "exact_center_recall": analytics["exact_center_recall"],
        "zone_precision": analytics["zone_precision"],
        "event_hits": analytics["event_hits"],
        "n_true_events": analytics["n_true_events"],
        "exact_hits": analytics["exact_hits"],
        "n_true_bases": analytics["n_true_bases"],
        "duplicate_rows": analytics["duplicate_rows"],
        "duplicate_mass_discarded": analytics["duplicate_mass_discarded"],
        "avg_entry_offset_days": analytics["avg_entry_offset_days"],
        "avg_entry_slippage": analytics["avg_entry_slippage"],
    }


def multi_budget_composite(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    budgets: tuple[float, ...] = BUDGET_FRACS,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> tuple[float, dict]:
    """Multi-budget, multi-horizon composite score with soft N-scaling.

    Evaluates model ranking at multiple signal budgets and holding horizons.
    Final score = mean of W * U across valid (budget, horizon) cells,
    where W = effective_n / (effective_n + 50). Missing cells are excluded.
    """
    valid_weighted = []
    total_cells = 0
    details = {}

    for q in budgets:
        signal_weights = _select_top_frac_weights(y_pred_proba, q)
        budget = float(q)
        details[budget] = {
            "budget": budget,
            "label": _format_budget_label(q),
            "signal_mass": float(signal_weights.sum()),
            "n_rows": int((signal_weights > 0).sum()),
            "tied_rows": int(((signal_weights > 0) & (signal_weights < 1)).sum()),
            "cells": {},
        }

        for h in horizons:
            total_cells += 1
            cell = _cell_score(df, signal_weights, h)
            if cell is None:
                continue
            valid_weighted.append(cell["weighted"])
            details[budget]["cells"][h] = cell

    score = np.mean(valid_weighted) if valid_weighted else float("-inf")
    details["_meta"] = {"valid_cells": len(valid_weighted), "total_cells": total_cells}
    return score, details


def _print_multi_budget(details: dict, horizons: tuple[int, ...], score: float) -> None:
    """Print multi-budget evaluation as a compact table."""
    h_str = "  ".join(f"{h}d W*U" for h in horizons)
    print(f"\n  {'Budget':>8} {'Mass':>6} {'Rows':>6} {'Dup':>6} | {h_str}")
    print("  " + "-" * (32 + 9 * len(horizons)))

    for key, info in details.items():
        if key == "_meta":
            continue
        cells = info["cells"]
        vals = []
        dup_mass = sum(cell["duplicate_mass_discarded"] for cell in cells.values())
        for h in horizons:
            cell = cells.get(h)
            vals.append(f"{cell['weighted']:>+7.2f}" if cell else f"{'n/a':>7}")
        print(
            f"  {info['label']:>8} {_format_count(info['signal_mass']):>6} "
            f"{info['n_rows']:>6} {_format_count(dup_mass):>6} | {'  '.join(vals)}"
        )

    ref = details.get(0.0025, {}).get("cells", {}).get(10)
    if ref:
        print(f"\n  Detail (0.25%, 10d):")
        print(
            f"    Excess: {ref['excess']:+.2%}  "
            f"Win: {ref['win_rate']:.1%}  "
            f"Knife: {ref['knife_rate']:.1%}  "
            f"TailMAE: {ref['tail_mae']:+.2%}  "
            f"Slip: {ref['entry_slippage']:+.2%}  "
            f"Eff.N: {_format_count(ref['effective_n'])}  "
            f"W: {ref['w']:.2f}"
        )
        print(
            f"    Event recall: {ref['event_recall']:.1%} "
            f"({ref['event_hits']}/{ref['n_true_events']})  "
            f"Exact recall: {ref['exact_center_recall']:.1%} "
            f"({ref['exact_hits']}/{ref['n_true_bases']})"
        )
        print(
            f"    Zone precision: {ref['zone_precision']:.1%}  "
            f"Avg offset: {ref['avg_entry_offset_days']:+.2f}d  "
            f"Dup mass: {ref['duplicate_mass_discarded']:.2f}"
        )

    if any(info["tied_rows"] > 0 for k, info in details.items() if k != "_meta"):
        print("\n  Note: cutoff ties are scored with fractional weights; Rows can exceed Mass.")

    meta = details.get("_meta", {})
    valid_cells = meta.get("valid_cells", 0)
    total_cells = meta.get("total_cells", 0)
    if valid_cells < total_cells:
        print(f"\n  Composite Score: {score:.4f} (mean of {valid_cells} of {total_cells} cells)")
    else:
        print(f"\n  Composite Score: {score:.4f} (mean of {total_cells} cells)")


# ---------------------------------------------------------------------------
# Full tiered evaluation
# ---------------------------------------------------------------------------


def tiered_eval(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_ap: float = 0.05,
) -> dict:
    """Run threshold-free multi-budget evaluation.

    Tier 1: Ranking quality (AP, AUC) -- gates on AP.
    Tier 2: Multi-budget, multi-horizon composite score.

    The model is evaluated at multiple signal budgets (top 0.1% to 2%)
    and multiple horizons (5d, 10d, 20d). Each cell is weighted by
    effective_n / (effective_n + 50) to penalize low-evidence scores.
    """
    results = {"passed": False}
    horizons = (5, 10, 20)

    # Tier 1: Ranking quality
    print("=" * 60)
    print("TIER 1: Ranking Quality")
    print("=" * 60)

    ap, auc = 0.0, 0.0
    base_ap, base_auc = 0.0, 0.0
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)

    annotated = ensure_pivot_low_event_columns(df)
    y_base = annotated[PIVOT_LOW_BASE_COL].to_numpy(dtype=int)
    if len(np.unique(y_base)) > 1:
        base_auc = roc_auc_score(y_base, y_pred_proba)
        base_ap = average_precision_score(y_base, y_pred_proba)

    results["tier1"] = {
        "avg_precision": ap,
        "roc_auc": auc,
        "base_avg_precision": base_ap,
        "base_roc_auc": base_auc,
    }
    print(f"  roc_auc:        {auc:.4f}")
    print(f"  avg_precision:  {ap:.4f}")
    print(f"  base_roc_auc:   {base_auc:.4f}")
    print(f"  base_ap:        {base_ap:.4f}")

    if ap < min_ap:
        print(f"\n  FAIL: avg_precision {ap:.4f} < {min_ap}. Stopping.")
        return results

    # Tier 2: Multi-budget composite
    print("\n" + "=" * 60)
    print("TIER 2: Multi-Budget Composite Score")
    print("=" * 60)

    score, details = multi_budget_composite(df, y_pred_proba, horizons=horizons)
    results["tier2"] = details
    results["tier3"] = score

    _print_multi_budget(details, horizons, score)

    results["passed"] = bool(score > float("-inf"))
    return results


# ---------------------------------------------------------------------------
# Benchmark: random entry on same stocks
# ---------------------------------------------------------------------------


def benchmark_random_entry(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    horizon: int = 10,
    n_simulations: int = 500,
    seed: int = 42,
) -> dict:
    """Compare model signals against random entry on the same stocks.

    For each simulation, picks random entry dates for each signal's stock
    (within the same date range), measures forward returns, and compares.
    Tests whether the model has timing skill vs just picking volatile stocks.

    Not part of the autoresearch gate -- use for post-research validation.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_pred"] = y_pred

    signals = df[df["_pred"] == 1]
    if len(signals) == 0:
        return {"error": "no signals"}

    # Pre-build per-stock arrays once
    stock_data = {}
    for stock_id, grp in df.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        stock_data[stock_id] = (grp["open"].values, {d: i for i, d in enumerate(grp["date"])})

    # Compute forward returns for each signal
    signal_returns = []
    signal_stocks = []
    valid_mask = pd.Series(False, index=df.index, dtype=bool)
    for stock_id, grp in df.groupby("stock_id", sort=False):
        grp = grp.sort_values("date")
        idx = grp.index.to_numpy()
        valid_mask.loc[idx] = np.arange(len(grp)) + horizon + 1 < len(grp)

    collapsed_signals = _collapsed_binary_prediction_rows(df, "_pred", valid_mask=valid_mask)
    for stock_id, group in collapsed_signals.groupby("stock_id"):
        opens, dates_idx = stock_data[stock_id]
        sig_dates = group["date"].values

        for d in sig_dates:
            idx = dates_idx.get(d)
            if idx is None or idx + horizon + 1 >= len(opens):
                continue
            entry = opens[idx + 1]
            exit_ = opens[idx + horizon + 1]
            signal_returns.append((exit_ - entry) / entry)
            signal_stocks.append(stock_id)

    if not signal_returns:
        return {"error": "no valid forward returns"}

    signal_returns = np.array(signal_returns)
    model_mean = signal_returns.mean()

    # Pre-compute per-stock opens and max valid indices for simulation
    stock_opens = {sid: stock_data[sid][0] for sid in set(signal_stocks)}
    stock_max_idx = {sid: len(opens) - horizon - 2 for sid, opens in stock_opens.items()}

    # Random simulations: for each signal, pick a random date in the same stock
    random_means = []
    for _ in range(n_simulations):
        sim_returns = []
        for stock_id in signal_stocks:
            max_idx = stock_max_idx[stock_id]
            if max_idx <= 0:
                continue
            opens = stock_opens[stock_id]
            rand_idx = rng.integers(0, max_idx)
            entry = opens[rand_idx + 1]
            exit_ = opens[rand_idx + horizon + 1]
            sim_returns.append((exit_ - entry) / entry)
        if sim_returns:
            random_means.append(np.mean(sim_returns))

    random_means = np.array(random_means)
    random_mean = random_means.mean()
    random_std = random_means.std()

    z = (model_mean - random_mean) / random_std if random_std > 0 else 0.0
    p = 1 - stats.norm.cdf(z)

    result = {
        "model_mean": model_mean,
        "random_mean": random_mean,
        "excess_return": model_mean - random_mean,
        "z_score": z,
        "p_value": p,
        "significant": p < 0.05,
        "n_signals": len(signal_returns),
        "n_simulations": n_simulations,
    }

    print(f"\nBenchmark: Random Entry (same stocks, {horizon}d)")
    print(f"  Model mean:    {model_mean:+.2%}")
    print(f"  Random mean:   {random_mean:+.2%}")
    print(f"  Excess return: {result['excess_return']:+.2%}")
    print(f"  z-score:       {z:.2f}")
    print(f"  p-value:       {p:.4f}")
    print(f"  Significant:   {result['significant']}")
    print(f"  ({len(signal_returns)} signals, {n_simulations} simulations)")

    return result
