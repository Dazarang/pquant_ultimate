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
def _backtest_single_trade(highs, lows, closes, entry_price, target_pct, stop_pct, max_hold, start_idx):
    """Find exit for a single trade. Returns (exit_idx, exit_price, exit_type).

    exit_type: 1=target, -1=stop, 0=max_hold.
    """
    target_price = entry_price * (1 + target_pct)
    stop_price = entry_price * (1 - stop_pct)
    end_idx = min(start_idx + max_hold, len(highs))
    for i in range(start_idx, end_idx):
        if highs[i] >= target_price:
            return i, target_price, 1
        if lows[i] <= stop_price:
            return i, stop_price, -1
    last_idx = end_idx - 1
    if last_idx < start_idx:
        last_idx = start_idx
    return last_idx, closes[last_idx], 0


_EXIT_REASONS = {1: "target", -1: "stop", 0: "max_hold"}


def backtest_quick(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    target_pct: float = 0.10,
    stop_pct: float = 0.05,
    max_hold_days: int = 30,
) -> pd.DataFrame:
    """Simple vectorized backtest on predicted pivot bottoms.

    For each predicted bottom (pred_col == 1), simulates buying at next-day open
    and exiting at target, stop, or max hold. Operates per stock.

    Returns DataFrame with one row per trade: stock_id, entry_date, exit_date,
    return_pct, exit_reason.
    """
    trades = []

    for stock_id, stock_df in df.groupby("stock_id"):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        signals = stock_df.index[stock_df[pred_col] == 1].tolist()
        if not signals:
            continue

        highs = stock_df["high"].values.astype(np.float64)
        lows = stock_df["low"].values.astype(np.float64)
        closes = stock_df["close"].values.astype(np.float64)
        opens = stock_df["open"].values.astype(np.float64)
        dates = stock_df["date"].values
        n = len(stock_df)

        for idx in signals:
            entry_idx = idx + 1
            if entry_idx >= n:
                continue

            entry_price = opens[entry_idx]
            exit_idx, exit_price, exit_type = _backtest_single_trade(
                highs, lows, closes, entry_price, target_pct, stop_pct, max_hold_days, entry_idx,
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
    wins = trades[trades["return_pct"] > 0]
    print(f"\nBacktest: {n} trades")
    print(f"  Win rate:    {len(wins) / n:.1%}")
    print(f"  Avg return:  {trades['return_pct'].mean():.2%}")
    print(f"  Med return:  {trades['return_pct'].median():.2%}")
    print(f"  Exit reasons: {trades['exit_reason'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Shared: per-signal analytics
# ---------------------------------------------------------------------------


def _signal_analytics(df: pd.DataFrame, y_pred: np.ndarray, horizon: int) -> dict | None:
    """Compute forward returns, MAE, and market benchmark per buy signal.

    Returns None if no valid signals exist for the given horizon.
    """
    buy_mask = pd.Series(y_pred == 1, index=df.index)

    entry = df.groupby("stock_id")["open"].shift(-1)
    exit_ = df.groupby("stock_id")["open"].shift(-(horizon + 1))
    fwd = exit_ / entry - 1

    # MAE: min low during holding period (entry day through day before exit)
    min_low = df.groupby("stock_id")["low"].transform(
        lambda x: x.rolling(horizon).min().shift(-horizon)
    )
    mae = min_low / entry - 1

    # Market benchmark: equal-weight avg forward return per date
    market_by_date = fwd.groupby(df["date"]).mean()
    market = df["date"].map(market_by_date)

    valid_idx = fwd.loc[buy_mask].dropna().index
    if len(valid_idx) == 0:
        return None

    valid_mask = mae.loc[valid_idx].notna() & market.loc[valid_idx].notna()
    valid_idx = valid_idx[valid_mask]
    if len(valid_idx) == 0:
        return None

    return {
        "returns": fwd.loc[valid_idx],
        "mae": mae.loc[valid_idx],
        "market": market.loc[valid_idx],
        "dates": df.loc[valid_idx, "date"],
    }


# ---------------------------------------------------------------------------
# Tier 2: Forward Returns
# ---------------------------------------------------------------------------


def forward_returns(df: pd.DataFrame, y_pred: np.ndarray, horizons: list[int] | None = None) -> dict:
    """Measure actual returns at fixed horizons for every positive prediction.

    Entry at next-day open, exit at open N days later. Includes MAE (max adverse
    excursion), excess return vs equal-weight market, and effective N (unique
    signal dates as proxy for independent bets).
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
        excess = fwd - market

        wins = fwd[fwd > 0]
        losses = fwd[fwd < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")

        results[f"mean_{h}d"] = fwd.mean()
        results[f"excess_{h}d"] = excess.mean()
        results[f"win_rate_{h}d"] = (fwd > 0).mean()
        results[f"avg_win_{h}d"] = wins.mean() if len(wins) else 0.0
        results[f"avg_loss_{h}d"] = losses.mean() if len(losses) else 0.0
        results[f"profit_factor_{h}d"] = pf
        results[f"mae_{h}d"] = mae.mean()
        results[f"worst_mae_{h}d"] = mae.min()
        results[f"n_signals_{h}d"] = len(fwd)
        results[f"effective_n_{h}d"] = dates.nunique()

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

    excess_return = (fwd - market).mean()
    mean_return = fwd.mean()
    market_mean = market.mean()
    win_rate = (fwd > 0).mean()
    worst_decile = fwd.quantile(0.1)
    knife_rate = (fwd < -0.05).mean()
    mean_mae = mae.mean()
    effective_n = dates.nunique()

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
    print(f"  Signals:       {len(fwd)} (effective: {effective_n})")

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

BUDGET_FRACS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02)


def select_top_frac(y_pred_proba: np.ndarray, frac: float) -> np.ndarray:
    """Select top fraction of predictions as signals. Returns binary array."""
    k = max(1, int(len(y_pred_proba) * frac))
    indices = np.argsort(-y_pred_proba)[:k]
    y_pred = np.zeros(len(y_pred_proba), dtype=int)
    y_pred[indices] = 1
    return y_pred


def _cell_score(df: pd.DataFrame, y_pred: np.ndarray, horizon: int) -> dict | None:
    """Score one (budget, horizon) cell. Returns metrics dict or None."""
    analytics = _signal_analytics(df, y_pred, horizon)
    if analytics is None:
        return None

    fwd = analytics["returns"]
    market = analytics["market"]
    mae = analytics["mae"]
    dates = analytics["dates"]

    effective_n = dates.nunique()
    excess = (fwd - market).mean()
    win_rate = (fwd > 0).mean()
    worst_decile = fwd.quantile(0.1)
    knife_rate = (fwd < -0.05).mean()
    mean_mae = mae.mean()

    raw = (
        0.50 * excess * 100
        + 0.15 * (win_rate - 0.5) * 100
        - 0.15 * abs(min(0.0, worst_decile)) * 100
        - 0.10 * knife_rate * 100
        - 0.10 * abs(mean_mae) * 100
    )

    w = np.sqrt(effective_n / (effective_n + 20))

    return {
        "raw": raw, "weighted": w * raw, "w": w,
        "n_signals": len(fwd), "effective_n": effective_n,
        "excess": excess, "win_rate": win_rate,
        "worst_decile": worst_decile, "knife_rate": knife_rate,
        "mean_mae": mean_mae,
    }


def multi_budget_composite(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    budgets: tuple[float, ...] = BUDGET_FRACS,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> tuple[float, dict]:
    """Multi-budget, multi-horizon composite score with soft N-scaling.

    Evaluates model ranking at multiple signal budgets and holding horizons.
    Final score = mean of W * U across all (budget, horizon) pairs,
    where W = sqrt(effective_n / (effective_n + 20)).
    """
    all_weighted = []
    details = {}

    for q in budgets:
        y_pred = select_top_frac(y_pred_proba, q)
        label = f"{q * 100:.2f}%"
        details[label] = {"n_signals": int(y_pred.sum()), "cells": {}}

        for h in horizons:
            cell = _cell_score(df, y_pred, h)
            if cell is None:
                all_weighted.append(0.0)  # missing cell = neutral, prevents gaming
                continue
            all_weighted.append(cell["weighted"])
            details[label]["cells"][h] = cell

    score = np.mean(all_weighted) if all_weighted else float("-inf")
    return score, details


def _print_multi_budget(details: dict, horizons: tuple[int, ...], score: float) -> None:
    """Print multi-budget evaluation as a compact table."""
    h_str = "  ".join(f"{h}d W*U" for h in horizons)
    print(f"\n  {'Budget':>8} {'Sig':>6} | {h_str}")
    print("  " + "-" * (18 + 9 * len(horizons)))

    for label, info in details.items():
        cells = info["cells"]
        vals = []
        for h in horizons:
            cell = cells.get(h)
            vals.append(f"{cell['weighted']:>+7.2f}" if cell else f"{'n/a':>7}")
        print(f"  {label:>8} {info['n_signals']:>6} | {'  '.join(vals)}")

    ref = details.get("0.25%", {}).get("cells", {}).get(10)
    if ref:
        print(f"\n  Detail (0.25%, 10d):")
        print(
            f"    Excess: {ref['excess']:+.2%}  "
            f"Win: {ref['win_rate']:.1%}  "
            f"Knife: {ref['knife_rate']:.1%}  "
            f"MAE: {ref['mean_mae']:+.2%}  "
            f"Eff.N: {ref['effective_n']}  "
            f"W: {ref['w']:.2f}"
        )

    n_cells = sum(len(info["cells"]) for info in details.values())
    print(f"\n  Composite Score: {score:.4f} (mean of {n_cells} cells)")


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

    The model is evaluated at multiple signal budgets (top 0.05% to 2%)
    and multiple horizons (5d, 10d, 20d). Each cell is weighted by
    sqrt(effective_n / (effective_n + 20)) to penalize low-evidence scores.
    """
    results = {"passed": False}
    horizons = (5, 10, 20)

    # Tier 1: Ranking quality
    print("=" * 60)
    print("TIER 1: Ranking Quality")
    print("=" * 60)

    ap, auc = 0.0, 0.0
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)

    results["tier1"] = {"avg_precision": ap, "roc_auc": auc}
    print(f"  roc_auc:        {auc:.4f}")
    print(f"  avg_precision:  {ap:.4f}")

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
    for stock_id, group in signals.groupby("stock_id"):
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
