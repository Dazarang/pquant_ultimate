"""Model evaluation: 3-tier framework (classification, forward returns, composite)."""

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

    if y_pred_proba is not None:
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

        for idx in signals:
            # Entry at next day's open
            entry_idx = idx + 1
            if entry_idx >= len(stock_df):
                continue

            entry_price = stock_df.loc[entry_idx, "open"]
            entry_date = stock_df.loc[entry_idx, "date"]
            target_price = entry_price * (1 + target_pct)
            stop_price = entry_price * (1 - stop_pct)

            exit_reason = "max_hold"
            exit_date = entry_date
            exit_price = entry_price

            for hold_idx in range(entry_idx, min(entry_idx + max_hold_days, len(stock_df))):
                row = stock_df.loc[hold_idx]
                if row["high"] >= target_price:
                    exit_reason = "target"
                    exit_price = target_price
                    exit_date = row["date"]
                    break
                if row["low"] <= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price
                    exit_date = row["date"]
                    break
                exit_price = row["close"]
                exit_date = row["date"]

            trades.append({
                "stock_id": stock_id,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": (exit_price - entry_price) / entry_price,
                "exit_reason": exit_reason,
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
# Tier 2: Forward Returns
# ---------------------------------------------------------------------------


def forward_returns(df: pd.DataFrame, y_pred: np.ndarray, horizons: list[int] | None = None) -> dict:
    """Measure actual returns at fixed horizons for every positive prediction.

    Entry at next-day open, exit at open N days later. Matches the decision-time
    rule: model predicts after close on day T, trade executes at open T+1.
    """
    if horizons is None:
        horizons = [5, 10, 20]

    df = df.copy()
    df["_pred"] = y_pred
    buy_mask = df["_pred"] == 1

    results = {}
    for h in horizons:
        entry = df.groupby("stock_id")["open"].shift(-1)
        exit_ = df.groupby("stock_id")["open"].shift(-(h + 1))
        fwd = (exit_ / entry - 1).loc[buy_mask].dropna()

        if len(fwd) == 0:
            continue

        wins = fwd[fwd > 0]
        losses = fwd[fwd < 0]

        results[f"mean_{h}d"] = fwd.mean()
        results[f"win_rate_{h}d"] = (fwd > 0).mean()
        results[f"avg_win_{h}d"] = wins.mean() if len(wins) else 0.0
        results[f"avg_loss_{h}d"] = losses.mean() if len(losses) else 0.0
        pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
        results[f"profit_factor_{h}d"] = pf
        results[f"n_signals_{h}d"] = len(fwd)

    _print_forward_returns(results, horizons)
    return results


def _print_forward_returns(results: dict, horizons: list[int]) -> None:
    print("\nForward Returns:")
    for h in horizons:
        key = f"mean_{h}d"
        if key not in results:
            continue
        n = int(results[f"n_signals_{h}d"])
        print(
            f"  {h:>2}d: mean={results[f'mean_{h}d']:+.2%}  "
            f"win={results[f'win_rate_{h}d']:.1%}  "
            f"PF={results[f'profit_factor_{h}d']:.2f}  "
            f"n={n}"
        )


# ---------------------------------------------------------------------------
# Tier 3: Composite Score
# ---------------------------------------------------------------------------


def composite_score(df: pd.DataFrame, y_pred: np.ndarray, horizon: int = 10) -> float:
    """Risk-adjusted composite score. Rewards good entries, penalizes falling knives.

    Uses 10d forward return by default. Higher is better.
    """
    df = df.copy()
    df["_pred"] = y_pred

    entry = df.groupby("stock_id")["open"].shift(-1)
    exit_ = df.groupby("stock_id")["open"].shift(-(horizon + 1))
    fwd = (exit_ / entry - 1)

    signal_returns = fwd.loc[df["_pred"] == 1].dropna()
    if len(signal_returns) == 0:
        return float("-inf")

    mean_return = signal_returns.mean()
    win_rate = (signal_returns > 0).mean()
    worst_decile = signal_returns.quantile(0.1)
    knife_rate = (signal_returns < -0.05).mean()

    score = (
        0.4 * mean_return * 100
        + 0.3 * win_rate
        - 0.2 * abs(worst_decile) * 100
        - 0.1 * knife_rate * 100
    )

    print(f"\nComposite Score ({horizon}d): {score:.2f}")
    print(f"  Mean return: {mean_return:+.2%}")
    print(f"  Win rate:    {win_rate:.1%}")
    print(f"  Worst 10%:   {worst_decile:+.2%}")
    print(f"  Knife rate:  {knife_rate:.1%} (>{5}% loss)")

    return score


# ---------------------------------------------------------------------------
# Full tiered evaluation
# ---------------------------------------------------------------------------


def tiered_eval(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    min_ap: float = 0.05,
) -> dict:
    """Run all 3 tiers. Returns combined results dict.

    Tier 1: Classification metrics (sanity check).
    Tier 2: Forward returns (does buying on signals make money?).
    Tier 3: Composite score (risk-adjusted).

    Stops early if Tier 1 AP < min_ap.
    """
    results = {"passed": False}

    # Tier 1
    print("=" * 60)
    print("TIER 1: Classification")
    print("=" * 60)
    t1 = evaluate(y_true, y_pred, y_pred_proba)
    results["tier1"] = t1
    for k, v in t1.items():
        print(f"  {k}: {v:.4f}")

    ap = t1.get("avg_precision", 0.0)
    if ap < min_ap:
        print(f"\n  FAIL: avg_precision {ap:.4f} < {min_ap} threshold. Stopping.")
        return results

    # Tier 2
    print("\n" + "=" * 60)
    print("TIER 2: Forward Returns")
    print("=" * 60)
    t2 = forward_returns(df, y_pred)
    results["tier2"] = t2

    mean_10d = t2.get("mean_10d", 0.0)
    if mean_10d <= 0:
        print(f"\n  FAIL: mean 10d return {mean_10d:+.2%} <= 0. Stopping.")
        return results

    # Tier 3
    print("\n" + "=" * 60)
    print("TIER 3: Composite Score")
    print("=" * 60)
    t3 = composite_score(df, y_pred)
    results["tier3"] = t3

    results["passed"] = True
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

    # Compute forward returns for each signal
    signal_returns = []
    signal_stocks = []
    for stock_id, group in signals.groupby("stock_id"):
        stock_df = df[df["stock_id"] == stock_id].sort_values("date").reset_index(drop=True)
        opens = stock_df["open"].values
        dates_idx = {d: i for i, d in enumerate(stock_df["date"])}

        for _, row in group.iterrows():
            idx = dates_idx.get(row["date"])
            if idx is None or idx + 1 >= len(opens) or idx + horizon + 1 >= len(opens):
                continue
            entry = opens[idx + 1]
            exit_ = opens[idx + horizon + 1]
            signal_returns.append((exit_ - entry) / entry)
            signal_stocks.append(stock_id)

    if not signal_returns:
        return {"error": "no valid forward returns"}

    signal_returns = np.array(signal_returns)
    model_mean = signal_returns.mean()

    # Random simulations: for each signal, pick a random date in the same stock
    random_means = []
    for _ in range(n_simulations):
        sim_returns = []
        for stock_id in signal_stocks:
            stock_df = df[df["stock_id"] == stock_id].sort_values("date").reset_index(drop=True)
            opens = stock_df["open"].values
            max_idx = len(opens) - horizon - 2
            if max_idx <= 0:
                continue
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
