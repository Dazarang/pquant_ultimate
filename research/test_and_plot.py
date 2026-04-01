#!/usr/bin/env python3
"""Evaluate the current model on the last walk-forward fold and visualize buy signals.

1. Train model on last fold's train split
2. Evaluate on last fold's val split
3. Run benchmark_random_entry + backtest at a reference budget
4. Plot price timelines with buy-signal markers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from lib.data import LABEL_COL, list_features, load_dataset, scale, temporal_split
from lib.eval import (
    backtest_quick,
    benchmark_random_entry,
    forward_open_return,
    select_top_frac,
    tiered_eval,
)
from research.experiment import (
    DATASET_PATH,
    FEATURE_GROUPS,
    STOCKS,
    _WF_FOLDS,
    build_model,
)
from research.features_lab import add_custom_features

OUT_DIR = Path(__file__).resolve().parent

# Reference budget for visualization, benchmark, and backtest
VIZ_BUDGET = 0.0025  # top 0.25% of predictions


def train_and_evaluate():
    """Train model on last fold, evaluate on its val set, run benchmarks."""

    features = list_features(FEATURE_GROUPS)
    df, feature_cols = load_dataset(DATASET_PATH, stocks=STOCKS, features=features)
    df, new_features = add_custom_features(df)
    feature_cols = feature_cols + new_features
    df[new_features] = df[new_features].replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df = df[df["date"] >= "2021-01-01"].reset_index(drop=True)

    train_end, val_end = _WF_FOLDS[-1]
    train, val, _ = temporal_split(df, train_end=train_end, val_end=val_end, include_test=False)
    train_s, val_s, _, _ = scale(train, val, val.iloc[:0], feature_cols)

    X_train = train_s[feature_cols].values
    y_train = train_s[LABEL_COL].values

    print(f"\nTraining on {X_train.shape[0]:,} rows, {X_train.shape[1]} features...")
    model = build_model(y_train)
    model.fit(X_train, y_train)

    # --- Val evaluation ---
    print("\n" + "=" * 70)
    print(f"LAST FOLD: train_end={train_end}, val_end={val_end}")
    print("=" * 70)
    X_val = val_s[feature_cols].values
    y_val = val_s[LABEL_COL].values
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_results = tiered_eval(val, y_val, y_val_proba)

    # --- Benchmark & backtest at reference budget ---
    y_val_pred = select_top_frac(y_val_proba, VIZ_BUDGET)
    n_sig = int(y_val_pred.sum())

    print("\n" + "=" * 70)
    print(f"BENCHMARK: Random Entry vs Model (top {VIZ_BUDGET*100:.2f}%, {n_sig} signals)")
    print("=" * 70)
    benchmark_random_entry(val, y_val_pred)

    print("\n" + "=" * 70)
    print(f"BACKTEST: Quick (top {VIZ_BUDGET*100:.2f}%, {n_sig} signals)")
    print("=" * 70)
    val_bt = val.copy()
    val_bt["prediction"] = y_val_pred
    backtest_quick(val_bt)

    return val, y_val_proba, val_results


def plot_signals(val_df, y_proba, budget=VIZ_BUDGET, max_stocks=16):
    """Plot price timeline with buy signals for top stocks by signal count."""

    val = val_df.copy()
    y_pred = select_top_frac(y_proba, budget)
    val["pred"] = y_pred
    val["proba"] = y_proba

    signal_stocks = val[val["pred"] == 1].groupby("stock_id").size().sort_values(ascending=False)
    if len(signal_stocks) == 0:
        print("No signals!")
        return

    top_stocks = signal_stocks.head(max_stocks).index.tolist()
    n = len(top_stocks)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    date_range = f"{val['date'].min().date()} to {val['date'].max().date()}"
    n_total_signals = int(y_pred.sum())
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(
        f"Buy Signals ({date_range})  |  budget={budget*100:.2f}%  |  "
        f"{n_total_signals} total signals across {len(signal_stocks)} stocks",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for i, stock_id in enumerate(top_stocks):
        ax = axes[i // cols][i % cols]
        stock = val[val["stock_id"] == stock_id].sort_values("date").copy()
        dates = stock["date"]
        close = stock["close"]

        ax.plot(dates, close, color="#555", linewidth=0.7, alpha=0.9)

        true_pivots = stock[stock[LABEL_COL] == 1]
        ax.scatter(
            true_pivots["date"], true_pivots["close"],
            color="#1976d2", s=30, zorder=4, marker="o", alpha=0.5,
            label=f"{len(true_pivots)} true",
        )

        signals = stock[stock["pred"] == 1]
        ax.scatter(
            signals["date"], signals["close"],
            color="crimson", s=50, zorder=5, marker="^", edgecolors="black", linewidths=0.3,
            label=f"{len(signals)} pred",
        )

        for _, row in signals.iterrows():
            ret = forward_open_return(stock, row["date"], horizon=10)
            if ret is None:
                continue
            color = "#2e7d32" if ret > 0 else "#c62828"
            ax.annotate(
                f"{ret:+.0%}", (row["date"], row["close"]),
                textcoords="offset points", xytext=(0, 10),
                fontsize=7, color=color, ha="center", fontweight="bold",
            )

        ax.set_title(f"{stock_id}", fontsize=10)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.2)

    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "test_signals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)


def plot_returns_distribution(val_df, y_proba, budget=VIZ_BUDGET):
    """Plot distribution of forward returns for buy signals."""

    val = val_df.copy()
    y_pred = select_top_frac(y_proba, budget)
    val["pred"] = y_pred

    signals = val[val["pred"] == 1].copy()
    if len(signals) == 0:
        return

    stocks_by_id = {
        stock_id: stock.sort_values("date").copy()
        for stock_id, stock in val.groupby("stock_id")
    }
    fwd_returns = []
    for _, row in signals.iterrows():
        ret = forward_open_return(stocks_by_id[row["stock_id"]], row["date"], horizon=10)
        if ret is None:
            continue
        fwd_returns.append(ret)

    if not fwd_returns:
        return

    fwd_returns = np.array(fwd_returns)
    wins = (fwd_returns > 0).sum()
    total = len(fwd_returns)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fwd_returns, bins=40, color="#1976d2", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(fwd_returns.mean(), color="crimson", linewidth=1.5, label=f"mean={fwd_returns.mean():+.2%}")
    ax.axvline(np.median(fwd_returns), color="orange", linewidth=1.5, linestyle=":", label=f"median={np.median(fwd_returns):+.2%}")
    ax.set_title(
        f"10d Forward Returns (top {budget*100:.2f}%)  |  {wins}/{total} wins ({wins/total:.0%})",
        fontweight="bold",
    )
    ax.set_xlabel("10-day forward return")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = OUT_DIR / "test_returns_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    val, y_proba, results = train_and_evaluate()
    plot_signals(val, y_proba)
    plot_returns_distribution(val, y_proba)
