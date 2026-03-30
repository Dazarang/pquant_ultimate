#!/usr/bin/env python3
"""Evaluate the winning experiment on the held-out TEST set and visualize buy signals.

1. Train model (same config as experiment.py)
2. Evaluate on val (sanity) + test (2024+, never seen during research)
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
    select_top_frac,
    tiered_eval,
)
from research.experiment import (
    DATASET_PATH,
    FEATURE_GROUPS,
    STOCKS,
    TRAIN_END,
    VAL_END,
    build_model,
)
from research.features_lab import add_custom_features

OUT_DIR = Path(__file__).resolve().parent

# Reference budget for visualization, benchmark, and backtest
VIZ_BUDGET = 0.0025  # top 0.25% of predictions


def train_and_evaluate():
    """Train winning model, evaluate on val + test, run benchmarks."""

    # Load & prepare (same pipeline as experiment.py)
    features = list_features(FEATURE_GROUPS)
    df, feature_cols = load_dataset(DATASET_PATH, stocks=STOCKS, features=features)
    df, new_features = add_custom_features(df)
    feature_cols = feature_cols + new_features
    df[new_features] = df[new_features].replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Split & scale
    train, val, test = temporal_split(df, train_end=TRAIN_END, val_end=VAL_END)
    train_s, val_s, test_s, scaler = scale(train, val, test, feature_cols)

    X_train = train_s[feature_cols].values
    y_train = train_s[LABEL_COL].values

    # Train
    print(f"\nTraining on {X_train.shape[0]:,} rows, {X_train.shape[1]} features...")
    model = build_model(y_train)
    model.fit(X_train, y_train)

    # --- Val evaluation (sanity check) ---
    print("\n" + "=" * 70)
    print("VALIDATION SET (in-sample for research, 2023)")
    print("=" * 70)
    X_val = val_s[feature_cols].values
    y_val = val_s[LABEL_COL].values
    y_val_proba = model.predict_proba(X_val)[:, 1]
    tiered_eval(val, y_val, y_val_proba)

    # --- Test evaluation (never seen!) ---
    print("\n" + "=" * 70)
    print("TEST SET (held-out, 2024+, never seen during research)")
    print("=" * 70)
    X_test = test_s[feature_cols].values
    y_test = test_s[LABEL_COL].values
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_results = tiered_eval(test, y_test, y_test_proba)

    # --- Benchmark & backtest at reference budget ---
    y_test_pred = select_top_frac(y_test_proba, VIZ_BUDGET)
    n_sig = int(y_test_pred.sum())

    print("\n" + "=" * 70)
    print(f"BENCHMARK: Random Entry vs Model (Test, top {VIZ_BUDGET*100:.2f}%, {n_sig} signals)")
    print("=" * 70)
    benchmark_random_entry(test, y_test_pred)

    print("\n" + "=" * 70)
    print(f"BACKTEST: Quick (Test, top {VIZ_BUDGET*100:.2f}%, {n_sig} signals)")
    print("=" * 70)
    test_bt = test.copy()
    test_bt["prediction"] = y_test_pred
    backtest_quick(test_bt)

    return test, y_test_proba, test_results


def plot_signals(test_df, y_proba, budget=VIZ_BUDGET, max_stocks=16):
    """Plot price timeline with buy signals for top stocks by signal count."""

    test = test_df.copy()
    y_pred = select_top_frac(y_proba, budget)
    test["pred"] = y_pred
    test["proba"] = y_proba

    signal_stocks = test[test["pred"] == 1].groupby("stock_id").size().sort_values(ascending=False)
    if len(signal_stocks) == 0:
        print("No signals in test set!")
        return

    top_stocks = signal_stocks.head(max_stocks).index.tolist()
    n = len(top_stocks)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    n_total_signals = int(y_pred.sum())
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(
        f"Test Set Buy Signals (2024+)  |  budget={budget*100:.2f}%  |  "
        f"{n_total_signals} total signals across {len(signal_stocks)} stocks",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for i, stock_id in enumerate(top_stocks):
        ax = axes[i // cols][i % cols]
        stock = test[test["stock_id"] == stock_id].sort_values("date").copy()
        dates = stock["date"]
        close = stock["close"]

        # Price line
        ax.plot(dates, close, color="#555", linewidth=0.7, alpha=0.9)

        # True pivot lows (ground truth)
        true_pivots = stock[stock[LABEL_COL] == 1]
        ax.scatter(
            true_pivots["date"], true_pivots["close"],
            color="#1976d2", s=30, zorder=4, marker="o", alpha=0.5,
            label=f"{len(true_pivots)} true",
        )

        # Predicted buy signals
        signals = stock[stock["pred"] == 1]
        ax.scatter(
            signals["date"], signals["close"],
            color="crimson", s=50, zorder=5, marker="^", edgecolors="black", linewidths=0.3,
            label=f"{len(signals)} pred",
        )

        # Annotate forward 10d return per signal
        for _, row in signals.iterrows():
            future = stock[stock["date"] > row["date"]].head(11)
            if len(future) < 2:
                continue
            entry_price = future.iloc[0]["open"]
            exit_idx = min(10, len(future) - 1)
            exit_price = future.iloc[exit_idx]["open"]
            ret = (exit_price - entry_price) / entry_price
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


def plot_returns_distribution(test_df, y_proba, budget=VIZ_BUDGET):
    """Plot distribution of forward returns for buy signals."""

    test = test_df.copy()
    y_pred = select_top_frac(y_proba, budget)
    test["pred"] = y_pred

    signals = test[test["pred"] == 1].copy()
    if len(signals) == 0:
        return

    fwd_returns = []
    for _, row in signals.iterrows():
        stock = test[(test["stock_id"] == row["stock_id"]) & (test["date"] > row["date"])].head(11)
        if len(stock) < 2:
            continue
        entry = stock.iloc[0]["open"]
        exit_idx = min(10, len(stock) - 1)
        exit_ = stock.iloc[exit_idx]["open"]
        fwd_returns.append((exit_ - entry) / entry)

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
    test, y_proba, results = train_and_evaluate()
    plot_signals(test, y_proba)
    plot_returns_distribution(test, y_proba)
