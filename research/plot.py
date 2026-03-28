#!/usr/bin/env python3
"""Plot autoresearch progress from results.tsv.

Usage:
    uv run python research/plot.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILE = Path(__file__).parent / "results.tsv"
OUTPUT_FILE = Path(__file__).parent / "progress.png"


def plot():
    if not RESULTS_FILE.exists():
        print("No results.tsv found.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_FILE, sep="\t")
    if len(df) == 0:
        print("No results yet.")
        sys.exit(1)

    df = df[df["status"] != "crash"].reset_index(drop=True)
    df["experiment"] = range(len(df))

    kept = df[df["status"] == "keep"]
    discarded = df[df["status"].isin(["discard", "gate_fail"])]

    # Running best (step function over kept experiments only)
    if len(kept) > 0:
        running_best = kept["score"].cummax()
    else:
        running_best = pd.Series(dtype=float)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Discarded: gray dots
    if len(discarded) > 0:
        ax.scatter(
            discarded["experiment"], discarded["score"],
            c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded",
        )

    # Kept: green dots with black edge
    if len(kept) > 0:
        ax.scatter(
            kept["experiment"], kept["score"],
            c="#2ecc71", s=50, zorder=4, edgecolors="black", linewidth=0.5, label="Kept",
        )

        # Running best step line
        ax.step(
            kept["experiment"], running_best,
            where="post", color="#27ae60", linewidth=2, alpha=0.7, zorder=3, label="Running best",
        )

        # Annotations on kept experiments
        for _, row in kept.iterrows():
            desc = str(row["description"])
            if len(desc) > 45:
                desc = desc[:42] + "..."
            ax.annotate(
                desc,
                (row["experiment"], row["score"]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="#1a7a3a", alpha=0.9,
                rotation=30, ha="left", va="bottom",
            )

    # Axes
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Composite Score (higher is better)", fontsize=12)

    if len(df) > 0:
        all_scores = df["score"]
        margin = (all_scores.max() - all_scores.min()) * 0.15 if all_scores.max() != all_scores.min() else 1
        ax.set_ylim(all_scores.min() - margin, all_scores.max() + margin)

    n_total = len(df)
    n_kept = len(kept)
    ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
