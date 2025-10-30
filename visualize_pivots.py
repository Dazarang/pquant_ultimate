"""
Visualize pivot highs and lows on Apple stock chart.

Usage:
    python visualize_pivots.py                    # Default: lb=8, rb=8
    python visualize_pivots.py 5 5                # Custom: lb=5, rb=5
    python visualize_pivots.py 5 8                # Asymmetric: lb=5, rb=8
"""

import sys

import matplotlib.pyplot as plt
import yfinance as yf

from indicators.pattern import find_pivots
from indicators.trend import calculate_ema, calculate_sma
from indicators.volatility import calculate_bbands


def main(left_bars=21, right_bars=21):
    """Main visualization workflow."""

    # Fetch data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start="2020-01-01", end="2024-01-01", auto_adjust=True)
    df.columns = df.columns.str.lower()

    # Calculate indicators
    sma_50 = calculate_sma(df, period=50)
    sma_200 = calculate_sma(df, period=34)
    ema_50 = calculate_ema(df, period=50)
    ema_200 = calculate_ema(df, period=34)
    bb_upper, bb_middle, bb_lower = calculate_bbands(df, period=13)

    # Detect pivots
    pivot_high, pivot_low = find_pivots(df, lb=left_bars, rb=right_bars, return_boolean=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot price line
    ax.plot(df.index, df["close"], color="black", linewidth=1.5, alpha=0.8, label="Close", zorder=3)

    # Plot SMAs
    ax.plot(df.index, sma_50, color="orange", linewidth=1.2, alpha=0.7, label="SMA 50")
    ax.plot(df.index, sma_200, color="purple", linewidth=1.2, alpha=0.7, label="SMA 200")

    # Plot EMAs
    ax.plot(df.index, ema_50, color="green", linewidth=1.2, alpha=0.7, linestyle="--", label="EMA 50")
    ax.plot(df.index, ema_200, color="brown", linewidth=1.2, alpha=0.7, linestyle="--", label="EMA 200")

    # Plot Bollinger Bands
    ax.plot(df.index, bb_upper, color="gray", linewidth=1, alpha=0.5, linestyle=":", label="BB Upper")
    ax.plot(df.index, bb_middle, color="gray", linewidth=1, alpha=0.5, linestyle="-")
    ax.plot(df.index, bb_lower, color="gray", linewidth=1, alpha=0.5, linestyle=":", label="BB Lower")
    ax.fill_between(df.index, bb_lower, bb_upper, color="gray", alpha=0.1)

    # Get pivot locations
    pivot_high_dates = df[pivot_high].index
    pivot_low_dates = df[pivot_low].index

    # Plot pivot highs as red dots
    pivot_high_close_prices = df.loc[pivot_high_dates, "close"]
    ax.scatter(
        pivot_high_dates,
        pivot_high_close_prices,
        color="red",
        s=50,
        marker="o",
        zorder=5,
        edgecolors="darkred",
        linewidths=1,
        label="Pivot High",
    )

    # Plot pivot lows as blue dots
    pivot_low_close_prices = df.loc[pivot_low_dates, "close"]
    ax.scatter(
        pivot_low_dates,
        pivot_low_close_prices,
        color="blue",
        s=50,
        marker="o",
        zorder=5,
        edgecolors="darkblue",
        linewidths=1,
        label="Pivot Low",
    )

    # Styling
    ax.set_title(f"AAPL - Pivots (lb={left_bars}, rb={right_bars})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save and show
    output = "apple_pivots.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    print(f"Detected: {len(pivot_high_dates)} highs, {len(pivot_low_dates)} lows")

    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 3:
        left_bars = int(sys.argv[1])
        right_bars = int(sys.argv[2])
        print(f"Using: left_bars={left_bars}, right_bars={right_bars}")
    elif len(sys.argv) == 2:
        left_bars = right_bars = int(sys.argv[1])
        print(f"Using: left_bars={left_bars}, right_bars={right_bars}")
    else:
        left_bars = 8
        right_bars = 13
        print("Using default: left_bars=8, right_bars=8")
        print("Usage: python visualize_pivots.py [left_bars] [right_bars]")

    main(left_bars, right_bars)
