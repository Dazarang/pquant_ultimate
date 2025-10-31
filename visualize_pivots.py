"""
Visualize pivot highs and lows on Apple stock chart.

Usage:
    python visualize_pivots.py                    # Default: lb=8, rb=13
    python visualize_pivots.py 5 5                # Custom: lb=5, rb=5
    python visualize_pivots.py 5 8                # Asymmetric: lb=5, rb=8
    python visualize_pivots.py 8 13 -2,-1,1,2     # With window variations
"""

import sys

import plotly.graph_objects as go
import yfinance as yf

from indicators.pattern import find_pivots
from indicators.trend import calculate_ema, calculate_sma
from indicators.volatility import calculate_bbands


def main(left_bars=21, right_bars=21, window_variations=None):
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
    pivot_high, pivot_low = find_pivots(
        df, lb=left_bars, rb=right_bars, return_boolean=True, window_variations=window_variations
    )

    # Create figure
    fig = go.Figure()

    # Bollinger Bands fill
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=bb_upper,
            mode="lines",
            name="BB Upper",
            line={"color": "gray", "width": 1, "dash": "dot"},
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=bb_lower,
            mode="lines",
            name="BB Lower",
            line={"color": "gray", "width": 1, "dash": "dot"},
            opacity=0.5,
            fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=bb_middle,
            mode="lines",
            name="BB Middle",
            line={"color": "gray", "width": 1},
            opacity=0.5,
        )
    )

    # SMAs
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sma_50,
            mode="lines",
            name="SMA 50",
            line={"color": "orange", "width": 1.2},
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sma_200,
            mode="lines",
            name="SMA 200",
            line={"color": "purple", "width": 1.2},
            opacity=0.7,
        )
    )

    # EMAs
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ema_50,
            mode="lines",
            name="EMA 50",
            line={"color": "green", "width": 1.2, "dash": "dash"},
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ema_200,
            mode="lines",
            name="EMA 200",
            line={"color": "brown", "width": 1.2, "dash": "dash"},
            opacity=0.7,
        )
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            name="Close",
            line={"color": "black", "width": 1.5},
            opacity=0.8,
        )
    )

    # Get pivot locations - use close price
    pivot_high_dates = df[pivot_high].index
    pivot_low_dates = df[pivot_low].index
    pivot_high_values = df.loc[pivot_high_dates, "close"]
    pivot_low_values = df.loc[pivot_low_dates, "close"]

    # Pivot highs
    fig.add_trace(
        go.Scatter(
            x=pivot_high_dates,
            y=pivot_high_values,
            mode="markers",
            name="Pivot High",
            marker={"color": "red", "size": 12, "line": {"color": "darkred", "width": 2}},
            text=[f"Close: ${v:.2f}" for v in pivot_high_values],
            hovertemplate="Date: %{x}<br>%{text}<extra></extra>",
        )
    )

    # Pivot lows
    fig.add_trace(
        go.Scatter(
            x=pivot_low_dates,
            y=pivot_low_values,
            mode="markers",
            name="Pivot Low",
            marker={"color": "blue", "size": 12, "line": {"color": "darkblue", "width": 2}},
            text=[f"Close: ${v:.2f}" for v in pivot_low_values],
            hovertemplate="Date: %{x}<br>%{text}<extra></extra>",
        )
    )

    # Title with window_variations info
    title_suffix = f" (variations={window_variations})" if window_variations else ""
    title = f"AAPL - Pivots (lb={left_bars}, rb={right_bars}){title_suffix}"

    # Layout
    fig.update_layout(
        title={"text": title, "font": {"size": 14}},
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white",
        height=800,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    # Save and show
    output = "apple_pivots.html"
    fig.write_html(output)
    print(f"Saved: {output}")
    print(f"Detected: {len(pivot_high_dates)} highs, {len(pivot_low_dates)} lows")

    fig.show()


if __name__ == "__main__":
    # Parse command line arguments
    left_bars = 8
    right_bars = 13
    window_variations = None

    if len(sys.argv) >= 4:
        # Format: lb rb variations (e.g., 8 13 -2,-1,1,2)
        left_bars = int(sys.argv[1])
        right_bars = int(sys.argv[2])
        window_variations = [int(x) for x in sys.argv[3].split(",")]
        print(f"Using: lb={left_bars}, rb={right_bars}, variations={window_variations}")
    elif len(sys.argv) == 3:
        left_bars = int(sys.argv[1])
        right_bars = int(sys.argv[2])
        print(f"Using: lb={left_bars}, rb={right_bars}")
    elif len(sys.argv) == 2:
        left_bars = right_bars = int(sys.argv[1])
        print(f"Using: lb={left_bars}, rb={right_bars}")
    else:
        print(f"Using default: lb={left_bars}, rb={right_bars}")
        print("Usage: python visualize_pivots.py [lb] [rb] [variations]")
        print("Example: python visualize_pivots.py 8 13 -2,-1,1,2")

    main(left_bars, right_bars, window_variations)
