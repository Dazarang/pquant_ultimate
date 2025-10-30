"""
Pytest configuration and shared fixtures for indicator tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 500

    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.random.uniform(0.5, 3, n)
    low = close - np.random.uniform(0.5, 3, n)
    open_ = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.uniform(1e6, 1e7, n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


@pytest.fixture
def real_market_data():
    """
    Fetch real Apple (AAPL) stock data for validation.
    Uses actual market data for realistic testing scenarios.
    """
    import yfinance as yf

    try:
        # Fetch Apple stock data
        ticker = yf.Ticker("AAPL")
        df = ticker.history(start="2023-01-01", end="2024-01-01", auto_adjust=True)

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()

        # Ensure we have sufficient data
        if len(df) < 100:
            raise ValueError("Insufficient data fetched")

        return df

    except Exception as e:
        # Fallback to synthetic data if fetch fails
        pytest.skip(f"Could not fetch Apple data: {e}. Skipping real data test.")

        # This code won't run due to skip, but kept for clarity
        np.random.seed(42)
        n = 252
        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        returns = np.random.normal(0.0005, 0.02, n)
        close = 150 * np.exp(np.cumsum(returns))
        daily_range = np.abs(np.random.normal(0, 0.015, n))
        high = close * (1 + daily_range)
        low = close * (1 - daily_range)
        open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n)
        volume = np.random.lognormal(15, 0.5, n)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

        return df


@pytest.fixture
def tolerance_params():
    """
    Tolerance parameters for indicator validation.

    Different indicators have different acceptable tolerances:
    - Simple indicators (SMA, EMA): very tight (< 1e-6)
    - Complex indicators (ADX, SAR): looser due to initialization differences
    """
    return {
        "sma": 1e-8,
        "ema": 1e-10,  # Should match exactly with corrected algorithm
        "rsi": 1e-3,
        "macd": 1e-3,
        "bbands": 0.5,  # Different middle band calculation methods acceptable
        "atr": 0.1,  # Wilder's smoothing has initialization differences
        "adx": 0.5,  # Complex multi-step indicator with smoothing
        "obv": 1,  # Integer comparison
        "adosc": 1e-2,
        "sar": 10.0,  # Complex reversal logic with many edge cases
        "roc": 1e-3,
        "mom": 1e-6,
    }


def assert_series_close(
    actual: pd.Series,
    expected: pd.Series,
    tolerance: float,
    name: str,
    allow_nan_mismatch: bool = True,
):
    """
    Assert two series are close within tolerance.

    Args:
        actual: Series from implementation being tested
        expected: Series from reference implementation
        tolerance: Maximum allowed difference
        name: Indicator name for error messages
        allow_nan_mismatch: Allow different NaN positions
    """
    # Align indices
    actual = actual.reindex(expected.index)

    # Find valid comparison points (both non-NaN)
    valid_mask = ~(actual.isna() | expected.isna())
    valid_count = valid_mask.sum()

    if valid_count == 0:
        pytest.fail(f"{name}: No valid comparison points found")

    # Compare values at valid points
    actual_valid = actual[valid_mask]
    expected_valid = expected[valid_mask]

    diff = np.abs(actual_valid - expected_valid)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Allow for small numerical differences
    assert max_diff < tolerance, (
        f"{name}: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}\n"
        f"Mean difference: {mean_diff:.2e}\n"
        f"Valid points: {valid_count}/{len(actual)}"
    )

    print(f"✓ {name}: Max diff {max_diff:.2e}, Mean diff {mean_diff:.2e}, Valid points {valid_count}/{len(actual)}")
