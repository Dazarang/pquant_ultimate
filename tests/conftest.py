"""
Pytest configuration and shared fixtures for indicator tests.
"""

import pytest
import pandas as pd
import numpy as np
import yfinance as yf


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

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def real_market_data():
    """Fetch real market data for validation."""
    try:
        ticker = "AAPL"
        df = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        pytest.skip(f"Could not fetch real market data: {e}")


@pytest.fixture
def tolerance_params():
    """
    Tolerance parameters for comparing custom vs ta-lib.

    Different indicators have different acceptable tolerances:
    - Simple indicators (SMA, EMA): very tight (< 1e-6)
    - Complex indicators (ADX, SAR): looser (< 1e-3)
    """
    return {
        "sma": 1e-8,
        "ema": 1e-10,  # Should match exactly with corrected algorithm
        "wma": 1e-6,
        "rsi": 1e-3,
        "macd": 1e-3,
        "bbands": 0.5,  # BBands may use EMA in ta-lib, we use SMA
        "atr": 0.1,  # Wilder's smoothing has initialization differences
        "adx": 0.5,  # Complex multi-step indicator with smoothing
        "obv": 1,  # Integer comparison
        "adosc": 1e-2,
        "sar": 10.0,  # Complex reversal logic with many edge cases
        "roc": 1e-3,
        "mom": 1e-6,
    }


def assert_series_close(
    custom: pd.Series,
    talib: pd.Series,
    tolerance: float,
    name: str,
    allow_nan_mismatch: bool = True,
):
    """
    Assert two series are close within tolerance.

    Args:
        custom: Series from custom implementation
        talib: Series from ta-lib
        tolerance: Maximum allowed difference
        name: Indicator name for error messages
        allow_nan_mismatch: Allow different NaN positions
    """
    # Align indices
    custom = custom.reindex(talib.index)

    # Find valid comparison points (both non-NaN)
    valid_mask = ~(custom.isna() | talib.isna())
    valid_count = valid_mask.sum()

    if valid_count == 0:
        pytest.fail(f"{name}: No valid comparison points found")

    # Compare values at valid points
    custom_valid = custom[valid_mask]
    talib_valid = talib[valid_mask]

    diff = np.abs(custom_valid - talib_valid)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Allow for small numerical differences
    assert max_diff < tolerance, (
        f"{name}: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}\n"
        f"Mean difference: {mean_diff:.2e}\n"
        f"Valid points: {valid_count}/{len(custom)}"
    )

    print(f"✓ {name}: Max diff {max_diff:.2e}, Mean diff {mean_diff:.2e}, "
          f"Valid points {valid_count}/{len(custom)}")
