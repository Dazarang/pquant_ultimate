"""
Performance benchmarks: custom implementations vs ta-lib.
"""

import pytest
import time
import pandas as pd
import numpy as np
import talib

from indicators.trend import SMA, EMA, WMA
from indicators.momentum import RSI, MACD, ADX
from indicators.volatility import BBands, ATR, SAR
from indicators.volume import OBV, ADOSC
from indicators.calculator import IndicatorCalculator


@pytest.fixture
def large_dataset():
    """Generate large dataset for benchmarking."""
    np.random.seed(42)
    n = 5000  # 5000 bars

    dates = pd.date_range(start="2010-01-01", periods=n, freq="D")
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


class TestPerformanceBenchmarks:
    """Performance benchmarks."""

    def test_sma_performance(self, large_dataset, benchmark):
        """Benchmark SMA performance."""
        df = large_dataset

        def custom_impl():
            return SMA().calculate(df, period=50)

        def talib_impl():
            return talib.SMA(df["close"], timeperiod=50)

        # Warmup
        custom_impl()
        talib_impl()

        # Benchmark custom
        start = time.perf_counter()
        for _ in range(10):
            custom_impl()
        custom_time = (time.perf_counter() - start) / 10

        # Benchmark talib
        start = time.perf_counter()
        for _ in range(10):
            talib_impl()
        talib_time = (time.perf_counter() - start) / 10

        speedup = talib_time / custom_time
        print(f"\nSMA - Custom: {custom_time*1000:.2f}ms, Ta-lib: {talib_time*1000:.2f}ms, "
              f"Speedup: {speedup:.2f}x")

        # Note: Ta-lib is C-optimized. Custom is pure Python/NumPy.
        # We prioritize correctness and maintainability over raw speed.
        # Custom should be reasonable (within 5x is acceptable)
        assert custom_time < talib_time * 5, "Custom SMA too slow"

    def test_rsi_performance(self, large_dataset):
        """Benchmark RSI performance."""
        df = large_dataset

        def custom_impl():
            return RSI().calculate(df, period=14)

        def talib_impl():
            return talib.RSI(df["close"], timeperiod=14)

        # Warmup
        custom_impl()
        talib_impl()

        # Benchmark custom
        start = time.perf_counter()
        for _ in range(10):
            custom_impl()
        custom_time = (time.perf_counter() - start) / 10

        # Benchmark talib
        start = time.perf_counter()
        for _ in range(10):
            talib_impl()
        talib_time = (time.perf_counter() - start) / 10

        speedup = talib_time / custom_time
        print(f"\nRSI - Custom: {custom_time*1000:.2f}ms, Ta-lib: {talib_time*1000:.2f}ms, "
              f"Speedup: {speedup:.2f}x")

        # With Numba JIT, should be competitive
        assert custom_time < talib_time * 3

    def test_batch_calculation_performance(self, large_dataset):
        """Benchmark full indicator calculation."""
        df = large_dataset

        calculator = IndicatorCalculator()

        # Warmup with sufficient data
        calculator.calculate_all(df.iloc[:300])

        # Benchmark
        start = time.perf_counter()
        result = calculator.calculate_all(df)
        elapsed = time.perf_counter() - start

        print(f"\nBatch calculation: {elapsed:.3f}s for {len(df)} bars")
        print(f"  Result shape: {result.shape}")
        print(f"  Columns: {result.shape[1]}")
        print(f"  Throughput: {len(df)/elapsed:.0f} bars/sec")

        # Should complete in reasonable time
        assert elapsed < 5.0, "Batch calculation too slow"

    def test_memory_efficiency(self, large_dataset):
        """Test memory usage is reasonable."""
        import sys

        df = large_dataset

        calculator = IndicatorCalculator()

        # Calculate memory before
        initial_memory = sys.getsizeof(df)

        # Calculate indicators
        result = calculator.calculate_all(df)

        # Calculate memory after
        result_memory = sys.getsizeof(result)
        memory_ratio = result_memory / initial_memory

        print(f"\nMemory - Initial: {initial_memory/1024:.1f}KB, "
              f"Result: {result_memory/1024:.1f}KB, "
              f"Ratio: {memory_ratio:.2f}x")

        # Should not explode memory (indicators add columns but reuse index)
        assert memory_ratio < 20, "Memory usage too high"


class TestScalability:
    """Test performance at different scales."""

    @pytest.mark.parametrize("size", [100, 500, 1000, 5000])
    def test_scaling(self, size):
        """Test performance scales reasonably with data size."""
        np.random.seed(42)

        dates = pd.date_range(start="2020-01-01", periods=size, freq="D")
        close = 100 + np.cumsum(np.random.randn(size) * 2)

        df = pd.DataFrame({"close": close}, index=dates)

        # Benchmark SMA
        start = time.perf_counter()
        SMA().calculate(df, period=50)
        elapsed = time.perf_counter() - start

        throughput = size / elapsed

        print(f"\nSize {size}: {elapsed*1000:.2f}ms, "
              f"Throughput: {throughput:.0f} bars/sec")

        # Should maintain reasonable throughput
        assert throughput > 1000, f"Throughput too low at size {size}"
