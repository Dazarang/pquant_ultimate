"""
Test advanced ML features and Stochastic oscillator.

Tests the new advanced feature engineering functions
and the Stochastic momentum indicator.
"""

import pytest
import pandas as pd
import numpy as np

from indicators.momentum import Stochastic, calculate_stochastic
from indicators.pattern import find_pivots
from indicators.advanced import (
    detect_multi_indicator_divergence,
    detect_volume_exhaustion,
    detect_panic_selling,
    detect_support_tests,
    detect_exhaustion_sequence,
    detect_hidden_divergence,
    calculate_mean_reversion_signal,
    detect_bb_squeeze_breakdown,
    add_time_features,
    create_all_advanced_features,
)
from indicators.calculator import IndicatorCalculator, IndicatorConfig


class TestStochastic:
    """Test Stochastic Oscillator."""

    def test_stochastic_range(self, sample_ohlcv_data):
        """Verify Stochastic stays within 0-100 range."""
        df = sample_ohlcv_data

        slowk, slowd = Stochastic().calculate(df, fastk_period=14, slowk_period=3, slowd_period=3)

        # Both should be between 0 and 100
        valid_k = slowk.dropna()
        valid_d = slowd.dropna()

        assert valid_k.min() >= 0, "Stochastic %K below 0"
        assert valid_k.max() <= 100, "Stochastic %K above 100"
        assert valid_d.min() >= 0, "Stochastic %D below 0"
        assert valid_d.max() <= 100, "Stochastic %D above 100"

    def test_stochastic_extremes(self):
        """Test Stochastic at price extremes."""
        # Create test data with controlled extremes
        dates = pd.date_range('2023-01-01', periods=50, freq='D')

        # Price oscillating between 90-110
        close = np.array([100] * 50)
        high = np.array([110] * 50)
        low = np.array([90] * 50)

        # Make close hit high for several days
        close[30:35] = 110

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.ones(50) * 1000,
        }, index=dates)

        slowk, slowd = Stochastic().calculate(df, fastk_period=14)

        # At high of range, %K should be near 100
        assert slowk.iloc[34] > 80, "Stochastic not high when price at range high"

        # Make close hit low for several days
        close[40:45] = 90
        df['close'] = close

        slowk, slowd = Stochastic().calculate(df, fastk_period=14)

        # At low of range, %K should be near 0
        assert slowk.iloc[44] < 20, "Stochastic not low when price at range low"

    def test_stochastic_convenience_function(self, sample_ohlcv_data):
        """Test convenience function matches class."""
        df = sample_ohlcv_data

        k1, d1 = Stochastic().calculate(df)
        k2, d2 = calculate_stochastic(df)

        pd.testing.assert_series_equal(k1, k2, check_names=False)
        pd.testing.assert_series_equal(d1, d2, check_names=False)


class TestMultiIndicatorDivergence:
    """Test multi-indicator divergence detection."""

    def test_divergence_detection(self, sample_ohlcv_data):
        """Test basic divergence detection logic."""
        df = sample_ohlcv_data.copy()

        # Create pivots
        pivot_high, pivot_low = find_pivots(df, lb=8, rb=8, return_boolean=True)
        df['PivotHigh'] = pivot_high.astype(int)
        df['PivotLow'] = pivot_low.astype(int)

        # Run divergence detection
        df = detect_multi_indicator_divergence(df)

        # Should have the column
        assert 'multi_divergence_score' in df.columns

        # Scores should be 0-3
        assert df['multi_divergence_score'].min() >= 0
        assert df['multi_divergence_score'].max() <= 3

    def test_divergence_no_pivots(self, sample_ohlcv_data):
        """Test divergence when no pivots exist."""
        df = sample_ohlcv_data.copy()

        df = detect_multi_indicator_divergence(df)

        # Should return all zeros when no pivots
        assert 'multi_divergence_score' in df.columns
        assert df['multi_divergence_score'].sum() == 0


class TestVolumePatterns:
    """Test volume-based features."""

    def test_volume_exhaustion(self, sample_ohlcv_data):
        """Test volume exhaustion detection."""
        df = sample_ohlcv_data.copy()

        df = detect_volume_exhaustion(df)

        # Should have the columns
        assert 'volume_exhaustion' in df.columns
        assert 'exhaustion_strength' in df.columns

        # Binary column should be 0 or 1
        assert set(df['volume_exhaustion'].unique()).issubset({0, 1})

        # Strength should be non-negative
        assert df['exhaustion_strength'].min() >= 0

    def test_panic_selling(self, sample_ohlcv_data):
        """Test panic selling detection."""
        df = sample_ohlcv_data.copy()

        df = detect_panic_selling(df)

        # Should have the columns
        assert 'panic_selling' in df.columns
        assert 'panic_severity' in df.columns

        # Binary column should be 0 or 1
        assert set(df['panic_selling'].unique()).issubset({0, 1})

        # Severity should be between 0 and 10
        assert df['panic_severity'].min() >= 0
        assert df['panic_severity'].max() <= 10


class TestSupportLevels:
    """Test support level detection."""

    def test_support_test_count(self, sample_ohlcv_data):
        """Test support level counting."""
        df = sample_ohlcv_data.copy()

        # Create pivots
        pivot_high, pivot_low = find_pivots(df, lb=8, rb=8, return_boolean=True)
        df['PivotHigh'] = pivot_high.astype(int)
        df['PivotLow'] = pivot_low.astype(int)

        df = detect_support_tests(df, tolerance=0.02)

        # Should have the column
        assert 'support_test_count' in df.columns

        # Should be non-negative integers
        assert df['support_test_count'].min() >= 0
        assert df['support_test_count'].dtype in [np.int64, np.float64]

    def test_support_no_pivots(self, sample_ohlcv_data):
        """Test support when no pivots exist."""
        df = sample_ohlcv_data.copy()

        df = detect_support_tests(df)

        # Should return all zeros when no pivots
        assert 'support_test_count' in df.columns
        assert df['support_test_count'].sum() == 0


class TestExhaustionSequence:
    """Test exhaustion sequence detection."""

    def test_consecutive_down_days(self, sample_ohlcv_data):
        """Test consecutive down day counting."""
        df = sample_ohlcv_data.copy()

        df = detect_exhaustion_sequence(df)

        # Should have the columns
        assert 'consecutive_down_days' in df.columns
        assert 'exhaustion_signal' in df.columns
        assert 'selling_acceleration' in df.columns

        # Consecutive days should be non-negative
        assert df['consecutive_down_days'].min() >= 0

        # Signal should be binary
        assert set(df['exhaustion_signal'].dropna().unique()).issubset({0, 1})


class TestStatisticalFeatures:
    """Test statistical features."""

    def test_mean_reversion(self, sample_ohlcv_data):
        """Test mean reversion signal calculation."""
        df = sample_ohlcv_data.copy()

        df = calculate_mean_reversion_signal(df)

        # Should have the columns
        assert 'price_zscore' in df.columns
        assert 'statistical_bottom' in df.columns
        assert 'at_zscore_extreme' in df.columns

        # Z-score should be reasonable
        valid_z = df['price_zscore'].dropna()
        assert valid_z.min() > -10, "Z-score unreasonably low"
        assert valid_z.max() < 10, "Z-score unreasonably high"

        # Binary columns should be 0 or 1
        assert set(df['statistical_bottom'].dropna().unique()).issubset({0, 1})

    def test_bb_squeeze(self, sample_ohlcv_data):
        """Test Bollinger Band squeeze detection."""
        df = sample_ohlcv_data.copy()

        df = detect_bb_squeeze_breakdown(df)

        # Should have the columns
        assert 'bb_squeeze' in df.columns
        assert 'below_lower_band' in df.columns
        assert 'squeeze_breakdown' in df.columns

        # All should be binary
        assert set(df['bb_squeeze'].dropna().unique()).issubset({0, 1})
        assert set(df['below_lower_band'].dropna().unique()).issubset({0, 1})
        assert set(df['squeeze_breakdown'].dropna().unique()).issubset({0, 1})


class TestTimeFeatures:
    """Test time-based features."""

    def test_time_features(self, sample_ohlcv_data):
        """Test temporal feature extraction."""
        df = sample_ohlcv_data.copy()

        df = add_time_features(df)

        # Should have the columns
        assert 'day_of_week' in df.columns
        assert 'is_monday' in df.columns
        assert 'is_friday' in df.columns
        assert 'is_month_end' in df.columns
        assert 'is_quarter_end' in df.columns

        # Day of week should be 0-6
        assert df['day_of_week'].min() >= 0
        assert df['day_of_week'].max() <= 6

        # Binary features should be 0 or 1
        assert set(df['is_monday'].unique()).issubset({0, 1})
        assert set(df['is_friday'].unique()).issubset({0, 1})


class TestIntegration:
    """Test integration with calculator."""

    def test_create_all_advanced_features(self, sample_ohlcv_data):
        """Test master function creates all features."""
        df = sample_ohlcv_data.copy()

        # Add pivots first
        pivot_high, pivot_low = find_pivots(df, lb=8, rb=8, return_boolean=True)
        df['PivotHigh'] = pivot_high.astype(int)
        df['PivotLow'] = pivot_low.astype(int)

        df = create_all_advanced_features(df)

        # Check key features exist
        expected_features = [
            'multi_divergence_score',
            'volume_exhaustion',
            'panic_selling',
            'support_test_count',
            'consecutive_down_days',
            'price_zscore',
            'bb_squeeze',
            'day_of_week',
        ]

        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_calculator_integration(self, sample_ohlcv_data):
        """Test advanced features through calculator."""
        df = sample_ohlcv_data.copy()

        # Configure calculator with advanced features
        config = IndicatorConfig(
            calculate_advanced_features=True,
            sma_periods=[20],
            ema_periods=[20],
            rsi_periods=[14],
        )

        calculator = IndicatorCalculator(config)
        result = calculator.calculate_all(df)

        # Should have base indicators
        assert 'SMA_20' in result.columns
        assert 'RSI_14' in result.columns

        # Should have advanced features
        assert 'multi_divergence_score' in result.columns
        assert 'volume_exhaustion' in result.columns

    def test_calculator_without_advanced(self, sample_ohlcv_data):
        """Test calculator doesn't add advanced features when disabled."""
        df = sample_ohlcv_data.copy()

        # Configure calculator WITHOUT advanced features
        config = IndicatorConfig(
            calculate_advanced_features=False,
            sma_periods=[20],
        )

        calculator = IndicatorCalculator(config)
        result = calculator.calculate_all(df)

        # Should have base indicators
        assert 'SMA_20' in result.columns

        # Should NOT have advanced features
        assert 'multi_divergence_score' not in result.columns
        assert 'volume_exhaustion' not in result.columns


class TestMultiStockSupport:
    """Test multi-stock DataFrame support."""

    def test_multi_stock_divergence(self, sample_ohlcv_data):
        """Test divergence detection with multiple stocks."""
        # Create multi-stock DataFrame
        df1 = sample_ohlcv_data.copy()
        df1['stock_id'] = 'AAPL'

        df2 = sample_ohlcv_data.copy()
        df2['stock_id'] = 'MSFT'

        df = pd.concat([df1, df2], ignore_index=True)

        # Add pivots
        pivot_high, pivot_low = find_pivots(df, lb=8, rb=8, return_boolean=True)
        df['PivotHigh'] = pivot_high.astype(int)
        df['PivotLow'] = pivot_low.astype(int)

        df = detect_multi_indicator_divergence(df)

        # Should work for both stocks
        assert 'multi_divergence_score' in df.columns
        assert len(df) == len(df1) + len(df2)

    def test_single_stock_compatibility(self, sample_ohlcv_data):
        """Test features work without stock_id column."""
        df = sample_ohlcv_data.copy()

        # Should work without stock_id
        df = detect_volume_exhaustion(df)
        df = detect_panic_selling(df)
        df = detect_exhaustion_sequence(df)

        # Should have the features
        assert 'volume_exhaustion' in df.columns
        assert 'panic_selling' in df.columns
        assert 'consecutive_down_days' in df.columns
