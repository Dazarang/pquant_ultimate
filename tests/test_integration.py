"""
Integration tests for the complete indicator system.
"""

import pandas as pd
import pytest

from indicators.calculator import IndicatorCalculator, IndicatorConfig


class TestIntegration:
    """Integration tests for full system."""

    def test_calculate_all_indicators(self, sample_ohlcv_data):
        """Test calculating all indicators at once."""
        df = sample_ohlcv_data

        calculator = IndicatorCalculator()
        result = calculator.calculate_all(df)

        # Should have many more columns than original
        assert result.shape[1] > df.shape[1]

        # Original columns should be preserved
        for col in df.columns:
            assert col in result.columns

        # Should have key indicators
        assert "SMA_50" in result.columns
        assert "EMA_21" in result.columns
        assert "RSI_14" in result.columns
        assert "ATR_14" in result.columns
        assert "VWAP" in result.columns

        print(f"\nCalculated {result.shape[1]} columns from {df.shape[1]} input columns")

    def test_custom_config(self, sample_ohlcv_data):
        """Test custom indicator configuration."""
        df = sample_ohlcv_data

        # Custom config with fewer indicators
        config = IndicatorConfig(
            sma_periods=[20, 50],
            ema_periods=[12],
            rsi_periods=[14],
            calculate_vwap=True,
            calculate_hammer=False,
            calculate_ht_sine=False,
        )

        calculator = IndicatorCalculator(config)
        result = calculator.calculate_all(df)

        # Should have configured indicators
        assert "SMA_20" in result.columns
        assert "SMA_50" in result.columns
        assert "EMA_12" in result.columns

        # Should not have disabled indicators
        assert "Hammer" not in result.columns

    def test_no_nans_propagation(self, sample_ohlcv_data):
        """Test that NaNs don't propagate unexpectedly."""
        df = sample_ohlcv_data

        calculator = IndicatorCalculator()
        result = calculator.calculate_all(df)

        # Check that we have valid data in latter part
        latter_half = result.iloc[len(result) // 2 :]

        for col in result.columns:
            if col in ["open", "high", "low", "close", "volume"]:
                continue  # Skip original columns

            valid_count = latter_half[col].notna().sum()
            total_count = len(latter_half)

            # At least 50% should be valid in latter half
            assert valid_count > total_count * 0.5, (
                f"{col}: Only {valid_count}/{total_count} valid values in latter half"
            )

    def test_caching(self, sample_ohlcv_data):
        """Test that caching works."""
        df = sample_ohlcv_data

        calculator = IndicatorCalculator()

        # First calculation
        result1 = calculator.calculate_all(df, use_cache=True)

        # Second calculation (should use cache)
        result2 = calculator.calculate_all(df, use_cache=True)

        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)

        # Clear cache
        calculator.clear_cache()

        # Third calculation (no cache)
        result3 = calculator.calculate_all(df, use_cache=False)

        # Should still be identical
        pd.testing.assert_frame_equal(result1, result3)

    def test_real_world_workflow(self, real_market_data):
        """Test real-world workflow with actual market data."""
        df = real_market_data

        # Step 1: Calculate all indicators
        calculator = IndicatorCalculator()
        df_with_indicators = calculator.calculate_all(df)

        # Step 2: Verify key indicators for ML features
        required_for_ml = [
            "RSI_14",
            "MACD_12_26",
            "BBands_upper_20",
            "ATR_14",
            "ADX_14",
            "OBV_EMA_8",
            "PivotHigh",
            "PivotLow",
        ]

        for col in required_for_ml:
            assert col in df_with_indicators.columns, f"Missing {col}"

            # Should have mostly valid data
            valid_pct = df_with_indicators[col].notna().sum() / len(df_with_indicators)
            assert valid_pct > 0.7, f"{col}: Only {valid_pct:.1%} valid data"

        print("\n✓ Real-world workflow test passed")
        print(f"  Input: {df.shape}")
        print(f"  Output: {df_with_indicators.shape}")
        print(f"  Added {df_with_indicators.shape[1] - df.shape[1]} indicator columns")

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        calculator = IndicatorCalculator()

        # Empty DataFrame
        with pytest.raises((ValueError, Exception)):
            empty_df = pd.DataFrame()
            calculator.calculate_all(empty_df)

        # Missing columns
        with pytest.raises((ValueError, KeyError)):
            invalid_df = pd.DataFrame({"invalid": [1, 2, 3]})
            calculator.calculate_all(invalid_df)

        # Too little data - should raise error for indicators requiring more data
        df = pd.DataFrame(
            {
                "open": [1, 2],
                "high": [2, 3],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [1000, 2000],
            }
        )

        # Should raise error due to insufficient data
        with pytest.raises((ValueError, Exception)):
            calculator.calculate_all(df)
