"""
Example: Using Indicators with Custom Parameters

Demonstrates three approaches:
1. IndicatorCalculator with custom config (RECOMMENDED)
2. Individual indicator imports
3. Advanced ML features

Run: uv run python example_indicator_import.py
"""

import yfinance as yf

# ============================================================================
# APPROACH 1: IndicatorCalculator with Custom Config (RECOMMENDED)
# ============================================================================


def approach_1_calculator():
    """Use IndicatorCalculator - easiest way to get all indicators."""
    print("\n" + "=" * 70)
    print("APPROACH 1: IndicatorCalculator with Custom Config")
    print("=" * 70)

    from indicators import IndicatorCalculator, IndicatorConfig

    # Fetch sample data
    print("\nFetching AAPL data...")
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start="2024-01-01", end="2024-12-31", auto_adjust=True)
    df.columns = df.columns.str.lower()
    print(f"Downloaded {len(df)} bars")

    # Configure with custom parameters
    config = IndicatorConfig(
        # Trend indicators
        sma_periods=[20, 50, 200],  # 3 SMAs
        ema_periods=[9, 21],  # 2 EMAs
        calculate_vwap=True,
        # Momentum indicators
        rsi_periods=[14],  # Standard RSI
        macd_configs=[(12, 26, 9)],  # Standard MACD
        adx_periods=[14],
        stoch_configs=[
            (14, 3, 3),  # Standard Stochastic
            (5, 3, 3),  # Fast Stochastic
        ],
        # Volatility indicators
        bbands_configs=[
            (20, 2.0, 2.0),  # Standard BB
            (10, 1.5, 1.5),  # Tight BB
        ],
        atr_periods=[14],
        # Volume indicators
        obv_ema_periods=[21],
        # Pattern recognition
        pivot_configs=[(8, 13)],  # Pivot detection
        calculate_hammer=True,
        # Advanced ML features (optional)
        calculate_advanced_features=False,  # Set to True to enable
    )

    # ONE LINE calculates everything
    print("\nCalculating all indicators...")
    calculator = IndicatorCalculator(config)
    result = calculator.calculate_all(df)

    # Display results
    print(f"\nOriginal columns: {list(df.columns)}")
    print(f"\nNew indicator columns added: {len(result.columns) - len(df.columns)}")

    indicator_cols = [col for col in result.columns if col not in df.columns]
    print(f"\nIndicator columns: {indicator_cols[:20]}...")  # First 20

    # Show latest values
    print("\nLatest values:")
    latest = result.iloc[-1]
    print(f"  Close: ${latest['close']:.2f}")
    print(f"  SMA_20: ${latest['SMA_20']:.2f}")
    print(f"  SMA_50: ${latest['SMA_50']:.2f}")
    print(f"  SMA_200: ${latest['SMA_200']:.2f}")
    print(f"  RSI_14: {latest['RSI_14']:.2f}")
    print(f"  MACD_12_26: {latest['MACD_12_26']:.2f}")
    print(f"  STOCH_K_14: {latest['STOCH_K_14']:.2f}")
    print(f"  STOCH_K_5: {latest['STOCH_K_5']:.2f}")
    print(f"  BBands_upper_20: ${latest['BBands_upper_20']:.2f}")
    print(f"  ATR_14: ${latest['ATR_14']:.2f}")

    return result


# ============================================================================
# APPROACH 2: Individual Indicator Imports
# ============================================================================


def approach_2_individual():
    """Import and use individual indicators for fine control."""
    print("\n" + "=" * 70)
    print("APPROACH 2: Individual Indicator Imports")
    print("=" * 70)

    from indicators.momentum import calculate_macd, calculate_rsi, calculate_stochastic
    from indicators.pattern import find_pivots
    from indicators.trend import calculate_ema, calculate_sma, calculate_vwap
    from indicators.volatility import calculate_atr, calculate_bbands
    from indicators.volume import OBV

    # Fetch sample data
    print("\nFetching MSFT data...")
    ticker = yf.Ticker("MSFT")
    df = ticker.history(start="2024-01-01", end="2024-12-31", auto_adjust=True)
    df.columns = df.columns.str.lower()
    print(f"Downloaded {len(df)} bars")

    # Calculate indicators individually with custom params
    print("\nCalculating indicators individually...")

    # Trend indicators
    df["sma_10"] = calculate_sma(df, period=10)
    df["sma_30"] = calculate_sma(df, period=30)
    df["ema_12"] = calculate_ema(df, period=12)
    df["vwap"] = calculate_vwap(df)

    # Momentum indicators
    df["rsi_7"] = calculate_rsi(df, period=7)  # Fast RSI
    df["rsi_21"] = calculate_rsi(df, period=21)  # Slow RSI

    macd, signal, hist = calculate_macd(df, fastperiod=5, slowperiod=13, signalperiod=5)
    df["macd_fast"] = macd
    df["macd_signal_fast"] = signal

    stoch_k, stoch_d = calculate_stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # Volatility indicators
    bb_upper, bb_middle, bb_lower = calculate_bbands(df, period=20, nbdevup=2.0, nbdevdn=2.0)
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_middle

    df["atr_14"] = calculate_atr(df, period=14)

    # Volume indicators
    obv, obv_ema = OBV().calculate(df, ema_period=21)
    df["obv"] = obv
    df["obv_ema"] = obv_ema

    # Pattern recognition
    pivot_high, pivot_low = find_pivots(df, lb=8, rb=8, return_boolean=True)
    df["pivot_high"] = pivot_high.astype(int)
    df["pivot_low"] = pivot_low.astype(int)

    # Display results
    print(f"\nTotal columns: {len(df.columns)}")

    # Show latest values
    print("\nLatest values:")
    latest = df.iloc[-1]
    print(f"  Close: ${latest['close']:.2f}")
    print(f"  SMA_10: ${latest['sma_10']:.2f}")
    print(f"  SMA_30: ${latest['sma_30']:.2f}")
    print(f"  RSI_7: {latest['rsi_7']:.2f}")
    print(f"  RSI_21: {latest['rsi_21']:.2f}")
    print(f"  MACD Fast: {latest['macd_fast']:.2f}")
    print(f"  Stoch %K: {latest['stoch_k']:.2f}")
    print(f"  BB Width: {latest['bb_width']:.4f}")
    print(f"  ATR: ${latest['atr_14']:.2f}")

    return df


# ============================================================================
# APPROACH 3: Advanced ML Features
# ============================================================================


def approach_3_advanced_features():
    """Use advanced ML features for bottom detection."""
    print("\n" + "=" * 70)
    print("APPROACH 3: Advanced ML Features for Bottom Detection")
    print("=" * 70)

    from indicators import ADVANCED_FEATURE_COLUMNS, IndicatorCalculator, IndicatorConfig

    # Fetch sample data (need more history for advanced features)
    print("\nFetching NVDA data...")
    ticker = yf.Ticker("NVDA")
    df = ticker.history(start="2023-01-01", end="2024-12-31", auto_adjust=True)
    df.columns = df.columns.str.lower()
    print(f"Downloaded {len(df)} bars")

    # Configure with advanced features enabled
    config = IndicatorConfig(
        sma_periods=[20, 50],
        ema_periods=[20],
        rsi_periods=[14],
        macd_configs=[(12, 26, 9)],
        bbands_configs=[(20, 2.0, 2.0)],
        pivot_configs=[(8, 8)],
        calculate_advanced_features=True,  # ENABLE ADVANCED FEATURES
        support_tolerance=0.02,
    )

    # Calculate all indicators including advanced features
    print("\nCalculating indicators + advanced ML features...")
    calculator = IndicatorCalculator(config)
    result = calculator.calculate_all(df)

    # Show advanced feature columns
    print(f"\nAdvanced feature columns ({len(ADVANCED_FEATURE_COLUMNS)}):")
    for i, col in enumerate(ADVANCED_FEATURE_COLUMNS, 1):
        if col in result.columns:
            print(f"  {i:2d}. {col}")

    # Show latest advanced feature values
    print("\nLatest advanced feature values:")
    latest = result.iloc[-1]

    print("\n  Divergence Signals:")
    print(f"    Multi-divergence score: {latest['multi_divergence_score']:.0f}/3")
    print(f"    Hidden divergence: {latest['hidden_bullish_divergence']:.0f}")

    print("\n  Volume Patterns:")
    print(f"    Volume exhaustion: {latest['volume_exhaustion']:.0f}")
    print(f"    Exhaustion strength: {latest['exhaustion_strength']:.2f}")
    print(f"    Panic selling: {latest['panic_selling']:.0f}")
    print(f"    Panic severity: {latest['panic_severity']:.2f}")

    print("\n  Support & Exhaustion:")
    print(f"    Support test count: {latest['support_test_count']:.0f}")
    print(f"    Consecutive down days: {latest['consecutive_down_days']:.0f}")
    print(f"    Exhaustion signal: {latest['exhaustion_signal']:.0f}")

    print("\n  Statistical:")
    print(f"    Price Z-score: {latest['price_zscore']:.2f}")
    print(f"    Statistical bottom: {latest['statistical_bottom']:.0f}")

    print("\n  Volatility:")
    print(f"    BB squeeze: {latest['bb_squeeze']:.0f}")
    print(f"    Below lower band: {latest['below_lower_band']:.0f}")
    print(f"    Squeeze breakdown: {latest['squeeze_breakdown']:.0f}")

    print("\n  Temporal:")
    print(f"    Day of week: {latest['day_of_week']:.0f} (0=Mon, 4=Fri)")
    print(f"    Is Monday: {latest['is_monday']:.0f}")
    print(f"    Days since last pivot: {latest['days_since_last_pivot']:.0f}")

    # Find recent bottom signals (multi-divergence score > 0)
    bottoms = result[result["multi_divergence_score"] > 0].tail(5)
    print("\n\nRecent bottom signals (multi-divergence):")
    if len(bottoms) > 0:
        for idx, row in bottoms.iterrows():
            print(
                f"  {idx.date()}: Score={row['multi_divergence_score']:.0f}, "
                f"Close=${row['close']:.2f}, RSI={row['RSI_14']:.1f}"
            )
    else:
        print("  No divergence signals found recently")

    return result


# ============================================================================
# COMPARISON: Show Performance Difference
# ============================================================================


def compare_approaches():
    """Compare calculation speed between approaches."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    import time

    from indicators import IndicatorCalculator, IndicatorConfig

    # Get data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start="2020-01-01", end="2024-12-31", auto_adjust=True)
    df.columns = df.columns.str.lower()
    print(f"\nDataset: {len(df)} bars")

    # Test calculator approach
    config = IndicatorConfig(
        sma_periods=[20, 50, 200],
        ema_periods=[9, 21],
        rsi_periods=[14],
        macd_configs=[(12, 26, 9)],
        stoch_configs=[(14, 3, 3)],
        bbands_configs=[(20, 2.0, 2.0)],
        atr_periods=[14],
    )

    calculator = IndicatorCalculator(config)

    start = time.perf_counter()
    result = calculator.calculate_all(df)
    elapsed = time.perf_counter() - start

    print("\nCalculator approach:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Indicators calculated: {len(result.columns) - len(df.columns)}")
    print(f"  Throughput: {len(df) / elapsed:.0f} bars/sec")


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("INDICATOR IMPORT EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates three ways to use indicators:")
    print("  1. IndicatorCalculator (recommended - easiest)")
    print("  2. Individual imports (fine control)")
    print("  3. Advanced ML features (bottom detection)")

    try:
        # Run examples
        approach_1_calculator()
        approach_2_individual()
        approach_3_advanced_features()
        compare_approaches()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\nRecommended approach: Use IndicatorCalculator with custom config")
        print("  - Clean, maintainable code")
        print("  - All settings in one place")
        print("  - Consistent column naming")
        print("  - Easy to add/remove indicators")
        print("\nFor ML model training:")
        print("  - Enable calculate_advanced_features=True")
        print("  - Use ADVANCED_FEATURE_COLUMNS for feature list")
        print("  - Combine with pivot labels for supervised learning")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have internet connection for yfinance data download.")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
