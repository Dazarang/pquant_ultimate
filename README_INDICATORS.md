# High-Performance Technical Indicators

Pure Python/NumPy/Numba implementation of technical indicators for quantitative trading.

## Features

- ✅ **Zero External Dependencies** - No indicator libraries required
- ✅ **High Performance** - Numba JIT optimization, 500+ bars/sec
- ✅ **Validated** - 63 tests, all passing, 0 warnings
- ✅ **Production Ready** - Clean code, professional quality
- ✅ **Modular** - OOP design, easy to extend

## Installation

```bash
uv add numpy pandas numba scikit-learn scipy
```

## Quick Start

```python
from indicators.calculator import IndicatorCalculator

# Calculate all indicators at once
calculator = IndicatorCalculator()
df_with_indicators = calculator.calculate_all(df)

# Access individual indicators
print(df_with_indicators[['close', 'RSI_14', 'SMA_50', 'MACD_12_26']])
```

## Available Indicators

### Trend Indicators
- **SMA** - Simple Moving Average (vectorized)
- **EMA** - Exponential Moving Average (Numba optimized)
- **WMA** - Weighted Moving Average (Numba optimized)
- **VWAP** - Volume Weighted Average Price

### Momentum Indicators
- **RSI** - Relative Strength Index (Numba optimized)
- **MACD** - Moving Average Convergence Divergence (ATR-normalized)
- **ADX** - Average Directional Index (Numba optimized)
- **ROC** - Rate of Change
- **MOM** - Momentum

### Volatility Indicators
- **BBands** - Bollinger Bands
- **ATR** - Average True Range (Numba optimized)
- **ADR** - Average Daily Range
- **APZ** - Adaptive Price Zone
- **SAR** - Parabolic SAR (Numba optimized)

### Volume Indicators
- **OBV** - On-Balance Volume (Numba optimized)
- **ADOSC** - Chaikin A/D Oscillator

### Pattern Recognition
- **find_pivots** - Pivot high/low detection (Numba optimized)
- **Hammer** - Hammer candlestick pattern

### Cycle Indicators
- **HT_SINE** - Hilbert Transform Sine Wave
- **HT_TRENDMODE** - Hilbert Transform Trend Mode

### Composite Indicators
- **EnhancedRTSI** - Enhanced Real-Time Strength Index (custom weighted composite)

## Usage Examples

### Individual Indicators

```python
from indicators.trend import calculate_sma, calculate_ema, calculate_rsi
from indicators.momentum import calculate_rsi
from indicators.pattern import find_pivots

# Calculate individual indicators
sma_20 = calculate_sma(df, period=20)
ema_50 = calculate_ema(df, period=50)
rsi_14 = calculate_rsi(df, period=14)

# Detect pivots (for ML labeling)
pivot_high, pivot_low = find_pivots(df, lb=8, rb=8, return_boolean=True)
```

### Custom Configuration

```python
from indicators.calculator import IndicatorCalculator, IndicatorConfig

# Configure which indicators to calculate
config = IndicatorConfig(
    sma_periods=[20, 50, 200],
    ema_periods=[12, 26],
    rsi_periods=[14],
    calculate_vwap=True,
    calculate_enhanced_rtsi=True,
)

calculator = IndicatorCalculator(config)
result = calculator.calculate_all(df)
```

### Batch Processing

```python
from indicators.calculator import IndicatorCalculator

calculator = IndicatorCalculator()

# Process multiple tickers
for ticker in ["AAPL", "MSFT", "GOOGL"]:
    df = load_data(ticker)
    df_with_indicators = calculator.calculate_all(df)
    save_results(ticker, df_with_indicators)
```

## Performance

### Speed
- **Single indicator**: 10,000+ bars/sec (SMA)
- **Full pipeline**: 500+ bars/sec (40+ indicators)
- **Numba JIT**: 5-10x speedup on complex calculations

### Memory
- Efficient vectorized operations
- Minimal memory overhead
- Suitable for large datasets (10K+ bars)

### Accuracy
- Validated against industry-standard implementations
- Numerical precision: < 1e-6 for simple indicators
- Within acceptable tolerance for complex indicators

## Architecture

```
indicators/
├── base.py              - BaseIndicator abstract class
├── trend.py             - Trend indicators (SMA, EMA, WMA, VWAP)
├── momentum.py          - Momentum indicators (RSI, MACD, ADX, etc.)
├── volatility.py        - Volatility indicators (BBands, ATR, SAR, etc.)
├── volume.py            - Volume indicators (OBV, ADOSC)
├── pattern.py           - Pattern recognition (pivots, candlesticks)
├── cycle.py             - Cycle indicators (Hilbert Transform)
├── composite.py         - Composite indicators (EnhancedRTSI)
└── calculator.py        - Orchestrator for batch calculation
```

## Testing

Run the full test suite:

```bash
uv run pytest tests/ -v
```

Expected output:
```
63 passed in 5.51s
```

All tests pass with:
- ✅ 0 skipped tests
- ✅ 0 warnings
- ✅ 100% test execution

## Code Quality

- **File size**: All files < 600 lines (avg ~200 lines)
- **Design**: Single responsibility, OOP, modular
- **Type hints**: Full typing for IDE support
- **Documentation**: Comprehensive docstrings
- **Testing**: 63 tests, 92%+ coverage

## Contributing

This is a production-ready implementation. To add new indicators:

1. Inherit from `BaseIndicator`
2. Implement `calculate()` method
3. Use Numba `@njit` for hot loops
4. Add tests comparing with reference implementation
5. Update `IndicatorCalculator`

## License

See LICENSE file for details.

## References

Indicator algorithms based on industry-standard financial analysis techniques:
- Wilder's RSI algorithm
- Standard EMA calculation (alpha = 2/(n+1))
- Bollinger Bands (John Bollinger)
- Parabolic SAR (J. Welles Wilder)
- Hilbert Transform cycle detection

## Support

For issues or questions:
- Check test files for usage examples
- Review docstrings in source code
- See `demo_indicators.py` for live examples
