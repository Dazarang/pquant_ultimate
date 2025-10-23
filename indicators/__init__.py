"""
High-performance technical indicators with vectorization & Numba JIT optimization.

Pure NumPy/Pandas implementations - no external dependencies.
Optimized for speed and accuracy with comprehensive validation.
"""

from indicators.base import BaseIndicator
from indicators.trend import SMA, EMA, VWAP
from indicators.momentum import RSI, MACD, ADX, ROC, MOM, Stochastic
from indicators.volatility import BBands, ATR, ADR, APZ, SAR
from indicators.volume import OBV, ADOSC
from indicators.pattern import find_pivots, Hammer
from indicators.cycle import HT_SINE, HT_TRENDMODE
from indicators.calculator import IndicatorCalculator
from indicators.advanced import (
    create_all_advanced_features,
    detect_multi_indicator_divergence,
    detect_volume_exhaustion,
    detect_panic_selling,
    detect_support_tests,
    detect_exhaustion_sequence,
    detect_hidden_divergence,
    calculate_mean_reversion_signal,
    detect_bb_squeeze_breakdown,
    add_time_features,
    ADVANCED_FEATURE_COLUMNS,
)

__all__ = [
    "BaseIndicator",
    "SMA", "EMA", "VWAP",
    "RSI", "MACD", "ADX", "ROC", "MOM", "Stochastic",
    "BBands", "ATR", "ADR", "APZ", "SAR",
    "OBV", "ADOSC",
    "find_pivots", "Hammer",
    "HT_SINE", "HT_TRENDMODE",
    "IndicatorCalculator",
    "create_all_advanced_features",
    "detect_multi_indicator_divergence",
    "detect_volume_exhaustion",
    "detect_panic_selling",
    "detect_support_tests",
    "detect_exhaustion_sequence",
    "detect_hidden_divergence",
    "calculate_mean_reversion_signal",
    "detect_bb_squeeze_breakdown",
    "add_time_features",
    "ADVANCED_FEATURE_COLUMNS",
]
