"""
High-performance technical indicators with vectorization & Numba JIT optimization.

Pure NumPy/Pandas implementations - no external dependencies.
Optimized for speed and accuracy with comprehensive validation.
"""

from indicators.advanced import (
    ADVANCED_FEATURE_COLUMNS,
    add_time_features,
    calculate_mean_reversion_signal,
    create_all_advanced_features,
    detect_bb_squeeze_breakdown,
    detect_exhaustion_sequence,
    detect_hidden_divergence,
    detect_multi_indicator_divergence,
    detect_panic_selling,
    detect_support_tests,
    detect_volume_exhaustion,
)
from indicators.base import BaseIndicator
from indicators.calculator import IndicatorCalculator
from indicators.cycle import HT_SINE, HT_TRENDMODE
from indicators.momentum import ADX, MACD, MOM, ROC, RSI, Stochastic
from indicators.pattern import Hammer, find_pivots
from indicators.trend import EMA, SMA, VWAP
from indicators.volatility import ADR, APZ, ATR, SAR, BBands
from indicators.volume import ADOSC, OBV

__all__ = [
    "BaseIndicator",
    "SMA",
    "EMA",
    "VWAP",
    "RSI",
    "MACD",
    "ADX",
    "ROC",
    "MOM",
    "Stochastic",
    "BBands",
    "ATR",
    "ADR",
    "APZ",
    "SAR",
    "OBV",
    "ADOSC",
    "find_pivots",
    "Hammer",
    "HT_SINE",
    "HT_TRENDMODE",
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
