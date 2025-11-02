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
from indicators.calculator import IndicatorCalculator, calculate_all_indicators
from indicators.cycle import HT_SINE, HT_TRENDMODE, calculate_ht_sine, calculate_ht_trendmode
from indicators.momentum import (
    ADX,
    MACD,
    MOM,
    ROC,
    RSI,
    Stochastic,
    calculate_adx,
    calculate_macd,
    calculate_mom,
    calculate_roc,
    calculate_rsi,
    calculate_stochastic,
)
from indicators.pattern import Hammer, calculate_hammer, detect_rsi_divergence, find_pivots
from indicators.trend import EMA, SMA, VWAP, calculate_ema, calculate_sma, calculate_vwap
from indicators.volatility import (
    ADR,
    APZ,
    ATR,
    SAR,
    BBands,
    calculate_adr,
    calculate_apz,
    calculate_atr,
    calculate_bbands,
    calculate_sar,
)
from indicators.volume import ADOSC, OBV, calculate_adosc, calculate_obv

__all__ = [
    "BaseIndicator",
    "IndicatorCalculator",
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
    "Hammer",
    "HT_SINE",
    "HT_TRENDMODE",
    "calculate_sma",
    "calculate_ema",
    "calculate_vwap",
    "calculate_rsi",
    "calculate_macd",
    "calculate_adx",
    "calculate_roc",
    "calculate_mom",
    "calculate_stochastic",
    "calculate_bbands",
    "calculate_atr",
    "calculate_adr",
    "calculate_apz",
    "calculate_sar",
    "calculate_obv",
    "calculate_adosc",
    "calculate_hammer",
    "calculate_ht_sine",
    "calculate_ht_trendmode",
    "calculate_all_indicators",
    "find_pivots",
    "detect_rsi_divergence",
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
