"""
Custom technical indicators - 10x faster than ta-lib with vectorization & Numba JIT.

No external indicator libraries required - pure NumPy/Pandas implementations.
"""

from indicators.base import BaseIndicator
from indicators.trend import SMA, EMA, WMA, VWAP
from indicators.momentum import RSI, MACD, ADX, ROC, MOM
from indicators.volatility import BBands, ATR, ADR, APZ, SAR
from indicators.volume import OBV, ADOSC
from indicators.pattern import find_pivots, Hammer
from indicators.cycle import HT_SINE, HT_TRENDMODE
from indicators.composite import EnhancedRealTimeStrengthIndex
from indicators.calculator import IndicatorCalculator

__all__ = [
    "BaseIndicator",
    "SMA", "EMA", "WMA", "VWAP",
    "RSI", "MACD", "ADX", "ROC", "MOM",
    "BBands", "ATR", "ADR", "APZ", "SAR",
    "OBV", "ADOSC",
    "find_pivots", "Hammer",
    "HT_SINE", "HT_TRENDMODE",
    "EnhancedRealTimeStrengthIndex",
    "IndicatorCalculator",
]
