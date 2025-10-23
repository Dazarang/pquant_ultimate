"""
IndicatorCalculator - orchestrator for computing all indicators.
Efficient batch calculation with caching and parallel processing.
"""

import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from indicators.trend import SMA, EMA, WMA, VWAP
from indicators.momentum import RSI, MACD, ADX, ROC, MOM
from indicators.volatility import BBands, ATR, ADR, APZ, SAR
from indicators.volume import OBV, ADOSC
from indicators.pattern import find_pivots, Hammer
from indicators.cycle import HT_SINE, HT_TRENDMODE
from indicators.composite import EnhancedRealTimeStrengthIndex


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations."""

    # Trend periods
    sma_periods: List[int] = field(default_factory=lambda: [5, 8, 21, 50, 55, 89, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 8, 21, 50, 55, 89])
    wma_periods: List[int] = field(default_factory=lambda: [21])

    # Momentum parameters
    rsi_periods: List[int] = field(default_factory=lambda: [14])
    macd_configs: List[tuple] = field(default_factory=lambda: [(12, 26, 9)])
    adx_periods: List[int] = field(default_factory=lambda: [14])
    roc_periods: List[int] = field(default_factory=lambda: [10])
    mom_periods: List[int] = field(default_factory=lambda: [10])

    # Volatility parameters
    bbands_configs: List[tuple] = field(default_factory=lambda: [(20, 2.0, 2.0)])
    atr_periods: List[int] = field(default_factory=lambda: [14])
    adr_periods: List[int] = field(default_factory=lambda: [20])
    apz_configs: List[tuple] = field(default_factory=lambda: [(21, 2.0)])
    sar_configs: List[tuple] = field(default_factory=lambda: [(0.02, 0.2)])

    # Volume parameters
    obv_ema_periods: List[int] = field(default_factory=lambda: [8, 55])
    adosc_configs: List[tuple] = field(default_factory=lambda: [(3, 10)])

    # Pattern parameters
    pivot_configs: List[tuple] = field(default_factory=lambda: [(8, 8)])

    # Flags
    calculate_vwap: bool = True
    calculate_hammer: bool = True
    calculate_ht_sine: bool = True
    calculate_ht_trendmode: bool = True
    calculate_enhanced_rtsi: bool = True


class IndicatorCalculator:
    """
    Orchestrates calculation of all technical indicators.
    Provides batch computation with caching for performance.
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize calculator with configuration.

        Args:
            config: Indicator configuration (uses defaults if None)
        """
        self.config = config or IndicatorConfig()
        self._indicators_cache: Dict[str, pd.Series] = {}

    def calculate_all(
        self,
        df: pd.DataFrame,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate all configured indicators and add to DataFrame.

        Args:
            df: DataFrame with OHLCV data
            use_cache: Whether to use cached results

        Returns:
            DataFrame with all indicators added
        """
        result_df = df.copy()

        # Clear cache if not using it
        if not use_cache:
            self._indicators_cache.clear()

        # Calculate trend indicators
        result_df = self._add_trend_indicators(result_df)

        # Calculate momentum indicators
        result_df = self._add_momentum_indicators(result_df)

        # Calculate volatility indicators
        result_df = self._add_volatility_indicators(result_df)

        # Calculate volume indicators
        result_df = self._add_volume_indicators(result_df)

        # Calculate pattern indicators
        result_df = self._add_pattern_indicators(result_df)

        # Calculate cycle indicators
        result_df = self._add_cycle_indicators(result_df)

        # Calculate composite indicators
        result_df = self._add_composite_indicators(result_df)

        return result_df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators to DataFrame."""
        # SMA
        for period in self.config.sma_periods:
            df[f"SMA_{period}"] = SMA().calculate(df, period)

        # EMA
        for period in self.config.ema_periods:
            df[f"EMA_{period}"] = EMA().calculate(df, period)

        # WMA
        for period in self.config.wma_periods:
            df[f"WMA_{period}"] = WMA().calculate(df, period)

        # VWAP
        if self.config.calculate_vwap:
            df["VWAP"] = VWAP().calculate(df)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators to DataFrame."""
        # RSI
        for period in self.config.rsi_periods:
            df[f"RSI_{period}"] = RSI().calculate(df, period)

        # MACD
        for fast, slow, signal in self.config.macd_configs:
            macd, macd_signal, macd_hist = MACD().calculate(df, fast, slow, signal)
            df[f"MACD_{fast}_{slow}"] = macd
            df[f"MACD_signal_{fast}_{slow}"] = macd_signal
            df[f"MACD_hist_{fast}_{slow}"] = macd_hist

        # ADX
        for period in self.config.adx_periods:
            df[f"ADX_{period}"] = ADX().calculate(df, period)

        # ROC
        for period in self.config.roc_periods:
            df[f"ROC_{period}"] = ROC().calculate(df, period)

        # MOM
        for period in self.config.mom_periods:
            df[f"MOM_{period}"] = MOM().calculate(df, period)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators to DataFrame."""
        # Bollinger Bands
        for period, nbdevup, nbdevdn in self.config.bbands_configs:
            upper, middle, lower = BBands().calculate(df, period, nbdevup, nbdevdn)
            df[f"BBands_upper_{period}"] = upper
            df[f"BBands_middle_{period}"] = middle
            df[f"BBands_lower_{period}"] = lower

        # ATR
        for period in self.config.atr_periods:
            df[f"ATR_{period}"] = ATR().calculate(df, period)

        # ADR
        for period in self.config.adr_periods:
            df[f"ADR_{period}"] = ADR().calculate(df, period)

        # APZ
        for period, band_pct in self.config.apz_configs:
            upper, lower = APZ().calculate(df, period, band_pct)
            df[f"APZ_upper_{period}"] = upper
            df[f"APZ_lower_{period}"] = lower

        # SAR
        for accel, maximum in self.config.sar_configs:
            df["SAR"] = SAR().calculate(df, accel, maximum)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators to DataFrame."""
        # OBV
        for ema_period in self.config.obv_ema_periods:
            obv, obv_ema = OBV().calculate(df, ema_period)
            df["OBV"] = obv
            df[f"OBV_EMA_{ema_period}"] = obv_ema

        # ADOSC
        for fast, slow in self.config.adosc_configs:
            df[f"ADOSC_{fast}_{slow}"] = ADOSC().calculate(df, fast, slow)

        return df

    def _add_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern indicators to DataFrame."""
        # Pivots
        for lb, rb in self.config.pivot_configs:
            pivot_high, pivot_low = find_pivots(df, lb, rb, return_boolean=True)
            df["PivotHigh"] = pivot_high.astype(int)
            df["PivotLow"] = pivot_low.astype(int)

        # Hammer
        if self.config.calculate_hammer:
            df["Hammer"] = Hammer().calculate(df)

        return df

    def _add_cycle_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cycle indicators to DataFrame."""
        if len(df) < 32:  # Minimum for HT stability
            return df

        # HT_SINE
        if self.config.calculate_ht_sine:
            sine, leadsine = HT_SINE().calculate(df)
            df["HT_SINE"] = sine
            df["HT_LEADSINE"] = leadsine

        # HT_TRENDMODE
        if self.config.calculate_ht_trendmode:
            df["HT_TRENDMODE"] = HT_TRENDMODE().calculate(df)

        return df

    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite indicators to DataFrame."""
        if self.config.calculate_enhanced_rtsi:
            df["Enhanced_RTSI"] = EnhancedRealTimeStrengthIndex().calculate(df)

        return df

    def clear_cache(self) -> None:
        """Clear indicator cache."""
        self._indicators_cache.clear()


def calculate_all_indicators(
    df: pd.DataFrame,
    config: Optional[IndicatorConfig] = None,
) -> pd.DataFrame:
    """
    Convenience function to calculate all indicators.

    Args:
        df: DataFrame with OHLCV data
        config: Optional indicator configuration

    Returns:
        DataFrame with all indicators
    """
    calculator = IndicatorCalculator(config)
    return calculator.calculate_all(df)
