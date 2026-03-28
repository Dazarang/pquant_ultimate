"""
IndicatorCalculator - orchestrator for computing all indicators.
Efficient batch calculation with caching and parallel processing.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from indicators.advanced import create_all_advanced_features
from indicators.cycle import HT_SINE, HT_TRENDMODE
from indicators.momentum import ADX, MACD, MOM, ROC, RSI, Stochastic
from indicators.pattern import Hammer, find_pivots
from indicators.trend import EMA, SMA, VWAP
from indicators.volatility import ADR, APZ, ATR, SAR, BBands
from indicators.volume import ADOSC, OBV

_MAX_WORKERS = min(8, os.cpu_count() or 4)


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations."""

    # Trend periods
    sma_periods: list[int] = field(default_factory=lambda: [5, 8, 21, 50, 55, 89, 200])
    ema_periods: list[int] = field(default_factory=lambda: [5, 8, 21, 50, 55, 89])

    # Momentum parameters
    rsi_periods: list[int] = field(default_factory=lambda: [14])
    macd_configs: list[tuple] = field(default_factory=lambda: [(12, 26, 9)])
    adx_periods: list[int] = field(default_factory=lambda: [14])
    roc_periods: list[int] = field(default_factory=lambda: [10])
    mom_periods: list[int] = field(default_factory=lambda: [10])
    stoch_configs: list[tuple] = field(default_factory=lambda: [(14, 3, 3)])  # (fastk, slowk, slowd)

    # Volatility parameters
    bbands_configs: list[tuple] = field(default_factory=lambda: [(20, 2.0, 2.0)])
    atr_periods: list[int] = field(default_factory=lambda: [14])
    adr_periods: list[int] = field(default_factory=lambda: [20])
    apz_configs: list[tuple] = field(default_factory=lambda: [(21, 2.0)])
    sar_configs: list[tuple] = field(default_factory=lambda: [(0.02, 0.2)])

    # Volume parameters
    obv_ema_periods: list[int] = field(default_factory=lambda: [8, 55])
    adosc_configs: list[tuple] = field(default_factory=lambda: [(3, 10)])

    # Pattern parameters
    pivot_configs: list[tuple] = field(default_factory=lambda: [(8, 8)])

    # Flags
    calculate_vwap: bool = True
    calculate_hammer: bool = True
    calculate_ht_sine: bool = True
    calculate_ht_trendmode: bool = True
    calculate_advanced_features: bool = False

    # Advanced feature parameters
    support_tolerance: float = 0.02  # Price tolerance for support testing


class IndicatorCalculator:
    """
    Orchestrates calculation of all technical indicators.
    Provides batch computation with caching for performance.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """
        Initialize calculator with configuration.

        Args:
            config: Indicator configuration (uses defaults if None)
        """
        self.config = config or IndicatorConfig()
        self._indicators_cache: dict[str, pd.Series] = {}

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

        # Calculate advanced features (if enabled)
        if self.config.calculate_advanced_features:
            result_df = self._add_advanced_features(result_df)

        return result_df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators to DataFrame."""
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        futures = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for period in self.config.sma_periods:
                futures[pool.submit(SMA().calculate, ohlcv, period)] = f"SMA_{period}"
            for period in self.config.ema_periods:
                futures[pool.submit(EMA().calculate, ohlcv, period)] = f"EMA_{period}"
            if self.config.calculate_vwap:
                futures[pool.submit(VWAP().calculate, ohlcv)] = "VWAP"

        for future in as_completed(futures):
            df[futures[future]] = future.result()
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators to DataFrame."""
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        futures = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for period in self.config.rsi_periods:
                futures[pool.submit(RSI().calculate, ohlcv, period)] = ("single", f"RSI_{period}")
            for fast, slow, signal in self.config.macd_configs:
                futures[pool.submit(MACD().calculate, ohlcv, fast, slow, signal)] = (
                    "macd",
                    (f"MACD_{fast}_{slow}", f"MACD_signal_{fast}_{slow}", f"MACD_hist_{fast}_{slow}"),
                )
            for period in self.config.adx_periods:
                futures[pool.submit(ADX().calculate, ohlcv, period)] = ("single", f"ADX_{period}")
            for period in self.config.roc_periods:
                futures[pool.submit(ROC().calculate, ohlcv, period)] = ("single", f"ROC_{period}")
            for period in self.config.mom_periods:
                futures[pool.submit(MOM().calculate, ohlcv, period)] = ("single", f"MOM_{period}")
            for fastk, slowk, slowd in self.config.stoch_configs:
                futures[pool.submit(Stochastic().calculate, ohlcv, fastk, slowk, slowd)] = (
                    "pair",
                    (f"STOCH_K_{fastk}", f"STOCH_D_{fastk}"),
                )

        for future in as_completed(futures):
            kind, names = futures[future]
            result = future.result()
            if kind == "single":
                df[names] = result
            elif kind == "pair":
                df[names[0]], df[names[1]] = result
            elif kind == "macd":
                df[names[0]], df[names[1]], df[names[2]] = result
        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators to DataFrame."""
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        futures = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for period, nbdevup, nbdevdn in self.config.bbands_configs:
                futures[pool.submit(BBands().calculate, ohlcv, period, nbdevup, nbdevdn)] = (
                    "triple",
                    (f"BBands_upper_{period}", f"BBands_middle_{period}", f"BBands_lower_{period}"),
                )
            for period in self.config.atr_periods:
                futures[pool.submit(ATR().calculate, ohlcv, period)] = ("single", f"ATR_{period}")
            for period in self.config.adr_periods:
                futures[pool.submit(ADR().calculate, ohlcv, period)] = ("single", f"ADR_{period}")
            for period, band_pct in self.config.apz_configs:
                futures[pool.submit(APZ().calculate, ohlcv, period, band_pct)] = (
                    "pair",
                    (f"APZ_upper_{period}", f"APZ_lower_{period}"),
                )
            for accel, maximum in self.config.sar_configs:
                futures[pool.submit(SAR().calculate, ohlcv, accel, maximum)] = ("single", "SAR")

        for future in as_completed(futures):
            kind, names = futures[future]
            result = future.result()
            if kind == "single":
                df[names] = result
            elif kind == "pair":
                df[names[0]], df[names[1]] = result
            elif kind == "triple":
                df[names[0]], df[names[1]], df[names[2]] = result
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators to DataFrame."""
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        futures = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for ema_period in self.config.obv_ema_periods:
                futures[pool.submit(OBV().calculate, ohlcv, ema_period)] = (
                    "obv",
                    ("OBV", f"OBV_EMA_{ema_period}"),
                )
            for fast, slow in self.config.adosc_configs:
                futures[pool.submit(ADOSC().calculate, ohlcv, fast, slow)] = ("single", f"ADOSC_{fast}_{slow}")

        for future in as_completed(futures):
            kind, names = futures[future]
            result = future.result()
            if kind == "single":
                df[names] = result
            elif kind == "obv":
                df[names[0]], df[names[1]] = result
        return df

    def _add_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern indicators to DataFrame."""
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        futures = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for lb, rb in self.config.pivot_configs:
                futures[pool.submit(find_pivots, ohlcv, lb, rb, return_boolean=True)] = (
                    "pivot",
                    ("PivotHigh", "PivotLow"),
                )
            if self.config.calculate_hammer:
                futures[pool.submit(Hammer().calculate, ohlcv)] = ("single", "Hammer")

        for future in as_completed(futures):
            kind, names = futures[future]
            result = future.result()
            if kind == "single":
                df[names] = result
            elif kind == "pivot":
                pivot_high, pivot_low = result
                df[names[0]] = pivot_high.astype(int)
                df[names[1]] = pivot_low.astype(int)
        return df

    def _add_cycle_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cycle indicators to DataFrame."""
        if len(df) < 32:  # Minimum for HT stability
            return df

        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        futures = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            if self.config.calculate_ht_sine:
                futures[pool.submit(HT_SINE().calculate, ohlcv)] = (
                    "pair",
                    ("HT_SINE", "HT_LEADSINE"),
                )
            if self.config.calculate_ht_trendmode:
                futures[pool.submit(HT_TRENDMODE().calculate, ohlcv)] = ("single", "HT_TRENDMODE")

        for future in as_completed(futures):
            kind, names = futures[future]
            result = future.result()
            if kind == "single":
                df[names] = result
            elif kind == "pair":
                df[names[0]], df[names[1]] = result
        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced ML features for bottom detection.

        Applies all advanced feature engineering functions.
        These features combine multiple indicators to detect patterns.
        """
        df = create_all_advanced_features(
            df,
            support_tolerance=self.config.support_tolerance,
        )
        return df

    def clear_cache(self) -> None:
        """Clear indicator cache."""
        self._indicators_cache.clear()


def calculate_all_indicators(
    df: pd.DataFrame,
    config: IndicatorConfig | None = None,
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
