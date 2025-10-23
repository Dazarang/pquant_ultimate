"""
Composite indicators: Enhanced Real-Time Strength Index.
Custom weighted composite of multiple indicators with optimization.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from indicators.base import BaseIndicator
from indicators.momentum import calculate_rsi, calculate_macd, calculate_roc, calculate_mom
from indicators.volatility import calculate_bbands
from indicators.volume import calculate_obv


class EnhancedRealTimeStrengthIndex(BaseIndicator):
    """
    Enhanced Real-Time Strength Index.

    Composite indicator combining:
    - RSI (normalized)
    - MACD (normalized)
    - Bollinger Bands position
    - OBV EMA (normalized)
    - Short-term EMA deviation
    - ROC (normalized)
    - Momentum (normalized)

    Weights are optimized to maximize correlation with future returns.
    """

    def __init__(self):
        super().__init__("EnhancedRTSI")
        self._cached_weights = None

    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """
        Calculate and normalize all component indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict of normalized indicator series
        """
        # Calculate raw indicators
        rsi = calculate_rsi(df, period=14)
        macd, macd_signal, _ = calculate_macd(df)
        upper_band, middle_band, lower_band = calculate_bbands(df, period=20)
        _, obv_ema = calculate_obv(df, ema_period=8)
        short_term_ema = df["close"].ewm(span=8, adjust=False).mean()
        roc = calculate_roc(df, period=10)
        mom = calculate_mom(df, period=10)

        # Normalize indicators to [-1, 1] or similar range
        rsi_normalized = (rsi - 50) / 50

        macd_diff = macd - macd_signal
        macd_max = np.abs(macd_diff).replace(0, np.nan).max()
        macd_normalized = macd_diff / macd_max if macd_max > 0 else macd_diff

        bb_range = (upper_band - lower_band).replace(0, np.nan)
        bb_value = (df["close"] - middle_band) / bb_range

        obv_max = np.abs(obv_ema).replace(0, np.nan).max()
        obv_normalized = obv_ema / obv_max if obv_max > 0 else obv_ema

        ema_normalized = (df["close"] - short_term_ema) / short_term_ema

        roc_max = np.abs(roc).replace(0, np.nan).max()
        roc_normalized = roc / roc_max if roc_max > 0 else roc

        mom_max = np.abs(mom).replace(0, np.nan).max()
        mom_normalized = mom / mom_max if mom_max > 0 else mom

        return {
            "rsi": rsi_normalized.fillna(0),
            "macd": macd_normalized.fillna(0),
            "bb": bb_value.fillna(0),
            "obv": obv_normalized.fillna(0),
            "ema": ema_normalized.fillna(0),
            "roc": roc_normalized.fillna(0),
            "mom": mom_normalized.fillna(0),
        }

    def _composite_index(
        self,
        weights: np.ndarray,
        indicators: dict,
    ) -> pd.Series:
        """
        Calculate composite index from weighted indicators.

        Args:
            weights: Array of 7 weights
            indicators: Dict of indicator series

        Returns:
            Composite index series
        """
        return (
            weights[0] * indicators["rsi"] +
            weights[1] * indicators["macd"] +
            weights[2] * indicators["bb"] +
            weights[3] * indicators["obv"] +
            weights[4] * indicators["ema"] +
            weights[5] * indicators["roc"] +
            weights[6] * indicators["mom"]
        )

    def _objective_function(
        self,
        weights: np.ndarray,
        indicators: dict,
        returns: pd.Series,
    ) -> float:
        """
        Objective function for weight optimization.
        Minimizes negative correlation with future returns.

        Args:
            weights: Array of 7 weights
            indicators: Dict of indicator series
            returns: Price returns series

        Returns:
            Negative correlation (to minimize)
        """
        index = self._composite_index(weights, indicators)

        # Correlation with next-period returns
        future_returns = returns.shift(-1)

        # Handle NaN and infinite values
        valid_mask = ~(index.isna() | future_returns.isna() |
                      np.isinf(index) | np.isinf(future_returns))

        if valid_mask.sum() < 10:  # Need minimum data points
            return 0.0

        correlation = index[valid_mask].corr(future_returns[valid_mask])

        # Minimize negative correlation (maximize positive correlation)
        return -correlation if not np.isnan(correlation) else 0.0

    def _optimize_weights(
        self,
        indicators: dict,
        returns: pd.Series,
    ) -> np.ndarray:
        """
        Optimize indicator weights using scipy.optimize.

        Args:
            indicators: Dict of indicator series
            returns: Price returns series

        Returns:
            Optimized weights array
        """
        initial_weights = np.array([1/7] * 7)
        bounds = [(0, 1)] * 7

        result = minimize(
            self._objective_function,
            initial_weights,
            args=(indicators, returns),
            bounds=bounds,
            method="SLSQP",
        )

        return result.x if result.success else initial_weights

    def calculate(
        self,
        df: pd.DataFrame,
        optimize: bool = True,
        weights: np.ndarray = None,
    ) -> pd.Series:
        """
        Calculate Enhanced Real-Time Strength Index.

        Args:
            df: DataFrame with OHLCV data
            optimize: Whether to optimize weights (default True)
            weights: Pre-computed weights (overrides optimization)

        Returns:
            Strength index series (0-100 scale)
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        self.validate_dataframe(df, required_cols)
        self.validate_data_length(df, 50)  # Need sufficient history

        # Calculate normalized indicators
        indicators = self._calculate_indicators(df)

        # Determine weights
        if weights is not None:
            optimized_weights = weights
        elif optimize and self._cached_weights is None:
            returns = df["close"].pct_change()
            optimized_weights = self._optimize_weights(indicators, returns)
            self._cached_weights = optimized_weights
        elif optimize and self._cached_weights is not None:
            optimized_weights = self._cached_weights
        else:
            optimized_weights = np.array([1/7] * 7)

        # Calculate composite index
        composite = self._composite_index(optimized_weights, indicators)

        # Scale to [0, 100] range
        strength_index = (composite + 1) * 50

        # Smooth with EMA
        smoothed = strength_index.ewm(span=3, adjust=False).mean()

        return smoothed.fillna(50)  # Default to neutral 50


def enhanced_real_time_strength_index(
    df: pd.DataFrame,
    optimize: bool = True,
    weights: np.ndarray = None,
) -> pd.Series:
    """
    Calculate Enhanced Real-Time Strength Index - convenience function.

    Args:
        df: DataFrame with OHLCV data
        optimize: Whether to optimize weights
        weights: Pre-computed weights

    Returns:
        Strength index series (0-100 scale)
    """
    return EnhancedRealTimeStrengthIndex().calculate(df, optimize, weights)


def calculate_indicators(df: pd.DataFrame) -> tuple:
    """
    Calculate all indicators used in composite index.
    Helper function for external analysis.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Tuple of normalized indicators
    """
    calculator = EnhancedRealTimeStrengthIndex()
    indicators = calculator._calculate_indicators(df)

    return (
        indicators["rsi"],
        indicators["macd"],
        indicators["bb"],
        indicators["obv"],
        indicators["ema"],
        indicators["roc"],
        indicators["mom"],
    )


def optimize_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Optimize weights for composite index.
    Helper function for weight analysis.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Optimized weights array
    """
    calculator = EnhancedRealTimeStrengthIndex()
    indicators = calculator._calculate_indicators(df)
    returns = df["close"].pct_change()

    return calculator._optimize_weights(indicators, returns)
