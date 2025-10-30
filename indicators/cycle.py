"""
Cycle indicators: HT_SINE, HT_TRENDMODE.
Hilbert Transform-based indicators for cycle detection.
"""

import numpy as np
import pandas as pd
from numba import njit

from indicators.base import BaseIndicator, ensure_numpy_array


@njit
def _hilbert_transform_numba(data: np.ndarray) -> tuple:
    """
    Numba-optimized Hilbert Transform for cycle detection.

    Implements standard Hilbert Transform with high-pass filter detrending
    and 7-period weighted sum for 90-degree phase shift approximation.

    Args:
        data: Price array

    Returns:
        Tuple of (inphase, quadrature, phase)
    """
    n = len(data)

    # Detrend with high-pass filter
    detrend = np.empty(n)
    detrend[:7] = 0.0

    for i in range(7, n):
        detrend[i] = (0.0962 * data[i] + 0.5769 * data[i - 2] - 0.5769 * data[i - 4] - 0.0962 * data[i - 6]) * (
            0.075 * (i - 6) + 0.54
        )

    # InPhase and Quadrature components
    inphase = np.empty(n)
    quadrature = np.empty(n)
    inphase[:7] = 0.0
    quadrature[:7] = 0.0

    for i in range(7, n):
        # Weighted sum for Hilbert Transform
        inphase[i] = 1.25 * (detrend[i - 4] - 0.33 * detrend[i - 6])
        quadrature[i] = detrend[i - 2]

    # Smooth InPhase and Quadrature
    smooth_inphase = np.empty(n)
    smooth_quadrature = np.empty(n)
    smooth_inphase[:] = inphase[:]
    smooth_quadrature[:] = quadrature[:]

    for i in range(7, n):
        smooth_inphase[i] = 0.33 * inphase[i] + 0.67 * smooth_inphase[i - 1]
        smooth_quadrature[i] = 0.33 * quadrature[i] + 0.67 * smooth_quadrature[i - 1]

    # Calculate phase
    phase = np.empty(n)
    phase[:7] = 0.0

    for i in range(7, n):
        if smooth_inphase[i] != 0:
            phase[i] = np.arctan(smooth_quadrature[i] / smooth_inphase[i])
        else:
            phase[i] = phase[i - 1]

    return smooth_inphase, smooth_quadrature, phase


@njit
def _calculate_sine_wave_numba(phase: np.ndarray) -> tuple:
    """
    Calculate sine and lead sine from phase.

    Args:
        phase: Phase array from Hilbert Transform

    Returns:
        Tuple of (sine, leadsine)
    """
    n = len(phase)
    sine = np.empty(n)
    leadsine = np.empty(n)

    for i in range(n):
        sine[i] = np.sin(phase[i])
        leadsine[i] = np.sin(phase[i] + np.pi / 4)  # 45-degree lead

    return sine, leadsine


@njit
def _detect_trend_mode_numba(inphase: np.ndarray, quadrature: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Detect trend vs cycle mode using Hilbert Transform components.

    Args:
        inphase: InPhase component
        quadrature: Quadrature component
        threshold: Threshold for trend detection

    Returns:
        Trend mode array (1 = trend, 0 = cycle)
    """
    n = len(inphase)
    result = np.zeros(n, dtype=np.int32)

    for i in range(7, n):
        # Calculate instantaneous amplitude
        amplitude = np.sqrt(inphase[i] ** 2 + quadrature[i] ** 2)

        # Compare with smoothed amplitude
        if i >= 14:
            avg_amplitude = 0.0
            for j in range(i - 7, i):
                avg_amplitude += np.sqrt(inphase[j] ** 2 + quadrature[j] ** 2)
            avg_amplitude /= 7

            # Trend if amplitude is consistently high
            if amplitude > threshold * avg_amplitude:
                result[i] = 1

    return result


class HT_SINE(BaseIndicator):
    """Hilbert Transform - Sine Wave Indicator."""

    def __init__(self):
        super().__init__("HT_SINE")

    def calculate(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Hilbert Transform Sine Wave.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Tuple of (sine, leadsine) series
        """
        self.validate_dataframe(df, ["close"])
        self.validate_data_length(df, 32)  # Minimum for HT stability

        data = ensure_numpy_array(df["close"])

        # Calculate Hilbert Transform
        inphase, quadrature, phase = _hilbert_transform_numba(data)

        # Calculate sine waves
        sine, leadsine = _calculate_sine_wave_numba(phase)

        sine_series = pd.Series(sine, index=df.index, name="HT_SINE")
        leadsine_series = pd.Series(leadsine, index=df.index, name="HT_LEADSINE")

        return sine_series, leadsine_series


class HT_TRENDMODE(BaseIndicator):
    """Hilbert Transform - Trend vs Cycle Mode."""

    def __init__(self):
        super().__init__("HT_TRENDMODE")

    def calculate(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Calculate Hilbert Transform Trend Mode.

        Args:
            df: DataFrame with 'close' column
            threshold: Threshold for trend detection

        Returns:
            Trend mode series (1 = trend, 0 = cycle)
        """
        self.validate_dataframe(df, ["close"])
        self.validate_data_length(df, 32)  # Minimum for HT stability

        data = ensure_numpy_array(df["close"])

        # Calculate Hilbert Transform components
        inphase, quadrature, _ = _hilbert_transform_numba(data)

        # Detect trend mode
        trendmode = _detect_trend_mode_numba(inphase, quadrature, threshold)

        return pd.Series(trendmode, index=df.index, name="HT_TRENDMODE")


# Convenience functions
def calculate_ht_sine(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate HT_SINE - convenience function."""
    return HT_SINE().calculate(df)


def calculate_ht_trendmode(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Calculate HT_TRENDMODE - convenience function."""
    return HT_TRENDMODE().calculate(df, threshold)
