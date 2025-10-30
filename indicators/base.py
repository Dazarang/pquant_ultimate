"""
Base indicator class with validation and caching.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    Provides validation, caching, and standardized interface.
    """

    def __init__(self, name: str):
        self.name = name
        self._cache = {}

    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series | tuple:
        """
        Calculate the indicator.

        Args:
            df: DataFrame with OHLCV data
            **kwargs: Indicator-specific parameters

        Returns:
            pd.Series or tuple of pd.Series
        """
        pass

    def validate_dataframe(self, df: pd.DataFrame, required_columns: list[str]) -> None:
        """
        Validate DataFrame has required columns.

        Args:
            df: Input DataFrame
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: Missing required columns: {missing}. Available: {list(df.columns)}")

        if len(df) == 0:
            raise ValueError(f"{self.name}: DataFrame is empty")

    def validate_period(self, period: int, min_period: int = 1) -> None:
        """
        Validate period parameter.

        Args:
            period: Period value to validate
            min_period: Minimum allowed period

        Raises:
            ValueError: If period is invalid
        """
        if not isinstance(period, int):
            raise TypeError(f"{self.name}: Period must be int, got {type(period)}")

        if period < min_period:
            raise ValueError(f"{self.name}: Period must be >= {min_period}, got {period}")

    def validate_data_length(self, df: pd.DataFrame, min_length: int) -> None:
        """
        Validate DataFrame has sufficient data.

        Args:
            df: Input DataFrame
            min_length: Minimum required length

        Raises:
            ValueError: If insufficient data
        """
        if len(df) < min_length:
            raise ValueError(f"{self.name}: Insufficient data. Need {min_length} rows, got {len(df)}")

    def handle_nan(self, series: pd.Series, method: str = "ffill") -> pd.Series:
        """
        Handle NaN values in series.

        Args:
            series: Input series
            method: 'ffill', 'bfill', 'drop', or 'zero'

        Returns:
            Cleaned series
        """
        if method == "ffill":
            return series.fillna(method="ffill")
        elif method == "bfill":
            return series.fillna(method="bfill")
        elif method == "drop":
            return series.dropna()
        elif method == "zero":
            return series.fillna(0)
        else:
            raise ValueError(f"Unknown NaN handling method: {method}")

    def get_cache_key(self, df: pd.DataFrame, **kwargs) -> str:
        """
        Generate cache key for indicator calculation.

        Args:
            df: Input DataFrame
            **kwargs: Indicator parameters

        Returns:
            Cache key string
        """
        df_hash = hash((len(df), df.index[0], df.index[-1]))
        params_hash = hash(frozenset(kwargs.items()))
        return f"{self.name}_{df_hash}_{params_hash}"

    def calculate_with_cache(self, df: pd.DataFrame, **kwargs) -> pd.Series | tuple:
        """
        Calculate with caching for performance.

        Args:
            df: Input DataFrame
            **kwargs: Indicator parameters

        Returns:
            Cached or freshly calculated result
        """
        cache_key = self.get_cache_key(df, **kwargs)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self.calculate(df, **kwargs)
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Clear indicator calculation cache."""
        self._cache.clear()


def ensure_numpy_array(data: pd.Series | np.ndarray) -> np.ndarray:
    """
    Convert input to numpy array.

    Args:
        data: pandas Series or numpy array

    Returns:
        numpy array
    """
    if isinstance(data, pd.Series):
        return data.values
    return data
