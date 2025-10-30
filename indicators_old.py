import numpy as np
import pandas as pd
import talib
from scipy.optimize import minimize


# Technical Indicator Functions
def calculate_cma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Calculates the Cumulative Moving Average (CMA).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'Close' prices.
    period (int): The period for the CMA calculation.

    Returns:
    pd.Series: Series containing the CMA.
    """
    return talib.SMA(df["close"], timeperiod=period)


def calculate_ema(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'Close' prices.
    period (int): The period for the EMA calculation.

    Returns:
    pd.Series: Series containing the EMA.
    """
    return talib.EMA(df["close"], timeperiod=period)


def calculate_sma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'Close' prices.
    period (int): The period for the SMA calculation.

    Returns:
    pd.Series: Series containing the SMA.
    """
    return talib.SMA(df["close"], timeperiod=period)


def calculate_wma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Calculates the Weighted Moving Average (WMA).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'Close' prices.
    period (int): The period for the WMA calculation.

    Returns:
    pd.Series: Series containing the WMA.
    """
    return talib.WMA(df["close"], timeperiod=period)


def calculate_bbands(
    df: pd.DataFrame,
    period: int = 20,
    nbdevup: int = 2,
    nbdevdn: int = 2,
    matype: int = 2,
) -> tuple:
    """
    Calculates the Bollinger Bands.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'Close' prices.
    timeperiod (int): The time period for the Bollinger Bands calculation.
    nbdevup (int): The number of standard deviations to add to the middle band for the upper band calculation.
    nbdevdn (int): The number of standard deviations to subtract from the middle band for the lower band calculation.
    matype (int): The type of moving average to use for the middle band calculation.

    Returns:
    tuple: A tuple containing three Series (upper, middle, lower bands).
    """
    upper, middle, lower = talib.BBANDS(
        df["close"],
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
        matype=matype,
    )
    return upper, middle, lower


def calculate_adx(df: pd.DataFrame, period=14) -> pd.Series:
    """
    Calculates the Average Directional Index (ADX).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'Close' prices.

    Returns:
    pd.Series: Series containing the ADX.
    """
    return talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)


def calculate_macd(df: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> tuple:
    """
    Calculates the Moving Average Convergence Divergence (MACD).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'Close' prices.
    fastperiod (int): The fast period for MACD calculation.
    slowperiod (int): The slow period for MACD calculation.
    signalperiod (int): The signal period for MACD calculation.

    Returns:
    tuple: A tuple containing three Series (MACD, MACD signal, MACD histogram).
    """
    # macd, macd_signal, macd_hist = talib.MACD(
    #     df["close"],
    #     fastperiod=fastperiod,
    #     slowperiod=slowperiod,
    #     signalperiod=signalperiod,
    # )
    ema_fast = talib.EMA(df["close"], timeperiod=fastperiod)
    ema_slow = talib.EMA(df["close"], timeperiod=slowperiod)
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=slowperiod)

    macd_v = ((ema_fast - ema_slow) / atr) * 100
    signal_line = talib.EMA(macd_v, timeperiod=signalperiod)
    histogram = macd_v - signal_line

    return macd_v, signal_line, histogram


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'Close' prices.
    period (int): The period for the RSI calculation.

    Returns:
    pd.Series: Series containing the RSI.
    """
    return talib.RSI(df["close"], timeperiod=period)


def calculate_adr(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """
    Calculates the Average Daily Range (ADR).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'high' and 'low' prices.
    length (int): The period for the ADR calculation.

    Returns:
    pd.Series: Series containing the ADR.
    """
    df["high/low"] = df["high"] / df["low"]
    adr = round(100 * (talib.SMA(df["high/low"], timeperiod=length) - 1), 2)
    df.drop(columns=["high/low"], inplace=True)  # Clean up the temporary column
    return adr


def find_pivots(df: pd.DataFrame, lb: int = 8, rb: int = 8, return_boolean: bool = True) -> tuple:
    """
    Identifies pivot highs and lows in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'high' and 'low' columns.
    lb (int): Number of bars to the left of the pivot.
    rb (int): Number of bars to the right of the pivot.
    return_boolean (bool): If True, return booleans indicating pivots; otherwise, return original values and NaNs.

    Returns:
    tuple: A tuple containing two Series (Pivothigh, PivotLow).
    """

    def is_pivot_high(window):
        center = len(window) // 2
        if window[center] == max(window):
            return window[center]
        return np.nan

    def is_pivot_low(window):
        center = len(window) // 2
        if window[center] == min(window):
            return window[center]
        return np.nan

    # Create rolling windows to find pivots
    pivot_high = df["high"].rolling(window=lb + rb + 1, center=True).apply(is_pivot_high, raw=True)
    pivot_low = df["low"].rolling(window=lb + rb + 1, center=True).apply(is_pivot_low, raw=True)

    if return_boolean:
        pivot_high = pivot_high.notna()
        pivot_low = pivot_low.notna()

    return pivot_high, pivot_low


def calculate_sar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """
    Calculates the Parabolic SAR (Stop and Reverse).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'high' and 'low' prices.
    acceleration (float): The acceleration factor.
    maximum (float): The maximum value for the acceleration factor.

    Returns:
    pd.Series: Series containing the SAR values.
    """
    return talib.SAR(df["high"], df["low"], acceleration=acceleration, maximum=maximum)


# Define the calculate_obv function with OBV EMA
def calculate_obv(df: pd.DataFrame, ema_period: int = 55) -> tuple:
    """
    Calculates the On-Balance Volume (OBV) and its EMA.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'close' and 'volume' columns.
    ema_period (int): The period for the OBV EMA calculation.

    Returns:
    tuple: A tuple containing two Series (OBV, OBV EMA).
    """
    obv = talib.OBV(df["close"], df["volume"])
    obv_ema = talib.EMA(obv, timeperiod=ema_period)
    return obv, obv_ema


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
    period (int): The period for the ATR calculation.

    Returns:
    pd.Series: Series containing the ATR.
    """
    return talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)


def calculate_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Hammer candlestick pattern.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.Series: Series containing the Hammer pattern.
    """
    return talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"])


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Volume Weighted Average Price (VWAP).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'high', 'low', 'close', and 'volume' columns.

    Returns:
    pd.Series: Series containing the VWAP.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return vwap


def calculate_adosc(df: pd.DataFrame, fastperiod: int = 3, slowperiod: int = 10) -> pd.Series:
    """
    Calculates the Chaikin A/D Oscillator (ADOSC).

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'high', 'low', 'close', and 'volume' columns.
    fastperiod (int): The fast period for ADOSC calculation.
    slowperiod (int): The slow period for ADOSC calculation.

    Returns:
    pd.Series: Series containing the ADOSC values.
    """
    return talib.ADOSC(
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
    )


def calculate_ht_sine(df: pd.DataFrame) -> tuple:
    sine, leadsine = talib.HT_SINE(df["close"])
    return sine, leadsine


def calculate_ht_trendmode(df: pd.DataFrame) -> pd.Series:
    return talib.HT_TRENDMODE(df["close"])


def calculate_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    return talib.ROC(df["close"], timeperiod=period)


def calculate_mom(df: pd.DataFrame, period: int = 10) -> pd.Series:
    return talib.MOM(df["close"], timeperiod=period)


# Adaptive Price Zone (APZ) Indicator
def double_smooth_ema(price: pd.Series, length: int) -> pd.Series:
    """
    Calculates the Double Smooth Exponential Moving Average.

    Parameters:
    :param (pd.Series): The input price series.
    length (int): The period for the EMA calculation.

    Returns:
    pd.Series: Series containing the Double Smooth EMA.
    """
    period = np.sqrt(length)
    ema1 = talib.EMA(price, timeperiod=period)
    ema2 = talib.EMA(ema1, timeperiod=period)
    return ema2


def calculate_apz(df: pd.DataFrame, period: int = 21, band_pct: float = 2.0) -> tuple:
    """
    Calculates the Adaptive Price Zone (APZ) indicator.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'close', 'high', and 'low' columns.
    period (int): The period for the APZ calculation.
    band_pct (float): The percentage for the band width calculation.

    Returns:
    tuple: A tuple containing two Series (upper band, lower band).
    """
    ds_ema = double_smooth_ema(df["close"], period)
    range_ds_ema = double_smooth_ema(df["high"] - df["low"], period)
    upper_band = ds_ema + band_pct * range_ds_ema
    lower_band = ds_ema - band_pct * range_ds_ema
    return upper_band, lower_band


def calculate_indicators(df):
    rsi = calculate_rsi(df)
    macd, macd_signal, _ = calculate_macd(df)
    upper_band, middle_band, lower_band = calculate_bbands(df)
    _, obv_ema = calculate_obv(df, ema_period=8)
    short_term_ema = calculate_ema(df, period=8)
    roc = calculate_roc(df)
    mom = calculate_mom(df)

    # Normalize indicators
    rsi_normalized = (rsi - 50) / 50  # Normalize RSI to range [-1, 1]
    macd_diff = macd - macd_signal
    macd_normalized = macd_diff / np.max(np.abs(macd_diff))  # Normalize MACD to range [-1, 1]
    bb_value = (df["close"] - middle_band) / (upper_band - lower_band)  # Normalize Bollinger Bands
    obv_normalized = obv_ema / np.max(np.abs(obv_ema))  # Normalize OBV EMA
    ema_normalized = (df["close"] - short_term_ema) / short_term_ema  # Normalize short-term EMA
    roc_normalized = roc / np.max(np.abs(roc))  # Normalize ROC to range [-1, 1]
    mom_normalized = mom / np.max(np.abs(mom))  # Normalize Momentum to range [-1, 1]

    return (
        rsi_normalized,
        macd_normalized,
        bb_value,
        obv_normalized,
        ema_normalized,
        roc_normalized,
        mom_normalized,
    )


def composite_index(weights, rsi, macd, bb, obv, ema, roc, mom):
    return (
        weights[0] * rsi
        + weights[1] * macd
        + weights[2] * bb
        + weights[3] * obv
        + weights[4] * ema
        + weights[5] * roc
        + weights[6] * mom
    )


def objective_function(weights, rsi, macd, bb, obv, ema, roc, mom, returns):
    index = composite_index(weights, rsi, macd, bb, obv, ema, roc, mom)
    # Calculate the correlation between the index and future returns (1-step ahead)
    future_returns = returns.shift(-1)
    correlation = index.corr(future_returns)
    # Minimize negative correlation to maximize positive correlation
    return -correlation


def optimize_weights(df):
    rsi, macd, bb, obv, ema, roc, mom = calculate_indicators(df)
    returns = df["close"].pct_change()

    initial_weights = np.array([1 / 7] * 7)
    bounds = [(0, 1)] * 7

    result = minimize(
        objective_function,
        initial_weights,
        args=(rsi, macd, bb, obv, ema, roc, mom, returns),
        bounds=bounds,
        method="SLSQP",
    )

    return result.x


def enhanced_real_time_strength_index(df: pd.DataFrame) -> pd.Series:
    rsi, macd, bb, obv, ema, roc, mom = calculate_indicators(df)

    # Optimize weights
    optimized_weights = optimize_weights(df)

    # Compute composite index with optimized weights
    composite_index = (
        optimized_weights[0] * rsi
        + optimized_weights[1] * macd
        + optimized_weights[2] * bb
        + optimized_weights[3] * obv
        + optimized_weights[4] * ema
        + optimized_weights[5] * roc
        + optimized_weights[6] * mom
    )

    # Scale composite index to range [0, 100]
    strength_index = (composite_index + 1) * 50

    # Apply EMA to smooth the composite index
    smoothed_strength_index = talib.EMA(strength_index, timeperiod=3)

    return smoothed_strength_index


def detect_rsi_divergence(df):
    # Initialize the new column with zeros
    df["RSI_Divergence"] = 0

    # Iterate through the DataFrame to find RSI divergence
    i = 0
    while i < len(df):
        if df.loc[i, "PivotLow"] == 1:
            first_pivot_low_index = i
            first_pivot_low_price = df.loc[first_pivot_low_index, "close"]
            first_pivot_low_rsi = df.loc[first_pivot_low_index, "RSI"]

            # Find the next PivotLow
            for j in range(first_pivot_low_index + 1, len(df)):
                if df.loc[j, "PivotLow"] == 1:
                    second_pivot_low_index = j
                    second_pivot_low_price = df.loc[second_pivot_low_index, "close"]
                    second_pivot_low_rsi = df.loc[second_pivot_low_index, "RSI"]

                    # Check for RSI divergence
                    if second_pivot_low_price < first_pivot_low_price and second_pivot_low_rsi > first_pivot_low_rsi:
                        df.loc[second_pivot_low_index, "RSI_Divergence"] = 1

                    # Update the first pivot low to the current one and continue the search
                    i = second_pivot_low_index
                    break
        i += 1

    return df


# # Composite Strength Index Function
# def real_time_strength_index(df: pd.DataFrame) -> pd.Series:
#     # Calculate technical indicators
#     rsi = calculate_rsi(df)
#     macd, macd_signal, _ = calculate_macd(df)
#     upper_band, middle_band, lower_band = calculate_bbands(df)
#     _, obv_ema = calculate_obv(df, ema_period=8)
#     short_term_ema = calculate_ema(df, period=8)

#     # Normalize indicators
#     rsi_normalized = (rsi - 50) / 50  # Normalize RSI to range [-1, 1]
#     macd_diff = macd - macd_signal
#     macd_normalized = macd_diff / np.max(
#         np.abs(macd_diff)
#     )  # Normalize MACD to range [-1, 1]
#     bb_value = (df["close"] - middle_band) / (
#         upper_band - lower_band
#     )  # Normalize Bollinger Bands
#     obv_normalized = obv_ema / np.max(np.abs(obv_ema))  # Normalize OBV EMA
#     ema_normalized = (
#         df["close"] - short_term_ema
#     ) / short_term_ema  # Normalize short-term EMA

#     # Normalize indicators
#     rsi_normalized = (rsi - 50) / 50  # Normalize RSI to range [-1, 1]
#     macd_diff = macd - macd_signal
#     macd_normalized = macd_diff / np.max(
#         np.abs(macd_diff)
#     )  # Normalize MACD to range [-1, 1]
#     bb_value = (df["close"] - middle_band) / (
#         upper_band - lower_band
#     )  # Normalize Bollinger Bands
#     obv_normalized = obv_ema / np.max(np.abs(obv_ema))  # Normalize OBV EMA
#     ema_normalized = (
#         df["close"] - short_term_ema
#     ) / short_term_ema  # Normalize short-term EMA

#     return rsi_normalized, macd_normalized, bb_value, obv_normalized, ema_normalized


# def composite_index(weights, rsi, macd, bb, obv, ema):
#     return (
#         weights[0] * rsi
#         + weights[1] * macd
#         + weights[2] * bb
#         + weights[3] * obv
#         + weights[4] * ema
#     )


# def objective_function(weights, rsi, macd, bb, obv, ema, returns):
#     index = composite_index(weights, rsi, macd, bb, obv, ema)
#     # Calculate the correlation between the index and future returns (1-step ahead)
#     future_returns = returns.shift(-1)
#     correlation = index.corr(future_returns)
#     # Minimize negative correlation to maximize positive correlation
#     return -correlation


# def optimize_weights(df):
#     rsi, macd, bb, obv, ema = real_time_strength_index(df)
#     returns = df["close"].pct_change()

#     initial_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
#     bounds = [(0, 1)] * 5

#     result = minimize(
#         objective_function,
#         initial_weights,
#         args=(rsi, macd, bb, obv, ema, returns),
#         bounds=bounds,
#         method="SLSQP",
#     )

#     return result.x


# def enhanced_real_time_strength_index(df: pd.DataFrame) -> pd.Series:
#     rsi, macd, bb, obv, ema = real_time_strength_index(df)

#     # Optimize weights
#     optimized_weights = optimize_weights(df)

#     # Compute composite index with optimized weights
#     composite_index = (
#         optimized_weights[0] * rsi
#         + optimized_weights[1] * macd
#         + optimized_weights[2] * bb
#         + optimized_weights[3] * obv
#         + optimized_weights[4] * ema
#     )

#     # Scale composite index to range [0, 100]
#     strength_index = (composite_index + 1) * 50

#     return strength_index
