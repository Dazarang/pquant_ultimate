"""Feature engineering for ML training dataset.

Extracted from indicators/notebooks/training-dataset-structure.ipynb.
All features are backward-looking (no lookahead bias).
"""

import numpy as np
import pandas as pd

from indicators import (
    calculate_adosc,
    calculate_adr,
    calculate_adx,
    calculate_apz,
    calculate_atr,
    calculate_bbands,
    calculate_ema,
    calculate_hammer,
    calculate_ht_sine,
    calculate_ht_trendmode,
    calculate_macd,
    calculate_mom,
    calculate_obv,
    calculate_roc,
    calculate_rsi,
    calculate_sar,
    calculate_sma,
    calculate_stochastic,
    calculate_vwap,
    create_all_advanced_features,
    detect_rsi_divergence,
    find_pivots,
)


def _calculate_base_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all base indicators for a single stock."""
    # Price-based features
    stock_df["ret_1d"] = stock_df["close"].pct_change(1)
    stock_df["ret_5d"] = stock_df["close"].pct_change(5)
    stock_df["ret_10d"] = stock_df["close"].pct_change(10)
    stock_df["ret_20d"] = stock_df["close"].pct_change(20)

    # Trend indicators
    stock_df["sma_20"] = calculate_sma(stock_df, period=20)
    stock_df["sma_50"] = calculate_sma(stock_df, period=50)
    stock_df["sma_200"] = calculate_sma(stock_df, period=200)
    stock_df["ema_20"] = calculate_ema(stock_df, period=20)
    stock_df["vwap"] = calculate_vwap(stock_df)

    stock_df["price_to_sma20"] = (stock_df["close"] - stock_df["sma_20"]) / stock_df["sma_20"]
    stock_df["price_to_sma50"] = (stock_df["close"] - stock_df["sma_50"]) / stock_df["sma_50"]
    stock_df["price_to_sma200"] = (stock_df["close"] - stock_df["sma_200"]) / stock_df["sma_200"]
    stock_df["sma20_to_sma50"] = (stock_df["sma_20"] - stock_df["sma_50"]) / stock_df["sma_50"]
    stock_df["price_to_vwap"] = (stock_df["close"] - stock_df["vwap"]) / stock_df["vwap"]

    # Momentum indicators
    stock_df["rsi_14"] = calculate_rsi(stock_df, period=14)
    stock_df["rsi_30"] = calculate_rsi(stock_df, period=30)

    macd_result = calculate_macd(stock_df, fastperiod=12, slowperiod=26, signalperiod=9)
    stock_df["macd"] = macd_result[0]
    stock_df["macd_signal"] = macd_result[1]
    stock_df["macd_hist"] = macd_result[2]
    stock_df["macd_cross"] = (stock_df["macd_hist"] > 0).astype(int)

    stock_df["adx"] = calculate_adx(stock_df, period=14)
    stock_df["roc"] = calculate_roc(stock_df, period=10)
    stock_df["mom_10"] = calculate_mom(stock_df, period=10)
    stock_df["mom_20"] = calculate_mom(stock_df, period=20)

    stoch_result = calculate_stochastic(stock_df, fastk_period=14, slowk_period=3, slowd_period=3)
    stock_df["stoch_k"] = stoch_result[0]
    stock_df["stoch_d"] = stoch_result[1]

    # Volatility indicators
    bb_result = calculate_bbands(stock_df, period=20, nbdevup=2.0, nbdevdn=2.0)
    stock_df["bb_upper"] = bb_result[0]
    stock_df["bb_middle"] = bb_result[1]
    stock_df["bb_lower"] = bb_result[2]
    stock_df["bb_position"] = (stock_df["close"] - stock_df["bb_lower"]) / (stock_df["bb_upper"] - stock_df["bb_lower"])
    stock_df["bb_width"] = (stock_df["bb_upper"] - stock_df["bb_lower"]) / stock_df["bb_middle"]

    stock_df["atr_14"] = calculate_atr(stock_df, period=14)
    stock_df["adr"] = calculate_adr(stock_df, length=20)
    stock_df["sar"] = calculate_sar(stock_df, acceleration=0.02, maximum=0.2)

    apz_result = calculate_apz(stock_df, period=21, band_pct=2.0)
    stock_df["apz_upper"] = apz_result[0]
    stock_df["apz_lower"] = apz_result[1]

    stock_df["volatility_20d"] = stock_df["ret_1d"].rolling(20).std()
    stock_df["vol_z"] = (stock_df["volatility_20d"] - stock_df["volatility_20d"].rolling(60).mean()) / stock_df[
        "volatility_20d"
    ].rolling(60).std()

    # Volume indicators
    obv_result = calculate_obv(stock_df, ema_period=55)
    stock_df["obv"] = obv_result[0]
    stock_df["obv_ema"] = obv_result[1]

    stock_df["adosc"] = calculate_adosc(stock_df, fastperiod=3, slowperiod=10)

    stock_df["volume_ma20"] = stock_df["volume"].rolling(20).mean()
    stock_df["volume_ratio"] = stock_df["volume"] / stock_df["volume_ma20"]
    stock_df["volume_ratio"] = stock_df["volume_ratio"].replace([np.inf, -np.inf], np.nan)

    stock_df["volume_z"] = (stock_df["volume_ratio"] - stock_df["volume_ratio"].rolling(60).mean()) / stock_df[
        "volume_ratio"
    ].rolling(60).std()

    obv_ma20 = stock_df["obv"].rolling(20).mean()
    stock_df["obv_ratio"] = stock_df["obv"] / obv_ma20
    stock_df["obv_ratio"] = stock_df["obv_ratio"].replace([np.inf, -np.inf], np.nan)

    # Pattern indicators
    stock_df["hammer"] = calculate_hammer(stock_df)

    stock_df_temp = stock_df.copy()
    stock_df_temp["RSI"] = stock_df["rsi_14"]
    rsi_div = detect_rsi_divergence(
        stock_df_temp, rsi_col="RSI", price_col="close", lookback_window=5, max_lookback=60, min_distance=5
    )
    stock_df["rsi_bullish_div"] = rsi_div["Bullish_Divergence"]
    stock_df["rsi_bearish_div"] = rsi_div["Bearish_Divergence"]
    stock_df["rsi_div_strength"] = rsi_div["Divergence_Strength"]

    # Cycle indicators
    ht_sine_result = calculate_ht_sine(stock_df)
    stock_df["ht_sine"] = ht_sine_result[0]
    stock_df["ht_leadsine"] = ht_sine_result[1]
    stock_df["ht_trendmode"] = calculate_ht_trendmode(stock_df, threshold=0.5)

    # Contextual features
    stock_df["high_252"] = stock_df["high"].rolling(252, min_periods=20).max()
    stock_df["drawdown"] = (stock_df["close"] - stock_df["high_252"]) / stock_df["high_252"]

    return stock_df


def _add_lagged_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features for a single stock."""
    lag_features = [
        "close",
        "ret_1d",
        "rsi_14",
        "macd_hist",
        "adx",
        "stoch_k",
        "volatility_20d",
        "bb_position",
        "atr_14",
        "volume_ratio",
        "obv_ratio",
        "drawdown",
        "price_to_sma20",
        "price_to_sma50",
    ]
    lag_periods = [1, 2, 3, 5, 10]

    new_cols = {}
    for feature in lag_features:
        if feature in stock_df.columns:
            for lag in lag_periods:
                new_cols[f"{feature}_lag{lag}"] = stock_df[feature].shift(lag).values

    if new_cols:
        stock_df = pd.concat([stock_df, pd.DataFrame(new_cols, index=stock_df.index)], axis=1)

    return stock_df


def _add_rolling_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics for a single stock."""
    rolling_windows = [5, 10, 20, 60]
    rolling_features = ["ret_1d", "rsi_14", "volume_z", "volatility_20d"]

    new_cols = {}

    # Rolling window statistics
    for feature in rolling_features:
        if feature in stock_df.columns:
            for window in rolling_windows:
                new_cols[f"{feature}_rolling_mean_{window}"] = stock_df[feature].rolling(window).mean().values
                new_cols[f"{feature}_rolling_std_{window}"] = stock_df[feature].rolling(window).std().values
                new_cols[f"{feature}_rolling_min_{window}"] = stock_df[feature].rolling(window).min().values
                new_cols[f"{feature}_rolling_max_{window}"] = stock_df[feature].rolling(window).max().values

    # Rate of change features
    if "rsi_14" in stock_df.columns:
        new_cols["rsi_change_5d"] = stock_df["rsi_14"].diff(5).values
        new_cols["rsi_change_10d"] = stock_df["rsi_14"].diff(10).values

    if "volume_ratio" in stock_df.columns:
        vol_change = stock_df["volume_ratio"].pct_change(10).replace([np.inf, -np.inf], np.nan)
        new_cols["volume_change_10d"] = vol_change.values

    if "atr_14" in stock_df.columns:
        new_cols["atr_change_20d"] = stock_df["atr_14"].pct_change(20).values

    if "macd_hist" in stock_df.columns:
        new_cols["macd_change_5d"] = stock_df["macd_hist"].diff(5).values

    # Percentile rank features
    new_cols["close_percentile_252"] = (
        stock_df["close"]
        .rolling(252, min_periods=50)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan)
    ).values
    new_cols["rsi_percentile_60"] = (
        stock_df["rsi_14"]
        .rolling(60, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan)
    ).values

    # Interaction features
    if "rsi_14" in stock_df.columns and "volume_z" in stock_df.columns:
        new_cols["rsi_volume_interaction"] = (stock_df["rsi_14"] * stock_df["volume_z"]).values

    if "drawdown" in stock_df.columns and "panic_severity" in stock_df.columns:
        new_cols["drawdown_panic_interaction"] = (stock_df["drawdown"] * stock_df["panic_severity"]).values

    if "rsi_14" in stock_df.columns and "vol_z" in stock_df.columns:
        new_cols["rsi_volatility_interaction"] = (stock_df["rsi_14"] * stock_df["vol_z"]).values

    if new_cols:
        stock_df = pd.concat([stock_df, pd.DataFrame(new_cols, index=stock_df.index)], axis=1)

    return stock_df


def _add_pivot_labels(stock_df: pd.DataFrame, lb: int = 8, rb: int = 13) -> pd.DataFrame:
    """Add pivot labels for a single stock."""
    pivot_high, pivot_low = find_pivots(stock_df, lb=lb, rb=rb, return_boolean=True)
    new_cols = {
        "PivotHigh": pivot_high.astype(int).values,
        "PivotLow": pivot_low.astype(int).values,
    }
    return pd.concat([stock_df, pd.DataFrame(new_cols, index=stock_df.index)], axis=1)


def build_features(df: pd.DataFrame, verbose: bool = True, min_rows: int = 252) -> pd.DataFrame:
    """Build all ML features from OHLCV data.

    Args:
        df: DataFrame with columns: date, stock_id, open, high, low, close, volume
        verbose: Print progress
        min_rows: Minimum rows required per stock (default 252 for yearly indicators)

    Returns:
        DataFrame with all features added
    """
    from tqdm import tqdm

    if verbose:
        print("\nBuilding features...")

    # Filter stocks with insufficient data
    stock_counts = df.groupby("stock_id").size()
    valid_stocks = stock_counts[stock_counts >= min_rows].index
    dropped_stocks = len(stock_counts) - len(valid_stocks)

    if dropped_stocks > 0:
        if verbose:
            print(f"  Dropping {dropped_stocks} stocks with <{min_rows} rows")
        df = df[df["stock_id"].isin(valid_stocks)]

    stocks = df["stock_id"].unique()
    if verbose:
        print(f"  Processing {len(stocks)} stocks...")

    # Step 1: Base indicators (per stock)
    if verbose:
        print("  [1/3] Base indicators...")
    result_dfs = []
    iterator = tqdm(stocks, desc="    Base", disable=not verbose, ncols=80)
    for stock_id in iterator:
        stock_df = df[df["stock_id"] == stock_id].copy()
        stock_df = _calculate_base_indicators(stock_df)
        result_dfs.append(stock_df)

    result = pd.concat(result_dfs, ignore_index=True)
    result = result.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # Step 2: Advanced ML features (operates on full dataset with groupby)
    if verbose:
        print("  [2/3] Advanced features...")
    result = create_all_advanced_features(result, verbose=verbose)

    # Step 3: Lagged, rolling, pivot labels (per stock)
    if verbose:
        print("  [3/3] Lagged, rolling, pivot labels...")
    final_dfs = []
    iterator = tqdm(stocks, desc="    Final", disable=not verbose, ncols=80)
    for stock_id in iterator:
        stock_df = result[result["stock_id"] == stock_id].copy()
        stock_df = _add_lagged_features(stock_df)
        stock_df = _add_rolling_features(stock_df)
        stock_df = _add_pivot_labels(stock_df)
        final_dfs.append(stock_df)

    result = pd.concat(final_dfs, ignore_index=True)
    result = result.sort_values(["date", "stock_id"]).reset_index(drop=True)

    if verbose:
        feature_cols = [
            col for col in result.columns if col not in ["date", "stock_id", "open", "high", "low", "close", "volume"]
        ]
        print(f"  Total features: {len(feature_cols)}")
        print(f"  Total rows: {len(result):,}")

    return result
