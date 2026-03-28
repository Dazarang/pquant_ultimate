"""Feature engineering for ML training dataset.

Extracted from indicators/notebooks/training-dataset-structure.ipynb.
All features are backward-looking (no lookahead bias).
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numba
import numpy as np
import pandas as pd

from indicators import (
    ADVANCED_FEATURE_COLUMNS,
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

# ---------------------------------------------------------------------------
# Feature catalog -- kept here next to the code that creates them.
# Update this when adding/removing features in the functions below.
# ---------------------------------------------------------------------------

# Features created by _calculate_base_indicators
_BASE_FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "sma_20", "sma_50", "sma_200", "ema_20", "vwap",
    "price_to_sma20", "price_to_sma50", "price_to_sma200", "price_to_vwap", "sma20_to_sma50",
    "rsi_14", "rsi_30",
    "macd", "macd_signal", "macd_hist", "macd_cross",
    "adx", "roc", "mom_10", "mom_20", "stoch_k", "stoch_d",
    "bb_upper", "bb_middle", "bb_lower", "bb_position", "bb_width",
    "atr_14", "adr", "sar", "apz_upper", "apz_lower",
    "volatility_20d", "vol_z",
    "obv", "obv_ema", "obv_ratio", "adosc",
    "volume_ma20", "volume_ratio", "volume_z",
    "hammer", "rsi_bullish_div", "rsi_bearish_div", "rsi_div_strength",
    "ht_sine", "ht_leadsine", "ht_trendmode",
    "high_252", "drawdown",
]

# Features from _add_lagged_features
_LAG_BASE_FEATURES = [
    "close", "ret_1d", "rsi_14", "macd_hist", "adx", "stoch_k",
    "volatility_20d", "bb_position", "atr_14", "volume_ratio",
    "obv_ratio", "drawdown", "price_to_sma20", "price_to_sma50",
]
_LAG_PERIODS = [1, 2, 3, 5, 10]

# Features from _add_rolling_features
_ROLLING_BASE_FEATURES = ["ret_1d", "rsi_14", "volatility_20d", "volume_z"]
_ROLLING_STATS = ["mean", "std", "min", "max"]
_ROLLING_WINDOWS = [5, 10, 20, 60]

_ROC_FEATURES = [
    "rsi_change_5d", "rsi_change_10d", "volume_change_10d",
    "atr_change_20d", "macd_change_5d",
]
_PERCENTILE_FEATURES = ["close_percentile_252", "rsi_percentile_60"]
_INTERACTION_FEATURES = [
    "rsi_volume_interaction", "drawdown_panic_interaction", "rsi_volatility_interaction",
]

# Assembled catalog
FEATURES: dict[str, list[str]] = {
    "base": _BASE_FEATURES,
    "advanced": list(ADVANCED_FEATURE_COLUMNS),
    "lag": [f"{f}_lag{n}" for f in _LAG_BASE_FEATURES for n in _LAG_PERIODS],
    "rolling": [
        f"{f}_rolling_{s}_{w}"
        for f in _ROLLING_BASE_FEATURES for s in _ROLLING_STATS for w in _ROLLING_WINDOWS
    ],
    "roc": _ROC_FEATURES,
    "percentile": _PERCENTILE_FEATURES,
    "interaction": _INTERACTION_FEATURES,
}


@numba.njit(cache=True)
def _rolling_percentile_rank(values, window, min_periods):
    n = len(values)
    result = np.empty(n)
    result[:] = np.nan
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        count = 0
        total = 0
        val = values[i]
        for j in range(start, i + 1):
            if values[j] == values[j]:
                total += 1
                if values[j] <= val:
                    count += 1
        if total >= min_periods:
            result[i] = count / total
    return result


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

    # Percentile rank features (numba-accelerated)
    new_cols["close_percentile_252"] = _rolling_percentile_rank(stock_df["close"].values, 252, 50)
    new_cols["rsi_percentile_60"] = _rolling_percentile_rank(stock_df["rsi_14"].values, 60, 20)

    # Interaction features
    if "rsi_14" in stock_df.columns and "volume_z" in stock_df.columns:
        new_cols["rsi_volume_interaction"] = (stock_df["rsi_14"] * stock_df["volume_z"]).values

    if "drawdown" in stock_df.columns and "panic_severity" in stock_df.columns:
        new_cols["drawdown_panic_interaction"] = (stock_df["drawdown"] * stock_df["panic_severity"]).values
    elif "drawdown" in stock_df.columns:
        new_cols["drawdown_panic_interaction"] = np.zeros(len(stock_df))

    if "rsi_14" in stock_df.columns and "vol_z" in stock_df.columns:
        new_cols["rsi_volatility_interaction"] = (stock_df["rsi_14"] * stock_df["vol_z"]).values

    if new_cols:
        stock_df = pd.concat([stock_df, pd.DataFrame(new_cols, index=stock_df.index)], axis=1)

    return stock_df


def _add_pivot_labels(stock_df: pd.DataFrame, lb: int = 8, rb: int = 13) -> pd.DataFrame:
    """Add pivot labels for a single stock. Uses [-1, +1] window with 1% tolerance."""
    pivot_high, pivot_low = find_pivots(
        stock_df, lb=lb, rb=rb, return_boolean=True,
        window_variations=[-1, 1], price_tolerance=0.01,
    )
    new_cols = {
        "PivotHigh": pivot_high.astype(int).values,
        "PivotLow": pivot_low.astype(int).values,
    }
    return pd.concat([stock_df, pd.DataFrame(new_cols, index=stock_df.index)], axis=1)


def _process_step3(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 per-stock: lagged + rolling + pivot labels."""
    stock_df = _add_lagged_features(stock_df)
    stock_df = _add_rolling_features(stock_df)
    stock_df = _add_pivot_labels(stock_df)
    return stock_df


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

    max_workers = max(1, (os.cpu_count() or 1) - 1)

    # Step 1: Base indicators (per stock, parallel)
    if verbose:
        print("  [1/3] Base indicators...")
    groups = {sid: sdf.copy() for sid, sdf in df.groupby("stock_id")}
    result_dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_calculate_base_indicators, sdf): sid for sid, sdf in groups.items()}
        with tqdm(total=len(futures), desc="    Base", disable=not verbose, ncols=80) as pbar:
            for future in as_completed(futures):
                result_dfs.append(future.result())
                pbar.update(1)

    result = pd.concat(result_dfs, ignore_index=True)
    result = result.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # Step 2: Advanced ML features (operates on full dataset with groupby)
    if verbose:
        print("  [2/3] Advanced features...")
    result = create_all_advanced_features(result, verbose=verbose)

    # Step 3: Lagged, rolling, pivot labels (per stock, parallel)
    if verbose:
        print("  [3/3] Lagged, rolling, pivot labels...")
    groups = {sid: sdf.copy() for sid, sdf in result.groupby("stock_id")}
    final_dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_step3, sdf): sid for sid, sdf in groups.items()}
        with tqdm(total=len(futures), desc="    Final", disable=not verbose, ncols=80) as pbar:
            for future in as_completed(futures):
                final_dfs.append(future.result())
                pbar.update(1)

    result = pd.concat(final_dfs, ignore_index=True)
    result = result.sort_values(["date", "stock_id"]).reset_index(drop=True)

    if verbose:
        feature_cols = [
            col for col in result.columns if col not in ["date", "stock_id", "open", "high", "low", "close", "volume"]
        ]
        print(f"  Total features: {len(feature_cols)}")
        print(f"  Total rows: {len(result):,}")

    return result
