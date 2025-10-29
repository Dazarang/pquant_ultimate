"""
Complete Training Dataset Builder for Stock Bottom Detection
Uses yfinance with smart filtering and stratification
"""

import json
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import pickle
from datetime import datetime, timedelta
from collections import Counter

# ============================================================================
# STEP 1: LOAD AND FILTER TICKERS
# ============================================================================

def load_and_filter_tickers(json_path):
    """
    Load tickers and filter out noise (warrants, units, preferred, ETFs)
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND FILTERING TICKERS")
    print("=" * 70)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter US tickers
    us_tickers = data['US']
    print(f"\nUS tickers (raw): {len(us_tickers)}")
    
    # Remove derivatives and junk
    us_filtered = [
        t for t in us_tickers
        if not (
            t.endswith('W') or  # Warrants
            t.endswith('U') or  # Units
            t.endswith('R') or  # Rights
            '.W' in t or        # Warrants (alt format)
            '.U' in t or        # Units (alt format)
            '+' in t or         # Special shares
            '=' in t or         # When-issued
            '^' in t or         # Preferred (some formats)
            len(t) > 6 or       # Likely derivatives
            t.endswith('WW')    # Double warrants
        )
    ]
    
    # Additional filters for preferred shares (complex patterns)
    us_filtered = [
        t for t in us_filtered
        if not any(t.endswith(suffix) for suffix in ['P', 'PR', 'PRA', 'PRB', 'PRC'])
    ]
    
    # Filter common ETF patterns
    etf_patterns = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VT', 'VO', 'AGG', 'BND',
        'XL', 'IWR', 'IWV', 'VEA', 'VWO', 'GLD', 'SLV', 'USO'
    ]
    us_filtered = [
        t for t in us_filtered
        if not any(t.startswith(pattern) for pattern in etf_patterns)
    ]
    
    print(f"US tickers (filtered): {len(us_filtered)}")
    print(f"  Removed: {len(us_tickers) - len(us_filtered)} tickers")
    
    # S&P 500 (already clean)
    sp500 = data['SP500']
    print(f"\nS&P 500: {len(sp500)} tickers (will prioritize these)")
    
    # Swedish stocks (optional, very clean)
    sweden = data.get('Sweden', [])
    print(f"Swedish stocks: {len(sweden)} tickers")
    
    return {
        'us_filtered': us_filtered,
        'sp500': sp500,
        'sweden': sweden
    }


# ============================================================================
# STEP 2: PRIORITIZED DOWNLOAD STRATEGY
# ============================================================================

def create_download_list(tickers_dict, target_us_sample=2000):
    """
    Smart prioritization: S&P 500 first, then sample from US list
    """
    print("\n" + "=" * 70)
    print("STEP 2: CREATING PRIORITIZED DOWNLOAD LIST")
    print("=" * 70)
    
    download_list = []
    
    # Priority 1: All S&P 500
    sp500 = tickers_dict['sp500']
    download_list.extend(sp500)
    print(f"\n✓ Added {len(sp500)} S&P 500 tickers (Priority 1)")
    
    # Priority 2: Sample from filtered US list (exclude S&P 500 to avoid duplicates)
    us_filtered = [t for t in tickers_dict['us_filtered'] if t not in sp500]
    
    # Smart sampling: prefer shorter tickers (usually main exchanges, not OTC)
    us_scored = [(t, len(t)) for t in us_filtered]
    us_scored.sort(key=lambda x: x[1])  # Shorter tickers first
    
    # Take up to target_us_sample, prioritizing short tickers
    us_sample_size = min(target_us_sample, len(us_filtered))
    us_sample = [t for t, _ in us_scored[:us_sample_size]]
    download_list.extend(us_sample)
    print(f"✓ Added {len(us_sample)} sampled US tickers (Priority 2)")
    
    # Priority 3: Swedish stocks (optional - comment out if not needed)
    # sweden = tickers_dict['sweden']
    # download_list.extend(sweden)  # Take top 200 Swedish
    # print(f"✓ Added Swedish tickers (Priority 3)")
    
    # Remove duplicates
    download_list = list(set(download_list))
    
    print(f"\n📊 TOTAL DOWNLOAD LIST: {len(download_list)} tickers")
    
    return download_list


# ============================================================================
# STEP 3: DOWNLOAD WITH QUALITY CHECKS
# ============================================================================

def check_data_quality(ticker, df, current_date):
    """
    Determine if stock is valid, delisted, or should be rejected
    """
    if df is None or df.empty or len(df) < 100:
        return {'status': 'rejected', 'reason': 'insufficient_data'}
    
    # Basic stats
    last_date = df.index[-1]
    first_date = df.index[0]
    days_of_data = len(df)
    
    # Handle Volume column (might be uppercase or lowercase)
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    avg_volume = df[volume_col].mean()
    avg_price = df[close_col].mean()
    last_price = df[close_col].iloc[-1]
    
    days_since_last = (pd.to_datetime(current_date) - last_date).days
    
    stats = {
        'last_date': last_date,
        'first_date': first_date,
        'days_of_data': days_of_data,
        'avg_volume': avg_volume,
        'avg_price': avg_price,
        'last_price': last_price,
        'days_since_last': days_since_last,
    }
    
    # === REJECTION CRITERIA ===
    # Penny stock
    if avg_price < 1.0:
        return {'status': 'rejected', 'reason': 'penny_stock', 'stats': stats}
    
    # Illiquid
    if avg_volume < 50_000:
        return {'status': 'rejected', 'reason': 'illiquid', 'stats': stats}
    
    # Too little data
    if days_of_data < 252:  # Less than 1 year
        return {'status': 'rejected', 'reason': 'insufficient_history', 'stats': stats}
    
    # === DELISTED DETECTION (KEEP THESE!) ===
    # Stopped trading >90 days ago
    if days_since_last > 90:
        return {
            'status': 'delisted',
            'reason': f'stopped_trading_{days_since_last}_days_ago',
            'stats': stats
        }
    
    # Price collapsed (bankruptcy signal)
    if last_price < 1.0 and avg_price > 5.0:
        return {
            'status': 'delisted',
            'reason': 'price_collapsed_to_penny_stock',
            'stats': stats
        }
    
    # === VALID ===
    return {'status': 'valid', 'stats': stats}


def download_batch(tickers, start_date, end_date, current_date):
    """
    Download a batch of tickers with error handling
    """
    results = {
        'valid': [],
        'delisted': [],
        'rejected': []
    }
    
    try:
        # Batch download (faster)
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            threads=True,
            progress=False
        )
        
        # Process each ticker
        for ticker in tickers:
            try:
                # Extract ticker data
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker] if ticker in data else None
                
                if ticker_data is None or ticker_data.empty:
                    results['rejected'].append(ticker)
                    continue
                
                # Quality check
                quality = check_data_quality(ticker, ticker_data, current_date)
                
                if quality['status'] == 'valid':
                    results['valid'].append({
                        'ticker': ticker,
                        'data': ticker_data,
                        'stats': quality['stats']
                    })
                elif quality['status'] == 'delisted':
                    results['delisted'].append({
                        'ticker': ticker,
                        'data': ticker_data,
                        'stats': quality['stats'],
                        'delisting_reason': quality['reason']
                    })
                else:
                    results['rejected'].append(ticker)
                    
            except Exception as e:
                results['rejected'].append(ticker)
        
    except Exception as e:
        print(f"  Batch download error: {e}")
        results['rejected'].extend(tickers)
    
    return results


def download_all_tickers(ticker_list, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Download all tickers in batches with progress bar
    """
    print("\n" + "=" * 70)
    print("STEP 3: DOWNLOADING DATA (This will take 15-30 minutes)")
    print("=" * 70)

    batch_size = 100
    all_results = {
        'valid': [],
        'delisted': [],
        'rejected': []
    }

    # Use end_date as reference (not today) for historical data
    current_date = end_date
    
    # Process in batches
    for i in tqdm(range(0, len(ticker_list), batch_size), desc="Downloading batches"):
        batch = ticker_list[i:i+batch_size]
        
        batch_results = download_batch(batch, start_date, end_date, current_date)
        
        all_results['valid'].extend(batch_results['valid'])
        all_results['delisted'].extend(batch_results['delisted'])
        all_results['rejected'].extend(batch_results['rejected'])
        
        # Rate limiting
        time.sleep(0.5)
        
        # Progress update every 500 tickers
        if (i + batch_size) % 500 == 0:
            print(f"\n  Progress: Valid={len(all_results['valid'])}, "
                  f"Delisted={len(all_results['delisted'])}, "
                  f"Rejected={len(all_results['rejected'])}")
    
    print("\n✓ Download complete!")
    print(f"  Valid stocks: {len(all_results['valid'])}")
    print(f"  Delisted/failed: {len(all_results['delisted'])}")
    print(f"  Rejected: {len(all_results['rejected'])}")
    
    return all_results


# ============================================================================
# STEP 4: ENRICH WITH METADATA
# ============================================================================

def enrich_with_metadata(stock_list, max_stocks=None):
    """
    Get market cap, sector info (slow, so limit if needed)
    """
    print("\n" + "=" * 70)
    print("STEP 4: ENRICHING WITH METADATA (Getting sector/market cap)")
    print("=" * 70)
    
    if max_stocks:
        stock_list = stock_list[:max_stocks]
        print(f"Limiting to {max_stocks} stocks for faster processing")
    
    enriched = []
    
    for stock_info in tqdm(stock_list, desc="Getting metadata"):
        ticker = stock_info['ticker']
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            stock_info['metadata'] = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'country': info.get('country', 'Unknown'),
            }
            
            enriched.append(stock_info)
            
        except Exception as e:
            # Still include, just without metadata
            stock_info['metadata'] = {
                'sector': 'Unknown',
                'market_cap': 0,
            }
            enriched.append(stock_info)
        
        time.sleep(0.1)  # Rate limiting
    
    print(f"✓ Enriched {len(enriched)} stocks with metadata")
    
    return enriched


# ============================================================================
# STEP 5: STRATIFIED SELECTION
# ============================================================================

def stratified_selection(enriched_stocks, failed_stocks, target_total=1500):
    """
    Select balanced training set by market cap and sector
    """
    print("\n" + "=" * 70)
    print("STEP 5: STRATIFIED SELECTION")
    print("=" * 70)

    if not enriched_stocks:
        print("⚠️  No valid stocks to select from!")
        return []

    # Convert to dataframe for easier manipulation
    df = pd.DataFrame([
        {
            'ticker': s['ticker'],
            'market_cap': s['metadata'].get('market_cap', 0),
            'sector': s['metadata'].get('sector', 'Unknown'),
            'avg_volume': s['stats']['avg_volume'],
            'days_of_data': s['stats']['days_of_data'],
        }
        for s in enriched_stocks
    ])
    
    # Market cap buckets
    df['cap_bucket'] = pd.cut(
        df['market_cap'],
        bins=[0, 2e9, 10e9, 200e9, 1e15],
        labels=['small', 'mid', 'large', 'mega']
    )
    
    # Target allocation (save 10% for failures)
    target_by_bucket = {
        'mega': int(target_total * 0.15),   # 15%
        'large': int(target_total * 0.25),  # 25%
        'mid': int(target_total * 0.30),    # 30%
        'small': int(target_total * 0.20),  # 20%
    }
    
    selected = []
    
    for bucket, n_target in target_by_bucket.items():
        bucket_df = df[df['cap_bucket'] == bucket]
        
        if len(bucket_df) == 0:
            continue
        
        # Sample within bucket
        n_sample = min(n_target, len(bucket_df))
        
        # Try to balance by sector within bucket
        sampled = bucket_df.sample(n=n_sample, random_state=42)
        selected.extend(sampled['ticker'].tolist())
    
    # Add failures (up to 10% of total)
    n_failures = min(len(failed_stocks), int(target_total * 0.10))
    failure_tickers = [s['ticker'] for s in failed_stocks[:n_failures]]
    selected.extend(failure_tickers)
    
    print(f"\n✓ Selected {len(selected)} stocks:")
    print(f"  Active stocks: {len(selected) - len(failure_tickers)}")
    print(f"  Failed stocks: {len(failure_tickers)} ({len(failure_tickers)/len(selected):.1%})")
    
    # Distribution report
    selected_df = df[df['ticker'].isin(selected)]
    print(f"\n📊 Market Cap Distribution:")
    print(selected_df['cap_bucket'].value_counts().sort_index())
    
    print(f"\n📊 Sector Distribution (top 10):")
    print(selected_df['sector'].value_counts().head(10))
    
    return selected


# ============================================================================
# STEP 6: PREPARE FINAL DATASET
# ============================================================================

def prepare_final_dataset(selected_tickers, enriched_stocks, failed_stocks):
    """
    Create final dataset with all selected stocks
    """
    print("\n" + "=" * 70)
    print("STEP 6: PREPARING FINAL DATASET")
    print("=" * 70)
    
    final_data = {}
    
    # Combine enriched and failed stocks
    all_stocks = enriched_stocks + failed_stocks
    
    for stock_info in all_stocks:
        if stock_info['ticker'] in selected_tickers:
            final_data[stock_info['ticker']] = {
                'data': stock_info['data'],
                'metadata': stock_info.get('metadata', {}),
                'stats': stock_info['stats'],
                'is_failed': stock_info in failed_stocks
            }
    
    print(f"✓ Final dataset: {len(final_data)} stocks")
    
    return final_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Complete pipeline execution
    """
    print("\n" + "=" * 70)
    print("STOCK BOTTOM DETECTION - TRAINING DATASET BUILDER")
    print("=" * 70)
    
    # Configuration
    TICKER_JSON_PATH = 'data/tickers_data/tickers_cleaned_20251023.json'
    START_DATE = '2015-01-01'
    END_DATE = '2024-12-31'
    TARGET_US_SAMPLE = 100  # How many US stocks to try downloading (TEST: reduced from 2000)
    TARGET_FINAL = 50      # Final training set size (TEST: reduced from 1500)
    
    # Step 1: Load and filter
    tickers_dict = load_and_filter_tickers(TICKER_JSON_PATH)
    
    # Step 2: Create download list
    download_list = create_download_list(tickers_dict, TARGET_US_SAMPLE)
    
    # Step 3: Download data
    download_results = download_all_tickers(download_list, START_DATE, END_DATE)
    
    # Step 4: Enrich with metadata
    enriched = enrich_with_metadata(download_results['valid'], max_stocks=2000)
    
    # Step 5: Stratified selection
    selected_tickers = stratified_selection(
        enriched,
        download_results['delisted'],
        TARGET_FINAL
    )
    
    # Step 6: Prepare final dataset
    final_data = prepare_final_dataset(
        selected_tickers,
        enriched,
        download_results['delisted']
    )
    
    # Save to disk
    print("\n" + "=" * 70)
    print("SAVING TO DISK")
    print("=" * 70)
    
    # Save as pickle (preserves dataframes)
    with open('training_data.pkl', 'wb') as f:
        pickle.dump(final_data, f)
    print("✓ Saved: training_data.pkl")
    
    # Also create combined dataframe
    all_dfs = []
    for ticker, info in final_data.items():
        df = info['data'].copy()
        df['ticker'] = ticker
        df['stock_id'] = ticker
        df['is_failed'] = info['is_failed']
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_parquet('training_stocks_data.parquet')
        print("✓ Saved: training_stocks_data.parquet")

        print(f"\n✅ COMPLETE!")
        print(f"   Total stocks: {len(final_data)}")
        print(f"   Total rows: {len(combined_df):,}")
        print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"   Failed stocks: {sum(1 for v in final_data.values() if v['is_failed'])}")
    else:
        print(f"\n⚠️  NO DATA - All downloads failed or were rejected")
        print(f"   Try adjusting the quality filters or ticker selection")


if __name__ == "__main__":
    main()