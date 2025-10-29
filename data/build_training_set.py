"""
Complete Training Dataset Builder for Stock Bottom Detection
Uses yfinance with smart filtering and stratification
"""

import json
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time
import pickle
from pathlib import Path
import re
import subprocess
from datetime import datetime

# ============================================================================
# STEP 1: LOAD AND FILTER TICKERS
# ============================================================================

def load_tickers(json_path):
    """
    Load pre-filtered and validated tickers.
    No filtering - assumes already filtered by filter_tickers.py
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOADING VALIDATED TICKERS")
    print("=" * 70)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Load US tickers (already filtered and validated)
    us_tickers = data.get('US', [])
    if us_tickers:
        print(f"\nUS tickers: {len(us_tickers)} (pre-filtered & validated)")
    else:
        print("\nWarning: No US tickers found in JSON")

    # S&P 500 (already validated)
    sp500 = data.get('SP500', [])
    if sp500:
        print(f"S&P 500: {len(sp500)} tickers (will prioritize these)")
    else:
        print("Warning: No S&P 500 tickers found in JSON")

    # Swedish stocks (already validated)
    sweden = data.get('Sweden', [])
    if sweden:
        print(f"Swedish stocks: {len(sweden)} tickers")
        print("  Note: Swedish market caps smaller than US (large-cap ~$5-10B vs US $10B+)")

    total = len(us_tickers) + len(sp500) + len(sweden)
    print(f"\nTotal tickers loaded: {total}")

    return {
        'us_filtered': us_tickers,
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
    sweden = tickers_dict['sweden']
    download_list.extend(sweden)
    print(f"✓ Added {len(sweden)} Swedish tickers (Priority 3)")
    
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

    # Handle column access (yfinance returns MultiIndex or simple columns)
    # For batch downloads: columns are ('Close', 'AAPL'), ('Volume', 'AAPL')
    # For single ticker: columns are 'Close', 'Volume'
    try:
        volume_data = df['Volume']
        close_data = df['Close']
    except KeyError:
        # Try lowercase
        try:
            volume_data = df['volume']
            close_data = df['close']
        except KeyError:
            return {'status': 'rejected', 'reason': 'missing_price_columns'}

    # Handle both Series and DataFrame (MultiIndex case)
    if isinstance(volume_data, pd.DataFrame):
        avg_volume = volume_data.iloc[:, 0].mean()
        last_price = close_data.iloc[-1, 0]
        avg_price = close_data.iloc[:, 0].mean()
    else:
        avg_volume = volume_data.mean()
        last_price = close_data.iloc[-1]
        avg_price = close_data.mean()
    
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
            progress=False,
            auto_adjust=True  # Silence FutureWarning
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
    Download all tickers in batches with progress bar and rate limiting.
    """
    print("\n" + "=" * 70)
    print("STEP 3: DOWNLOADING DATA (This will take 15-30 minutes)")
    print("=" * 70)
    print(f"Note: Using {start_date} to {end_date} date range")

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

        # Increased rate limiting to avoid 429 errors
        # Yahoo Finance rate limits more aggressively now
        time.sleep(1.0)  # Increased from 0.5 to 1.0 seconds

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
    Get market cap, sector info using fast_info (more reliable than info).
    """
    print("\n" + "=" * 70)
    print("STEP 4: ENRICHING WITH METADATA (Getting sector/market cap)")
    print("=" * 70)

    if max_stocks:
        stock_list = stock_list[:max_stocks]
        print(f"Limiting to {max_stocks} stocks for faster processing")

    enriched = []
    failed_count = 0

    for stock_info in tqdm(stock_list, desc="Getting metadata"):
        ticker = stock_info['ticker']

        try:
            ticker_obj = yf.Ticker(ticker)

            # Try fast_info first (more reliable for market cap)
            # fast_info is an object with attributes, not a dict
            try:
                fast_info = ticker_obj.fast_info
                market_cap = getattr(fast_info, 'market_cap', None) or getattr(fast_info, 'marketCap', None)
                if market_cap is None:
                    market_cap = 0
            except:
                # Fallback to info if fast_info fails
                try:
                    info = ticker_obj.info
                    market_cap = info.get('marketCap', 0) or info.get('market_cap', 0)
                except:
                    market_cap = 0

            # Get sector/industry/country from info (not available in fast_info)
            try:
                info = ticker_obj.info
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                country = info.get('country', 'Unknown')
            except:
                sector = 'Unknown'
                industry = 'Unknown'
                country = 'Unknown'

            stock_info['metadata'] = {
                'sector': sector,
                'industry': industry,
                'market_cap': market_cap if market_cap else 0,
                'country': country,
            }

            enriched.append(stock_info)

        except Exception as e:
            # Still include, just without metadata
            failed_count += 1
            stock_info['metadata'] = {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'country': 'Unknown',
            }
            enriched.append(stock_info)

        time.sleep(0.2)  # Increased rate limiting to avoid 429 errors

    print(f"✓ Enriched {len(enriched)} stocks with metadata")
    if failed_count > 0:
        print(f"  ⚠️  Failed to get full metadata for {failed_count} stocks")

    return enriched


# ============================================================================
# STEP 5: STRATIFIED SELECTION
# ============================================================================

def stratified_selection(enriched_stocks, failed_stocks, target_total=1500):
    """
    Select balanced training set by market cap and sector.
    Handles Swedish vs US market cap differences.
    Falls back to volume-based selection if market cap unavailable.
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
            'country': s['metadata'].get('country', 'Unknown'),
            'avg_volume': s['stats']['avg_volume'],
            'days_of_data': s['stats']['days_of_data'],
        }
        for s in enriched_stocks
    ])

    # Separate Swedish stocks (different market cap scale)
    # Swedish large-cap ~$5-10B vs US large-cap $10B+
    is_swedish = df['country'] == 'Sweden'
    df_us = df[~is_swedish].copy()
    df_swedish = df[is_swedish].copy()

    print(f"\nStock distribution:")
    print(f"  US/International: {len(df_us)}")
    print(f"  Swedish: {len(df_swedish)}")

    # Check if we have market cap data
    valid_market_caps = df[df['market_cap'] > 0]
    use_market_cap = len(valid_market_caps) > (len(df) * 0.5)  # At least 50% have market cap

    selected = []

    if use_market_cap:
        print(f"Using market cap stratification ({len(valid_market_caps)}/{len(df)} stocks have market cap)")

        # Apply different market cap buckets for US vs Swedish
        # US: small <$2B, mid $2-10B, large $10-200B, mega $200B+
        # Swedish: adjusted down (large-cap there ~$5-10B)
        if len(df_us) > 0:
            df_us['cap_bucket'] = pd.cut(
                df_us['market_cap'],
                bins=[0, 2e9, 10e9, 200e9, 1e15],
                labels=['small', 'mid', 'large', 'mega']
            )

        if len(df_swedish) > 0:
            # Lower thresholds for Swedish market
            df_swedish['cap_bucket'] = pd.cut(
                df_swedish['market_cap'],
                bins=[0, 500e6, 2e9, 10e9, 1e15],  # Adjusted: 500M, 2B, 10B
                labels=['small', 'mid', 'large', 'mega']
            )

        # Recombine
        df = pd.concat([df_us, df_swedish], ignore_index=True)

        # Target allocation (save 10% for failures)
        target_by_bucket = {
            'mega': int(target_total * 0.15),   # 15%
            'large': int(target_total * 0.25),  # 25%
            'mid': int(target_total * 0.30),    # 30%
            'small': int(target_total * 0.20),  # 20%
        }

        for bucket, n_target in target_by_bucket.items():
            bucket_df = df[df['cap_bucket'] == bucket]

            if len(bucket_df) == 0:
                continue

            # Sample within bucket
            n_sample = min(n_target, len(bucket_df))

            # Try to balance by sector within bucket
            sampled = bucket_df.sample(n=n_sample, random_state=42)
            selected.extend(sampled['ticker'].tolist())

        # If we didn't get enough from bucketed selection, add random samples
        if len(selected) < int(target_total * 0.90):
            remaining_needed = int(target_total * 0.90) - len(selected)
            unselected = df[~df['ticker'].isin(selected)]
            if len(unselected) > 0:
                additional = unselected.sample(n=min(remaining_needed, len(unselected)), random_state=42)
                selected.extend(additional['ticker'].tolist())
                print(f"  Added {len(additional)} random samples to reach target")

    else:
        # Fallback: use volume-based stratification
        print(f"⚠️  Insufficient market cap data, using volume-based selection")

        # Volume buckets
        df['volume_bucket'] = pd.qcut(
            df['avg_volume'],
            q=4,
            labels=['low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )

        target_per_bucket = int(target_total * 0.90 / 4)

        for bucket in ['low', 'medium', 'high', 'very_high']:
            bucket_df = df[df['volume_bucket'] == bucket]

            if len(bucket_df) == 0:
                continue

            n_sample = min(target_per_bucket, len(bucket_df))
            sampled = bucket_df.sample(n=n_sample, random_state=42)
            selected.extend(sampled['ticker'].tolist())

    # Add failures (up to 10% of total)
    n_failures = min(len(failed_stocks), int(target_total * 0.10))
    failure_tickers = [s['ticker'] for s in failed_stocks[:n_failures]]
    selected.extend(failure_tickers)

    # Final report
    print(f"\n✓ Selected {len(selected)} stocks:")
    print(f"  Active stocks: {len(selected) - len(failure_tickers)}")

    if len(selected) > 0:  # Fix division by zero
        print(f"  Failed stocks: {len(failure_tickers)} ({len(failure_tickers)/len(selected):.1%})")
    else:
        print(f"  Failed stocks: {len(failure_tickers)}")

    # Distribution report and stats collection
    stats = {
        'total_selected': len(selected),
        'active_stocks': len(selected) - len(failure_tickers),
        'failed_stocks': len(failure_tickers),
        'failed_pct': len(failure_tickers)/len(selected) if len(selected) > 0 else 0,
        'us_international': len(df_us),
        'swedish': len(df_swedish),
        'market_cap_distribution': {},
        'volume_distribution': {},
        'sector_distribution': {}
    }

    if len(selected) > 0:
        selected_df = df[df['ticker'].isin(selected)]

        if use_market_cap and 'cap_bucket' in selected_df.columns:
            print(f"\n📊 Market Cap Distribution:")
            cap_dist = selected_df['cap_bucket'].value_counts().sort_index()
            print(cap_dist)
            stats['market_cap_distribution'] = cap_dist.to_dict()
        elif 'volume_bucket' in selected_df.columns:
            print(f"\n📊 Volume Distribution:")
            vol_dist = selected_df['volume_bucket'].value_counts().sort_index()
            print(vol_dist)
            stats['volume_distribution'] = vol_dist.to_dict()

        print(f"\n📊 Sector Distribution (top 10):")
        sector_dist = selected_df['sector'].value_counts()
        print(sector_dist.head(10))
        stats['sector_distribution'] = sector_dist.to_dict()

    return selected, stats


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
    # Track run timing
    run_start_time = datetime.now()

    print("\n" + "=" * 70)
    print("STOCK BOTTOM DETECTION - TRAINING DATASET BUILDER")
    print("=" * 70)

    # Configuration
    TICKER_JSON_PATH = '/Users/deaz/Developer/project_quant/pQuant_ultimate/data/tickers_data/tickers_validated_20251029.json'
    START_DATE = '2015-01-01'  # 10 years of data for pattern recognition
    END_DATE = '2024-12-31'    # ~2,500 trading days (250/year * 10)
    TARGET_US_SAMPLE = 2000    # How many US stocks to try downloading
    TARGET_FINAL = 1500        # Final training set size

    # Extract date from ticker filename (tickers_validated_20251029.json -> 20251029)
    ticker_filename = Path(TICKER_JSON_PATH).name
    date_match = re.search(r'(\d{8})', ticker_filename)
    dataset_date = date_match.group(1) if date_match else datetime.now().strftime('%Y%m%d')

    # Get git commit hash
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except:
        git_commit = 'unknown'

    print(f"\nConfiguration:")
    print(f"  Dataset date: {dataset_date}")
    print(f"  Git commit: {git_commit}")
    print(f"  Input: {ticker_filename}")
    print(f"  Date range: {START_DATE} to {END_DATE} (10 years, ~2500 trading days)")
    print(f"  Data frequency: Daily (default yfinance interval)")
    print(f"  Target sample: {TARGET_US_SAMPLE} US stocks")
    print(f"  Target final: {TARGET_FINAL} stocks")
    print(f"\nNote: Input should be filtered & validated (run filter_tickers.py -> validate_tickers.py first)")

    # Step 1: Load validated tickers
    tickers_dict = load_tickers(TICKER_JSON_PATH)
    
    # Step 2: Create download list
    download_list = create_download_list(tickers_dict, TARGET_US_SAMPLE)
    
    # Step 3: Download data
    download_results = download_all_tickers(download_list, START_DATE, END_DATE)
    
    # Step 4: Enrich with metadata
    enriched = enrich_with_metadata(download_results['valid'], max_stocks=2500)
    
    # Step 5: Stratified selection
    selected_tickers, selection_stats = stratified_selection(
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

    # Create dated output directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'training_data' / dataset_date
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create combined dataframe first (needed for metadata)
    all_dfs = []
    for ticker, info in final_data.items():
        if info['data'] is not None:
            df = info['data'].copy()
            df['ticker'] = ticker
            df['stock_id'] = ticker
            df['is_failed'] = info['is_failed']
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            all_dfs.append(df)

    if not all_dfs:
        print(f"\n⚠️  NO DATA - All downloads failed or were rejected")
        print(f"   Try adjusting the quality filters or ticker selection")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Calculate run duration
    run_end_time = datetime.now()
    run_duration_seconds = int((run_end_time - run_start_time).total_seconds())

    # Generate comprehensive metadata
    metadata = {
        'created_at': run_end_time.isoformat(),
        'dataset_date': dataset_date,
        'data_date_range': {
            'start': START_DATE,
            'end': END_DATE,
            'actual_min': str(combined_df['date'].min()),
            'actual_max': str(combined_df['date'].max())
        },
        'ticker_source': ticker_filename,
        'parameters': {
            'target_us_sample': TARGET_US_SAMPLE,
            'target_final': TARGET_FINAL,
            'start_date': START_DATE,
            'end_date': END_DATE
        },
        'results': {
            'total_stocks': len(final_data),
            'active_stocks': len([v for v in final_data.values() if not v['is_failed']]),
            'failed_stocks': sum(1 for v in final_data.values() if v['is_failed']),
            'total_rows': len(combined_df)
        },
        'git_commit': git_commit,
        'run_duration_seconds': run_duration_seconds,
        'run_duration_human': f"{run_duration_seconds // 3600}h {(run_duration_seconds % 3600) // 60}m {run_duration_seconds % 60}s",
        'selection_stats': selection_stats
    }

    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Saved: {metadata_path.name}")

    # Save as pickle (preserves dataframes)
    pkl_path = output_dir / 'training_data.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"✓ Saved: {pkl_path.name}")

    # Save parquet
    parquet_path = output_dir / 'training_stocks_data.parquet'
    combined_df.to_parquet(parquet_path)
    print(f"✓ Saved: {parquet_path.name}")

    # Save selection stats (legacy compatibility)
    stats_path = output_dir / 'selection_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(selection_stats, f, indent=2, default=str)
    print(f"✓ Saved: {stats_path.name}")

    print(f"\n✅ COMPLETE!")
    print(f"   Output: {output_dir}")
    print(f"   Total stocks: {len(final_data)}")
    print(f"   Total rows: {len(combined_df):,}")
    print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"   Failed stocks: {sum(1 for v in final_data.values() if v['is_failed'])}")
    print(f"   Run duration: {metadata['run_duration_human']}")


if __name__ == "__main__":
    main()