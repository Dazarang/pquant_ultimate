"""
Incremental update for training dataset.
Fetches new data from latest available date to now.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


class TrainingDataUpdater:
    """Handles incremental updates to existing training datasets."""

    def __init__(self, dataset_dir: Path, buffer_days: int = 7):
        """
        Initialize updater.

        Args:
            dataset_dir: Path to existing dataset directory (e.g., data/training_data/20251031)
            buffer_days: Number of days to subtract from 'now' to avoid incomplete data
        """
        self.dataset_dir = Path(dataset_dir)
        self.buffer_days = buffer_days
        self.metadata = None
        self.tickers = None
        self.latest_date = None

    def load_metadata(self) -> dict:
        """Load existing dataset metadata."""
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        print("Dataset metadata loaded:")
        print(f"  Dataset date: {self.metadata['dataset_date']}")
        print(f"  Total stocks: {self.metadata['results']['total_stocks']}")
        print(f"  Date range: {self.metadata['data_date_range']['start']} to {self.metadata['data_date_range']['end']}")

        return self.metadata

    def load_existing_data(self) -> pd.DataFrame:
        """Load existing training data from parquet."""
        parquet_path = self.dataset_dir / "training_stocks_data.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Training data not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        print(f"\nExisting data loaded: {len(df):,} rows")

        self.tickers = sorted(df["ticker"].unique())
        print(f"Tickers found: {len(self.tickers)}")

        df["date"] = pd.to_datetime(df["date"])
        self.latest_date = df["date"].max()
        print(f"Latest date in dataset: {self.latest_date}")

        return df

    def calculate_update_range(self) -> tuple[str, str]:
        """
        Calculate date range for update.

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        if self.latest_date is None:
            raise ValueError("Must load existing data first")

        start_date = (self.latest_date + timedelta(days=1)).strftime("%Y-%m-%d")

        end_date = (datetime.now() - timedelta(days=self.buffer_days)).strftime("%Y-%m-%d")

        print(f"\nUpdate range: {start_date} to {end_date}")

        return start_date, end_date

    def fetch_new_data(self, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        """
        Fetch new data for all tickers in batches.

        Args:
            start_date: Start date for new data
            end_date: End date for new data

        Returns:
            Dict mapping ticker -> DataFrame of new data
        """
        print(f"\nFetching new data for {len(self.tickers)} tickers...")

        batch_size = 100
        new_data = {}
        failed_tickers = []

        for i in tqdm(range(0, len(self.tickers), batch_size), desc="Downloading batches"):
            batch = self.tickers[i : i + batch_size]

            try:
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    auto_adjust=True,
                )

                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None

                        if ticker_data is not None and not ticker_data.empty:
                            new_data[ticker] = ticker_data
                        else:
                            failed_tickers.append(ticker)

                    except Exception:
                        failed_tickers.append(ticker)

            except Exception as e:
                print(f"  Batch error: {e}")
                failed_tickers.extend(batch)

            time.sleep(1.0)

        print("\nFetch complete:")
        print(f"  Success: {len(new_data)} tickers")
        print(f"  Failed: {len(failed_tickers)} tickers")

        if failed_tickers:
            print(f"  Failed tickers: {', '.join(failed_tickers[:10])}")
            if len(failed_tickers) > 10:
                print(f"  ... and {len(failed_tickers) - 10} more")

        return new_data

    def merge_data(self, existing_df: pd.DataFrame, new_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge new data with existing dataset.

        Args:
            existing_df: Existing training data
            new_data: Dict of new data per ticker

        Returns:
            Merged DataFrame
        """
        print("\nMerging new data with existing dataset...")

        new_dfs = []
        for ticker, ticker_data in new_data.items():
            df = ticker_data.copy()
            df["ticker"] = ticker
            df["stock_id"] = ticker
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            existing_ticker_data = existing_df[existing_df["ticker"] == ticker]
            if not existing_ticker_data.empty:
                df["is_failed"] = existing_ticker_data["is_failed"].iloc[0]
            else:
                df["is_failed"] = False

            new_dfs.append(df)

        if not new_dfs:
            print("  No new data to merge")
            return existing_df

        new_combined = pd.concat(new_dfs, ignore_index=True)
        print(f"  New data: {len(new_combined):,} rows")

        merged = pd.concat([existing_df, new_combined], ignore_index=True)

        merged = merged.drop_duplicates(subset=["ticker", "date"], keep="last")

        print(f"  Merged total: {len(merged):,} rows")
        print(f"  Added: {len(merged) - len(existing_df):,} rows")

        return merged

    def save_updated_data(self, merged_df: pd.DataFrame, output_dir: Path) -> None:
        """
        Save updated dataset to new directory.

        Args:
            merged_df: Merged DataFrame with updated data
            output_dir: Directory to save updated dataset
        """
        print(f"\nSaving updated dataset to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = output_dir / "training_stocks_data.parquet"
        merged_df.to_parquet(parquet_path)
        print(f"  Saved: {parquet_path.name}")

        updated_metadata = self.metadata.copy()
        updated_metadata["updated_at"] = datetime.now().isoformat()
        updated_metadata["data_date_range"]["actual_max"] = str(merged_df["date"].max())
        updated_metadata["results"]["total_rows"] = len(merged_df)

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(updated_metadata, f, indent=2, default=str)
        print(f"  Saved: {metadata_path.name}")

        tickers_path = output_dir / "all_tickers.json"
        tickers = sorted(merged_df["ticker"].unique())
        with open(tickers_path, "w") as f:
            json.dump(tickers, f, indent=2)
        print(f"  Saved: {tickers_path.name}")

        print("\nUpdate complete!")


def main():
    """Main execution."""
    print("=" * 70)
    print("TRAINING DATA INCREMENTAL UPDATE")
    print("=" * 70)

    base_dir = Path(__file__).parent
    training_data_dir = base_dir / "training_data"

    existing_datasets = sorted(training_data_dir.glob("????????"), reverse=True)
    if not existing_datasets:
        print("ERROR: No existing training datasets found in data/training_data/")
        return

    latest_dataset = existing_datasets[0]
    print(f"Latest dataset: {latest_dataset.name}")

    buffer_days = 7
    print(f"Buffer days: {buffer_days} (to avoid incomplete recent data)")

    updater = TrainingDataUpdater(latest_dataset, buffer_days=buffer_days)

    updater.load_metadata()
    existing_df = updater.load_existing_data()

    start_date, end_date = updater.calculate_update_range()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if start_dt >= end_dt:
        print(f"\nNo update needed. Latest data is from {updater.latest_date.date()}")
        print(f"Current date (with {buffer_days} day buffer): {end_date}")
        return

    new_data = updater.fetch_new_data(start_date, end_date)

    if not new_data:
        print("\nNo new data fetched. Nothing to update.")
        return

    merged_df = updater.merge_data(existing_df, new_data)

    output_date = datetime.now().strftime("%Y%m%d")
    output_dir = training_data_dir / output_date
    updater.save_updated_data(merged_df, output_dir)

    print(f"\nNew dataset: {output_dir}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Total rows: {len(merged_df):,}")
    print(f"Total tickers: {len(merged_df['ticker'].unique())}")


if __name__ == "__main__":
    main()
