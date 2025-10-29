"""
Ticker validation for filtering out delisted and invalid stocks.
Single responsibility: validate ticker availability and data quality.
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time


class TickerValidator:
    """
    Validates stock tickers against Yahoo Finance to filter out:
    - Delisted stocks
    - Invalid tickers
    - Tickers with timezone errors
    - Tickers with insufficient data
    """

    def __init__(self, validation_days: int = 7, rate_limit_delay: float = 0.1):
        """
        Initialize validator.

        Args:
            validation_days: Number of recent days to check for data
            rate_limit_delay: Delay between requests in seconds
        """
        self.validation_days = validation_days
        self.rate_limit_delay = rate_limit_delay
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=validation_days)

    def validate_single_ticker(self, ticker: str) -> Tuple[bool, str]:
        """
        Validate a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (is_valid, reason)
            - is_valid: True if ticker is valid and has data
            - reason: Explanation if invalid
        """
        try:
            # Quick test: try to download recent data
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                show_errors=False
            )

            # Check if data exists
            if data is None or data.empty:
                return False, "no_data"

            # Check if we got at least 1 day of data
            if len(data) < 1:
                return False, "insufficient_data"

            # Check for valid close price
            close_col = 'Close' if 'Close' in data.columns else 'close'
            if close_col not in data.columns:
                return False, "missing_close_column"

            if data[close_col].isna().all():
                return False, "all_prices_nan"

            # Valid ticker
            return True, "valid"

        except Exception as e:
            error_str = str(e).lower()

            # Categorize common errors
            if 'delisted' in error_str or 'no price data' in error_str:
                return False, "delisted"
            elif 'timezone' in error_str or 'no timezone' in error_str:
                return False, "no_timezone"
            elif 'rate limit' in error_str or '429' in error_str:
                return False, "rate_limited"
            elif 'not found' in error_str or '404' in error_str:
                return False, "not_found"
            elif 'timeout' in error_str:
                return False, "timeout"
            else:
                return False, f"error_{type(e).__name__}"

    def validate_batch(
        self,
        tickers: List[str],
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Validate a batch of tickers.

        Args:
            tickers: List of ticker symbols
            verbose: Print progress

        Returns:
            Dictionary with results:
            {
                'valid': [ticker1, ticker2, ...],
                'invalid': {ticker: reason, ...},
                'stats': {...}
            }
        """
        results = {
            'valid': [],
            'invalid': {},
        }

        for i, ticker in enumerate(tickers, 1):
            is_valid, reason = self.validate_single_ticker(ticker)

            if is_valid:
                results['valid'].append(ticker)
            else:
                results['invalid'][ticker] = reason

            # Progress update
            if verbose and i % 100 == 0:
                valid_count = len(results['valid'])
                invalid_count = len(results['invalid'])
                print(f"  Progress: {i}/{len(tickers)} | "
                      f"Valid: {valid_count} | Invalid: {invalid_count}")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        # Add statistics
        results['stats'] = self._compute_stats(results)

        return results

    def _compute_stats(self, results: Dict) -> Dict:
        """Compute validation statistics."""
        total = len(results['valid']) + len(results['invalid'])

        # Count reasons
        reason_counts = {}
        for reason in results['invalid'].values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            'total_checked': total,
            'valid_count': len(results['valid']),
            'invalid_count': len(results['invalid']),
            'valid_rate': len(results['valid']) / total if total > 0 else 0,
            'rejection_reasons': reason_counts
        }

    def filter_ticker_list(
        self,
        tickers: List[str],
        verbose: bool = True
    ) -> List[str]:
        """
        Filter a list of tickers, returning only valid ones.

        Args:
            tickers: List of ticker symbols
            verbose: Print progress

        Returns:
            List of valid tickers
        """
        results = self.validate_batch(tickers, verbose=verbose)

        if verbose:
            self._print_summary(results)

        return results['valid']

    def _print_summary(self, results: Dict) -> None:
        """Print validation summary."""
        stats = results['stats']

        print(f"\n{'=' * 70}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total checked: {stats['total_checked']}")
        print(f"Valid: {stats['valid_count']} ({stats['valid_rate']:.1%})")
        print(f"Invalid: {stats['invalid_count']}")

        print(f"\nRejection reasons:")
        for reason, count in sorted(
            stats['rejection_reasons'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {reason}: {count}")
