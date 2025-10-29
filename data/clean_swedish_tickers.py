"""
swedish ticker cleaner - removes duplicate share classes

keeps one share class per company (e.g. VOLV-A.ST but not VOLV-B.ST)
preserves original naming format
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/tickers_data/swedish_clean.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SwedishTickerCleaner:
    """removes duplicate share classes from swedish tickers"""

    def __init__(self):
        self.raw_tickers: List[str] = []
        self.cleaned_tickers: List[str] = []
        self.removed_tickers: List[str] = []

    def _extract_base_name(self, ticker: str) -> str:
        """
        extract base company name from ticker

        examples:
            ERIC-A.ST -> ERIC
            ERIC-B.ST -> ERIC
            HM-B.ST -> HM
            NOKIA-SEK.ST -> NOKIA
        """
        # remove .ST suffix
        ticker_without_suffix = ticker.replace('.ST', '')

        # get base name (everything before first hyphen)
        if '-' in ticker_without_suffix:
            base = ticker_without_suffix.split('-')[0]
        else:
            base = ticker_without_suffix

        return base

    def clean(self, tickers: List[str]) -> List[str]:
        """
        clean tickers by keeping one share class per company

        args:
            tickers: list of swedish tickers

        returns:
            deduplicated list (first occurrence of each company)
        """
        logger.info("=" * 70)
        logger.info("cleaning swedish tickers")
        logger.info("=" * 70)
        logger.info(f"input tickers: {len(tickers)}")

        self.raw_tickers = tickers
        seen_bases: Dict[str, str] = {}  # base -> full ticker

        for ticker in tickers:
            base = self._extract_base_name(ticker)

            if base not in seen_bases:
                # first occurrence - keep it
                seen_bases[base] = ticker
                self.cleaned_tickers.append(ticker)
            else:
                # duplicate - remove it
                self.removed_tickers.append(ticker)
                logger.debug(f"removing {ticker} (keeping {seen_bases[base]})")

        logger.info(f"output tickers: {len(self.cleaned_tickers)}")
        logger.info(f"removed: {len(self.removed_tickers)}")

        if self.removed_tickers:
            logger.info("\nremoved tickers:")
            for ticker in self.removed_tickers[:10]:
                base = self._extract_base_name(ticker)
                kept = seen_bases[base]
                logger.info(f"  {ticker} -> kept {kept}")
            if len(self.removed_tickers) > 10:
                logger.info(f"  ... and {len(self.removed_tickers) - 10} more")

        return self.cleaned_tickers

    def load_from_file(self, filepath: Path) -> Dict[str, List[str]]:
        """load all tickers from json file"""
        logger.info(f"loading from {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'Sweden' not in data:
            raise ValueError("no 'Sweden' key in json file")

        return data

    def save_to_file(self, all_tickers: Dict[str, List[str]], filepath: Path):
        """save all tickers with cleaned swedish ones"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_tickers, f, indent=2, ensure_ascii=False)

        logger.info(f"\nsaved: {filepath}")


def main():
    """main entry point"""
    logger.info("swedish ticker cleaner")
    logger.info("")

    cleaner = SwedishTickerCleaner()

    try:
        # find latest tickers file
        tickers_dir = Path('data/tickers_data')
        ticker_files = sorted(tickers_dir.glob('tickers_*.json'))

        if not ticker_files:
            logger.error("no ticker files found")
            return

        latest_file = ticker_files[-1]

        # load all tickers
        all_tickers = cleaner.load_from_file(latest_file)

        # clean only swedish tickers
        raw_swedish = all_tickers['Sweden']
        cleaned_swedish = cleaner.clean(raw_swedish)

        # replace swedish tickers with cleaned ones
        all_tickers['Sweden'] = cleaned_swedish

        # save results
        timestamp = datetime.now().strftime('%Y%m%d')
        output_file = tickers_dir / f'tickers_cleaned_{timestamp}.json'
        cleaner.save_to_file(all_tickers, output_file)

        logger.info("\ndone")

    except KeyboardInterrupt:
        logger.info("\n\ninterrupted by user")

    except Exception as e:
        logger.error(f"error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
