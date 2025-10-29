"""
ticker downloader for us, sp500, and swedish markets

sources:
- us: nasdaq ftp (official)
- sp500: stockanalysis.com api
- sweden: stockanalysis.com api
"""

import ftplib
import requests
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/tickers_data/ticker_download.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class USTickerFetcher:
    """fetches us tickers from nasdaq ftp"""

    def __init__(self):
        self.tickers: List[str] = []

    def _convert_ticker(self, ticker: str) -> str:
        """
        convert ticker format for yahoo finance compatibility
        only converts dots in share class designations, not exchange suffixes

        examples:
            BRK.A -> BRK-A
            BRK.B -> BRK-B
            SFAST.ST -> SFAST.ST (unchanged)
        """
        # don't convert if ticker has exchange suffix
        exchange_suffixes = ['.ST', '.TO', '.L', '.AX', '.HK', '.T']
        for suffix in exchange_suffixes:
            if ticker.endswith(suffix):
                return ticker

        # convert remaining dots (share classes)
        return ticker.replace('.', '-')

    def fetch(self) -> List[str]:
        """
        download us market tickers from nasdaq ftp

        returns:
            list of ticker symbols
        """
        logger.info("=" * 70)
        logger.info("fetching us tickers from nasdaq ftp")
        logger.info("=" * 70)

        tickers = []

        try:
            ftp = ftplib.FTP("ftp.nasdaqtrader.com", timeout=30)
            ftp.login("anonymous", "")
            ftp.cwd("SymbolDirectory")

            logger.info("connected to nasdaq ftp server")

            # nasdaq-listed stocks
            logger.info("downloading nasdaqlisted.txt...")
            nasdaq_data = []
            ftp.retrlines('RETR nasdaqlisted.txt', nasdaq_data.append)

            for line in nasdaq_data[1:-1]:
                parts = line.split('|')
                if len(parts) >= 2:
                    symbol = parts[0].strip()
                    if symbol and symbol not in ['File Creation Time']:
                        tickers.append(self._convert_ticker(symbol))

            logger.info(f"  nasdaq: {len(tickers)} tickers")

            # other listings (nyse, amex, etc)
            logger.info("downloading otherlisted.txt...")
            other_data = []
            ftp.retrlines('RETR otherlisted.txt', other_data.append)

            for line in other_data[1:-1]:
                parts = line.split('|')
                if len(parts) >= 7:
                    symbol = parts[0].strip()
                    if symbol and symbol not in ['File Creation Time']:
                        tickers.append(self._convert_ticker(symbol))

            ftp.quit()
            logger.info(f"total us tickers: {len(tickers)}")

        except Exception as e:
            logger.error(f"error downloading from nasdaq ftp: {e}")

        self.tickers = tickers
        return tickers


class SP500TickerFetcher:
    """fetches s&p 500 tickers from stockanalysis.com api"""

    def __init__(self):
        self.tickers: List[str] = []
        self.session = requests.Session()
        self.base_url = "https://stockanalysis.com/api/screener/s/f"

    def _convert_ticker(self, ticker: str) -> str:
        """
        convert ticker format for yahoo finance compatibility
        only converts dots in share class designations, not exchange suffixes

        examples:
            BRK.A -> BRK-A
            BRK.B -> BRK-B
            SFAST.ST -> SFAST.ST (unchanged)
        """
        # don't convert if ticker has exchange suffix
        exchange_suffixes = ['.ST', '.TO', '.L', '.AX', '.HK', '.T']
        for suffix in exchange_suffixes:
            if ticker.endswith(suffix):
                return ticker

        # convert remaining dots (share classes)
        return ticker.replace('.', '-')

    def fetch(self) -> List[str]:
        """
        download s&p 500 tickers from stockanalysis.com api

        returns:
            list of ticker symbols
        """
        logger.info("=" * 70)
        logger.info("fetching s&p 500 tickers from stockanalysis.com")
        logger.info("=" * 70)

        tickers = []

        try:
            params = {
                'm': 'marketCap',
                's': 'desc',
                'c': 'no,s,tr1m,tr6m,trYTD,tr1y,tr5y,tr10y,marketCap',
                'sc': 'marketCap',
                'f': 'inIndex-includes-SP500',
                'i': 'stocks'
            }

            logger.info("fetching s&p 500 stocks...")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data['status'] != 200:
                logger.error(f"api returned status {data['status']}")
                return tickers

            items = data['data']['data']

            for item in items:
                ticker = item.get('s', '')
                if ticker:
                    tickers.append(self._convert_ticker(ticker))

            logger.info(f"total s&p 500 tickers: {len(tickers)}")

        except Exception as e:
            logger.error(f"error downloading s&p 500 tickers: {e}")

        self.tickers = tickers
        return tickers


class SwedishTickerFetcher:
    """fetches swedish tickers from stockanalysis.com api"""

    def __init__(self):
        self.tickers: List[str] = []
        self.session = requests.Session()
        self.base_url = "https://stockanalysis.com/api/screener/a/f"

    def _convert_ticker(self, ticker_str: str) -> str:
        """
        convert sto/XXX.X format to XXX-X.ST for yahoo finance

        examples:
            sto/QLOSR.B -> QLOSR-B.ST
            sto/SAFETY.B -> SAFETY-B.ST
            sto/ACE -> ACE.ST
        """
        # remove sto/ prefix
        ticker = ticker_str.replace('sto/', '')

        # replace . with - for share classes
        if '.' in ticker:
            ticker = ticker.replace('.', '-')

        # add .ST suffix
        ticker = f"{ticker}.ST"

        return ticker

    def fetch(self) -> List[str]:
        """
        download swedish market tickers from stockanalysis.com api

        returns:
            list of ticker symbols in yahoo finance format
        """
        logger.info("=" * 70)
        logger.info("fetching swedish tickers from stockanalysis.com")
        logger.info("=" * 70)

        tickers = []
        page = 1

        try:
            while True:
                params = {
                    'm': 'marketCap',
                    's': 'desc',
                    'c': 'no,s,n,marketCap,price,change,revenue',
                    'sc': 'marketCap',
                    'cn': '500',
                    'f': 'exchangeCode-is-STO,subtype-is-stock',
                    'p': str(page),
                    'i': 'symbols'
                }

                logger.info(f"fetching page {page}...")
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data['status'] != 200:
                    logger.error(f"api returned status {data['status']}")
                    break

                items = data['data']['data']

                if not items:
                    break

                for item in items:
                    ticker_raw = item.get('s', '')
                    if ticker_raw:
                        ticker_converted = self._convert_ticker(ticker_raw)
                        tickers.append(ticker_converted)

                logger.info(f"  page {page}: {len(items)} tickers")

                # check if we've fetched all
                total = data['data'].get('resultsCount', 0)
                if len(tickers) >= total:
                    break

                page += 1

            logger.info(f"total swedish tickers: {len(tickers)}")

        except Exception as e:
            logger.error(f"error downloading swedish tickers: {e}")

        self.tickers = tickers
        return tickers


class TickerDownloader:
    """main ticker downloader coordinating us, sp500, and swedish fetchers"""

    def __init__(self):
        self.us_fetcher = USTickerFetcher()
        self.sp500_fetcher = SP500TickerFetcher()
        self.swedish_fetcher = SwedishTickerFetcher()
        self.tickers: Dict[str, List[str]] = {}

    def download_all(self) -> Dict[str, List[str]]:
        """
        download tickers from all sources

        returns:
            dict with keys 'US', 'SP500', and 'Sweden' containing ticker lists
        """
        start_time = datetime.now()
        logger.info("starting ticker download")
        logger.info(f"started at: {start_time}")

        # fetch us tickers
        us_tickers = self.us_fetcher.fetch()

        # fetch sp500 tickers
        sp500_tickers = self.sp500_fetcher.fetch()

        # fetch swedish tickers
        swedish_tickers = self.swedish_fetcher.fetch()

        self.tickers = {
            'US': us_tickers,
            'SP500': sp500_tickers,
            'Sweden': swedish_tickers
        }

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 70)
        logger.info("download complete")
        logger.info("=" * 70)
        logger.info(f"us tickers: {len(us_tickers):,}")
        logger.info(f"sp500 tickers: {len(sp500_tickers):,}")
        logger.info(f"swedish tickers: {len(swedish_tickers):,}")
        logger.info(f"total: {len(us_tickers) + len(sp500_tickers) + len(swedish_tickers):,}")
        logger.info(f"time elapsed: {elapsed:.1f} seconds")

        return self.tickers

    def save_results(self, output_dir: Path = Path('data/tickers_data')):
        """save results to json"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d')

        json_file = output_dir / f'tickers_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.tickers, f, indent=2, ensure_ascii=False)

        logger.info(f"\nsaved: {json_file}")


def main():
    """main entry point"""
    logger.info("ticker downloader")
    logger.info("markets: us, sp500, sweden")
    logger.info("")

    downloader = TickerDownloader()

    try:
        downloader.download_all()
        downloader.save_results()

    except KeyboardInterrupt:
        logger.info("\n\ninterrupted by user")

    except Exception as e:
        logger.error(f"error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
