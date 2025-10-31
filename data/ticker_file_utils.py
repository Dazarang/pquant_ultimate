"""
utility for finding latest ticker json files by pattern
"""

import re
from pathlib import Path


class TickerFileFinder:
    """finds latest ticker json files based on date patterns"""

    def __init__(self, base_dir: Path | str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / "tickers_data"

    def _extract_date(self, filename: str) -> int:
        """extract date from filename like tickers_20251029.json -> 20251029"""
        match = re.search(r"(\d{8})", filename)
        return int(match.group(1)) if match else 0

    def find_latest(self, pattern: str) -> Path | None:
        """
        find latest file matching pattern with date

        args:
            pattern: glob pattern like "tickers_*.json" or "tickers_filtered_*.json"

        returns:
            path to latest file or none if not found

        examples:
            find_latest("tickers_*.json") -> tickers_20251031.json
            find_latest("tickers_filtered_*.json") -> tickers_filtered_20251031.json
            find_latest("tickers_validated_*.json") -> tickers_validated_20251029.json
        """
        if not self.base_dir.exists():
            return None

        matching_files = list(self.base_dir.glob(pattern))
        if not matching_files:
            return None

        # sort by extracted date descending
        latest = max(matching_files, key=lambda f: self._extract_date(f.name))
        return latest

    def get_latest_raw(self) -> Path | None:
        """get latest tickers_*.json file (excluding filtered/validated)"""
        if not self.base_dir.exists():
            return None

        # find tickers_*.json but exclude tickers_filtered_* and tickers_validated_*
        matching_files = [
            f
            for f in self.base_dir.glob("tickers_*.json")
            if not f.name.startswith("tickers_filtered_") and not f.name.startswith("tickers_validated_")
        ]

        if not matching_files:
            return None

        # sort by extracted date descending
        latest = max(matching_files, key=lambda f: self._extract_date(f.name))
        return latest

    def get_latest_filtered(self) -> Path | None:
        """get latest tickers_filtered_*.json file"""
        return self.find_latest("tickers_filtered_*.json")

    def get_latest_validated(self) -> Path | None:
        """get latest tickers_validated_*.json file"""
        return self.find_latest("tickers_validated_*.json")
