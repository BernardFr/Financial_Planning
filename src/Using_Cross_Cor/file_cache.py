"""On-disk cache of dated series, one JSON file per ticker.

File name: ``<TICKER>_<creation date>.json`` (date written with ``date_format``,
e.g. ``BRK.B_2026_11_09.json``).  Only the most recent file per ticker is kept.

File content::

    {
      "ticker": "BNDX",
      "creation_date": "2026-11-09",
      "source": "Yahoo Finance (Adj Close)",
      "data": [{"date": "2016-11-09", "adj_close": 40.32, "log_return": null}, ...]
    }

``data`` holds one record per date with every column of the cached DataFrame;
missing values are stored as ``null``.

An entry is stale once the calendar month is later than the month it was created in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from custom_logger import custom_logger as logger

CACHE_SUFFIX = ".json"
TMP_SUFFIX = ".json.tmp"
NAME_SEPARATOR = "_"  # between ticker and creation date in a file name
ISO_DATE_FORMAT = "%Y-%m-%d"
DATE_COLUMN = "date"
JSON_INDENT = 1


class CacheEntry(NamedTuple):
    ticker: str
    created: date
    path: Path


@dataclass
class FileCache:
    """Per-ticker JSON cache, refreshed monthly."""

    directory: Path
    date_format: str
    source: str = ""

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---- lookup ---------------------------------------------------------- #
    def _parse_name(self, path: Path) -> CacheEntry | None:
        """Split ``<TICKER>_<date>`` from the right; None when no date suffix parses.

        Splitting at each separator lets tickers contain it (e.g. ``BRK_B``).
        """
        stem = path.name.removesuffix(CACHE_SUFFIX)
        for i in (i for i, ch in enumerate(stem) if ch == NAME_SEPARATOR):
            try:
                created = datetime.strptime(stem[i + 1:], self.date_format).date()
            except ValueError:
                continue
            return CacheEntry(stem[:i], created, path)
        return None

    def entries(self) -> list[CacheEntry]:
        """Every cache file, sorted by ticker then newest first."""
        parsed = (self._parse_name(p) for p in self.directory.glob(f"*{CACHE_SUFFIX}"))
        found = [e for e in parsed if e is not None]
        return sorted(found, key=lambda e: (e.ticker, -e.created.toordinal()))

    def latest(self, ticker: str) -> CacheEntry | None:
        """Most recent entry for ``ticker``, whatever its age; older files are deleted."""
        files = [e for e in self.entries() if e.ticker == ticker]
        if not files:
            return None
        for old in files[1:]:
            old.path.unlink(missing_ok=True)
        return files[0]

    @staticmethod
    def is_stale(created: date, today: date) -> bool:
        """True when ``today`` is in a calendar month later than ``created``."""
        return (today.year, today.month) > (created.year, created.month)

    def get(self, ticker: str, today: date) -> pd.DataFrame | None:
        """The cached data for ``ticker``; None when missing or stale."""
        entry = self.latest(ticker)
        if entry is None or self.is_stale(entry.created, today):
            return None
        return self.read_file(entry.path)

    def split_fresh(self, ticker_lst: list[str], today: date) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Fresh cached data per ticker, and the tickers that need to be (re)downloaded."""
        cached: dict[str, pd.DataFrame] = {}
        to_refresh: list[str] = []
        for ticker in ticker_lst:
            df = self.get(ticker, today)
            if df is None:
                to_refresh.append(ticker)
            else:
                cached[ticker] = df

        if not to_refresh:
            logger.info("Using cached data for all %d tickers", len(ticker_lst))
        else:
            logger.info("Refreshing %d ticker(s): %s", len(to_refresh), ", ".join(to_refresh))
            if cached:
                logger.info("Using cached data for the other %d ticker(s)", len(cached))
        return cached, to_refresh

    # ---- read / write ---------------------------------------------------- #
    @staticmethod
    def read_file(path: Path) -> pd.DataFrame:
        """The DataFrame (indexed by date) stored in cache file ``path``."""
        payload = json.loads(path.read_text())
        df = pd.DataFrame(payload["data"])
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        return df.set_index(DATE_COLUMN)

    def put(self, ticker: str, df: pd.DataFrame, created: date) -> Path:
        """Write ``df`` for ``ticker``, replacing any older file for it."""
        path = self.directory / f"{ticker}{NAME_SEPARATOR}{created.strftime(self.date_format)}{CACHE_SUFFIX}"
        date_strs = pd.DatetimeIndex(df.index).strftime(ISO_DATE_FORMAT)
        records = [
            {
                DATE_COLUMN: day,
                **{col: None if pd.isna(v) else float(v) for col, v in row.items()},
            }
            for day, row in zip(date_strs, df.to_dict("records"))
        ]
        payload = {
            "ticker": ticker,
            "creation_date": created.isoformat(),
            "source": self.source,
            "data": records,
        }
        tmp = path.with_name(path.name.removesuffix(CACHE_SUFFIX) + TMP_SUFFIX)
        tmp.write_text(json.dumps(payload, indent=JSON_INDENT))
        tmp.replace(path)  # atomic write

        self.latest(ticker)  # removes older files for this ticker
        return path
