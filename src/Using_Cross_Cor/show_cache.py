#!/usr/local/bin/python3
"""Print a cached ticker series to the console, one ``date: values`` line per date.

Usage::

    show_cache.py BNDX     # exact ticker
    show_cache.py bnd      # part of a ticker: pick from the cached tickers that contain it

The cache folder and file-name date format come from ``holdings_cross_corr.toml``
(see ``file_cache.FileCache``).  Matching ignores case.  A single match is shown
directly; several are listed for the user to choose from.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from file_cache import CacheEntry, FileCache
from holdings_cross_corr import Config

CONFIG_SCRIPT = Path(__file__).resolve().with_name("holdings_cross_corr.py")
DATE_FORMAT = "%Y-%m-%d"
NUMBER_FORMAT = "{:,.6f}"
MISSING_VALUE_TEXT = "n/a"
COLUMN_SEPARATOR = "  "


class CachedSeriesFinder:
    """Finds cached series by full or partial ticker."""

    def __init__(self, cache: FileCache) -> None:
        self.entry_lst = cache.entries()

    def matches(self, query: str) -> list[CacheEntry]:
        """Entries whose ticker equals ``query``, ignoring case; failing that, those whose ticker contains it."""
        query = query.lower()
        exact = [e for e in self.entry_lst if e.ticker.lower() == query]
        return exact or [e for e in self.entry_lst if query in e.ticker.lower()]


def choose(options: list[CacheEntry]) -> CacheEntry | None:
    """Ask the user to pick one of ``options`` by number; None when they enter nothing."""
    for number, entry in enumerate(options, start=1):
        print(f"{number:>3}. {entry.ticker} ({entry.created})")
    while True:
        answer = input(f"Choose 1-{len(options)} (Enter to quit): ").strip()
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print(f"Please enter a number from 1 to {len(options)}.")


def format_value(value: float) -> str:
    return MISSING_VALUE_TEXT if pd.isna(value) else NUMBER_FORMAT.format(value)


def print_series(path: Path) -> None:
    df = FileCache.read_file(path)
    print(f"{'date':<10}: " + COLUMN_SEPARATOR.join(df.columns))
    date_strs = pd.DatetimeIndex(df.index).strftime(DATE_FORMAT)
    for day, row in zip(date_strs, df.itertuples(index=False)):
        print(f"{day}: " + COLUMN_SEPARATOR.join(format_value(v) for v in row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a cached ticker series, one 'date: values' line per date.")
    parser.add_argument("ticker", help="ticker, or part of one, e.g. BNDX or bnd")
    args = parser.parse_args(argv)

    cfg = Config.load(CONFIG_SCRIPT)
    cache = FileCache(cfg.cache_folder, cfg.date_format)
    matches = CachedSeriesFinder(cache).matches(args.ticker)
    if not matches:
        print(f"No cached ticker matches {args.ticker!r} in {cfg.cache_folder}", file=sys.stderr)
        return 1
    entry = matches[0] if len(matches) == 1 else choose(matches)
    if entry is None:
        return 0
    print(f"{entry.ticker} (cached {entry.created})")
    print_series(entry.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
