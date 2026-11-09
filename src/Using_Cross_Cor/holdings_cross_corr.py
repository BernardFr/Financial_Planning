#!/usr/local/bin/python3

"""
holdings_cross_corr.py

Load 10 years of daily Adjusted Close prices for every ticker in the most
recent Positions_YYYY_MM_DD.xlsx file, compute daily log returns, and cache
each series as JSON via file_cache.FileCache (one file per ticker:
<TICKER>_<YYYY_MM_DD>.json, refreshed when the calendar month changes).

Configuration is read from holdings_cross_corr.toml (same root as this script).
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

from custom_logger import custom_logger as logger, setup_logger
from file_cache import FileCache
from ticker_class import Ticker, ADJ_CLOSE_COLUMN, TRADING_DAYS_PER_YEAR

DEBUG = False
MAX_COUNT = 3
DATA_SOURCE = "Yahoo Finance (Adj Close)"
TICKER_COLUMN = "ticker"  # first column of the result file
OUTPUT_INFIX = "_out_"   # result file: <program stem>_out_<date>.xlsx
OUTPUT_SUFFIX = ".xlsx"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    positions_prefix: str
    input_folder: Path
    date_format: str
    cache_folder: Path
    output_folder: Path
    ticker_column: str = "Ticker"
    history_years: int = 10
    # Holdings ticker -> Yahoo Finance ticker (only where they differ)
    yf_ticker_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, script_path: Path) -> "Config":
        toml_path = script_path.with_suffix(".toml")
        if not toml_path.exists():
            raise FileNotFoundError(f"Config file not found: {toml_path}")
        with toml_path.open("rb") as f:
            raw = tomllib.load(f)

        # Relative folders are resolved against the script's directory so the
        # program behaves the same regardless of the current working directory.
        base = script_path.parent

        def resolve(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else (base / path).resolve()

        return cls(
            positions_prefix=raw["positions_prefix"],
            input_folder=resolve(raw["input_folder"]),
            date_format=raw["date_format"],
            cache_folder=resolve(raw["cache_folder"]),
            output_folder=resolve(raw["output_folder"]),
            ticker_column=raw.get("ticker_column", "Ticker"),
            history_years=int(raw.get("history_years", 10)),
            # Keys upper-cased to match Ticker.read_positions()
            yf_ticker_map={k.strip().upper(): v.strip()
                           for k, v in raw.get("yf_ticker_map", {}).items()},
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_date(text: str, fmt: str) -> date | None:
    try:
        return datetime.strptime(text, fmt).date()
    except ValueError:
        return None


def find_latest_positions_file(cfg: Config) -> tuple[Path, date]:
    """Return the most recent Positions file (by the date in its name)."""
    candidates: list[tuple[date, Path]] = []
    for p in cfg.input_folder.glob(f"{cfg.positions_prefix}*.xlsx"):
        if p.name.startswith("~$"):  # Excel lock file
            continue
        d = parse_date(p.stem[len(cfg.positions_prefix):], cfg.date_format)
        if d is not None:
            candidates.append((d, p))

    if not candidates:
        raise FileNotFoundError(
            f"No {cfg.positions_prefix}<date>.xlsx file found in {cfg.input_folder}"
        )
    d, p = max(candidates)
    return p, d

def compute_stats(ticker_lst: list[Ticker]) -> pd.DataFrame:
    """One row of annualized return statistics per ticker (see ``Ticker.stats``)."""
    return pd.DataFrame([ticker.stats() for ticker in ticker_lst]).rename_axis(TICKER_COLUMN).reset_index()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_holdings_data(cfg: Config, today: date | None = None) -> list[Ticker]:
    """
    The tickers of the latest positions file, each with its ``history`` loaded
    (from the cache, refreshed as needed).  Tickers without data are left out.
    """
    today = today or date.today()

    positions_file, positions_date = find_latest_positions_file(cfg)
    logger.info("Positions file: %s (dated %s)", positions_file.name, positions_date)

    ticker_lst = Ticker.read_positions(positions_file, cfg.ticker_column, cfg.yf_ticker_map)
    logger.info("Found %d tickers", len(ticker_lst))

    cache = FileCache(cfg.cache_folder, cfg.date_format, DATA_SOURCE)

    cached, _ = cache.split_fresh([ticker.symbol for ticker in ticker_lst], today)

    loaded_lst: list[Ticker] = []
    failed: list[str] = []

    counter = 0
    for ticker in ticker_lst:
        df = cached.get(ticker.symbol)
        if df is None:
            df = ticker.download_or_stale(cache, today, cfg.history_years)
        if df is None:
            failed.append(ticker.symbol)
            continue

        ticker.history = df
        loaded_lst.append(ticker)
        logger.info("%-8s first date: %s  data points: %d",
                    ticker.symbol, df.index[0].date(), len(df))
        counter += 1
        if DEBUG and counter >= MAX_COUNT:
            break

    if failed:
        logger.warning("%d ticker(s) without data: %s", len(failed), ", ".join(failed))

    return loaded_lst


def result_path(prog_name: str, cfg: Config, today: date) -> Path:
    """``<prog_name>_out_<date>.xlsx`` in the configured output folder."""
    name = f"{prog_name}{OUTPUT_INFIX}{today.strftime(cfg.date_format)}{OUTPUT_SUFFIX}"
    return cfg.output_folder / name

def compute_cross_corr(ticker_lst: list[Ticker]) -> pd.DataFrame:
    """Compute the cross-correlation matrix of log returns for the given tickers."""
    # create prices: DataFrame, one column per ticker, index = dates, Adj Close values
    prices = pd.DataFrame({
        ticker.symbol: ticker.history[ADJ_CLOSE_COLUMN]
        for ticker in ticker_lst
        if ticker.history is not None
    })
    logr = prices.apply(np.log).diff()        # daily log returns: log(p_t) - log(p_t-1)

    common = logr.dropna(how="any")          # only dates where every stock traded
    cross_corr_df = common.corr(method="pearson")
    cov    = common.cov() * TRADING_DAYS_PER_YEAR              # annualized covariance, if you need it
    return cross_corr_df


def main() -> int:
    script_path = Path(__file__).resolve()
    setup_logger(script_path.stem)
    cfg = Config.load(script_path)
    try:
        ticker_lst = load_holdings_data(cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    # for each ticker, compute its arithmetic mean of log returns and variance

    stats_df = compute_stats(ticker_lst)
    logger.info("Computed statistics for %d tickers", len(stats_df))
    logger.info("Resulting DataFrame:\n%s", stats_df)
    # Save the resulting DataFrame to a Excel file
    result_file = result_path(script_path.stem, cfg, date.today())
    stats_df.to_excel(result_file, index=False, engine="openpyxl", sheet_name="Stats")
    logger.info("Saved Stats to %s", result_file)

    corr_df = compute_cross_corr(ticker_lst)
    logger.info("Computed cross-correlation matrix:\n%s", corr_df)
    # Save the cross-correlation matrix to the same Excel file, different sheet
    with pd.ExcelWriter(result_file, engine="openpyxl", mode="a") as writer:
        corr_df.to_excel(writer, sheet_name="Correlation")
    logger.info("Saved Cross-Correlation matrix to %s", result_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
