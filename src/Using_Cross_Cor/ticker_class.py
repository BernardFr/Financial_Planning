"""A holding's ticker: its Yahoo Finance symbol, price history and return statistics.

``symbol`` is the ticker as it appears in the positions file (and names its cache
file); ``yf_symbol`` is what Yahoo Finance knows it as (e.g. ``BRK.B`` -> ``BRK-B``).
``history`` is a DataFrame indexed by date with ``adj_close`` and ``log_return``
columns, filled by the caller (from the cache or ``download_or_stale``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from custom_logger import custom_logger as logger
from file_cache import FileCache

YF_INTERVAL = "1d"
YF_PRICE_COLUMN = "Adj Close"
DATE_COLUMN = "date"
ADJ_CLOSE_COLUMN = "adj_close"
LOG_RETURN_COLUMN = "log_return"
TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25          # calendar days, for CAGR
YEAR_END_FREQ = "YE"            # resample rule: calendar year end
TICKER_EXCLUDE = ["total", 'ticker'] # list of names that are not tickers


class Ticker:
    """One ticker from the positions file."""

    def __init__(self, symbol: str, yf_symbol: str | None = None) -> None:
        self.symbol = symbol
        self.yf_symbol = yf_symbol or symbol
        self.history: pd.DataFrame | None = None

    def __repr__(self) -> str:
        return f"Ticker({self.symbol!r}, yf_symbol={self.yf_symbol!r})"

    @classmethod
    def read_positions(cls, positions_file: Path, ticker_column: str,
                       yf_ticker_map: dict[str, str]) -> list[Ticker]:
        """The unique, non-empty tickers of the positions file (order kept), upper-cased."""
        df = pd.read_excel(positions_file)
        if ticker_column not in df.columns:
            # Fall back to the first column, per the spec ("first column 'Ticker'")
            logger.warning(
                "Column '%s' not found in %s - using first column '%s'",
                ticker_column, positions_file.name, df.columns[0],
            )
            ticker_column = df.columns[0]

        symbol_ser = df[ticker_column].dropna().astype(str).str.strip().str.upper()
        symbol_ser = symbol_ser[symbol_ser != ""].drop_duplicates()
        # remove row that is in the exclusion list - in whatever case
        symbol_ser = symbol_ser[~symbol_ser.str.lower().isin(TICKER_EXCLUDE)]
        return [cls(symbol, yf_ticker_map.get(symbol)) for symbol in symbol_ser]

    def download(self, years: int) -> pd.DataFrame:
        """Daily Adj Close for the last ``years`` years, with daily log returns."""
        hist = yf.Ticker(self.yf_symbol).history(
            period=f"{years}y", interval=YF_INTERVAL, auto_adjust=False, actions=False
        )
        if hist.empty or YF_PRICE_COLUMN not in hist.columns:
            raise ValueError(f"No '{YF_PRICE_COLUMN}' data returned for {self.yf_symbol}")

        s = hist[YF_PRICE_COLUMN].dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize().rename(DATE_COLUMN)
        s = s[~s.index.duplicated(keep="last")].sort_index()

        df = pd.DataFrame({ADJ_CLOSE_COLUMN: s})
        df[LOG_RETURN_COLUMN] = np.log(df[ADJ_CLOSE_COLUMN] / df[ADJ_CLOSE_COLUMN].shift(1))  # daily log returns
        return df

    def download_or_stale(self, cache: FileCache, today: date, years: int) -> pd.DataFrame | None:
        """
        Download and cache the data.  On failure, fall back to the stale cache
        file; None when there is none.
        """
        try:
            df = self.download(years)
            cache.put(self.symbol, df, today)
            return df
        except Exception as exc:  # network error, delisted ticker, etc.
            stale = cache.latest(self.symbol)
            if stale is None:
                logger.error("%s: download failed (%s) - skipped", self.symbol, exc)
                return None
            logger.warning("%s: download failed (%s) - using stale cache from %s",
                           self.symbol, exc, stale.created)
            return cache.read_file(stale.path)

    def stats(self) -> pd.Series:
        """Annualized return and volatility statistics of ``history``, named by symbol."""
        assert self.history is not None, f"{self.symbol}: history not loaded"
        log_ret = self.history[LOG_RETURN_COLUMN]
        price = self.history[ADJ_CLOSE_COLUMN]

        mu_log = log_ret.mean() * TRADING_DAYS_PER_YEAR      # annualized log (continuously compounded) return
        mean_return = np.exp(mu_log) - 1
        # ddof=1 because we are estimating the sample standard deviation - i.e. we have an incomplete sample of the population
        sigma = log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)  # annualized volatility
        years = (price.index[-1] - price.index[0]).days / DAYS_PER_YEAR
        annual = price.resample(YEAR_END_FREQ).last().pct_change().dropna()  # calendar-year returns

        return pd.Series(
            {
                "mean_log_return": mu_log,
                "mean_return": mean_return,                  # approximate annualized simple return
                "sigma": sigma,
                "cagr": (price.iloc[-1] / price.iloc[0]) ** (1 / years) - 1,  # geometric; ≈ exp(mu_log) - 1
                "arith_mean": np.exp(mu_log + sigma**2 / 2) - 1,   # expected 1-yr simple return (lognormal)
                "arith_empirical": annual.mean(),                  # empirical cross-check
                # Columns below are used by monte carlo simulation
                "Expected Return": 100.0 * mean_return,  # in percentage
                "Standard Deviation": 100.0 * sigma  # in percentage
            },
            name=self.symbol,
        )
