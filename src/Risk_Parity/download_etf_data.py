#!/usr/local/bin/python3
"""
Downloads daily closing prices for ETFs listed in ETF.xlsx and maintains a
local cache in ETF_back_data.parquet: columns are ETF tickers, the index is
the sorted date of each price (as pandas Timestamps).

For each ticker:
  - not in cache yet     -> download full history
  - in cache but stale   -> download only the missing recent days
  - in cache and current -> skip

Settings (input_file, data_file) come from download_etf_data.toml, under the
[EtfDataDownloader] section.
"""

import sys
from functools import reduce
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Set import Path to include: '..'
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configuration_manager_class import ConfigurationManager
from logger import logger

DEBUG_FLAG = False
BDAY_US = CustomBusinessDay(calendar=USFederalHolidayCalendar())


class EtfDataDownloader:
    """
    Reads ETF tickers from an Excel file, refreshes a local Parquet cache of
    their daily closing prices (only downloading what's missing), and writes
    the updated cache back out.
    """

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.input_file = Path(self.config["input_file"])
        self.data_file = Path(self.config["data_file"])
        self.monthly_file = Path(self.config["monthly_file"])
        self.monthly_prices: pd.DataFrame = pd.DataFrame()
        self.new_tickers: list[str] = []
        self.prices: pd.DataFrame = pd.DataFrame()
        self.expected_last_day: pd.Timestamp = self._last_expected_trading_day()
        return None

    def run(self) -> None:
        if not self.input_file.exists():
            sys.exit(f"Input file not found: {self.input_file}")

        self.new_tickers = self._load_ticker_list()
        logger.info(f"Read {len(self.new_tickers)} tickers from {self.input_file}")

        self.prices = self._load_cache()
        logger.info(f"Read {len(self.prices)} tickers from {self.data_file}")
        logger.info(f"Updating data (expected last trading day: {self.expected_last_day.date()})...")

        series_by_ticker = {ticker: self.prices[ticker].dropna() for ticker in self.prices.columns}
        new_data_flag = False
        for ticker in self.new_tickers:
            try:
                updated = self._update_ticker(ticker, series_by_ticker)
                if updated:
                    new_data_flag = True
            except Exception as e:
                logger.error(f"  {ticker}: failed ({e})")

        # Make sure all series have the same index - fill in missing dates with NaN
        # all_dates is the sorted union of all indexes
        all_dates = reduce(
            lambda acc, idx: acc.union(idx),
            (series.index for series in series_by_ticker.values()),
            pd.Index([]),
        ) # Incrementally build the union of all indexes (idx) into acc
        all_dates = all_dates.sort_values() # sort all_dates to ensure chronological order
        logger.debug(f"All dates count: {len(all_dates)}")
        logger.info(f"# series: {len(series_by_ticker)}")
        for ticker, series in series_by_ticker.items():
            series_by_ticker[ticker] = series.reindex(all_dates)
            logger.info(f"{ticker}: {len(series_by_ticker[ticker])}")
        self.prices = pd.concat(series_by_ticker, axis=1, ignore_index=False)
        self.prices = self.prices.loc[self.prices.index.sort_values()]
        logger.info(f"Final prices shape: {self.prices.shape}")
        if new_data_flag:
            self._save_cache()
            self.monthly_prices = self.sample_monthly_prices()
            self.monthly_prices.to_excel(self.monthly_file, header=True, index=True, float_format="%.2f")  
        else:
            logger.info("No new data found. No files were updated.")
        return None

    def _load_ticker_list(self) -> list[str]:
        """First column of input_file, deduped and upper-cased."""
        df = pd.read_excel(self.input_file, header=0)
        tickers = df.loc[:, 'Ticker'].dropna().astype(str).str.strip().str.upper()
        return tickers.unique().tolist()

    def _load_cache(self) -> pd.DataFrame:
        if not self.data_file.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.data_file)
        df.index = pd.to_datetime(df.index)
        df = df.loc[df.index.sort_values()]
        self._log_date_summary(df, "Loaded")
        return df

    def _save_cache(self) -> None:
        logger.debug(f"Saving cache:\n{self.prices.head()}\n----\n{self.prices.tail()}")
        self.prices.to_parquet(self.data_file)
        self._log_date_summary(self.prices, "Saved")
        return None

    def _log_date_summary(self, df: pd.DataFrame, action: str) -> None:
        if df is None or df.empty:
            logger.info(f"{action} {self.data_file}: no data")
            return None
        common = df.dropna(how="any")
        oldest_common = common.index.min() if common is not None and not common.empty else None
        logger.info(
            f"{action} {self.data_file}: "
            f"number of rows={len(df)}, "
            f"number of tickers={len(df.columns)}\n"
            f"   most recent date={df.index.max().date()}, "
            f"oldest date={df.index.min().date()}, "
            f"oldest date with all tickers present="
            f"{oldest_common.date() if oldest_common is not None else 'N/A'}"
        )
        return None

    def sample_monthly_prices(self) -> pd.DataFrame:
        """self.prices resampled to month-end (last available price per ticker per month)."""
        monthly = self.prices.resample("ME").last()  # ME = month-end
        self._log_date_summary(monthly, "Computed monthly")
        return monthly

    def _last_expected_trading_day(self) -> pd.Timestamp:
        """Most recent business day strictly before today (avoids assuming today's close is posted)."""
        today = pd.Timestamp.now().normalize()
        return today - BDAY_US

    def _update_ticker( self, ticker: str, series_by_ticker: dict[str, pd.Series], ) -> bool:
        """
        Update the cached series for a given ticker with new data.
        Args:
            ticker (str): The ticker symbol to update.
            series_by_ticker (dict[str, pd.Series]): Dictionary of existing prices by ticker.

        Returns:
            bool: True if the series was updated, False otherwise.
        
        """
        existing = series_by_ticker.get(ticker, None)

        if existing is not None and not existing.empty:  # we already have some data for this ticker in the cache
            last_date = existing.index.max()
            first_date = existing.index.min()
            logger.info(f"  {ticker}: existing data from {first_date.date()} to {last_date.date()}")
            if last_date >= self.expected_last_day:
                logger.info(f"  {ticker}: up to date (last={last_date.date()})")
                return False
            start = last_date + pd.Timedelta(days=1)
            logger.info(f"  {ticker}: refreshing from {start.date()}")
        else:  # First time seeing this ticker
            start = None
            logger.info(f"  {ticker}: downloading full history")

        new_data = self._download_close(ticker, start)
        if new_data is None:
            logger.info(f"  {ticker}: no new data returned")
            return False

        combined = new_data if existing is None else pd.concat([existing, new_data])
        combined = combined.loc[combined.index.sort_values()]
        series_by_ticker[ticker] = combined[~combined.index.duplicated(keep="last")]
        return True

    def _download_close(self, ticker: str, start: pd.Timestamp | None) -> pd.Series | None:
        kwargs: dict[str, bool | str] = {"auto_adjust": True, "progress": False}
        if start is not None:
            kwargs["start"] = start.strftime("%Y-%m-%d")
        else:
            kwargs["period"] = "max"

        data = yf.download(ticker, **kwargs)
        if data is None or data.empty:
            return None
        logger.info(f"  {ticker}: downloaded data:\n{data.head()}")
        close = data["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance returns MultiIndex (Price, Ticker) columns
            close = close.iloc[:, 0]
        return close.rename(ticker)


def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    # Set logger log level based on configuration
    logger.setLevel(config_manager.config.get("LOG_LEVEL", "INFO"))
    downloader = EtfDataDownloader(config_manager)
    downloader.run()
    return None


if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
