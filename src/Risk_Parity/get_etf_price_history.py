#!/usr/local/bin/python3
"""
Downloads daily closing prices for ETFs listed in ETF.xlsx and maintains a
local cache in ETF_back_data.parquet (one row per ticker, one column per day).

For each ticker:
  - not in cache yet        -> download full history
  - in cache but stale      -> download only the missing recent days
  - in cache and current    -> skip

"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

INPUT_FILE = Path("ETF.xlsx")
DATA_FILE = Path("ETF_back_data.parquet")

BDAY_US = CustomBusinessDay(calendar=USFederalHolidayCalendar())


@dataclass
class Ticker_Price_History:
    ticker: str
    history_df: pd.DataFrame | pd.Series

    def __post_init__(self) -> None:
        self.ticker = self.ticker.strip().upper()
        self.history_df = self._normalize_history_df(self.history_df)

    @staticmethod
    def _normalize_history_df(history_df: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(history_df, pd.Series):
            normalized = history_df.to_frame(name="close")
        else:
            normalized = history_df.copy()

        if normalized.empty:
            return pd.DataFrame(columns=["close"], dtype=float)

        if "close" not in normalized.columns:
            if len(normalized.columns) != 1:
                raise ValueError("Ticker history must contain exactly one close-price column")
            normalized = normalized.rename(columns={normalized.columns[0]: "close"})

        normalized = normalized[["close"]].copy()
        normalized.index = pd.to_datetime(normalized.index)
        normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
        normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
        return normalized.dropna(subset=["close"])

    @property
    def close_ser(self) -> pd.Series:
        return self.history_df["close"]

    @property
    def last_date(self) -> pd.Timestamp | None:
        if self.history_df.empty:
            return None
        return pd.Timestamp(self.history_df.index.max())

    def append(self, newer_history: "Ticker_Price_History") -> "Ticker_Price_History":
        if self.ticker != newer_history.ticker:
            raise ValueError("Cannot merge ticker histories for different tickers")
        merged_df = pd.concat([self.history_df, newer_history.history_df])
        merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()
        return Ticker_Price_History(self.ticker, merged_df)

    @classmethod
    def from_download(cls, ticker: str, start: pd.Timestamp | None) -> "Ticker_Price_History | None":
        kwargs: dict[str, bool | str] = {"auto_adjust": True, "progress": False}
        if start is not None:
            kwargs["start"] = start.strftime("%Y-%m-%d")
        else:
            kwargs["period"] = "max"
        data = yf.download(ticker, **kwargs)
    
        close_ser = cls._extract_close_series(ticker, data)
        if close_ser is None or close_ser.empty:
            return None
        return cls(ticker=ticker, history_df=close_ser)

    @staticmethod
    def _extract_close_series(ticker: str, data: pd.DataFrame | None) -> pd.Series | None:
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            close_data = data["Close"]
            if isinstance(close_data, pd.Series):
                return close_data
            if ticker in close_data.columns:
                return close_data[ticker]
            if len(close_data.columns) == 1:
                return close_data.iloc[:, 0]
            raise ValueError(f"Unable to isolate Close prices for {ticker}")

        if "Close" not in data.columns:
            raise ValueError(f"No Close column returned for {ticker}")
        close_data = data["Close"]
        if isinstance(close_data, pd.Series):
            return close_data
        if ticker in close_data.columns:
            return close_data[ticker]
        if close_data.shape[1] == 1:
            return close_data.iloc[:, 0]
        raise ValueError(f"Unable to isolate Close prices for {ticker}")


class ETF_Price_History:
    def __init__(self, history_df: pd.DataFrame | None = None):
        self.history_df = self._normalize_history_df(history_df)

    @staticmethod
    def _normalize_history_df(history_df: pd.DataFrame | None) -> pd.DataFrame:
        if history_df is None or history_df.empty:
            return pd.DataFrame(dtype=float)

        normalized = history_df.copy()
        normalized.index = normalized.index.map(lambda ticker: str(ticker).strip().upper())
        normalized.columns = pd.to_datetime(normalized.columns)
        normalized = normalized.sort_index(axis=0).sort_index(axis=1)
        return normalized.astype(float)

    @classmethod
    def load(cls, path: Path) -> "ETF_Price_History":
        if not path.exists():
            return cls()
        return cls(pd.read_parquet(path))

    def save(self, path: Path) -> None:
        save_df = self.history_df.reindex(sorted(self.history_df.columns), axis=1)
        save_df = save_df.copy()
        save_df.columns = [pd.Timestamp(col).strftime("%Y-%m-%d") for col in save_df.columns]
        save_df.to_parquet(path)

    def __len__(self) -> int:
        return len(self.history_df.index)

    def get_ticker_history(self, ticker: str) -> Ticker_Price_History | None:
        ticker = ticker.strip().upper()
        if ticker not in self.history_df.index:
            return None

        row = self.history_df.loc[ticker]
        close_ser = row.dropna().sort_index()
        if close_ser.empty:
            return None
        return Ticker_Price_History(ticker=ticker, history_df=close_ser)

    def set_ticker_history(self, ticker_history: Ticker_Price_History) -> None:
        ticker_df = ticker_history.close_ser.to_frame().T
        ticker_df.index = pd.Index([ticker_history.ticker])
        combined = pd.concat([self.history_df, ticker_df], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        self.history_df = self._normalize_history_df(combined)

    def update_ticker(self, ticker: str, expected_last_day: pd.Timestamp) -> None:
        existing_history = self.get_ticker_history(ticker)

        if existing_history is not None and existing_history.last_date is not None:
            last_date = existing_history.last_date
            if last_date >= expected_last_day:
                print(f"  {ticker}: up to date (last={last_date.date()})")
                return
            start = last_date + pd.Timedelta(days=1)
            print(f"  {ticker}: refreshing from {start.date()}")
        else:
            start = None
            print(f"  {ticker}: downloading full history")

        new_history = Ticker_Price_History.from_download(ticker, start)
        if new_history is None:
            print(f"  {ticker}: no new data returned")
            return

        merged_history = new_history if existing_history is None else existing_history.append(new_history)
        self.set_ticker_history(merged_history)

    def update_all(self, tickers: list[str], expected_last_day: pd.Timestamp) -> None:
        for ticker in tickers:
            try:
                self.update_ticker(ticker, expected_last_day)
            except Exception as e:
                print(f"  {ticker}: failed ({e})")


def load_ticker_list(path: Path) -> list[str]:
    df = pd.read_excel(path, header=None)
    tickers = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper()
    return tickers.unique().tolist()


def last_expected_trading_day() -> pd.Timestamp:
    """Most recent business day strictly before today (avoids assuming today's close is posted)."""
    today = pd.Timestamp.now().normalize()
    return today - BDAY_US


def main() -> None:
    if not INPUT_FILE.exists():
        sys.exit(f"Input file not found: {INPUT_FILE}")

    tickers = load_ticker_list(INPUT_FILE)
    print(f"Read {len(tickers)} tickers from {INPUT_FILE}")

    cache = ETF_Price_History.load(DATA_FILE)
    expected_last_day = last_expected_trading_day()

    print(f"\nUpdating data (expected last trading day: {expected_last_day.date()})...")
    cache.update_all(tickers, expected_last_day)

    cache.save(DATA_FILE)
    print(f"\nSaved {len(cache)} tickers to {DATA_FILE}")


if __name__ == "__main__":
    main()
