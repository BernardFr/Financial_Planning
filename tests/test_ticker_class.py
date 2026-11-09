#!/usr/local/bin/python3
"""
Tests for Ticker: reading the positions file, Yahoo symbol mapping,
the stale-cache fallback when a download fails, and return statistics.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parent.parent / "src" / "Using_Cross_Cor"
sys.path.insert(0, str(CODE_DIR))

from file_cache import FileCache  # noqa: E402
import holdings_cross_corr as hcc  # noqa: E402
from ticker_class import DAYS_PER_YEAR, Ticker  # noqa: E402

DATE_FORMAT = "%Y_%m_%d"
TODAY = date(2026, 11, 9)
YEARS = 10


def sample_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-11-02", "2026-11-03"]), name="date")
    return pd.DataFrame({"adj_close": [100.0, 101.0], "log_return": [float("nan"), 0.00995]}, index=idx)


def test_read_positions_cleans_dedupes_and_maps(tmp_path: Path) -> None:
    positions = tmp_path / "Positions_2026_11_04.xlsx"
    pd.DataFrame({"Ticker": [" spy", "brk.b", None, "SPY", ""], "Qty": [1, 2, 3, 4, 5]}).to_excel(positions, index=False)

    tickers = Ticker.read_positions(positions, "Ticker", {"BRK.B": "BRK-B"})

    assert [(t.symbol, t.yf_symbol) for t in tickers] == [("SPY", "SPY"), ("BRK.B", "BRK-B")]


def test_read_positions_falls_back_to_first_column(tmp_path: Path) -> None:
    positions = tmp_path / "Positions_2026_11_04.xlsx"
    pd.DataFrame({"Symbol": ["QQQ"]}).to_excel(positions, index=False)

    assert [t.symbol for t in Ticker.read_positions(positions, "Ticker", {})] == ["QQQ"]


def test_download_or_stale_caches_successful_download(tmp_path: Path, monkeypatch) -> None:
    cache = FileCache(tmp_path, DATE_FORMAT)
    monkeypatch.setattr(Ticker, "download", lambda self, years: sample_df())

    df = Ticker("SPY").download_or_stale(cache, TODAY, YEARS)

    assert df is not None and len(df) == 2
    assert cache.get("SPY", TODAY) is not None


def test_download_or_stale_falls_back_to_stale_cache(tmp_path: Path, monkeypatch) -> None:
    cache = FileCache(tmp_path, DATE_FORMAT)
    cache.put("SPY", sample_df(), date(2026, 10, 1))

    def failing_download(self: Ticker, years: int) -> pd.DataFrame:
        raise ValueError("network down")

    monkeypatch.setattr(Ticker, "download", failing_download)

    stale = Ticker("SPY").download_or_stale(cache, TODAY, YEARS)
    assert stale is not None and len(stale) == 2           # stale file used
    assert Ticker("BND").download_or_stale(cache, TODAY, YEARS) is None  # nothing to fall back on


def ticker_with_prices(symbol: str, prices: list[float], days: list[str]) -> Ticker:
    price = pd.Series(prices, index=pd.DatetimeIndex(pd.to_datetime(days), name="date"))
    ticker = Ticker(symbol)
    ticker.history = pd.DataFrame({"adj_close": price, "log_return": np.log(price / price.shift(1))})
    return ticker


def test_stats_cagr_from_prices() -> None:
    days = ["2024-01-01", "2025-01-01", "2026-01-01"]
    ticker = ticker_with_prices("SPY", [100.0, 150.0, 200.0], days)

    stats = ticker.stats()

    years = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days / DAYS_PER_YEAR
    assert stats.name == "SPY"
    assert stats["cagr"] == pytest.approx(2 ** (1 / years) - 1)
    assert not stats.isna().any()


def test_compute_stats_one_row_per_ticker() -> None:
    days = ["2024-01-01", "2025-01-01", "2026-01-01"]
    ticker_lst = [ticker_with_prices("SPY", [100.0, 150.0, 200.0], days),
                  ticker_with_prices("BND", [50.0, 51.0, 52.0], days)]

    result_df = hcc.compute_stats(ticker_lst)

    assert list(result_df["ticker"]) == ["SPY", "BND"]
    assert list(result_df.columns) == ["ticker", "mean_log_return", "mean_return", "sigma",
                                       "cagr", "arith_mean", "arith_empirical",
                                       "Expected Return", "Standard Deviation"]
