#!/usr/local/bin/python3
"""
Tests for FileCache: one file per ticker (the most recent), monthly staleness,
round-trip of the JSON format, and tickers that contain the name separator.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from file_cache import CacheEntry, FileCache  # noqa: E402
from show_cache import CachedSeriesFinder  # noqa: E402

DATE_FORMAT = "%Y_%m_%d"
TODAY = date(2026, 11, 9)


def make_cache(folder: Path) -> FileCache:
    return FileCache(folder, DATE_FORMAT, "test source")


def sample_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-11-02", "2026-11-03"]), name="date")
    return pd.DataFrame({"adj_close": [100.0, 101.0], "log_return": [float("nan"), 0.00995]}, index=idx)


def test_latest_keeps_only_most_recent_file(tmp_path: Path) -> None:
    for name in ["SPY_2026_09_01.json", "SPY_2026_10_01.json", "SPY_2026_11_01.json"]:
        (tmp_path / name).write_text("{}")
    (tmp_path / "SPY_B_2026_01_01.json").write_text("{}")  # different ticker: kept

    entry = make_cache(tmp_path).latest("SPY")

    assert entry == CacheEntry("SPY", date(2026, 11, 1), tmp_path / "SPY_2026_11_01.json")
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [
        "SPY_2026_11_01.json",
        "SPY_B_2026_01_01.json",
    ]


def test_put_leaves_one_file_per_ticker_and_round_trips(tmp_path: Path) -> None:
    (tmp_path / "QQQ_2026_09_01.json").write_text("{}")
    cache = make_cache(tmp_path)

    path = cache.put("QQQ", sample_df(), TODAY)

    assert [p.name for p in tmp_path.glob("QQQ_*")] == [path.name]
    pd.testing.assert_frame_equal(cache.read_file(path), sample_df(), check_freq=False)


def test_get_returns_none_when_stale(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    cache.put("SPY", sample_df(), date(2026, 10, 31))

    assert cache.get("SPY", date(2026, 10, 31)) is not None
    assert cache.get("SPY", TODAY) is None  # new calendar month


def test_entries_parse_tickers_with_separator(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    cache.put("BRK_B", sample_df(), TODAY)
    cache.put("BRK.B", sample_df(), TODAY)
    (tmp_path / "notes.json").write_text("{}")  # no date: ignored

    assert [(e.ticker, e.created) for e in cache.entries()] == [("BRK.B", TODAY), ("BRK_B", TODAY)]


def test_show_cache_finder_exact_then_partial(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    for t in ["BND", "BNDX", "SPY"]:
        cache.put(t, sample_df(), TODAY)
    finder = CachedSeriesFinder(cache)

    assert [e.ticker for e in finder.matches("bnd")] == ["BND"]
    assert [e.ticker for e in finder.matches("bn")] == ["BND", "BNDX"]
