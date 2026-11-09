#!/usr/local/bin/python3
"""
Tests for yf_ticker_map: holdings tickers are looked up on Yahoo Finance
under their mapped symbol, while data and cache stay keyed by the holdings ticker.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import holdings_cross_corr as hcc  # noqa: E402
from ticker_class import Ticker  # noqa: E402

DATE_FORMAT = "%Y_%m_%d"
TODAY = date(2026, 11, 9)


def sample_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-11-02", "2026-11-03"]), name="date")
    return pd.DataFrame({"adj_close": [100.0, 101.0], "log_return": [float("nan"), 0.00995]}, index=idx)


def test_config_loads_map_with_uppercase_keys(tmp_path: Path) -> None:
    script = tmp_path / "prog.py"
    (tmp_path / "prog.toml").write_text(
        'positions_prefix = "Positions_"\ninput_folder = "."\n'
        f'date_format = "{DATE_FORMAT}"\ncache_folder = "cache"\n'
        'yf_ticker_map = { "brk.b" = "BRK-B" }\n'
    )

    cfg = hcc.Config.load(script)

    assert cfg.yf_ticker_map == {"BRK.B": "BRK-B"}


def test_download_uses_mapped_yahoo_ticker(tmp_path: Path, monkeypatch) -> None:
    cfg = hcc.Config(
        positions_prefix="Positions_",
        input_folder=tmp_path,
        date_format=DATE_FORMAT,
        cache_folder=tmp_path / "cache",
        yf_ticker_map={"BRK.B": "BRK-B"},
    )
    requested: list[str] = []

    def fake_download(self: Ticker, years: int) -> pd.DataFrame:
        requested.append(self.yf_symbol)
        return sample_df()

    monkeypatch.setattr(hcc, "find_latest_positions_file", lambda c: (tmp_path / "p.xlsx", TODAY))
    monkeypatch.setattr(Ticker, "read_positions",
                        classmethod(lambda cls, f, col, m: [cls(s, m.get(s)) for s in ["BRK.B", "SPY"]]))
    monkeypatch.setattr(Ticker, "download", fake_download)

    data = hcc.load_holdings_data(cfg, today=TODAY)

    assert requested == ["BRK-B", "SPY"]
    assert list(data) == ["BRK.B", "SPY"]
    assert (tmp_path / "cache" / "BRK.B_2026_11_09.json").exists()
