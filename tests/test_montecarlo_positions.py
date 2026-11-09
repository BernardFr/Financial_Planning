#!/usr/local/bin/python3
"""
Tests for read_positions_market_values: the Monte Carlo portfolio is read by ticker
from the positions file (no mapping to asset classes).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parent.parent / "src" / "Using_Cross_Cor"
sys.path.insert(0, str(CODE_DIR))

from montecarlo_simulation_class import read_positions_market_values  # noqa: E402

TICKER_COLUMN = "Ticker"
VALUE_COLUMN = "Market Value"


def write_positions(path: Path, tickers: list, values: list[float]) -> Path:
    pd.DataFrame({TICKER_COLUMN: tickers, "Shares": [1] * len(tickers), VALUE_COLUMN: values}).to_excel(path, index=False)
    return path


def test_reads_market_value_by_ticker_and_drops_total(tmp_path: Path) -> None:
    positions = write_positions(tmp_path / "Positions_2026_11_04.xlsx",
                                [" vti", "BRK.B", None, "Total"], [100.0, 50.0, 7.0, 157.0])

    holdings_ser = read_positions_market_values(positions, TICKER_COLUMN, VALUE_COLUMN)

    assert list(holdings_ser.index) == ["VTI", "BRK.B"]
    assert list(holdings_ser) == [100.0, 50.0]


def test_duplicate_tickers_exit(tmp_path: Path) -> None:
    positions = write_positions(tmp_path / "Positions_2026_11_04.xlsx", ["VTI", "vti"], [1.0, 2.0])

    with pytest.raises(SystemExit):
        read_positions_market_values(positions, TICKER_COLUMN, VALUE_COLUMN)
