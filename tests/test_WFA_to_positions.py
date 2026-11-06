#!/usr/local/bin/python3
"""
Tests for WFAtoPositionsClass: positions are extracted from every configured section (ETFs and Stocks)
"""
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from WFA_to_positions import WFAtoPositionsClass  # noqa: E402

HEADER = ["Description", "Symbol", "Effective Yield", "Shares", "Market Value"]
BLANK = [None] * len(HEADER)
POSITIONS_SECTIONS = {"ETFs": "Total ETFs", "Stocks": "Total Stocks"}


class FakeConfigManager:
    """Minimal stand-in for ConfigurationManager"""
    def __init__(self, positions_sections: dict[str, str]) -> None:
        self.positions_sections = positions_sections

    def get_class_config(self, class_name: str) -> dict[str, Any]:
        return {
            "input_directory": ".", "holdings_file_prefix": "WFA", "holdings_date_format": "%m%d%y",
            "output_directory": ".", "output_file_prefix": "Positions", "ETF_for_cash": "SHY",
            "cash_balance_description": "Total Cash", "positions_sections": self.positions_sections,
            "total_portfolio_label": "Total Portfolio",
        }


def section_rows(label: str, total_label: str, holdings: list[list[Any]]) -> list[list[Any]]:
    """A section as WFA lays it out: label row, header row, holdings rows, total row"""
    return [[label] + BLANK[1:], HEADER, *holdings, [total_label] + BLANK[1:], BLANK]


def make_raw_df(include_stocks: bool = True) -> pd.DataFrame:
    rows: list[list[Any]] = []
    if include_stocks:
        rows += section_rows("Stocks", "Total Stocks", [
            ["Common Stock", None, None, None, None],
            ["BERKSHIRE HATHAWAY INC", "BRK.B", "Detail", "56", "32,429.04"],  # summary of the lots below
            ["BERKSHIRE HATHAWAY INC", "BRK.B", None, "52", "30,112.68"],
            ["BERKSHIRE HATHAWAY INC", "BRK.B", None, "4", "2,316.36"],
        ])
    rows += section_rows("ETFs", "Total ETFs", [
        ["VANGUARD TOTAL STOCK", "VTI", None, "10", "3,000.00"],
        ["ISHARES 1-3 YR TREASURY", "SHY", None, "5", "400.00"],
    ])
    return pd.DataFrame(rows, dtype=object)


def make_converter(positions_sections: dict[str, str] = POSITIONS_SECTIONS) -> WFAtoPositionsClass:
    return WFAtoPositionsClass(FakeConfigManager(positions_sections))  # type: ignore[arg-type]


def test_positions_include_stocks_and_etfs() -> None:
    positions = make_converter()._extract_positions_table(make_raw_df()).set_index("Ticker")

    assert sorted(positions.index) == ["BRK.B", "SHY", "VTI"]
    assert positions.loc["BRK.B", "Shares"] == 56  # lots only - the 'Detail' summary row is not double counted
    assert positions.loc["BRK.B", "Market Value"] == pytest.approx(32_429.04)
    assert positions.loc["VTI", "Market Value"] == pytest.approx(3_000.00)


def test_missing_section_is_skipped() -> None:
    positions = make_converter()._extract_positions_table(make_raw_df(include_stocks=False))

    assert sorted(positions["Ticker"]) == ["SHY", "VTI"]


def test_no_section_found_raises() -> None:
    with pytest.raises(ValueError, match="None of the position sections were found"):
        make_converter({"Bonds": "Total Bonds"})._extract_positions_table(make_raw_df())


def test_missing_total_row_raises() -> None:
    raw_df = make_raw_df()
    raw_df = raw_df[raw_df.iloc[:, 0] != "Total Stocks"].reset_index(drop=True)
    with pytest.raises(ValueError, match="'Total Stocks' not found after 'Stocks'"):
        make_converter()._extract_positions_table(raw_df)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
