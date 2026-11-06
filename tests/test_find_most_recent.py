#!/usr/local/bin/python3
"""
Tests for find_most_recent - in particular prefixes that include a subdirectory
"""
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from find_most_recent import find_most_recent  # noqa: E402

DATE_FORMAT = "%Y_%m_%d"


def test_prefix_with_subdirectory_finds_most_recent_file(tmp_path: Path) -> None:
    sub_dir = tmp_path / "JPM"
    sub_dir.mkdir()
    for name in ["portfolio_2026_01_15.xlsx", "portfolio_2026_06_10.xlsx"]:
        (sub_dir / name).touch()
    (tmp_path / "portfolio_2026_12_31.xlsx").touch()  # top-level file must be ignored

    result, date_str = find_most_recent(str(tmp_path), "JPM/portfolio", DATE_FORMAT)

    assert result == sub_dir / "portfolio_2026_06_10.xlsx"
    assert date_str == "2026_06_10"


def test_prefix_with_dot_slash_subdirectory(tmp_path: Path) -> None:
    sub_dir = tmp_path / "JPM"
    sub_dir.mkdir()
    (sub_dir / "stats_2026_06_16.xlsx").touch()

    result, _ = find_most_recent(str(tmp_path), "./JPM/stats", DATE_FORMAT)

    assert result == sub_dir / "stats_2026_06_16.xlsx"


def test_plain_prefix_still_searches_top_level(tmp_path: Path) -> None:
    (tmp_path / "life_plan_2026_02_09.xlsx").touch()

    result, date_str = find_most_recent(str(tmp_path), "life_plan", DATE_FORMAT)

    assert result == tmp_path / "life_plan_2026_02_09.xlsx"
    assert date_str == "2026_02_09"


def test_xls_files_are_candidates(tmp_path: Path) -> None:
    for name in ["WFA_Portfolio_Positions_081425_2052.xlsx", "WFA_Portfolio_Positions_082426_2122.xls"]:
        (tmp_path / name).touch()

    result, date_str = find_most_recent(str(tmp_path), "WFA_Portfolio_Positions", "%m%d%y")

    assert result == tmp_path / "WFA_Portfolio_Positions_082426_2122.xls"
    assert date_str == "2026_08_24"


def test_non_excel_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "life_plan_2026_02_09.xlsx").touch()
    (tmp_path / "life_plan_2026_03_01.pdf").touch()

    result, _ = find_most_recent(str(tmp_path), "life_plan", DATE_FORMAT)

    assert result == tmp_path / "life_plan_2026_02_09.xlsx"


def test_no_dated_match_returns_none_instead_of_a_stale_file(tmp_path: Path) -> None:
    # A file dated in another format must not be picked up (it used to be returned, stamped with today's date)
    (tmp_path / "WFA_Portfolio_Positions_081425_2052.xlsx").touch()
    (tmp_path / "WFA_Portfolio_Positions.xlsx").touch()

    result, date_str = find_most_recent(str(tmp_path), "WFA_Portfolio_Positions", DATE_FORMAT)

    assert result is None
    assert date_str == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
