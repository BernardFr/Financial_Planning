#!/usr/local/bin/python3
"""
Tests for the ETF -> asset class mappers (Morningstar and JPM versions) and their shared etf_mapping_merge module:
re-use vs refresh, merging new classifications, asking old vs new on differences, and fixed mappings (BRK.B)
No Claude API calls or web scraping: classification results are built directly
"""
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import etf_mapping_merge  # noqa: E402
import map_etf_asset_class  # noqa: E402
import map_etf_asset_class_jpm  # noqa: E402
from etf_mapping_merge import (  # noqa: E402
    FIXED_MATCH_STEP, apply_fixed_mappings, ask_keep_new_mapping, ask_reuse_existing_mappings, merge_mappings,
)

MAP_PREFIX = "etf_asset_class_map"
EXISTING_MAP_NAME = f"{MAP_PREFIX}_2026_06_11.xlsx"
MAP_COLUMNS = ["Ticker", "Asset_class", "category", "themes", "confidence", "reasoning", "match_step"]
POSITION_TICKERS = ["VTI", "BND", "NEWETF", "BRK.B"]
FIXED_MAPPINGS = {"BRK.B": "Large Value"}
MAPPER_MODULES = [map_etf_asset_class, map_etf_asset_class_jpm]


class FakeConfigManager:
    """Minimal stand-in for ConfigurationManager, pointing every directory at tmp_path"""
    def __init__(self, directory: Path) -> None:
        self.directory = str(directory)

    def get_class_config(self, class_name: str) -> dict[str, Any]:
        return {
            "input_directory": self.directory, "output_directory": self.directory,
            "output_file_prefix": MAP_PREFIX, "maps_date_format": "%Y_%m_%d",
            "positions_file_prefix": "Positions", "positions_date_format": "%Y_%m_%d",
            "fixed_mappings": FIXED_MAPPINGS,
        }


def map_row(ticker: str, asset_class: str | None, reasoning: str = "previous run") -> dict[str, Any]:
    return {"Ticker": ticker, "Asset_class": asset_class, "category": "", "themes": "",
            "confidence": "high", "reasoning": reasoning, "match_step": "themes"}


def keep_new_always(*_args: str) -> bool:
    return True


def keep_old_always(*_args: str) -> bool:
    return False


def make_mapper(module: ModuleType, directory: Path) -> Any:
    pd.DataFrame({"Ticker": POSITION_TICKERS + ["Total"], "Market Value": [1.0, 2.0, 3.0, 4.0, 10.0]}).to_excel(
        directory / "Positions_2026_11_04.xlsx", index=False)
    pd.DataFrame([map_row("VTI", "Large Blend"), map_row("BND", "Intermediate Core Bond")], columns=MAP_COLUMNS).to_excel(
        directory / EXISTING_MAP_NAME, index=False)
    return module.MapETFToAssetClass(FakeConfigManager(directory))


def result(module: ModuleType, ticker: str, asset_class: str | None) -> Any:
    return module.ClassificationResult(Ticker=ticker, Category="", Themes={}, AssetClass=asset_class,
                                       Confidence="high", Reasoning="this run", MatchStep="themes")


def read_map(saved_file: str) -> pd.Series:
    return pd.read_excel(saved_file).set_index("Ticker")["Asset_class"]


# ── Mappers (both versions) ─────────────────────────────────────────────────────

@pytest.mark.parametrize("module", MAPPER_MODULES, ids=lambda m: m.__name__)
def test_reuse_classifies_only_unmapped_non_fixed_tickers(module: ModuleType, tmp_path: Path) -> None:
    tickers, _, _ = make_mapper(module, tmp_path).get_tickers(reuse_existing=True)
    assert tickers == ["NEWETF"]


@pytest.mark.parametrize("module", MAPPER_MODULES, ids=lambda m: m.__name__)
def test_refresh_classifies_every_non_fixed_ticker(module: ModuleType, tmp_path: Path) -> None:
    tickers, _, _ = make_mapper(module, tmp_path).get_tickers(reuse_existing=False)
    assert tickers == ["VTI", "BND", "NEWETF"]  # BRK.B is never sent to Claude


@pytest.mark.parametrize("module", MAPPER_MODULES, ids=lambda m: m.__name__)
def test_reuse_adds_new_mappings_to_existing_file(module: ModuleType, tmp_path: Path) -> None:
    mapper = make_mapper(module, tmp_path)
    mapper.get_tickers(reuse_existing=True)
    saved_file = mapper.save_results([result(module, "NEWETF", "Small Value")], keep_new=keep_old_always)

    assert Path(saved_file) == tmp_path / EXISTING_MAP_NAME
    assert read_map(saved_file).to_dict() == {
        "VTI": "Large Blend", "BND": "Intermediate Core Bond", "NEWETF": "Small Value", "BRK.B": "Large Value"}


@pytest.mark.parametrize("module", MAPPER_MODULES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("keep_new, expected_vti", [(keep_new_always, "Large Growth"), (keep_old_always, "Large Blend")])
def test_refresh_writes_new_file_and_asks_on_differences(module: ModuleType, tmp_path: Path, keep_new: Any, expected_vti: str) -> None:
    asked: list[str] = []

    def recording_keep_new(ticker: str, old_class: str, new_class: str, reasoning: str) -> bool:
        asked.append(ticker)
        return keep_new(ticker, old_class, new_class, reasoning)

    mapper = make_mapper(module, tmp_path)
    mapper.get_tickers(reuse_existing=False)
    saved_file = mapper.save_results(
        [result(module, "VTI", "Large Growth"), result(module, "BND", "Intermediate Core Bond"), result(module, "NEWETF", "Small Value")],
        keep_new=recording_keep_new)

    assert Path(saved_file) != tmp_path / EXISTING_MAP_NAME  # refresh: new dated file, existing one untouched
    assert asked == ["VTI"]  # only the ticker whose asset class changed
    assert read_map(saved_file).to_dict() == {
        "VTI": expected_vti, "BND": "Intermediate Core Bond", "NEWETF": "Small Value", "BRK.B": "Large Value"}


@pytest.mark.parametrize("module", MAPPER_MODULES, ids=lambda m: m.__name__)
def test_positions_use_fixed_mapping_even_if_map_file_lacks_it(module: ModuleType, tmp_path: Path) -> None:
    mapper = make_mapper(module, tmp_path)
    out_file = module.map_positions_to_Asset_classes(
        str(tmp_path / "Positions_2026_11_04.xlsx"), str(tmp_path / EXISTING_MAP_NAME), "2026_11_04", mapper.fixed_mappings)

    by_class = pd.read_excel(out_file).set_index("Asset_class")["Market Value"]
    assert by_class["Large Value"] == 4.0


# ── Shared merge logic ──────────────────────────────────────────────────────────

def test_merge_keeps_previous_mapping_when_new_classification_failed() -> None:
    previous = pd.DataFrame([map_row("VTI", "Large Blend")])
    new = pd.DataFrame([map_row("VTI", None, "this run")])

    merged = merge_mappings(previous, new, keep_new_always)

    assert merged.set_index("Ticker")["Asset_class"].to_dict() == {"VTI": "Large Blend"}


def test_merge_same_class_takes_new_row_without_asking() -> None:
    previous = pd.DataFrame([map_row("VTI", "Large Blend")])
    new = pd.DataFrame([map_row("VTI", "Large Blend", "this run")])

    merged = merge_mappings(previous, new, lambda *_: pytest.fail("must not ask when the class is unchanged"))

    assert merged["reasoning"].tolist() == ["this run"]


def test_merge_fills_previously_empty_mapping() -> None:
    previous = pd.DataFrame([map_row("VTI", None)])
    new = pd.DataFrame([map_row("VTI", "Large Blend", "this run")])

    merged = merge_mappings(previous, new, lambda *_: pytest.fail("no existing class to compare with"))

    assert merged.set_index("Ticker")["Asset_class"].to_dict() == {"VTI": "Large Blend"}


def test_fixed_mapping_replaces_existing_row() -> None:
    mapping = pd.DataFrame([map_row("BRK.B", "Large Growth"), map_row("VTI", "Large Blend")], columns=MAP_COLUMNS)

    fixed = apply_fixed_mappings(mapping, FIXED_MAPPINGS).set_index("Ticker")

    assert fixed.loc["BRK.B", "Asset_class"] == "Large Value"
    assert fixed.loc["BRK.B", "match_step"] == FIXED_MATCH_STEP
    assert fixed.loc["VTI", "Asset_class"] == "Large Blend"


# ── Prompts ─────────────────────────────────────────────────────────────────────

def answer_with(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    answer_iter = iter(answers)
    monkeypatch.setattr(etf_mapping_merge.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answer_iter))


@pytest.mark.parametrize("answers, expected", [
    ([""], True), (["y"], True), (["YES"], True), (["r"], False), (["Refresh"], False), (["maybe", "r"], False),
])
def test_reuse_prompt_answers(monkeypatch: pytest.MonkeyPatch, answers: list[str], expected: bool) -> None:
    answer_with(monkeypatch, answers)
    assert ask_reuse_existing_mappings(Path("map.xlsx")) is expected


@pytest.mark.parametrize("answers, expected", [
    ([""], False), (["o"], False), (["OLD"], False), (["n"], True), (["New"], True), (["x", "n"], True),
])
def test_keep_new_prompt_answers(monkeypatch: pytest.MonkeyPatch, answers: list[str], expected: bool) -> None:
    answer_with(monkeypatch, answers)
    assert ask_keep_new_mapping("VTI", "Large Blend", "Large Growth", "reason") is expected


def test_non_interactive_run_reuses_and_keeps_old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(etf_mapping_merge.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("must not prompt without a terminal"))

    assert ask_reuse_existing_mappings(Path("map.xlsx")) is True
    assert ask_keep_new_mapping("VTI", "Large Blend", "Large Growth", "reason") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
