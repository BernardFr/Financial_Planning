#!/usr/local/bin/python3
"""Standalone validation script for MapETFToAssetClass.dedupe_results.

This script creates a copy of Data/etf_asset_class_map_2026_06_04.xlsx as a test seed,
adds scenario-specific rows (including normal entries), then validates dedupe behavior
for:
1) Duplicate ticker with only empty asset classes.
2) Duplicate ticker with different asset classes.
3a) Duplicate ticker with same asset class.
3b) Duplicate ticker with same asset class plus empty rows.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import pandas as pd
from yfinance import ticker
from yfinance import ticker

from map_etf_asset_class import MapETFToAssetClass
from find_most_recent import find_most_recent
from logger import logger

PRINT_COL = ["ticker", "asset_class", "category"]  # columns to include in printouts for debugging


@dataclass
class ScenarioInput:
    ticker: str
    asset_class: str | None
    category: str = "test-category"
    expected_result: str = "ERROR"  # "VALID", "EMPTY", "MISMATCH", "EXTRA" 

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    message: str


# def _build_row(
#     ticker: str,
#     asset_class: str | None,
#     category: str = "test-category",
#     confidence: str = "high",
#     reasoning: str = "test reasoning",
#     match_step: str = "rules",
#     themes: dict[str, str] | None = None,
# ) -> dict[str, object]:
#     return {
#         "ticker": ticker,
#         "category": category,
#         "themes": themes or {},
#         "asset_class": asset_class,
#         "confidence": confidence,
#         "reasoning": reasoning,
#         "match_step": match_step,
#     }

def _build_row(scenario_input: ScenarioInput) -> dict[str, object]:
    return {
        "ticker": scenario_input.ticker,
        "category": scenario_input.category,
        "themes":  {},  # empty dict for themes since it's not relevant to these scenarios
        "asset_class": scenario_input.asset_class,
        "confidence": "test",
        "reasoning": "test",    
        "match_step": "test",
    }

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "ticker",
        "category",
        "themes",
        "asset_class",
        "confidence",
        "reasoning",
        "match_step",
    ]
    out = df.copy(deep=True)
    for col in expected:
        if col not in out.columns:
            out[col] = ""
    return out

# Create a list of ScenarioInput objects for each test scenario,

TestScenarios = [
    ScenarioInput(ticker="NORMAL_A", asset_class="Cash", expected_result="VALID"),
    ScenarioInput(ticker="NORMAL_B", asset_class="Commodities", expected_result="VALID"),
    ScenarioInput(ticker="UC1_EMPTY", asset_class="", expected_result="UNMAPPED"),
    ScenarioInput(ticker="UC1_EMPTY", asset_class=None, expected_result="UNMAPPED"),
    ScenarioInput(ticker="UC2_MIXED", asset_class="Cash", expected_result="MISMATCH"),
    ScenarioInput(ticker="UC2_MIXED", asset_class="Real Estate", expected_result="MISMATCH"),
    ScenarioInput(ticker="UC3_SAME", asset_class="US Large Cap Growth", expected_result="VALID"),
    ScenarioInput(ticker="UC3_SAME", asset_class="US Large Cap Growth", expected_result="EXTRA"),
    ScenarioInput(ticker="UC3_MIX_EMPTY", asset_class="", expected_result="EMPTY"),
    ScenarioInput(ticker="UC3_MIX_EMPTY", asset_class="US Mid Cap Value", expected_result="VALID"),
    ScenarioInput(ticker="UC3_MIX_EMPTY", asset_class=None, expected_result="EMPTY"),
    ScenarioInput(ticker="UC3_MIX_EMPTY", asset_class="US Mid Cap Value", expected_result="EXTRA"),
    ScenarioInput(ticker="UC3_MIX_EMPTYSTR", asset_class="", expected_result="EMPTY"),
    ScenarioInput(ticker="UC3_MIX_EMPTYSTR", asset_class="US Mid Cap Value", expected_result="VALID"),
    ScenarioInput(ticker="UC3_MIX_EMPTYSTR", asset_class="US Mid Cap Value", expected_result="EXTRA"),
]


def _seed_test_file(repo_root: Path) -> tuple[pd.DataFrame, Path]:
    data_dir = repo_root / "Data" # data_dir = repo_root/Data
    src_file, _ = find_most_recent( directory_path=str(data_dir), filename_prefix="etf_asset_class_map", date_format="%Y_%m_%d" )
    if src_file is None or not Path(src_file).exists():
        raise FileNotFoundError(f"Seed file not found: {src_file}")

    artifacts_dir = data_dir / "dedupe_test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dest_file = artifacts_dir / "etf_asset_class_map_dedupe_test.xlsx"
    shutil.copy2(src_file, dest_file)

    seed_df = pd.read_excel(dest_file, header=0)
    seed_df = _ensure_columns(seed_df)

    # Keep only one row per existing ticker so scenario outcomes are driven by
    # the explicit test rows added below, not by historical duplicates.
    if "ticker" in seed_df.columns:
        seed_df = seed_df.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)

    return seed_df, dest_file

def _add_scenario_rows(in_df: pd.DataFrame, scenario_inputs: list[ScenarioInput]) -> pd.DataFrame:
    # create a DataFrame from the list of ScenarioInput objects
    scenario_rows = pd.DataFrame([_build_row(s) for s in scenario_inputs])
    return pd.concat([in_df, scenario_rows], ignore_index=True)


def _run_dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, bool]:
    mapper = MapETFToAssetClass.__new__(MapETFToAssetClass)
    out_df, dupe_df, dupe_error_count, fatal_error_flag = mapper.dedupe_results(df)
    return out_df, dupe_df, dupe_error_count, fatal_error_flag



def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    try:
        seed_df, copied_file = _seed_test_file(repo_root)
    except Exception as exc:
        print(f"ERROR: failed to create seed test file: {exc}")
        return 2

    print(f"Seed workbook copied to: {copied_file}")
    print(f"Seed rows loaded: {len(seed_df)}")
    test_df = _add_scenario_rows(seed_df, TestScenarios)
    out_df, dupe_df, dupe_error_count, fatal_error_flag = _run_dedupe(test_df)
    print(f"Dedupe completed. Output rows: {len(out_df)}, Duplicates found: {len(dupe_df)}, Dupe count: {dupe_error_count}, Fatal Error flag: {fatal_error_flag}")
    print("\nOutput DataFrame:")
    print(out_df[PRINT_COL])
    print("\nDuplicates DataFrame:")
    print(dupe_df[PRINT_COL])

    #  add assertions or checks to validate that the output matches expected results for each scenario.
    # Make sure NORMAL_A and NORMAL_B are in the output with their asset classes intact
    valid_as_expected = 0
    error_as_expected = 0
    error_error = 0
    for scenario in TestScenarios:
        ticker = scenario.ticker
        expected = scenario.expected_result
        if expected == "VALID":
            if ticker in out_df["ticker"].values and out_df[out_df["ticker"] == ticker]["asset_class"].iloc[0] == scenario.asset_class:
                valid_as_expected += 1
            else:
                error_error += 1
                print(f"ERROR: Scenario {ticker} expected VALID but was not found with correct asset class in output.")
        elif expected == "EMPTY":
            if ticker in dupe_df["ticker"].values and "EMPTY" in dupe_df[dupe_df["ticker"] == ticker]["asset_class"].values:
                error_as_expected += 1
            else:
                error_error += 1
                print(f"ERROR: Scenario {ticker} expected EMPTY but was not found with empty asset class in duplicates.")
        elif expected == "MISMATCH":
            if ticker in dupe_df["ticker"].values and not all(dupe_df[dupe_df["ticker"] == ticker]["asset_class"].isin(["", None])):
                error_as_expected += 1
            else:
                error_error += 1
                print(f"ERROR: Scenario {ticker} expected MISMATCH but was not found with mismatched asset classes in duplicates.")
        elif expected == "EXTRA":
            # count_in_output = len(out_df[out_df["ticker"] == ticker])
            if ticker in dupe_df["ticker"].values and "EXTRA" in dupe_df[dupe_df["ticker"] == ticker]["asset_class"].values:
                error_as_expected += 1
            # if count_in_output != 1:
            #     error_error += 1
            #     print(f"ERROR: Scenario {ticker} expected EXTRA but was found {count_in_output} times in output instead of exactly once.")
            else:     
                error_error += 1
                print(f"ERROR: Scenario {ticker} expected EXTRA but was NOT found in dupes")

    print(f"\nSummary: {valid_as_expected} scenarios passed as expected\n{error_as_expected} error scenarios identified correctly")
    if error_error > 0:
        print(f"FAIL: {error_error} scenarios did not match expected results.")
    else:
        print("SUCCESS: All scenarios matched expected results.")



if __name__ == "__main__":
    raise SystemExit(main())
