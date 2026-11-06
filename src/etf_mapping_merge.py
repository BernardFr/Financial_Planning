#!/usr/local/bin/python3
"""
Shared logic for the ETF -> asset class mappers (map_etf_asset_class.py and map_etf_asset_class_jpm.py):
- ask whether to re-use the existing mappings or refresh them all
- merge new classifications into the existing mappings, asking which to keep when they differ
- apply fixed mappings from the configuration (e.g. individual stocks such as BRK.B, which are not ETFs)

A mapping table is a DataFrame with (at least) the columns 'Ticker' and 'Asset_class'.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from logger import logger

REUSE_ANSWERS = ("", "y", "yes")       # Enter defaults to re-using the existing mappings
REFRESH_ANSWERS = ("r", "refresh")
KEEP_OLD_ANSWERS = ("", "o", "old")     # Enter defaults to keeping the existing mapping
KEEP_NEW_ANSWERS = ("n", "new")
FIXED_MATCH_STEP = "fixed"
FIXED_REASONING = "Fixed mapping from configuration (fixed_mappings)"

# (ticker, existing asset class, new asset class, new reasoning) -> True to keep the new mapping
KeepNewMapping = Callable[[str, str, str, str], bool]


def _has_asset_class(value: object) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def _ask(prompt: str, first_answers: tuple[str, ...], second_answers: tuple[str, ...], retry_message: str) -> bool:
    """Ask until the answer is in first_answers (-> True) or second_answers (-> False)"""
    while True:
        answer = input(prompt).strip().lower()
        if answer in first_answers:
            return True
        if answer in second_answers:
            return False
        print(retry_message)


def ask_reuse_existing_mappings(mapped_file: Path) -> bool:
    """Ask whether to re-use the existing mappings (True) or re-classify every ETF (False).
    Non-interactive runs (no terminal on stdin) re-use the existing mappings."""
    if not sys.stdin.isatty():
        logger.info(f"Non-interactive run: re-using existing mappings from {mapped_file.name}")
        return True
    prompt = (f"Existing ETF -> asset class mappings found in {mapped_file.name}.\n"
              f"  [Y] re-use them and classify only new ETFs (default)\n"
              f"  [r] refresh: re-classify every ETF in the positions file with Claude\n"
              f"Choice [Y/r]: ")
    return _ask(prompt, REUSE_ANSWERS, REFRESH_ANSWERS, "Please answer 'y' (re-use) or 'r' (refresh).")


def ask_keep_new_mapping(ticker: str, old_class: str, new_class: str, new_reasoning: str) -> bool:
    """Ask which mapping to keep when a re-classification differs from the existing one: True keeps the new one.
    Non-interactive runs keep the existing mapping."""
    if not sys.stdin.isatty():
        logger.info(f"Non-interactive run: {ticker} keeps existing mapping {old_class!r} (new: {new_class!r})")
        return False
    prompt = (f"\n{ticker}: new mapping differs from the existing one\n"
              f"  existing: {old_class}\n"
              f"  new:      {new_class}  ({new_reasoning})\n"
              f"Keep [O]ld (default) or [n]ew? ")
    return not _ask(prompt, KEEP_OLD_ANSWERS, KEEP_NEW_ANSWERS, "Please answer 'o' (old) or 'n' (new).")


def merge_mappings(previous_df: pd.DataFrame, new_df: pd.DataFrame, keep_new: KeepNewMapping) -> pd.DataFrame:
    """Merge new classifications into the existing mappings - one decision per ticker in new_df:
    - new ticker, or existing row without an asset class: the new row is added
    - same asset class as before: the new row replaces the existing one
    - different asset class: keep_new() decides which row is kept
    - Claude could not classify it (no asset class): the existing mapping is kept
    Existing tickers that are not in new_df are kept unchanged."""
    previous_classes = {
        ticker: asset_class for ticker, asset_class in zip(previous_df["Ticker"], previous_df["Asset_class"])
        if _has_asset_class(asset_class)
    }
    replaced_tickers: set[str] = set()
    accepted_rows = []
    for _, row in new_df.iterrows():
        ticker, new_class = row["Ticker"], row["Asset_class"]
        old_class = previous_classes.get(ticker)
        if not _has_asset_class(new_class):
            if old_class is None:
                accepted_rows.append(row)  # no mapping at all: keep the row so it is reported as unmapped
            continue
        if old_class is None or old_class == new_class or keep_new(ticker, old_class, new_class, str(row.get("reasoning", ""))):
            accepted_rows.append(row)
            replaced_tickers.add(ticker)
    kept_previous_df = previous_df[~previous_df["Ticker"].isin(replaced_tickers)]
    accepted_df = pd.DataFrame(accepted_rows, columns=new_df.columns)
    return pd.concat([kept_previous_df, accepted_df], ignore_index=True)


def apply_fixed_mappings(mapping_df: pd.DataFrame, fixed_mappings: dict[str, str]) -> pd.DataFrame:
    """Set each fixed ticker's asset class, replacing any existing row for that ticker"""
    if not fixed_mappings:
        return mapping_df
    fixed_df = pd.DataFrame({column: [""] * len(fixed_mappings) for column in mapping_df.columns})
    fixed_df["Ticker"] = list(fixed_mappings)
    fixed_df["Asset_class"] = list(fixed_mappings.values())
    for column, value in {"confidence": "high", "reasoning": FIXED_REASONING, "match_step": FIXED_MATCH_STEP}.items():
        if column in fixed_df.columns:
            fixed_df[column] = value
    kept_df = mapping_df[~mapping_df["Ticker"].isin(fixed_mappings)]
    return pd.concat([kept_df, fixed_df], ignore_index=True)
