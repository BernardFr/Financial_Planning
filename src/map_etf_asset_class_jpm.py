#!/usr/local/bin/python3
"""
map_etf_Asset_class.py

Maps a list of ETFs to Morningstar Expanded asset classes using:
  1. ETF category from etfdb.com  (fast path — name-similarity match via Claude)
  2. ETF Database Themes table     (fallback if category alone is ambiguous)

Matching is delegated to Claude (model/prompt configured in etf_classifier.toml).
Results are written to a CSV file.

Requirements:
    pip install playwright anthropic openpyxl
    playwright install chromium

Usage:
    python map_etf_Asset_class.py                     # uses etf_classifier.toml
    python map_etf_Asset_class.py --config my.toml
    python map_etf_Asset_class.py --tickers SPY QQQ   # override ticker list
"""

import argparse
import json
import re
import sys
import time
import tomllib
from io import StringIO
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List
import pandas as pd
import anthropic
from openpyxl import Workbook, load_workbook
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
from yfinance import ticker
from configuration_manager_class import ConfigurationManager
from find_most_recent import find_most_recent
from logger import logger
from claude_model_manager import ClaudeModelManager, ModelSelectionError
from utilities import clean_excel_text

BASE_RESULT_FIELDS = [ "Ticker", "Category", "AssetClass", "Confidence", "Reasoning", "MatchStep" ]

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ETFData:
    Ticker:   str
    Category: str = ""
    Themes:   dict[str, str] = field(default_factory=dict)
    Error:    str = ""


@dataclass
class ClassificationResult:
    Ticker:       str
    Category:     str
    Themes:       dict[str, str]
    AssetClass:   str | None
    Confidence:   str
    Reasoning:    str
    MatchStep:    str   # "rules" | "category" | "themes" | "unmatched"


def get_claude_model(config_manager: ConfigurationManager) -> str:
    """Helper to resolve the Claude model to use from config, with error handling."""
    model_manager = ClaudeModelManager(config_manager)
    try:
        selected_model = model_manager.resolve_model()
        logger.info(f"Using Claude model: {selected_model}")
        return selected_model
    except ModelSelectionError as e:
        logger.error(f"[error] {e}")
        sys.exit(1)


class MapETFToAssetClass:
    """ MAP ETF positions to Asset Classes."""
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.script_directory = Path(__file__).resolve().parent
        self.input_directory = self.config["input_directory"]
        self.output_directory = self.config["output_directory"]
        self.output_file_prefix = self.config["output_file_prefix"]
        self.maps_date_format = self.config["maps_date_format"]

        self.positions_file_prefix = str(self.config["positions_file_prefix"])
        self.positions_date_format = str(self.config["positions_date_format"])
        # Read the existing mapped tickers to avoid re-processing
        self.mapped_file, self.mapped_date_str = find_most_recent(
            self.output_directory,
            filename_prefix=self.output_file_prefix,
            date_format=self.maps_date_format,
        )
        if self.mapped_file:
            logger.info(f"Most recent mapped file: {self.mapped_file} (date: {self.mapped_date_str})")
        else:
            logger.info(f"No existing mapped file found with prefix '{self.output_file_prefix}' in {self.output_directory}")

        return None
    

    # ── Find unmapped tickers in Positions ──────────────────────────────────────────

    def get_tickers(self) -> tuple[list[str], str, str]:
                
        positions_file, date_str = find_most_recent(
            self.input_directory,
            filename_prefix=self.positions_file_prefix.rstrip("*."),
            date_format=self.positions_date_format,
        )
        logger.info(f"Most recent positions file: {positions_file} (date: {date_str})")
        if not positions_file:
            logger.error(f"No positions file found with prefix '{self.positions_file_prefix}' in {self.input_directory}")
            sys.exit(1)

        # Load tickers from positions file
        try:
            df = clean_excel_text(pd.read_excel(positions_file, header=0, usecols=["Ticker"]))
            position_tickers = df["Ticker"].dropna().astype(str).str.strip().tolist()
            position_tickers = [t for t in position_tickers if t.lower() not in ['total', 'cash']]
            logger.info(f"Loaded position tickers:\n{position_tickers}")
        except Exception as e:        
            logger.info(f"Error loading tickers from {positions_file}: {e}")
            sys.exit(1)
        
        # Load mapped tickers from the existing results file to avoid re-mapping them.
        try:
            if self.mapped_file:
                self.df_mapped = clean_excel_text(pd.read_excel(self.mapped_file, header=0))
                # Mapped tickers are those already present in the results file, where the 'Asset_class' column is not empty.
                mapped_tickers = self.df_mapped[self.df_mapped["Asset_class"].notna()]["Ticker"].str.strip().tolist()
                logger.info(f"Loaded mapped tickers:\n{mapped_tickers}")
            else:
                mapped_tickers = []
            tickers_to_map = [t for t in position_tickers if t not in mapped_tickers]
        except Exception as e:
            logger.info(f"Error loading mapped tickers from {self.mapped_file}: {e}")
            sys.exit(1)

        if not tickers_to_map:  # All tickers in positions file are already mapped.
            logger.info(f"No unmapped tickers found in {positions_file} - {len(position_tickers)} total.")
        else:
            logger.info(f"Found {len(tickers_to_map)} unmapped tickers in {positions_file} (out of {len(position_tickers)} total).")
        return tickers_to_map, positions_file.name, date_str  
    
    
    def dedupe_results(self, in_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, bool]:
        """
        Check for mismatched (different asset classes) - and unmapped (empty asset class) tickers in the results DataFrame.
        Use cases:
        1. if there are tickers with only empty asset classes, leave the entries, but report error for manual review, 
        2. If there are duplicate tickers with different asset classes, log an error and return a flag indicating a dupe error, along with the duplicated entries for manual review. This indicates a data issue that needs to be resolved manually in the map_etf_Asset_class function before re-running the script, as it may lead to duplicate entries in the output file.    
        3. If there are duplicate tickers with the same asset class, keep the last one and drop the rest. Also drop any entries with the same ticker but empty asset class, if they exist.    

        Incorrect rows are removed from the output DataFrame and collected in a dupe_df for reporting. Asset class is set to:
        "UNMAPPED": No entry for the ticker has an asset class value (all are empty)
        "EMPTY": The ticker has one (or more)   entries with an empty asset class, but at least one entry with a non-empty asset class 
        "EXTRA": Multiple entries with the same ticker have the same non-empty asset class (keep only one, mark the rest as EXTRA)
        "MISMATCHED": 2 or more entries with the same ticker have different non-empty asset classes
        The function returns the cleaned output DataFrame, the dupe_df for reporting, and a count of dupe errors found. The

        dupe_error_count counts the number of tickers for which there are mismatched or unmapped entries, with the exception of tickers 
        that have both multiple entries with the same non-empty asset class and entries with empty asset class
        In this case, dupe_error_count += 2 for this ticker (1 for dupe identical asset class entries, 1 for empty asset class entries)

        """
        dupe_error_count = 0
        fatal_dupe_error_flag = False
        dupe_df = pd.DataFrame(columns=in_df.columns)
        out_df = in_df.copy(deep=True)

        if in_df["Ticker"].duplicated().any(): 
            duplicated_tickers = in_df[in_df["Ticker"].duplicated()]["Ticker"].tolist()
            logger.info(f"Duplicate tickers found in combined results: {duplicated_tickers}. Attempting to resolve duplicates ...")  
            # for each duplicated ticker, check if the Asset_class is the same for all entries with that ticker. If not, log a warning.
            for ticker in set(duplicated_tickers):
                ticker_entries = in_df[in_df["Ticker"] == ticker].copy(deep=True)
                # list unique non-empty asset classes for this ticker
                dupe_Asset_classes = ticker_entries["Asset_class"][
                    ticker_entries["Asset_class"].notna() & (ticker_entries["Asset_class"] != "")
                ].unique()
                #  count how many entries have empty asset class for this ticker
                empty_Asset_classes_count = ticker_entries["Asset_class"].isna().sum() + (ticker_entries["Asset_class"] == "").sum()
                if len(dupe_Asset_classes) == 0:  # Use case 1
                    dupe_error_count += 1
                    fatal_dupe_error_flag = True  # This indicates a data issue that needs to be resolved manually
                    # move all the rows with this ticker to the dupe_df for reporting, and drop them from the out_df
                    # Assign Category UNMAPPED to these entries in the dupe_df 
                    ticker_entries.loc[:, "Asset_class"] = "UNMAPPED"
                    dupe_df = pd.concat([dupe_df, ticker_entries], ignore_index=True)
                    out_df = out_df[out_df["Ticker"] != ticker]
                    logger.error (f"Ticker {ticker} is UNMAPPED")
                elif len(dupe_Asset_classes) > 1:  # Use case 2
                    dupe_error_count += 1  
                    fatal_dupe_error_flag = True  # This indicates a data issue that needs to be resolved manually
                    # move all the rows with this ticker to the dupe_df for reporting, and drop them from the out_df
                    ticker_entries.loc[:, "Asset_class"] = "MISMATCHED" # 
                    dupe_df = pd.concat([dupe_df, ticker_entries], ignore_index=True)
                    out_df = out_df[out_df["Ticker"] != ticker]
                    logger.error(f"Ticker {ticker} is MISMATCHED: multiple different asset classes: {dupe_Asset_classes}")
                elif len(dupe_Asset_classes) == 1:  # Use case 3  - Not fatal
                    if empty_Asset_classes_count > 0:
                        dupe_error_count += 1  
                        # move the rows with empty asset class for this ticker to the dupe_df for reporting, and drop them from the out_df
                        empty_entries = ticker_entries[ticker_entries["Asset_class"].isna() | (ticker_entries["Asset_class"] == "")].copy(deep=True)
                        empty_entries.loc[:, "Asset_class"] = "EMPTY" 
                        dupe_df = pd.concat([dupe_df, empty_entries], ignore_index=True)
                        out_df = out_df[~((out_df["Ticker"] == ticker) & (out_df["Asset_class"].isna() | (out_df["Asset_class"] == "")))]
                        logger.warning(f"Ticker {ticker} has {empty_Asset_classes_count} entries with empty asset class that will be dropped.")
                    # also move the non-empty duplicate entries to the dupe_df for reporting, and mark them as EXTRA, then drop them from the out_df
                    # Keep the last entry in out_df, and mark the rest as EXTRA in the dupe_df
                    non_empty_entries = ticker_entries[ticker_entries["Asset_class"].notna() & (ticker_entries["Asset_class"] != "")].copy(deep=True)
                    if len(non_empty_entries) > 1:
                        dupe_error_count += 1
                        extra_entries = non_empty_entries.iloc[:-1].copy()
                        extra_entries.loc[:, "Asset_class"] = "EXTRA"
                        dupe_df = pd.concat([dupe_df, extra_entries], ignore_index=True)
                        out_df = out_df[~((out_df["Ticker"] == ticker) & (out_df.index.isin(extra_entries.index)))]
                        logger.warning(f"Ticker {ticker} has {len(extra_entries)} duplicate entries with the same asset class that will be dropped.")

        return out_df, dupe_df, dupe_error_count, fatal_dupe_error_flag


    def save_results(self, results: list[ClassificationResult]) -> str:
        """ Save classification results to an Excel file, appending to existing file if it exists.
         If save_results is called, it means there are new mappings to save, so we create a new mappings file, with today's date. """
        if not results:
            logger.warning("No new results to save.")
            return ""
        previous_map_flag = True if self.mapped_file else False

        # Create new mapped file with today's date 
        date_str = time.strftime(self.maps_date_format)  # Today's date
        result_file  = str(Path(self.output_directory) / f"{self.output_file_prefix}_{date_str}.xlsx")
        logger.info(f"No existing mapped file found. Will create new file: {result_file}")

        # Keep output schema stable even though dataclass attributes are capitalized.
        new_results_df = pd.DataFrame([
            {
                "Ticker": r.Ticker,
                "Asset_class": r.AssetClass,
                "category": r.Category,
                "themes": r.Themes,
                "confidence": r.Confidence,
                "reasoning": r.Reasoning,
                "match_step": r.MatchStep,
            }
            for r in results
        ])
        # Make sure that new_results_df has the same columns as the existing mapped file, if it exists, to avoid issues when appending.
        if previous_map_flag:
                missing_cols = set(self.df_mapped.columns) - set(new_results_df.columns)
                for col in missing_cols:
                    new_results_df[col] = ""  # Add missing columns with empty values
                new_results_df = new_results_df[self.df_mapped.columns]  # Reorder columns to match existing file
                new_results_df = pd.concat([self.df_mapped, new_results_df], ignore_index=True)

        # check that all entries in new_results_df are unique by ticker, to avoid duplicates in the output file
        deduped_result_df, duplicates_df, dupe_error_count,fatal_error_flag = self.dedupe_results(new_results_df)

        # Save the successfully deduped results first, even if some tickers need manual review.
        try:
            deduped_result_df.to_excel(result_file, index=False)
            logger.info(f"Results saved → {result_file} (new file created with today's date: {date_str})")
        except Exception as e:
            logger.error(f"Error saving results to {result_file}: {e}")

        if fatal_error_flag > 0:
            logger.error(f"{dupe_error_count} Mismatched - or unmatched - tickers found in results: fix the map_etf_Asset_class MANUALLY and re-run this script")
            tmp = duplicates_df[duplicates_df["Asset_class"] == "MISMATCHED"]
            if len(tmp) > 0:
                logger.error(f"Mismatched tickers and their asset classes:\n{tmp[['Ticker', 'Asset_class']]}")
            tmp = duplicates_df[duplicates_df["Asset_class"] == "UNMAPPED"]
            if len(tmp) > 0:
                # just print each ticker once
                tmp = tmp.drop_duplicates(subset=["Ticker"])
                logger.error(f"Unmapped tickers:\n{tmp[['Ticker', 'Asset_class']]}")
            sys.exit("-- Exit due Mismatched or unmapped ETFs that require manual fix")

        return result_file

# ── Scraper ───────────────────────────────────────────────────────────────────
class ETFDataScraper:

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)   
        self.url_template = self.config["etfdb_url_template"]     
        self.wait_s    = self.config["page_load_wait_s"]
        self.timeout   = self.config["timeout_ms"]
        self.headless  = self.config["headless"]        
        return None
    

    def scrape_etf(self, ticker: str) -> ETFData:
        """Use Playwright to fetch category and themes from etfdb.com."""
        ticker_url  = self.url_template.replace("{ticker}", ticker.upper())

        data = ETFData(Ticker=ticker)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1440, "height": 900},
            )
            page = context.new_page()
            themes: dict[str, str] = {}

            try:
                page.goto(ticker_url, wait_until="domcontentloaded", timeout=self.timeout)
                time.sleep(self.wait_s)

                # ── Category (top profile section) ──────────────────────────────
                # etfdb renders "ETF Database Category" as a labeled field in the
                # vitals / profile table near the top of the page.
                category = ""

                def extract_category_value(text: str) -> str:
                    """Return only the category value, not the 'Category:' label."""
                    if not text:
                        return ""
                    t = " ".join(text.split()).strip()
                    m = re.search(r"(?:ETF\s+Database\s+)?Category\s*:\s*(.+)$", t, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
                    return "" if t.lower() in {"category", "category:", "etf database category", "etf database category:"} else t

                for sel in [
                    "td:has-text('ETF Database Category') + td",
                    "th:has-text('Category') + td",
                    "[data-th='Category']",
                    "span:has-text('Category')",
                ]:
                    try:
                        loc = page.locator(sel).first
                        if loc.count() > 0:
                            raw = (loc.text_content(timeout=3000) or "").strip()
                            category = extract_category_value(raw)

                            # If selector matched only the label (e.g., "Category:"),
                            # parse the nearest row/container where value appears on same line.
                            if not category:
                                for container_xpath in [
                                    "xpath=ancestor::tr[1]",
                                    "xpath=ancestor::li[1]",
                                    "xpath=ancestor::div[1]",
                                ]:
                                    container = loc.locator(container_xpath).first
                                    if container.count() == 0:
                                        continue
                                    container_text = (container.inner_text() or "").strip()
                                    category = extract_category_value(container_text)
                                    if category:
                                        break

                            if category:
                                break
                    except Exception:
                        pass

                # Broader fallback: scan all table cells for a "Category" label
                if not category:
                    rows = page.locator("tr").all()
                    for row in rows:
                        cells = row.locator("td,th").all_text_contents()
                        if len(cells) >= 2 and "category" in cells[0].lower():
                            category = cells[1].strip()
                            break

                data.Category = category

                # ── ETF Database Themes table ────────────────────────────────────

                # The "ETF Database Themes" section has a table/list of theme tags.
                # Common patterns: a <table> or <ul> following an h* "ETF Database Themes".
                try:
                    # Find the section header
                    section = page.locator(
                        "h3:has-text('ETF Database Themes'), "
                        "h3:has-text('ETF Database Themes'), "
                        "h4:has-text('ETF Database Themes'), "
                        "th:has-text('ETF Database Themes')"
                    ).first

                    if section.count() > 0:
                        # Grab the next sibling table or list
                        parent = section.locator("xpath=..").first
                        # Preferred: parse row-wise key/value pairs.
                        for tr in parent.locator("tr").all():
                            cells = [c.strip() for c in tr.locator("td,th").all_text_contents() if c.strip()]
                            if len(cells) >= 2:
                                key = cells[0].rstrip(":").strip()
                                value = cells[1].strip()
                                if key and value:
                                    themes[key] = value

                        # Fallback: parse flat sequence as key/value pairs.
                        if not themes:
                            flat = [t.strip() for t in parent.locator("td, li").all_text_contents() if t.strip()]
                            for i in range(0, len(flat) - 1, 2):
                                themes[flat[i].rstrip(":").strip()] = flat[i + 1].strip()

                    # Fallback: find any table row that mentions "Theme"
                    if not themes:
                        for row in page.locator("tr").all():
                            cells = row.locator("td,th").all_text_contents()
                            if len(cells) >= 2 and "theme" in cells[0].lower():
                                clean = [c.strip() for c in cells if c.strip()]
                                for i in range(0, len(clean) - 1, 2):
                                    key = clean[i].rstrip(":").strip()
                                    value = clean[i + 1].strip()
                                    if key and value:
                                        themes[key] = value
                                break
                        

                    # Second fallback: look for links/tags inside a themes-labelled div
                    if not themes:
                        div = page.locator(
                            "div:has-text('ETF Database Themes')"
                        ).last
                        if div.count() > 0:
                            raw_text = [
                                a.strip()
                                for a in div.locator("a, span, td").all_text_contents()
                                if a.strip() and "ETF Database Themes" not in a
                            ]

                            # Preferred path: each entry may contain a full line
                            # where first/second text elements are key/value.
                            for line in raw_text:
                                parts = [p.strip() for p in line.splitlines() if p.strip()]
                                if len(parts) >= 2:
                                    themes[parts[0].rstrip(":").strip()] = parts[1].strip()

                            # Fallback: flat list interpreted as key/value/value triplets.
                            # In this block, the value is repeated; keep only the first value.
                            if not themes:
                                for i in range(0, len(raw_text) - 2, 3):
                                    key = raw_text[i].rstrip(":").strip()
                                    value = raw_text[i + 1].strip()
                                    if key and value:
                                        themes[key] = value

                except Exception as e:
                    data.Error = f"themes scrape error: {e}"

            except PlaywrightTimeout:
                data.Error = f"Timeout loading {ticker_url}"
            except Exception as e:
                data.Error = str(e)
            finally:
                browser.close()

                # Cap to avoid noise
                data.Themes = dict(list(themes.items())[:20])

            return data



# ── Claude classifier ─────────────────────────────────────────────────────────

class ClaudeClassifier:
    """
    Use Claude to classify ETFs to Morningstar asset classes based on category and themes.    
    Returns (Asset_class, confidence, reasoning, match_step).
    match_step is  "category", "themes", or "unmatched".
    """
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)          
        self.client  = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.request_delay_s   = self.config.get("request_delay_s", 1)      # polite delay between ETF requests

        self.prompt_cfg = self.config_manager.get_class_config("Prompts")
        self.system_prompt = self.prompt_cfg["system_prompt"]
        self.full_match_prompt = self.prompt_cfg["full_match"]
        self.category_match_prompt = self.prompt_cfg["category_match"]
        self.jpm_categories = self._get_jpm_categories()

        try:
            self.claude_model = get_claude_model(config_manager)
            logger.info(f"Using Claude model: {self.claude_model}")
        except ModelSelectionError as e:
            logger.error(f"[error] {e}")
            sys.exit(1)
            return None
        
        # Insert the JPM categories into the prompts
        self.system_prompt = self.system_prompt.replace("{jpm_categories}", ", ".join(self.jpm_categories))
        logger.info(f"System prompt after inserting JPM categories:\n{self.system_prompt}")

    def _get_jpm_categories(self) -> List[str]:
        # Find the most recent JPM stats file to extract the list of valid JPM asset classes for prompt injection.
        stats_file, date_str = find_most_recent(
            self.config["jpm_directory"],
            filename_prefix=self.config["jpm_file_prefix"].rstrip("*."),
            date_format=self.config["jpm_date_format"])
        if not stats_file:
            logger.error(f"No JPM stats file found with prefix '{self.config['jpm_file_prefix']}' in {self.config['jpm_directory']}")
            sys.exit(1)
        logger.info(f"Most recent JPM stats file: {stats_file} (date: {date_str})")
        try:
            jpm_stats = clean_excel_text(pd.read_excel(stats_file, sheet_name=self.config["jpm_sheet_name"], header=0, index_col=0))
            idx = jpm_stats.index.tolist()
            logger.info(f"Extracted JPM asset classes for prompt injection:\n{idx}")
        except Exception as e:
            logger.error(f"Error extracting JPM categories from {stats_file}: {e}")
            sys.exit(1)
        
        return idx  

        

    def _classify_with_claude(self, etf_data: ETFData) -> tuple[str | None, str, str, str]:
        # ── Step 1: deterministic rules for obvious mappings ───────────────────
        logger.info(f"classify_with_claude:  {etf_data}")
        # rule_Asset_class, rule_reason = self._rule_based_Asset_class(etf_data)
        # if rule_Asset_class:
        #     return rule_Asset_class, "high", rule_reason, "rules"

        # ── Step 1: category-only (fast path) ────────────────────────────────────
        # if etf_data.Category:
        #     user_msg = self.category_match_prompt.format(
        #         ticker=etf_data.Ticker,
        #         category=etf_data.Category,
        #     )
        #     result = self._call_claude(user_msg)
        #     if result and result.get("asset_class") and result.get("confidence") != "low":
        #         return (
        #             result["asset_class"],
        #             result["confidence"],
        #             result.get("reasoning", ""),
        #             "category",
        #         )

        # ── Step 2: full context (themes fallback) ───────────────────────────────
        themes_str = ", ".join(
            f"{k}: {v}" for k, v in etf_data.Themes.items()
        ) if etf_data.Themes else "(none found)"

        # Keep compatibility with prompts that still reference {region}.
        region_hint = etf_data.Themes.get("Region (General)") or etf_data.Themes.get("Region (Specific)") or "(not found in themes)"
        user_msg = self.full_match_prompt.format(
            ticker=etf_data.Ticker,
            category=etf_data.Category or "",
            region=region_hint,
            themes=themes_str,
        )
        result = self._call_claude( user_msg)
        if result and result.get("asset_class"):
            return (
                result["asset_class"],
                result.get("confidence", "low"),
                result.get("reasoning", ""),
                "themes",
            )
        else:
            logger.info(f"  [warn] Claude could not classify {etf_data.Ticker} with category or themes")
            logger.info(f"user_msg:\n{user_msg}")

        return None, "none", "Claude could not determine a match.", "unmatched"


    def _call_claude(self, user_msg: str ) -> dict | None:
        """Call the Claude API and parse the JSON response."""
        response_text = ""
        try:
            response = self.client.messages.create(
                model=self.claude_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            # Extract text from the first TextBlock in response content
            response_text = next(
                (block.text for block in response.content if block.type == "text"),
                None
            )
            if response_text is None:
                raise ValueError("No text content in response")
            response_text = response_text.strip()

            # Strip markdown code fences if present
            response_text = response_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.info(f"  [warn] JSON parse error: {e}  Raw: {response_text[:200] if response_text else 'None'}")
            return None
        except anthropic.APIError as e:
            logger.info(f"  [error] Claude API error: {e}")
            return None
        

    def run(self,etf_scraper,tickers_to_map: list[str]) -> list[ClassificationResult]:
        """ Run the classification process for a list of tickers, including scraping and Claude classification.
        Returns a list of ClassificationResult objects. """
        
        results : list[ClassificationResult] = []
        unmatched_count = 0

        for i, ticker in enumerate(tickers_to_map):
            ticker = ticker.upper()
            logger.info(f"[{i+1}/{len(tickers_to_map)}] {ticker}")

            # 1 — Scrape etfdb.com
            logger.info(f"  Scraping etfdb.com…")
            etf_data = etf_scraper.scrape_etf(ticker)
            if etf_data.Error:
                logger.warning(f"{etf_data.Error}")

            logger.info(f"  Category : {etf_data.Category or '(none)'}")
            theme_items = list(etf_data.Themes.items())
            theme_preview = ", ".join(f"{k}: {v}" for k, v in theme_items[:5]) if theme_items else "(none)"
            logger.info(f"  Themes   : {theme_preview}{'…' if len(theme_items) > 5 else ''}")

            # 2 — Classify with Claude
            logger.info(f"  Classifying…")
            Asset_class, confidence, reasoning, step = self._classify_with_claude( etf_data )

            result = ClassificationResult(
                Ticker=ticker,
                Category=etf_data.Category,
                Themes=etf_data.Themes,
                AssetClass=Asset_class,
                Confidence=confidence,
                Reasoning=reasoning,
                MatchStep=step,
            )
            results.append(result)

            if step == "unmatched":
                print_unmatched(result)
            else:
                icon = "✓" if Asset_class else "✗"
                logger.info(f"  {icon} {Asset_class or 'UNMATCHED'}  [{confidence}]  via {step}")
                logger.info(f"    → {reasoning}")

            if step == "unmatched":
                unmatched_count += 1
                # if unmatched_count >= 3:
                #     logger.info("\nStopping early: reached 3 unmatched ETFs.")
                #     break

            # Polite delay between ETFs
            if i < len(tickers_to_map) - 1:
                time.sleep(self.request_delay_s)

        return results


# ── Output helpers ────────────────────────────────────────────────────────────

unmatched_etfs = []  # Collect unmatched ETF messages for later review.

def print_unmatched(result: ClassificationResult):
    """ Nicely format unmatched results for easier debugging and prompt iteration.
    Save this message and output at program completion for review for all unmatched ETFs. """
    themes_text = ", ".join(f"{k}: {v}" for k, v in result.Themes.items()) if result.Themes else "(none)"
    unmatched_msg = (
        "\n" + "═" * 60 + "\n"
        f"  ⚠  UNMATCHED: {result.Ticker}\n"
        f"  Category : {result.Category or '(none)'}\n"
        f"  Themes   : {themes_text}\n"
        f"  Reasoning: {result.Reasoning}\n"
        + "═" * 60 + "\n"
    )
    unmatched_etfs.append(unmatched_msg)
    logger.info(unmatched_msg)

def map_positions_to_Asset_classes(positions_file: str, etf_Asset_class_map_file: str, date_str: str) -> str:
    """Map the Positions to their asset classes, aggregate market values by asset class, save to output file, and print summary of asset class distribution."""
    try:
        positions_df = clean_excel_text(pd.read_excel(positions_file, header=0))
        # remove the "Total" row if it exists, which can interfere with mapping and aggregation
        if positions_df["Ticker"].str.lower().eq("total").any():
            positions_df = positions_df[~positions_df["Ticker"].str.lower().eq("total")]
        logger.info(f"Total Market value in positions file: ${positions_df['Market Value'].sum():,.0f}")
        map_df = clean_excel_text(pd.read_excel(etf_Asset_class_map_file, header=0))
        logger.info(f"Mapping positions to asset classes using positions file: {positions_file}, map file: {etf_Asset_class_map_file}")
        logger.info(f"{len(positions_df)} positions found, {len(map_df)} mapped ETFs found")
        merged_df = positions_df.merge(map_df[["Ticker", "Asset_class"]], on="Ticker", how="left")
        merged_df["Asset_class"] = merged_df["Asset_class"].fillna("UNMAPPED")
        # List any tickers in positions that did not get mapped to an asset class, for review
        unmapped_tickers = merged_df[merged_df["Asset_class"] == "UNMAPPED"]["Ticker"].tolist()
        if unmapped_tickers:
            logger.info(f"Unmapped tickers: {', '.join(unmapped_tickers)}")
        logger.info(f"After merging,    {merged_df['Asset_class'].isna().sum()} positions have UNMAPPED asset class")
        logger.info(f"Asset class distribution in merged data:\n{merged_df['Asset_class'].value_counts(dropna=False)}")

        # Aggregate market values by asset class
        agg_df = merged_df.groupby("Asset_class")["Market Value"].sum().reset_index()

        # Save to output file
        output_file = str(Path(positions_file).parent / f"portfolio_by_Asset_class_{date_str}.xlsx")
        agg_df.to_excel(output_file, index=False)
        logger.info(f"Portfolio by asset class saved → {output_file}")

        # Print summary of asset class distribution
        logger.info("\nAsset Class Distribution:")
        for _, row in agg_df.iterrows():
            logger.info(f"  {row['Asset_class']}: ${row['Market Value']:,.2f}")
        logger.info(f"  Total: ${agg_df['Market Value'].sum():,.0f}")

        return output_file

    except Exception as e:
        logger.error(f"Error mapping positions to asset classes: {e}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cmd_line: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Map ETFs to Morningstar asset classes.")
    parser.add_argument("--tickers", nargs="+", help="Override ticker list from config")
    args = parser.parse_args()
    from_cli_tickers = args.tickers is not None
    if from_cli_tickers:  # remover --tickers from cmd_line for config manager
        cmd_line = [arg for arg in cmd_line if arg not in ("--tickers",) and arg not in args.tickers]  
        logger.info(f"Tickers provided via command line: {', '.join(args.tickers)}")

    config_manager = ConfigurationManager(cmd_line)
    etf_mapper = MapETFToAssetClass(config_manager)
    etf_scraper = ETFDataScraper(config_manager)
    claude_classifier = ClaudeClassifier(config_manager)    


    # 1. Get tickers to classify (either from CLI or from positions file)
    if from_cli_tickers:
        tickers_to_map = [t.upper() for t in args.tickers]
        logger.info(f"Classifying {len(tickers_to_map)} tickers from command line: {', '.join(tickers_to_map)}")
    else:
        tickers_to_map, positions_file, date_str = etf_mapper.get_tickers()
        print(f"Found {len(tickers_to_map)} unique tickers from positions file: {positions_file} (as of {date_str})")

    if not tickers_to_map:
        logger.info("No tickers to classify")
    else:  # classify the new tickers
        # 2. Classify the tickers with Claude (with scraping in same loop)
        results = claude_classifier.run(etf_scraper, tickers_to_map)

        # 3 — Save results (skip for ad-hoc CLI ticker runs)
        if from_cli_tickers:
            logger.info("\nSkipping file save because tickers were provided via --tickers.")
        else:
            etf_Asset_class_map_file = etf_mapper.save_results(results)    

        # 4 — Print unmatched messages (if any) before exit
        if unmatched_etfs:
            logger.info("\nUnmatched ETFs:")
            logger.info("".join(unmatched_etfs))

        # 5 — Summary
        matched   = sum(1 for r in results if r.AssetClass)
        unmatched = len(results) - matched
        logger.info(f"\nSummary: {matched} matched, {unmatched} unmatched out of {len(results)} ETFs.")

    # Map the Positions to their asset classes,and aggregate market values by asset class, save to output file, and print summary of asset class distribution.
    # find the most recent mapped file to use for mapping positions to asset classes
    if not from_cli_tickers:
        etf_Asset_class_map_file, _ = find_most_recent(etf_mapper.output_directory, etf_mapper.output_file_prefix)
        positions_file, _ = find_most_recent(etf_mapper.input_directory, etf_mapper.positions_file_prefix, etf_mapper.positions_date_format)
        portfolio_by_Asset_class_file = map_positions_to_Asset_classes(positions_file, etf_Asset_class_map_file, date_str) # pyright: ignore[reportPossiblyUnboundVariable, reportArgumentType]
        logger.info(f"Done! Portfolio by asset class saved to: {portfolio_by_Asset_class_file}")


if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
