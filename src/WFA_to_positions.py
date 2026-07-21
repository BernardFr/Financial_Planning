#!/usr/local/bin/python3

"""
Find the most recent WFA holdings file and convert it to a positions spreadsheet.
"""

import sys
from pathlib import Path
from typing import cast

import pandas as pd

from configuration_manager_class import ConfigurationManager
from find_most_recent import find_most_recent
from logger import logger
from utilities import clean_excel_text


def dollar_str(x: float) -> str:
    """Return a currency string with commas and 2 decimal places."""
    return f"$ {x:,.2f}"


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object or series.dtype.name == "string":
        cleaned = series.astype("string").str.replace(r"[\$,]", "", regex=True)
    else:
        cleaned = series
    return pd.to_numeric(cleaned, errors="coerce")


class WFAtoPositionsClass:
    """Convert the latest WFA holdings file into a positions file."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.script_directory = Path(__file__).resolve().parent
        self.input_directory = self._resolve_path(self.config["input_directory"])
        self.holdings_file_prefix = str(self.config["holdings_file_prefix"])
        self.output_directory = self._resolve_path(self.config["output_directory"])
        self.output_file_prefix = str(self.config["output_file_prefix"])
        self.etf_for_cash = str(self.config["ETF_for_cash"])
        self.cash_balance_description = str(self.config["cash_balance_description"])
        self.positions_section_label = str(self.config["positions_section_label"])
        self.total_etf_label = str(self.config["total_etf_label"])
        self.total_portfolio_label = str(self.config["total_portfolio_label"])
        self.validation_tolerance = float(self.config.get("validation_tolerance", 0.01))
        return None

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.script_directory / path).resolve()

    def _find_cash_balance(self, raw_df: pd.DataFrame) -> float:
        normalized = raw_df.map(lambda value: "" if pd.isna(value) else str(value).strip())
        header_idx: int | None = None
        for idx, row in enumerate(normalized.itertuples(index=False, name=None)):
            if {"Description", "Account Number", "Market Value"}.issubset(set(row)):
                header_idx = idx
                break
        if header_idx is None:
            raise ValueError("Cash Balance table header not found (Description/Account Number/Market Value).")

        header = list(normalized.iloc[header_idx].tolist())
        table = raw_df.iloc[header_idx + 1 :].copy()
        table.columns = header
        cash_rows = table[_normalize_text(table["Description"]) == self.cash_balance_description]
        if cash_rows.empty:
            raise ValueError(f"Cash Balance row not found (Description={self.cash_balance_description!r}).")

        cash_value = _to_numeric(cash_rows["Market Value"]).iloc[0]
        if pd.isna(cash_value):
            raise ValueError("Cash Balance Market Value is not numeric.")
        return float(cash_value)

    def _find_total_portfolio_row(self, raw_df: pd.DataFrame) -> float:
        normalized = raw_df.map(lambda value: "" if pd.isna(value) else str(value).strip())
        total_portfolio_rows = normalized.apply(lambda row: row.eq(self.total_portfolio_label).any(), axis=1)
        if not total_portfolio_rows.any():
            raise ValueError(f"Row labeled {self.total_portfolio_label!r} not found.")

        total_portfolio_idx: int = int(total_portfolio_rows[total_portfolio_rows].index[0])
        market_value_row = cast(pd.Series, normalized.iloc[total_portfolio_idx + 1])
        if not market_value_row.astype(str).eq("Market Value").any():
            raise ValueError("Row labeled 'Market Value' not found.")

        total_portfolio_row = cast(pd.Series, normalized.iloc[total_portfolio_idx + 2])
        if not total_portfolio_row.astype(str).eq(self.total_portfolio_label).any():
            raise ValueError(f"Row labeled {self.total_portfolio_label!r} not found.")
        return float(_to_numeric(total_portfolio_row).iloc[1])

    def _extract_positions_table(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        normalized = raw_df.map(lambda value: "" if pd.isna(value) else str(value).strip())
        etf_rows = normalized.apply(lambda row: row.eq(self.positions_section_label).any(), axis=1)
        if not etf_rows.any():
            raise ValueError(f"Row labeled {self.positions_section_label!r} not found.")

        etf_idx: int = int(etf_rows[etf_rows].index[0])
        header_idx: int = etf_idx + 1
        header = list(raw_df.iloc[header_idx].tolist())

        total_etf_rows = normalized.apply(lambda row: row.eq(self.total_etf_label).any(), axis=1)
        if not total_etf_rows.any():
            raise ValueError(f"Row labeled {self.total_etf_label!r} not found.")

        total_etf_idx: int = int(total_etf_rows[total_etf_rows].index[0])
        raw_df = raw_df.iloc[: total_etf_idx + 1]
        positions = raw_df.iloc[header_idx + 1 :].copy()
        positions.columns = header
        positions = positions.dropna(how="all")

        required_columns = {"Symbol", "Shares", "Market Value"}
        if not required_columns.issubset(set(positions.columns)):
            raise ValueError(f"Missing required columns: {sorted(required_columns - set(positions.columns))}")

        for col in ["Effective Yield", "Tax Terms"]:
            if col in positions.columns:
                mask = _normalize_text(positions[col]).str.upper().isin(["DETAIL", "N/A", "NA"])
                positions = positions[~mask]

        positions["Shares"] = _to_numeric(positions["Shares"])
        positions["Market Value"] = _to_numeric(positions["Market Value"])
        positions = positions.dropna(subset=["Symbol"])
        positions = positions[["Symbol", "Shares", "Market Value"]].rename(columns={"Symbol": "Ticker"})
        positions["Market Value"] = positions["Market Value"].astype(float)
        return positions.groupby("Ticker", as_index=False).sum()

    def _apply_cash_balance(self, positions_df: pd.DataFrame, cash_balance: float) -> pd.DataFrame:
        positions_df = positions_df.copy()
        if self.etf_for_cash in positions_df["Ticker"].values:
            positions_df.loc[positions_df["Ticker"] == self.etf_for_cash, "Market Value"] += cash_balance
        else:
            positions_df = pd.concat(
                [
                    positions_df,
                    pd.DataFrame(
                        [{"Ticker": self.etf_for_cash, "Shares": 0.0, "Market Value": cash_balance}]
                    ),
                ],
                ignore_index=True,
            )
        return positions_df.sort_values("Ticker").reset_index(drop=True)

    def _add_total_row(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        total_value = positions_df["Market Value"].sum()
        total_row = pd.DataFrame([{"Ticker": "Total", "Shares": 0.0, "Market Value": total_value}])
        if positions_df.empty:
            return total_row
        total_row = total_row.astype(
            {col: positions_df[col].dtype for col in total_row.columns if col in positions_df.columns},
            errors="ignore",
        )
        return pd.concat([positions_df, total_row], ignore_index=True)

    def run(self) -> None:
        logger.info(f"Scanning directory: {self.input_directory}")
        holdings_file, date_str = find_most_recent(str(self.input_directory), self.holdings_file_prefix)
        if holdings_file is None:
            raise FileNotFoundError(
                f"No holdings file found in {self.input_directory} with prefix {self.holdings_file_prefix!r}."
            )

        logger.info(f"Using holdings file: {holdings_file.name}")
        raw_df = clean_excel_text(pd.read_excel(holdings_file, header=None, dtype=object))
        cash_balance = self._find_cash_balance(raw_df)
        logger.info(f"Cash Balance: {dollar_str(cash_balance)}")

        positions_df = self._extract_positions_table(raw_df)
        positions_df = self._apply_cash_balance(positions_df, cash_balance)
        positions_df["$ Market Value"] = positions_df["Market Value"].map(dollar_str)

        logger.info('Positions after cash balance:\n%s', positions_df[["Ticker", "Shares", "$ Market Value"]])

        positions_df = self._add_total_row(positions_df)
        total_rows = positions_df.loc[positions_df["Ticker"] == "Total", "Market Value"]
        if total_rows.empty:
            raise ValueError("Total row not found in positions output.")

        computed_total_value = float(total_rows.iloc[0])
        total_portfolio_value = float(self._find_total_portfolio_row(raw_df))
        logger.info(f"Total Portfolio Value: {dollar_str(total_portfolio_value)}")
        logger.info(f"Computed Total Value: {dollar_str(computed_total_value)}")

        delta = computed_total_value - total_portfolio_value
        logger.info(f"Delta: {dollar_str(delta)}")
        if abs(delta) > self.validation_tolerance:
            raise ValueError(
                f"Computed Total Value {computed_total_value} does not match Total Portfolio Value {total_portfolio_value}"
            )

        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / (self.output_file_prefix + f"_{date_str}.xlsx")
        positions_df["$ Market Value"] = positions_df["Market Value"].map(dollar_str)
        positions_df.to_excel(output_path, index=False)
        logger.info(f"Saved positions to: {output_path}")
        return None


def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    positions = WFAtoPositionsClass(config_manager)
    positions.run()
    return None


if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
