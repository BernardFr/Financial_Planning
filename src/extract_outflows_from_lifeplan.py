#!/usr/local/bin/python3
"""
Extract Total Withdrawals by year from the latest Life Plan PDF.
"""

import re
import sys
import datetime as dt
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pdfplumber as _pdfplumber  # type: ignore[import-not-found]

pdfplumber: Any = _pdfplumber

from configuration_manager_class import ConfigurationManager
from logger import logger


class ExtractorClass:
    """Extracts yearly withdrawals and exports them to an Excel file."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager: ConfigurationManager = config_manager
        self.config: dict[str, Any] = self.config_manager.get_class_config(self.__class__.__name__)

        self.year_col_pattern: re.Pattern[str] = re.compile(str(self.config["year_col_pattern"]))
        self.pdf_name_pattern: re.Pattern[str] = re.compile(str(self.config["pdf_filename_pattern"]))

        self.section_title: str = str(self.config["section_title"])
        self.section_subtitle: str = str(self.config["section_subtitle"])
        self.table_header: str = str(self.config["table_header"])
        self.table_row_name: str = str(self.config["table_row_name"])

        self.input_directory: Path = Path(self.config["input_directory"]).expanduser().resolve()
        self.output_directory: Path = Path(self.config["output_directory"]).expanduser().resolve()
        self.output_file_prefix: str = str(self.config["output_file_prefix"])
        return None

    def _find_latest_pdf(self) -> tuple[Path | None, dt.date | None]:
        """Return latest matching PDF path and date parsed from filename."""
        latest_pdf: Path | None = None
        latest_date: dt.date | None = None

        for pdf_path in self.input_directory.glob("*.pdf"):
            if pdf_path.name.startswith("fraenkel"):
                print(f"{pdf_path.name}")
            match: re.Match[str] | None = self.pdf_name_pattern.fullmatch(pdf_path.name)
            if not match:
                continue

            mo: int
            dy: int
            yr_2digit: int
            mo, dy, yr_2digit = map(int, match.groups())
            yr: int = 2000 + yr_2digit
            plan_date = dt.date(yr, mo, dy)
            if latest_date is None or plan_date > latest_date:
                latest_date = plan_date
                latest_pdf = pdf_path

        return latest_pdf, latest_date

    def _currency_str_2_float(self, raw_value: str) -> float:
        """Convert a currency-like string into a float."""
        clean_value: str = (
            raw_value.replace("$", "")
            .replace(",", "")
            .replace("(", "-")
            .replace(")", "")
            .strip()
        )
        if clean_value in {"", "-", "--"}:
            return 0.0
        return float(clean_value)

    def _extract_withdrawals(self, pdf_path: Path) -> pd.DataFrame:
        """Return a one-row DataFrame with integer year columns and float withdrawals."""
        results: dict[int, float] = {}
        with pdfplumber.open(pdf_path) as pdf:
            in_section: bool = False
            pages: list[Any] = cast(list[Any], pdf.pages)
            for page in pages:
                raw_text: Any = page.extract_text()
                text: str = str(raw_text) if raw_text is not None else ""

                if not in_section:
                    if self.section_title in text and self.section_subtitle in text and self.table_header in text:
                        in_section = True
                    else:
                        continue

                if self.table_header not in text:
                    break

                years: list[str] = self.year_col_pattern.findall(text)
                if not years:
                    continue

                tables: list[Any] = cast(list[Any], page.extract_tables())
                for table in tables:
                    for row in table:
                        if row and str(row[0]).strip() == self.table_row_name:
                            values: list[str] = [str(cell) for cell in row[1:] if cell is not None]
                            for year, value in zip(years, values):
                                results[int(year)] = self._currency_str_2_float(value)  # year is int, value is float
                            break

        if not results:
            return pd.DataFrame()

        result_df: pd.DataFrame = pd.DataFrame([results])
        result_df = result_df.reindex(sorted(result_df.columns), axis=1)
        return result_df.astype(float)

    def _save_to_excel(self, data: pd.DataFrame, date_str: str) -> Path:
        """Save year -> withdrawals as a single-row spreadsheet."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / f"{self.output_file_prefix}_{date_str}.xlsx"

        data.to_excel(output_path, index=False)
        return output_path

    def run(self) -> None:
        logger.info(f"Scanning directory: {self.input_directory}")
        pdf_path, plan_date = self._find_latest_pdf()
        if pdf_path is None or plan_date is None:
            raise FileNotFoundError("No matching Life Plan PDF files found.")

        date_str: str = plan_date.strftime("%Y_%m_%d")
        logger.info(f"Latest plan: {pdf_path.name} ({date_str})")

        data: pd.DataFrame = self._extract_withdrawals(pdf_path)
        if data.empty:
            raise ValueError("No Total Withdrawals row found in PDF.")

        output_path: Path = self._save_to_excel(data, date_str)
        logger.info(f"Saved: {output_path}")
        logger.info(f"Years extracted: {len(data.columns)} ({min(data.columns)}-{max(data.columns)})")
        return None


def main(cmd_line: list[str]) -> None:
    config_manager: ConfigurationManager = ConfigurationManager(cmd_line)
    extractor: ExtractorClass = ExtractorClass(config_manager)
    extractor.run()
    return None


if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
