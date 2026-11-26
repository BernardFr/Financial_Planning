#!/usr/local/bin/python3
"""
Compute the market value of the portfolio for today.

The program:
    1. Loads the holdings (symbols and shares) and the cash amount with DataLoader.
    2. Gets the last close price of each symbol from Yahoo Finance.
    3. Writes the holdings with prices to a sheet named with today's date.
    4. Adds today's total to the 'Market Value' history sheet.
    5. Logs today's value and compares it with the historical min and max.
    6. Asks the user if they want to plot the market value over time.

All the modules that this program imports have the prefix 'mv_'.

Classes:
    - MarketValueProcessor: Main orchestrator class
"""

import datetime as dt
import sys
import traceback
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from mv_configuration_manager_class import ConfigurationManager
from mv_custom_logger import custom_logger as logger, setup_logger
from mv_data_loader_class import DataLoader
from mv_utils import clean_excel_text, get_prog_name, to_dollar_str

MARKET_VALUE_SHEET = 'Market Value'
MARKET_VALUE_COL = 'Market Value'
DATE_COL = 'Date'
CASH_SYMBOL = 'Cash'
DATE_FORMAT = '%Y-%m-%d'
PLOT_FIG_SIZE = (10, 6)
PLOT_PROMPT = "Plot market value over time? (y/n): "
DEBUG_FLAG = False


class MarketValueProcessor:
    """Main orchestrator class for the market value computation."""

    def __init__(self, cmd_line: List[str]):
        self.prog_name = get_prog_name()
        self.config_manager = ConfigurationManager(self.prog_name, cmd_line)
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.data_loader = DataLoader(self.config_manager)

    @staticmethod
    def get_quote(symbol: str) -> float | None:
        """
        Get the last close price of a symbol from Yahoo Finance.
        Return None if no price is available.
        """
        try:
            history_df = yf.Ticker(symbol).history(period='1d')
            if history_df.empty:
                logger.error(f"Error getting quote for {symbol}: no data returned")
                return None
            return float(history_df['Close'].iloc[-1])
        except Exception as e:  # yfinance raises many exception types on network or data errors
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    def _get_price_dict(self, symbol_lst: List[str]) -> dict[str, float]:
        """Map each symbol to its last close price. Use 0.0 for a symbol without a quote."""
        price_dict = {}
        symbols_with_errors = []
        for symbol in symbol_lst:
            price = self.get_quote(symbol)
            if price is None:
                symbols_with_errors.append(symbol)
                price = 0.0
            price_dict[symbol] = price

        if symbols_with_errors:
            logger.error(f"symbols_with_errors: {symbols_with_errors}")
        if DEBUG_FLAG:
            logger.info(f"symbol_to_price: { {k: to_dollar_str(v) for k, v in price_dict.items()} }")
        return price_dict

    def process(self) -> None:
        """Main processing method."""
        cumul_df, symbol_lst, cash_amount = self.data_loader.load_all_transaction_data()

        # Compute the market value of each symbol, then add a row for cash
        cumul_df['Price'] = cumul_df['Symbol'].map(self._get_price_dict(symbol_lst))
        cumul_df['Sub_Total'] = cumul_df['Price'] * cumul_df['Shares']
        if DEBUG_FLAG:
            logger.info(f"cumul_df:\n{cumul_df.head()}")
        cash_row = pd.DataFrame([{'Symbol': CASH_SYMBOL, 'Sub_Total': cash_amount}])
        cumul_df = pd.concat([cumul_df, cash_row], ignore_index=True)

        today_market_value = cumul_df['Sub_Total'].sum()
        market_value_df = self._save_results(cumul_df)
        self._log_summary(market_value_df, today_market_value)

        if input(PLOT_PROMPT).strip().lower() == 'y':
            self._plot_market_value(market_value_df)

    def _save_results(self, cumul_df: pd.DataFrame) -> pd.DataFrame:
        """
        Write cumul_df to a sheet named with today's date.
        Add today's total to the market value history sheet, and return the history.
        """
        market_value = cumul_df['Sub_Total'].sum()
        today = dt.datetime.now().strftime(DATE_FORMAT)
        xl_path = Path(self.config_manager.output_paths['xl_out_filename'])

        # Replace today's sheet if it already exists
        if xl_path.is_file():
            writer = pd.ExcelWriter(xl_path, mode='a', if_sheet_exists='replace', engine='openpyxl')
        else:
            writer = pd.ExcelWriter(xl_path, mode='w', engine='openpyxl')
        with writer:
            cumul_df.to_excel(writer, sheet_name=today, index=False)

        # Add or update today's row in the market value history
        with pd.ExcelFile(xl_path) as xls:
            if MARKET_VALUE_SHEET in xls.sheet_names:
                market_value_df = clean_excel_text(pd.read_excel(xls, sheet_name=MARKET_VALUE_SHEET))
            else:
                market_value_df = None
        if market_value_df is None:
            market_value_df = pd.DataFrame({DATE_COL: [today], MARKET_VALUE_COL: [market_value]})
        elif today in market_value_df[DATE_COL].values:
            market_value_df.loc[market_value_df[DATE_COL] == today, MARKET_VALUE_COL] = market_value
        else:
            market_value_df.loc[len(market_value_df)] = {DATE_COL: today, MARKET_VALUE_COL: market_value}
        market_value_df = market_value_df.sort_values(by=DATE_COL, ignore_index=True)

        with pd.ExcelWriter(xl_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            market_value_df.to_excel(writer, sheet_name=MARKET_VALUE_SHEET, index=False)

        return market_value_df

    @staticmethod
    def _log_summary(market_value_df: pd.DataFrame, today_market_value: float) -> None:
        """Log the market value history, and compare today's value with the historical min and max."""
        display_df = market_value_df[[DATE_COL, MARKET_VALUE_COL]].copy()
        display_df[MARKET_VALUE_COL] = display_df[MARKET_VALUE_COL].map(to_dollar_str)
        logger.info(f"Market Values:\n{display_df}")

        value_series = market_value_df[MARKET_VALUE_COL]
        min_idx = value_series.idxmin()
        min_market_value = float(value_series.min())
        min_date = market_value_df.at[min_idx, DATE_COL]
        delta_min = today_market_value - min_market_value
        delta_min_percent = delta_min / today_market_value * 100

        max_idx = value_series.idxmax()
        max_market_value = float(value_series.max())
        max_date = market_value_df.at[max_idx, DATE_COL]
        delta_max = max_market_value - today_market_value
        delta_max_percent = delta_max / today_market_value * 100

        if today_market_value <= min_market_value:
            new_str = "NEW LOW: "
        elif today_market_value >= max_market_value:
            new_str = "NEW HIGH: "
        else:
            new_str = ""
        logger.info(f"Today's market value: ${today_market_value:,.0f}")
        logger.info(f"{new_str} Min Market Value: ${min_market_value:,.0f} - Delta (${delta_min:,.0f}) ({delta_min_percent:.2f}%) on {min_date}")
        logger.info(f"{new_str} Max Market Value: ${max_market_value:,.0f} - Delta (${delta_max:,.0f}) ({delta_max_percent:.2f}%) on {max_date}")

    @staticmethod
    def _plot_market_value(market_value_df: pd.DataFrame) -> None:
        """
        Plot the market value over time.
        The dates are not evenly spaced, so the x-axis is a time axis: markers connected by lines.
        """
        # Dates are stored as 'YYYY-MM-DD' strings: convert them so matplotlib uses a time axis, not categories
        dates = pd.to_datetime(market_value_df[DATE_COL])
        _, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
        ax.scatter(dates, market_value_df[MARKET_VALUE_COL], marker='o')
        ax.plot(dates, market_value_df[MARKET_VALUE_COL], linestyle='-')
        ax.set_title('Market Value Over Time')
        ax.set_xlabel(DATE_COL)
        ax.set_ylabel('Market Value ($)')
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid()
        plt.tight_layout()
        plt.show()


def main(cmd_line: List[str]) -> None:
    """Run the market value computation, log the run time, and exit with a status code."""
    setup_logger(Path(__file__).stem)
    processor = MarketValueProcessor(cmd_line)
    exit_code = 0
    start_time = dt.datetime.now()
    try:
        processor.process()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received.")
        exit_code = 1
    except Exception as e:
        logger.error(f"\nError: {e}")
        logger.error(traceback.format_exc())
        logger.error("\n----\n")
        exit_code = -2
    finally:
        end_time = dt.datetime.now()
        logger.info(f'End: {str(end_time)} -- Run Time: {str(end_time - start_time)}\n')
    sys.exit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
