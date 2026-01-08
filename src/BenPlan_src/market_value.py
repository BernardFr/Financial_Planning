#!/usr/bin/env python3
"""
Object-Oriented BenPlan - Financial Analysis Tool

This module provides an object-oriented implementation of the BenPlan financial analysis system.
It processes transaction data from multiple sources, categorizes expenses, and generates reports.

Classes:
    - BenPlanProcessor: Main orchestrator class
    - DataLoader: Handles loading and processing transaction data
    - DataCleaner: Handles data cleaning and categorization
    - ReportGenerator: Handles report generation and Excel output
    - VisualizationManager: Handles plotting and visualization
    - ConfigurationManager: Handles configuration and settings
"""

import datetime as dt
import os
import re
import sys
import traceback
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

import yfinance as yf
import openpyxl

                             
from utils import get_prog_name, to_dollar_str
from mv_configuration_manager_class import ConfigurationManager
from mv_data_loader_class import DataLoader
from logger import logger

MARKET_VALUE_SHEET = 'Market Value'
MARKET_VALUE_COL = 'Market Value'
DEBUG_FLAG = False

class MarketValueProcessor:
    """Main orchestrator class for Market Value processing.
    Each processing step is a separate function returning a DataFrame
    the main function then calls the plotting functions - if applicable
    and calls the async writing functions
    """

    def __init__(self, cmd_line: List[str]):
        self.prog_name = get_prog_name()
        self.config_manager = ConfigurationManager(self.prog_name, cmd_line)
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.data_loader = DataLoader(self.config_manager)
        # self.data_cleaner = DataCleaner(self.config_manager)
        self.open_files = set()


    def _log_info_tick(self, msg: str, xtra_log_msg = None) -> None:
        """Log info and tick timer"""
        if xtra_log_msg is not None:
            logger.info(f"{msg}\n{xtra_log_msg}")
        else:
            logger.info(msg)
        self.config_manager._tick_timer(msg)
        return

    def get_last_day_close(self,ticker_symbol):
        """Get the last day close price for a given ticker symbol using yfinance """
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(period='1d')  # Fetch data for the last 1 day
        if not data.empty:
            return data['Close'].iloc[-1]  # Get the last closing price
        else:
            return None


    def get_quote(self,symbol: str) -> float:
        """Get the last day close price for a given ticker symbol 
        Returns None if error getting the quote """
        try:
            last_close = float(self.get_last_day_close(symbol))
            return last_close
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    
    def process(self) -> None:
        """Main processing method."""

        # TODO: move this to the ConfigurationManager class
        # Get current directory and setup paths
        paths = self.config_manager.output_paths
        # Setup output files
        self._open_output_files()
        xl_outf = self.config_manager.output_paths['xl_out_filename']
        self.xl_outf = xl_outf  # FIXME: this is a hack to get the xl_outf into the class


        # Load and process data
        cumul_df, symbols_list, cash_amount = self.data_loader.load_all_transaction_data()

        # Make a map of symbol to price
        symbol_to_price = {}
        symbols_with_errors = []
        for symbol in symbols_list:
            price = self.get_quote(symbol)
            if price is not None:
                symbol_to_price[symbol] = price
            else:
                symbols_with_errors.append(symbol)
                symbol_to_price[symbol] = 0.0

        if len(symbols_with_errors) > 0:
            logger.error(f"symbols_with_errors: {symbols_with_errors}")
        symbol_to_price_str = {k: f"${v:,.2f}" for k, v in symbol_to_price.items()}
        if DEBUG_FLAG:
            logger.info(f"symbol_to_price: {symbol_to_price_str}")

        # Add the price to the cumul_df
        cumul_df['Price'] = cumul_df['Symbol'].map(symbol_to_price)
        # Compute the market value of each symbol
        cumul_df['Sub_Total'] = cumul_df['Price'] * cumul_df['Shares']
        print(f"cumul_df: {cumul_df.head()}")

        # Add a row for cash: where 'Symbol' is 'Cash' and 'Sub_Total' is cash_amount
        cash_row = pd.DataFrame([{'Symbol': 'Cash', 'Sub_Total': cash_amount}])
        cumul_df = pd.concat([cumul_df, cash_row], ignore_index=True)

        # Compute the total market value
        market_value = cumul_df['Sub_Total'].sum()
        print(f"Market Value: ${market_value:,.0f}")

        self._save_results(cumul_df)

    def _save_results(self, cumul_df:pd.DataFrame) -> None:
        """Save the results to a file"""
        market_value = cumul_df['Sub_Total'].sum()
        # get today's date as a YYYY-MM-DD string
        today = dt.datetime.now().strftime('%Y-%m-%d')
        # save results to a file
        xl_outf = self.config_manager.output_paths['xl_out_filename']
        # check if xl_outf exists
        if Path(xl_outf).is_file():
            write_mode = 'a'
            # Use 'replace' to overwrite the sheet in case it already exists
            with pd.ExcelWriter(xl_outf, mode=write_mode, if_sheet_exists='replace', engine='openpyxl') as writer:
                cumul_df.to_excel(writer, sheet_name=today, index=False)
            
        else:
            write_mode = 'w'
            # Use 'replace' to overwrite the sheet in case it already exists
            with pd.ExcelWriter(xl_outf, mode=write_mode, engine='openpyxl') as writer:
                cumul_df.to_excel(writer, sheet_name=today, index=False)
        
        # Add market value to the 'Market_Value' sheet
        # read the existing DF of market_value from the sheet - if it exists
        with pd.ExcelFile(xl_outf) as xls:
            if MARKET_VALUE_SHEET in xls.sheet_names:
                # read the existing DF of market_value from the sheet
                market_value_df = pd.read_excel(xl_outf, sheet_name=MARKET_VALUE_SHEET)
                # Check if today is in the Date column - if not append it
                if today not in market_value_df['Date'].values:
                    market_value_df.loc[len(market_value_df)] = {'Date': today, MARKET_VALUE_COL: market_value}
                else:
                    # update the market value for today
                    market_value_df.loc[market_value_df['Date'] == today, MARKET_VALUE_COL] = market_value
                # sort the DF by Date
                market_value_df = market_value_df.sort_values(by='Date')
                # save the DF to the sheet
            else:
                # create a new sheet
                market_value_df = pd.DataFrame({'Date': [today], MARKET_VALUE_COL: [market_value]})
            # sort the DF by Date
            market_value_df = market_value_df.sort_values(by='Date')
            # save the market_value_df to the sheet
            with pd.ExcelWriter(xl_outf, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                market_value_df.to_excel(writer, sheet_name=MARKET_VALUE_SHEET, index=False)   
            
            # make tmp_df with only the Date and Market Value columns - and map market_value to a string
            tmp_df = market_value_df[['Date', MARKET_VALUE_COL]]
            tmp_df[MARKET_VALUE_COL] = tmp_df[MARKET_VALUE_COL].map(to_dollar_str)
            logger.info(f"Market Values:\n{tmp_df}")

        return

    def _open_output_files(self) -> None:
        """Setup output files and directories."""
        paths = self.config_manager.output_paths
        # Handle Excel output file
        self.xl_out_filename = paths['xl_out_filename']
        # Don't add string paths to open_files - only add actual file objects

        # Setup output file
        self.outf = open(paths['out_file'], mode='w', encoding='utf-8')
        self.open_files.add(self.outf)

        if DEBUG_FLAG:
            logger.info(
                'Processing files from: ' + os.path.basename(os.path.dirname(self.xl_out_filename)))
        return

   

    def cleanup(self, kill_flag: bool = False) -> None:
        """
        Cleanup resources.
        1. Send stop signal to write process
        2. If kill_flag is True (KeyboardInterrupt or Exception), kill the write process i.e. async tasks
        3. Close open files
        4. Stop timer
        """
        
        # Close open files
        logger.debug(f"Cleanup: closing open files: {len(self.open_files)}")
        for file in self.open_files:
            try:
                file.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")

        logger.debug(f"Cleanup: cleanup completed")
        return

    


def main(cmd_line: List[str]):
    """Main entry point for BenPlan.
    processor_kill_flag: if True, async processes will be killed by the cleanup function
    """
    processor = MarketValueProcessor(cmd_line)
    exit_code = 0
    # Start the timer for the entire program
    start_time = dt.datetime.now()
    try:
        processor.process()

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Cleaning up...")
        # Cancel all async tasks
        processor_kill_flag = True
        exit_code = 1
    except Exception as e:
        logger.error("\nError: {e}")
        # dump stack trace
        logger.error(traceback.format_exc())
        logger.error("\n----\n")
        processor_kill_flag = True
        exit_code = -2
    finally:
        # Only call cleanup if there was an exception or interrupt
        end_time = dt.datetime.now()
        logger.info(f'End: {str(end_time)} -- Run Time: {str(end_time - start_time)}\n')
        sys.exit(exit_code)

if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)