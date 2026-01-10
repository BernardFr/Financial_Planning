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
from pathlib import Path

import openpyxl

                             
import utilities as ut
from configuration_manager_class import ConfigurationManager
from logger import logger

MARKET_VALUE_SHEET = 'Market Value'
MARKET_VALUE_COL = 'Market Value'
DEBUG_FLAG = False

class Holdings:
    """Main orchestrator class for Market Value processing.
    Each processing step is a separate function returning a DataFrame
    the main function then calls the plotting functions - if applicable
    and calls the async writing functions
    """

    def __init__(self, cmd_line: List[str]):
        self.config_manager = ConfigurationManager(cmd_line)
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.data_dir =  self.config['DATA_DIR']
        self.cash_amount_string = self.config['Cash_Amount_String']
        self.cols_2_keep = self.config['Cols_2_Keep']
        self.detail_columns = self.config['Detail_Columns']

    def _load_all_transaction_data(self) -> tuple[pd.DataFrame, List[str], float]:
        """Load all transaction data from holdings file."""
        # Get the latest Holdings file from Holdings directory
        holdings_file, holdings_date = self._get_holdings_file()
        # convert holdings_date to a datetime object
        holdings_date_dt = dt.datetime.strptime(holdings_date, '%Y-%m-%d')
        # compute how old the holdings file is in days
        days_old = (dt.datetime.now() - holdings_date_dt).days
        logger.info(
            f"Holdings file: {holdings_file} - Holdings date: {holdings_date} - days old: "
            f"{days_old}")

        # Load the data from the holdings file
        holdings_df, cash_amount = self._load_holdings_data(holdings_file)
        return holdings_df, cash_amount  # return holdings_df

    def _get_holdings_file(self) -> tuple[str, str]:
        """Get Holdings file to process """
        os.chdir(self.data_dir)

        def _remap_date(date_str: str) -> str:
            """Remap the date string MMDDYY to the format YYYY-MM-DD"""
            return f"20{date_str[4:6]}-{date_str[0:2]}-{date_str[2:4]}"

        # Get XLS (not xlsx) files from current directory
        scandir_entries = list(os.scandir('.'))
        xl_files = [f.name for f in scandir_entries if f.is_file() and '.' in f.name]
        xl_files = [f for f in xl_files if f.split('.')[1] == 'xls']
        # Find the files that match the pattern
        pattern = self.config['FILE_NAME_PATTERN']
        holdings_files = [f for f in xl_files if re.match(pattern, f)]
        # extract the date from the file name and return a tuple: (file_name: date)
        h_tuples = [(f, f.split('_')[3]) for f in holdings_files]
        # apply remap_date to all values of h_tuples   
        holdings_files_lst = [(f[0], _remap_date(f[1])) for f in h_tuples]
        # sort the list of tuples by date - latest first
        holdings_files_lst = sorted(holdings_files_lst, key=lambda item: item[1], reverse=True)
        # Get the one with the latest date
        latest_file, latest_date = holdings_files_lst[0][0], holdings_files_lst[0][1]
        # TODO: there may be multiple files with the same date - need to check the last 4 digits
        #  of the file name to determine which one to use
        return latest_file, latest_date

    def _get_cash_amount(self, in_df: pd.DataFrame) -> float:
        """ Match the Cash_Amount_String in the input dataframe in the first column and return
        the value in the third
         column"""
        return float(in_df[in_df.iloc[:, 0] == self.cash_amount_string].iloc[:, 2].values[0])

    def _load_holdings_data(self, holdings_file: str) -> tuple[pd.DataFrame, float]:
        """Load the holdings data from the file"""
        in_file = os.path.join(self.data_dir, holdings_file)
        # openpyxl does not work with .xls files - use xlrd instead
        in_df = pd.read_excel(in_file, sheet_name='WFA_Positions', dtype=str, engine='xlrd')
        cash_amount = self._get_cash_amount(in_df)
        # Find the row where the first value is 'ETFs'
        etfs_row_idx = in_df[in_df.iloc[:, 0] == 'ETFs'].index[0]
        # Find the row where the first value is 'Total ETFs'
        total_etfs_row_idx = in_df[in_df.iloc[:, 0] == 'Total ETFs'].index[0]
        col_row = in_df.iloc[etfs_row_idx + 1]
        logger.debug(f"col_row: {col_row}")
        first_row = etfs_row_idx + 2
        last_row = total_etfs_row_idx  # don't need to subtract 1 - b/c of how range works
        working_df = pd.DataFrame(in_df.iloc[first_row:last_row, :])
        working_df.columns = col_row
        # keep only the columns in Cols_2_Keep
        working_df = working_df[self.cols_2_keep].copy(deep=True)
        # Convert the data in the Shares and Market Value columns to a float
        working_df['Shares'] = working_df['Shares'].astype(float)
        working_df['Market Value'] = working_df['Market Value'].astype(float)
        # Delete the rows that represent aggregate - they are labeled as 'Detail'
        # we cannot keep the aggregate rows, because some ETFs have only one entry - so no "Detail"
        holdings_df = self._del_detail_rows(working_df)
        # Drop Account Number column
        holdings_df = holdings_df.drop(columns=['Account Number'])
        # Sum Shares and Market Value by Symbol
        holdings_df = holdings_df.groupby('Symbol').sum().reset_index()
        # Sort holdings_df by Symbol 
        holdings_df = holdings_df.sort_values(by='Symbol')
        holdings_df.reset_index(drop=True, inplace=True)
        return holdings_df, cash_amount

    def _del_detail_rows(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """ Delete the rows where 'Detail' is in the Detail_Columns """
        working_df = in_df.copy(deep=True)
        # drop the rows where 'Detail' is in the Detail_Columns
        for col in self.detail_columns:
            working_df = working_df[working_df[col] != 'Detail']
        # drop the Detail_Columns columns
        # working_df = working_df.drop(columns=self.detail_columns)
        working_df = working_df.drop(columns=['Tax Term'])
        return working_df


def main(cmd_line: List[str]):
    """Main entry point for BenPlan.
    processor_kill_flag: if True, async processes will be killed by the cleanup function
    """
    portfolio = Holdings(cmd_line)
    holdings_df, cash_amount = portfolio._load_all_transaction_data()
    print(f"Holdings DataFrame:\n{holdings_df}")
    print(f"Cash amount: ${cash_amount:,.2f}")
    return


if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)