import datetime as dt
import os
import re
import sys
from typing import Dict, List

import pandas as pd

from logger import logger
from mv_configuration_manager_class import ConfigurationManager

Cols_2_Keep = ['Symbol', 'Account Number', 'Shares', 'Description', 'Tax Term']
account_map: Dict[str, str] = {'*8475': 'BRKG', '*3672': '401K'}
Detail_Columns = ['Account Number', 'Tax Term']
Cash_Amount_String = "Total Cash/Cash Alternatives & Margin"  # this is the string in the
# holdings file that contains the cash amount


# # Settings used in this file

class DataLoader:
    """Handles loading and processing transaction data from various sources."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.home_dir = self.config['HOME_DIR']
        self.data_dir = self.home_dir + self.config['DATA_DIR']

        self.open_files = set()

    def load_all_transaction_data(self) -> tuple[pd.DataFrame, List[str], float]:
        """Load all transaction data from various sources."""
        # Get the latest Holdings file from Holdings directory
        holdings_file, holdings_date = self._get_holdings_file()
        # Check if the holdings date is more than 30 days old
        # convert holdings_date to a datetime object
        holdings_date_dt = dt.datetime.strptime(holdings_date, '%Y-%m-%d')
        # compute how old the holdings file is in days
        days_old = (dt.datetime.now() - holdings_date_dt).days
        logger.info(
            f"Holdings file: {holdings_file} - Holdings date: {holdings_date} - days old: "
            f"{days_old}")

        # Load the data from the holdings file
        holdings_df, cash_amount = self._load_holdings_data(holdings_file)
        logger.info(f"Cash amount: ${cash_amount:,.2f}")
        logger.debug(f"holdings_df head: {holdings_df.head(10)}")
        logger.debug(f"holdings_df tail: {holdings_df.tail(10)}")

        # Aggregate the holdings data by Symbol and by Account Number   
        holdings_df = holdings_df.groupby(['Symbol', 'Account Number']).sum().reset_index()
        logger.debug(f"holdings_df head: {holdings_df.head(10)}")
        logger.debug(f"holdings_df tail: {holdings_df.tail(10)}")
        # get a unique list of symbols
        symbols_list = list(set(list(holdings_df['Symbol'])))
        logger.debug(f"symbols: {symbols_list}")

        return holdings_df, symbols_list, cash_amount  # return holdings_df

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
        return float(in_df[in_df.iloc[:, 0] == Cash_Amount_String].iloc[:, 2].values[0])

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
        working_df = working_df[Cols_2_Keep].copy(deep=True)
        # Convert the data in the Shares column to a float
        working_df['Shares'] = working_df['Shares'].astype(float)
        # Delete the rows that represent aggregate - they are labeled as 'Detail'
        working_df = self._del_detail_rows(working_df)
        # remap the Account Number to the short name
        holdings_df = self._remap_account_number(working_df)
        # Sort holdings_df by Account Number and Symbol 
        holdings_df = holdings_df.sort_values(by=['Symbol', 'Account Number'])
        holdings_df.reset_index(drop=True, inplace=True)
        return holdings_df, cash_amount

    def _del_detail_rows(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """ Delete the rows where 'Detail' is in the Detail_Columns """
        working_df = in_df.copy(deep=True)
        # find the rows where 'Detail' is in the Detail_Columns
        for col in Detail_Columns:
            working_df = working_df[working_df[col] != 'Detail']
        # drop the rows where 'Detail' is in the Detail_Columns
        # drop the "Tax Term" column
        working_df = working_df.drop(columns=['Tax Term'])
        return working_df

    def _remap_account_number(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """Remap the Account Number to the short name"""
        working_df = in_df.copy(deep=True)
        # verify that Account Numbers are as expected
        account_numbers = list(set(list(in_df['Account Number'])))
        known_account_numbers = list(account_map.keys())
        # verify that the account numbers are in the account_map
        unknown_account_numbers = [a for a in account_numbers if a not in known_account_numbers]
        if len(unknown_account_numbers) > 0:
            logger.error(f"Unknown account numbers: {unknown_account_numbers}")
            sys.exit(1)
        # remap the Account Number to the short name
        working_df['Account Number'] = working_df['Account Number'].map(account_map)
        return working_df
