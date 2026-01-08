import pandas as pd
import sys
from typing import Tuple
import time
import datetime as dt

from configuration_manager_class import ConfigurationManager
from logger import logger

COLUMN_ORDER = ['Date', 'Payee', 'OriginalCat', 'Category', 'Memo/Notes', 'Tags',  'Amount', 'Source', 'Month', 'MasterCat', 'BenPlan', 'Discretionary']

class DataCleaner:
    """Handles data cleaning, categorization, and preprocessing."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.qdate_format = self.config['QDATE_FORMAT']

    def clean_and_prepare_data(self, cumul_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the transaction data."""
        clean_df = cumul_df.copy(deep=True)
        # Fix dates
        clean_df['Date'] = clean_df['Date'].map(fix_the_date)

        # Create month labels
        clean_df['Month'] = clean_df['Date'].apply(lambda x: _date2month(x, self.qdate_format))

        # Check for weird dates
        z = clean_df[clean_df.Month == 'xxxx-xx']
        if len(z.index) > 0:
            logger.info(f'rows with weird dates: {repr(z)}\n')

        # Filter by date range
        first_mo = _date2month(self.config['FIRST_DATE'], self.qdate_format)
        last_mo = _date2month(self.config_manager.current_dir, self.config['DIR_DATE_FORMAT'])
        logger.info(f'First Mo: {first_mo} (included) - Last Mo: {last_mo} (excluded)')

        clean_df = clean_df.loc[clean_df.loc[:, 'Month'] >= first_mo]
        clean_df = clean_df.loc[clean_df.loc[:, 'Month'] < last_mo]

        # Convert Amount to float
        clean_df['Amount'] = clean_df['Amount'].apply(lambda x: float(str(x).replace(',', '')))

        # Remap categories
        clean_df, bad_cat_flag = self._remap_sub_cat(clean_df)
        clean_df, bad_travel_flag = self._remap_travel_cat(clean_df)
        if bad_cat_flag or bad_travel_flag:
            if bad_cat_flag:
                logger.error("Bad Categories")
            if bad_travel_flag:
                logger.error("Bad Travel Tags")
            sys.exit(-1)

        # Sort by date
        clean_df['Day'] = clean_df['Date'].apply(self._date2day)
        clean_df.sort_values(by='Day', ascending=True, inplace=True, na_position='last')
        clean_df.drop('Day', inplace=True, axis=1)

        return clean_df

    def _remap_sub_cat(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """ remap Categories/Subcategory to a new Category based on MasterCat file """
        master_cat_file = self.config_manager.master_cat_file
        bad_cat_flag = False
        # Read the Master Categories files
        read_df = pd.read_excel(master_cat_file, sheet_name='Categories', header=0,
                                engine='openpyxl')
        # Remove useless columns
        master_cat_df = read_df[
            ['Category', 'MappedCat', 'MasterCat', 'BenPlan', 'Discretionary']].copy(deep=True)
        # List of Categories
        master_cat_set = set(list(master_cat_df['Category']))

        def _bad_category(category: str) -> bool:
            """ Bad category is not in the list or Uncategorized """
            if (
                    category == "Uncategorized" or category == "" or category is None or category
                    not in master_cat_set):
                return True
            else:
                return False

        # When the Mapped Category is not specified (blank) we use 'Category'
        # fillna() replaces NaN in MappedCat with the corresponding value from Category
        master_cat_df['newCat'] = pd.Series(master_cat_df['MappedCat']).fillna(
                pd.Series(master_cat_df['Category']))

        # Create 2 dict to perform the mapping from Category to newCat and MasterCat
        cat_dict = dict(zip(master_cat_df['Category'], master_cat_df['newCat']))
        master_dict = dict(zip(master_cat_df['Category'], master_cat_df['MasterCat']))
        ben_dict = dict(zip(master_cat_df['Category'], master_cat_df['BenPlan']))
        discr_dict = dict(zip(master_cat_df['Category'], master_cat_df['Discretionary']))
        bad_cat = df[df['Category'].apply(_bad_category) == True]
        if len(bad_cat.index) > 0:
            bad_cat_flag = True
            logger.warning(f"Bad Categories: {len(bad_cat.index)}\n{bad_cat}")

        df['MasterCat'] = pd.Series(df['Category']).apply(lambda x: master_dict[x])
        df['BenPlan'] = df['Category'].apply(lambda x: ben_dict[x])
        df['Discretionary'] = df['Category'].apply(lambda x: discr_dict[x])
        # !!! IMPORTANT: since we are remapping Category - it has to be done last
        # Otherwise the mapping to masterCat won't work
        df['NewCat'] = df['Category'].apply(lambda x: cat_dict[x])
        # rename Category to OriginalCat and NewCat to Category
        df.rename(columns={'Category': 'OriginalCat'}, inplace=True)
        df.rename(columns={'NewCat': 'Category'}, inplace=True)
        df = df[COLUMN_ORDER]  # reorder the columns
        del master_cat_df

        return df, bad_cat_flag

    def _get_travel_map(self) -> Tuple[dict, list]:
        """
        Map Memo Entries to Vacation entries in the master category file / Travel sheet
        """
        travel_cat_df = pd.read_excel(self.config_manager.master_cat_file, sheet_name='Travel', header=0,
                                      engine='openpyxl')
        travel_map = dict(zip(travel_cat_df['Memo'], travel_cat_df['Vacation']))
        vacation_list = list(set(list(travel_cat_df['Vacation'])))
        return travel_map, vacation_list


    def _remap_travel_cat(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """ remap Travel tags based on Memo Entries (if Tag is not already present)
            Step1 : remap Travel tags based on Memo Entries (if Tag is not already present)
            Step2 : check that the Travel Tag is in the list of vacations
        """
        travel_map, vacation_list = self._get_travel_map()
        bad_travel_flag = False

        # remap Travel tags based on Memo Entries (if Tag is not already present)
        def _map_travel_tag(row: pd.Series) -> str:
            """ For BenPlan Travel, assign Memo/Notes to Tag if Tag is empty """
            [tag, memo, benplan] = row[['Tags', 'Memo/Notes', 'BenPlan']]
            if benplan == 'Travel':
                tag = tag.strip() if tag is not None else ''
                if tag == '':
                    return travel_map.get(memo, '')
                else:
                    return tag
            else:
                return tag

        df['Tags'] = df.apply(lambda row: _map_travel_tag(row), axis=1)

        # Get that the Travel Tag is in the list of vacations
        def _check_bad_travel_category(df: pd.DataFrame) -> bool:
            """ Check for bad Travel categories """
            # Get the list of vacations from the master file
            # logger.info(f"Vacation list: {vacation_list}")

            def _bad_travel_category(row: pd.Series) -> bool:
                """ Error if Tag is empty or is not in travel_map
                The assumption is that the Memo/Notes has been remapped to the tag if the tag is empty
                """
                [benplan, tag] = list(row[['BenPlan', 'Tags']])
                if benplan == 'Travel':
                    # set tag to empty string if it is None or strip it
                    tag = tag.strip() if tag is not None else ''
                    if tag == '' or tag is None or tag not in vacation_list:
                        return True
                    else:
                        return False
                else:
                    return False

            bad_cat = df[df.apply(_bad_travel_category, axis=1) == True]
            if len(bad_cat.index) > 0:  # show the bad entries grouped by Source
                display_col = ['Date', 'Payee', 'Category', 'BenPlan', 'Tags', 'Memo/Notes', 'Amount',
                            'Month']
                source_set = set(bad_cat['Source'])
                for src in source_set:
                    logger.info(f"\nSource: {src}\n")
                    display_df = bad_cat.loc[bad_cat['Source'] == src]
                    logger.warning(f"{display_df[display_col]}\n")
                return True
            else:
                return False

        bad_travel_flag = _check_bad_travel_category(df)
        return df, bad_travel_flag

    def _date2day(self, dd):
        """ Convert Quicken data to a day count"""
        try:
            return int(time.mktime(time.strptime(dd, self.qdate_format)) / (24 * 60 * 60))
        except ValueError:
            return int(time.mktime(time.strptime(dd, '%m/%d/%y')) / (24 * 60 * 60))


def fix_the_date(date_in):
    """ Convert all date strings to a mm/dd/yyyy string  """
    if ' ' in date_in:
        date_str, _ = str(date_in).split(' ')  # get rid of time if present
    else:
        date_str = date_in
    try:
        x_time = dt.datetime.strptime(date_str, '%m/%d/%Y')
    except ValueError:
        try:
            x_time = dt.datetime.strptime(date_str, '%m/%d/%y')
        except ValueError:
            try:
                x_time = dt.datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                print(f"Error in _date2month - invalid date: {date_str} - date_in = {date_in}")
                return 'xxxx-xx'
    return x_time.strftime('%m/%d/%Y')


def _date2month(time_str, date_format):
    """ Convert a date string to a yy-mm string  """
    # Protect against empty or otherwise weird row
    if ' ' in time_str:
        date_str, _ = str(time_str).split(' ')  # get rid of time if present
    else:
        date_str = time_str
    try:
        x_time = time.strptime(date_str, date_format)
    except ValueError:
        try:
            x_time = time.strptime(date_str, '%m/%d/%y')
        except ValueError:
            try:
                x_time = time.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                print(f"Error in _date2month - invalid date: {date_str} - time_str = {time_str} - "
                      f"expected format: {date_format}")
                # noinspection SpellCheckingInspection
                return 'xxxx-xx'
    month = x_time.tm_mon  # int
    if month < 10:
        out_str = str(x_time.tm_year) + '-0' + str(month)  # make month 2 digits
    else:
        out_str = str(x_time.tm_year) + '-' + str(month)
    return out_str


