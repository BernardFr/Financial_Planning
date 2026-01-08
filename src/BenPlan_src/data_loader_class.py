import pandas as pd
import os
from typing import List
import datetime as dt
import sys

from configuration_manager_class import ConfigurationManager
from logger import logger
import venmo_utilities as Vmo
from utils import this_function

# # Settings used in this file
brkg_wrksheet = ['WF_LoC', 'Brokerage', 'WF_3400']  # Worksheets in brokerage file
col_wrksheet = ['Date', 'Category', 'Payee', 'Amount', 'Tags', 'Memo/Notes']
col_2_keep = ['Date', 'Payee', 'Category', 'Tags', 'Memo/Notes', 'Amount']
brkg_col_2_keep = ['Date', 'Activity', 'Description', 'Amount', 'Notes']
# after deleting the brkg_to_del
loc_expected = ['Journal', 'Wire Transfer', 'Interest', 'Int Charged']
brokerage_expected = ['Journal', 'Wire Transfer', 'Interest', 'Invest Inc', 'Int Charged']
wf_3400_expected = ['Deposit', 'Interest']
# CORNELIA_DEBUG = True
VERBOSE = False
P_F_DEBUG = False
# List the Activities to remove - See ReedMe for details
brkg_to_del = ['Ach Activity', 'Margin Int', 'Transfer', 'Advisory Fee', 'Dividend',
               'Shrt Trm Gain', 'Lt Cap Gain', 'Charge', 'Buy', 'Sell']
CORNELIA_STRING = "DITISHEIM"  # Incoming wire from Cornelia/Dickie Trust
PPP_SALE_STRING = "INCOMING WIRE MERCURY"   # Incoming wire from PPP sale

class DataLoader:
    """Handles loading and processing transaction data from various sources."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.home_dir = self.config['HOME_DIR']
        self.data_dir = self.home_dir + self.config['DATA_DIR']
        self.closed_dir = self.data_dir + 'CLOSED/'
        self.brkg_file = self.data_dir + 'Brokerage_Activity.xlsx'  # LoC and Brokerage wires
        self.envision_file = self.data_dir + 'Envision/Envision_Projections.xlsx'
        self.venmo_dir = self.data_dir + self.config['VENMO_DIR']
        self.venmo_xl_file = self.venmo_dir + self.config['VENMO_XL_FILE']
        self.master_cat_file = self.home_dir + self.config['BENPLAN_MAP_FILE']

        self.open_files = set()

    def load_all_transaction_data(self) -> pd.DataFrame:
        """Load all transaction data from various sources."""
        # Get CSV files from current directory
        csv_files = self._get_csv_files()

        # Load and combine all transaction data
        cumul_df = pd.DataFrame()

        for file in csv_files:
            file_name = file.split('/')[-1].split('.')[0]
            q_flag_val = file_name.startswith('Quicken')
            logger.info(f"Processing file: {file}")
            df = _process_file(file, q_flag=q_flag_val)
            cumul_df = pd.concat([cumul_df, df], ignore_index=True)

        # Remove Venmo transactions (will be added from Venmo statements)
        cumul_df.reset_index(drop=True, inplace=True)
        venmo_mask = cumul_df['Source'] == "Venmo"
        venmo_index = cumul_df.loc[venmo_mask].index.tolist()
        logger.info(f"# Venmo: {len(venmo_index)}")
        cumul_df = cumul_df.drop(index=venmo_index, errors='ignore')

        # Add brokerage transactions
        df = _brkg_process_file(self.brkg_file)
        cumul_df = pd.concat([cumul_df, df], ignore_index=True)

        # Add Venmo transactions
        venmo_df = Vmo.process_all_venmo_files(self.venmo_dir, self.venmo_xl_file, self.master_cat_file,
                                               self.config_manager.config['SKIP_VENMO'])
        cumul_df = pd.concat([cumul_df, venmo_df], ignore_index=True)

        # Remove Venmo duplicates
        cumul_df.reset_index(drop=True, inplace=True)
        dupe_mask = cumul_df['Category'] == "Venmo_Dupe"
        dupe_index = cumul_df.loc[dupe_mask].index.tolist()
        logger.info(f"# Venmo Dupe: {len(dupe_index)}")
        cumul_df = cumul_df.drop(index=dupe_index, errors='ignore')

        assert isinstance(cumul_df, pd.DataFrame)
        return cumul_df

    def _get_csv_files(self) -> List[str]:
        """Get list of CSV files to process."""
        current_dir = self.config_manager.current_dir
        os.chdir(self.data_dir + current_dir)

        # Get CSV files from current directory
        scandir_entries = list(os.scandir('.'))
        logger.debug(f"scandir('.') returned {len(scandir_entries)} entries")
        for entry in scandir_entries:
            logger.debug(f"Entry: {entry.name}, is_file: {entry.is_file()}")

        csv_files = [f.name for f in scandir_entries if '.' in f.name]
        logger.debug(f"Files with dots: {csv_files}")
        csv_files = [f for f in csv_files if f.split('.')[1] == 'csv']
        logger.debug(f"CSV files: {csv_files}")

        # Exclude output files
        suppress_list = ['aggregate-' + current_dir + '.csv', 'combined-' + current_dir + '.csv',
                         'average-' + current_dir + '.csv', 'monthly-' + current_dir + '.csv',
                         'monthly-master-' + current_dir + '.csv',
                         'benPlan-' + current_dir + '.csv']
        csv_files = [f for f in csv_files if f not in suppress_list]
        logger.debug(f"After suppression: {csv_files}")

        # Add closed files
        closed_scandir_entries = list(os.scandir(self.closed_dir))
        logger.debug(f"closed scandir returned {len(closed_scandir_entries)} entries")
        closed_files = [f.name for f in closed_scandir_entries if f.name.split('.')[1] == 'csv']
        closed_files = [self.closed_dir + f for f in closed_files]
        logger.debug(f"Closed files: {closed_files}")
        csv_files.extend(closed_files)

        logger.debug(f"Final result: {csv_files}")
        return csv_files

    def get_travel_cat_df(self) -> pd.DataFrame:
        """ Get the travel category dataframe """
        return pd.read_excel(self.master_cat_file, sheet_name='Travel', header=0, dtype=str, engine='openpyxl')

def _process_file(in_file, q_flag=False):
    """ read account transactions from CSV file, clean up, and return in DF """
    # ToDo: make this smarter ... i.e. read row by row until you find the one that has
    #  the expected labels
    if P_F_DEBUG:
        print('Processing file:', in_file)
    if q_flag:
        header_val = 8
        col_2_keep_local = col_2_keep + ['Account']  # cannot modify col_2_keep which is global
    else:
        header_val = 6
        col_2_keep_local = col_2_keep

    df = pd.read_csv(in_file, sep=',', header=header_val)
    # Get rid of useless columns
    col2drop = df.columns[0:3].tolist()
    df = df.drop(columns=col2drop, axis=1)

    # Check that the necessary columns are in the file
    s1 = set(col_2_keep_local)
    s2 = set(df.columns)
    if not s1.issubset(s2):  # required columns are Not included in the file
        missing = [x for x in s1 if x not in s1 & s2]
        print(f'Error - File {in_file} is missing columns: {missing}')
        print('Columns present:', s2)
        print('Columns required:', s1)
        sys.exit(-1)
    # Get rid of un-desired columns (can't use dropna b/c some desired columns may be empty)
    col_2_del = [col for col in df.columns if col not in col_2_keep_local]
    if col_2_del:
        df = df.drop(col_2_del, axis=1)
    # Order the (selected) columns in consistent fashion
    # print(inFile, list(df)) # print column names
    df = df[col_2_keep_local]

    # Get rid of empty rows
    df.dropna(how='all', inplace=True, axis=0)
    if P_F_DEBUG:
        print(f'Columns: {df.columns}')
        print(f'\nTail\n{repr(df.tail(10))}')
        print('\n')
        print(df.iloc[len(df.index) - 1])

    def bad_date(val):
        return str(val).find('/') == -1  # True -> delete row

    df.reset_index(drop=True, inplace=True)  # Good practice before a filter & drop operation
    bad_date_mask = df.loc[:, 'Date'].apply(bad_date)
    df.drop(df.loc[bad_date_mask].index.tolist(), inplace=True)

    # Replace the annoying NaN w/ empty string
    df.fillna('', inplace=True)

    # Label the account from where the data came from
    if q_flag:  # replace column label 'Account' by 'Source'
        col_list = df.columns.tolist()
        idx = col_list.index('Account')
        col_list[idx] = 'Source'
        df.columns = col_list
    else:  # 'Source' is the name of the file
        # Add a column to indicate the source of the transaction
        src = os.path.split(in_file)[1]  # get the basename of the file
        src = src.strip('.csv')
        df['Source'] = src

    return df


def _map_loc_row(row):  
    """
    Assign the Category based on values on the row
    :param row: row in the LOC summary file
    :return: Category label
    """
    assert row['Category'] in loc_expected, (f"{this_function()} - Unexpected Activity label: "
                                             f"{row['Category']}")
    if row['Category'] == 'Journal':  # These are typically payments from Brokerage
        return 'to_delete'  # we'll delete it later
    if row['Category'] == 'Wire Transfer':
        return 'VC_Invest'
    if row['Category'] == 'Interest' or row['Category'] == 'Int Charged':
        return 'PCL_Interest'
    # else:
    print(f"{this_function()} - Unexpected Activity label: {row['Category']}")
    return 'Uncategorized'





def _map_brokerage_row(row, cornelia_debug: bool = False) -> str:
    """
    Assign the Category based on values on the row
    :param row: row in the brokerage summary file
    :param cornelia_debug: Debug flag
    :return: Category label
    """
    assert row['Category'] in brokerage_expected, f"Unexpected Activity label: {row['Category']}"
    # We track transfers to PCL in WF_LoC
    if row['Category'] == 'Journal':  # These are typically payments from Brokerage
        if (row['Payee'].startswith('MONTHLY JNL TRF TO') or row['Payee'].startswith('TO 61276797')
                or row['Payee'].startswith('TO:61276797')):  # PCL payment from Brokerage
            return 'Pay_off_PCL'
        # else:
        # Only print error message is Date is less than 2 years old
        if (dt.datetime.strptime(row['Date'], '%Y-%m-%d %H:%M:%S') > dt.datetime.now() -
                dt.timedelta(days=730)):
            print(f"{this_function()}- Unexpected Journal entry: {row}")
        return 'Invest Inc'  # randdom - just needed to pick a category
    if row['Category'] == 'Interest':
        return 'to_delete'  # we'll delete it later
    if row['Category'] == 'Wire Transfer':
        if row['Amount'] >= 0:  # Distribution from VC or Cornelia/Dickie Trust
            if CORNELIA_STRING in row['Payee']:  # Payment from Cornelia/Dickie Trust
                if cornelia_debug:
                    print(f"Cornelia/Dickie Trust wire:\n{row}`")
                return "Cornelia"
            if PPP_SALE_STRING in row['Payee']:  # Payment from PPP sale
                return "PPP_Sale"
            # else:
            return 'VC_Principal'
        # else:  # Payment to VC / drawdown
        return 'VC_Invest'
    # else:
    # ToDo: only print error message is Date is less than 1 year old
    print(f"{this_function()} - Unexpected Activity label: {row['Category']}")
    return row['Category']  # we return the original category b/c if is in brokerage_expected


def _map_wf_3400_row(row):
    """
    Assign the Category based on values on the row
    :param row: row in the brokerage summary file
    :return: Category label
    """
    assert row['Category'] in wf_3400_expected, f"Unexpected Activity label: {row['Category']}"
    if row['Category'] == 'Deposit':
        return 'XTRA_BRKG'
    if row['Category'] == 'Interest':
        return 'to_delete'
    # else:
    print(f'Unexpected Activity label: {row["Category"]}')
    return 'Uncategorized'


# Assign the mapping function to its corresponding worksheet
sheet_fct_map = {'WF_LoC': _map_loc_row, 'Brokerage': _map_brokerage_row, 'WF_3400': _map_wf_3400_row}


def _del_row_2_del(val):
    """ Kludge Delete the rows that have Category = 'to_delete' """
    return val == 'to_delete'  # True -> delete row

    # Delete the rows that contain activities we do not want


def _row2del(val):
    """ Delete the rows that contain non-relevant activities """
    # Need  strip() to get rid of leading/trailing spaces
    return val.strip() in brkg_to_del  # True -> delete row


def _brkg_process_file(in_file):
    """
    Special processing for brokerage file to capture all outgoing and incoming wire transfers
    Column names are: Date, Activity, Description, Amount
    """

    # Create an empty DF with the desired columns
    final_df = pd.DataFrame(columns=pd.Index(col_2_keep + ['Source']))
    final_col = final_df.columns.tolist()
    for sheet_name in brkg_wrksheet:  # read each worksheet
        print('Processing file:', in_file, 'Sheet: ', sheet_name)
        assert sheet_name in brkg_wrksheet, f'Unknown worksheet in brokerage file: {sheet_name}'
        # Forcing dtype to str to read the date properly
        df = pd.read_excel(in_file, sheet_name=sheet_name, header=0, dtype=str, engine='openpyxl')
        df = df.dropna(how='all', axis=1)  # Remove empty columns
        df = df.dropna(how='all', axis=0)  # Remove empty rows
        df['Amount'] = df['Amount'].map(float)  # Get amount back to float
        df = df.drop(df[df['Activity'].map(_row2del)].index.tolist())
        df.reset_index(drop=True, inplace=True)  # Good practice after a filter & drop operation

        # Map columns to be similar to other accounts: ['Date', 'Payee', 'Category',
        # 'Tags', 'Memo/Notes', 'Amount']
        map_dict = {'Date': 'Date', 'Activity': 'Category', 'Description': 'Payee',
                    'Notes': 'Memo/Notes', 'Amount': 'Amount'}
        col_names = list(df.columns)
        new_col_names = [map_dict[x] for x in col_names]
        df.columns = new_col_names

        # Map the Category based on the Activity
        # Select the mapping function according to the sheet
        map_fct = sheet_fct_map[sheet_name]
        df['Category'] = df.apply(map_fct, axis=1)
        # Kludge to eliminate anything that is not a recognized transaction type
        df = df.drop(df[df['Category'].map(_del_row_2_del)].index.tolist())
        df.reset_index(drop=True, inplace=True)  # Good practice after a filter & drop operation

        df['Tags'] = ''  # Add the Tags column
        df['Memo/Notes'] = ''
        df['Source'] = sheet_name
        df = df[final_col]  # Ensure both DF columns are in the same order
        df.reset_index(drop=True, inplace=True)
        # final_df = final_df.append(df, ignore_index=True)  # resets index
        final_df = pd.concat([final_df, df], ignore_index=True)  # resets index

    return final_df

