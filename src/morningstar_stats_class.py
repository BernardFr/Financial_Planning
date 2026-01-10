#!/usr/local/bin/python3

"""
Get major asset types cross-correlation from Morningstar Webpage
Ref: https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
Create N pseudo-random series with the same cross correlation
Correlated series Ref: https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html

# The URL of the ETF profile page
URL_STATS = "https://admainnew.morningstar.com/webhelp/dialog_boxes/cs_db_editassumptions.htm"
ULR_CORR = "https://admainnew.morningstar.com/webhelp/Practice/Plans/Correlation_Matrix_of_the_14_Asset_Classes.htm"
"""

import sys
import os
import re
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm as matrix_norm
from scipy.stats import norm
from scipy.linalg import cholesky
from sklearn.datasets import make_spd_matrix
import requests
from lxml import etree
from bs4 import BeautifulSoup
# Use helper functions from Color_Map
sys.path.insert(0, '../Color_Map/')  # Add path to the directory containing plot_color_matrix
from plot_color_matrix import plot_color_matrix as pcm
from configuration_manager_class import ConfigurationManager
from logger import logger
from typing import Any, List
from utilities import error_exit

XLabelLenDefault = 5  # Length of the X axis labels
QuickFlagDefault = False

# Random state for the random number generator used to generate the random variables to ensure we have the same
# numbers across runs
RvsRandomState = 42


def matrix_equal(df_1: pd.DataFrame, df_2: pd.DataFrame, error_margin: float) -> (bool, float):
    """
    Compares 2 DF with can be 1- or 2-dimensional and returns a tuple (flag, error)
    The error is normalized by the number of elements in the DF
    @param df_1: DF 1
    @param df_2: DF 2
    @param error_margin: normalized error margin to determine if the 2 DF are equal
    @return: (flag, error). Flag is True if the 2 DF are equal (i.e. normalized delta < error_margin). error is the
    computated normalized difference
    """
    fct_name = sys._getframe().f_code.co_name
    assert df_1.shape == df_2.shape, f"{fct_name}: df_1 and df_2  have different shapes\n{df_1.shape}\n{df_2.shape}"
    assert list(df_1.index) == list(df_2.index), f"{fct_name}: df_1 and df_2 don't have same index\n" \
                                                f"{df_1.index}\n{df_2.index}"
    delta = df_1 - df_2
    # denom is the total number of samples in each DF
    denom = len(df_1.index) if df_1.ndim == 1 else len(df_1.index) * len(df_1.columns)
    err = np.linalg.norm(delta) / denom
    flag = True if abs(err) <= error_margin else False
    return flag, err




class MorningstarStats:
    """
    Class to manage the Morningstar stats
    """
    def __init__(self, cmd_line: List[str]):
        self.config_manager = ConfigurationManager(cmd_line)
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        
        self.must_have_param = self.config['must_have_param']
        self.XLabelLen = self.config['XLabelLen']
        self.must_have_param = self.config['must_have_param']
        self.XLabelLen = self.config.get('XLabelLen', XLabelLenDefault)
        self.quick_flag = self.config.get('quick_flag', QuickFlagDefault)
        self.stat_corr_name_map = dict[str, str](self.config['STAT_CORR_NAME_MAP']) # map the list of 2-element lists to a dict
        self.nb_smpl = self.config['nb_smpl']
        self.validation_error_margin = self.config['validation_error_margin']
        self.validate_cross_correlation_flag = self.config['validate_cross_correlation_flag']

    def _make_xlabel(in_str: str, label_len) -> str:
        """
        Shorten the X axis labels to XLabelLen characters
        """
        # remove white space
        in_str = in_str.replace(' ', '')
        return in_str[0:label_len]

    def get_asset_stats(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Get the asset statistics and correlation matrix from Morningstar
        @param url_stats:
        @param url_corr:
        @return: 2 DF for asset statistics and correlation matrix
        """
        # if we already computed the asset statistics and correlation matrix, return them
        # test if df_stat and df_corr have been created
        if hasattr(self, 'df_stat') and hasattr(self, 'df_corr'):
            return self.df_stat, self.df_corr

        # else, we have to compute them
        if not hasattr(self, 'df_stat'):
            self.df_stat = self._get_morningstar_stats()
        if not hasattr(self, 'df_corr'):
            self.df_corr = self._get_morningstar_corr()

        # Remap the name of asset classes to match the names in df_corr
        self._remap_names()
        # Force the index of df_stat to be asset_class_names

        return self.df_stat, self.df_corr


    def _remap_names(self) -> None:
        """
        Remap the names in names according to the mapping of old_names new_names
        """
        # Check that df_stat and df_corr indexes match DF_STAT_NAMES and DF_CORR_NAMES
        error_flag = False
        if set(list(self.df_stat.index)) != set(self.stat_corr_name_map.keys()):
            error_flag = True
            logger.error(f"df_stat index: {list(self.df_stat.index)} does not match STAT_CORR_NAME_MAP keys: {self.stat_corr_name_map.keys()}")
        if set(list(self.df_corr.index)) != set(self.stat_corr_name_map.values()):
            error_flag = True
            logger.error(f"df_corr index: {list(self.df_corr.index)} does not match STAT_CORR_NAME_MAP values: {self.stat_corr_name_map.values()}")
        if error_flag:
            error_exit(f"Error in remapping names")

        # Remap the index of df_stat to match the index of df_corr
        asset_class_names = [self.stat_corr_name_map.get(x, "Error") for x in self.df_stat.index]  # remap
        if "Error" in asset_class_names:
            error_exit(f"Error(s) in remapping names: {self.df_stat.index} -> {asset_class_names}")
        self.df_stat['IDX'] = asset_class_names
        self.df_stat.set_index('IDX', drop=True, inplace=True)
        # Reorder  df_stat index to match df_corr index
        self.df_stat = self.df_stat.reindex(index=self.df_corr.index)

        # FIXME: HACK the Morningstar webpage has the wrong column names for the correlation matrix
        self.df_corr.columns = self.df_corr.index

        return

    def _get_morningstar_stats(self) -> pd.DataFrame:
        url_stats = self.config['url_stats']
        try:
            response = requests.get(url_stats, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve the page for URL_STATS Error: {e}\nURL: {url_stats}")
            return None
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the paragraph containing the Morningstar Basic Statistics
        element_all = soup.find_all("p", {"class": "Tip-Note-Heading"})
        elt = None
        for element in element_all:
            if element.text.strip() == "Morningstar Basic":
                elt = element
                break
        
        if elt is None:
            logger.error(f"Could not find 'Morningstar Basic' heading in page: {url_stats}")
            return None
        
        # Find the first table after elt containing the Morningstar Basic Statistics
        tbl = elt.find_next("table", {"class": "MsoTableGrid"})
        if tbl is None:
            logger.error(f"Could not find table with class 'MsoTableGrid' after 'Morningstar Basic' heading")
            return None

        # Build rows more efficiently - collect all rows first, then create DataFrame
        tbl_rows = tbl.find_all('tr')
        if not tbl_rows:
            logger.error("Table has no rows")
            return None
        
        # Extract column names from first row
        col_names = [td.text.strip() for td in tbl_rows[0].find_all('td')]
        if not col_names:
            logger.error("Table has no columns")
            return None
        
        # Build data rows using list comprehension for better performance
        data_rows = [[td.text.strip() for td in row.find_all('td')] for row in tbl_rows[1:]]
        
        # Create DataFrame in one operation instead of row-by-row
        result_df = pd.DataFrame(data_rows, columns=col_names)

        # Make Asset Class the index
        if 'Asset Class' not in result_df.columns:
            logger.error(f"'Asset Class' column not found. Available columns: {result_df.columns.tolist()}")
            return None
        
        result_df.set_index('Asset Class', inplace=True)
        
        # Convert to float, handling any non-numeric values
        try:
            result_df = result_df.astype(float)
        except ValueError as e:
            logger.error(f"Error converting table data to float: {e}")
            return None
        
        # Remove Inflation if present
        if 'Inflation' in result_df.index:
            logger.info('Removing Inflation from Stats')
            result_df.drop(labels='Inflation', axis=0, inplace=True)
        
        return result_df


    def _get_morningstar_corr(self) -> pd.DataFrame:
        url_corr = self.config['url_corr']
        try:
            response = requests.get(url_corr, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve the page for URL_CORR Error: {e}\nURL: {url_corr}")
            return None
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the h1 heading containing the correlation matrix title
        element_all = soup.find_all("h1")
        elt = None
        for element in element_all:
            if element.text.strip() == "Correlation Matrix for the 14 Asset Classes":
                elt = element
                break
        
        if elt is None:
            logger.error(f"Could not find 'Correlation Matrix for the 14 Asset Classes' heading in page: {url_corr}")
            return None
        
        # Find the first table after elt
        tbl = elt.find_next("table")
        if tbl is None:
            logger.error(f"Could not find table after 'Correlation Matrix for the 14 Asset Classes' heading")
            return None

        # Build rows more efficiently - collect all rows first, then create DataFrame
        tbl_rows = tbl.find_all('tr')
        if not tbl_rows:
            logger.error("Table has no rows")
            return None
        
        # Extract column names from first row and fix the first column name
        col_names = [td.text.strip() for td in tbl_rows[0].find_all('td')]
        if not col_names:
            logger.error("Table has no columns")
            return None
        
        # Fix the first column name to "Asset Class" (as per original logic)
        col_names[0] = "Asset Class"
        
        # Build data rows using list comprehension for better performance
        data_rows = [[td.text.strip() for td in row.find_all('td')] for row in tbl_rows[1:]]
        
        # Create DataFrame in one operation instead of row-by-row
        result_df = pd.DataFrame(data_rows, columns=col_names)

        # Make Asset Class the index
        if 'Asset Class' not in result_df.columns:
            logger.error(f"'Asset Class' column not found. Available columns: {result_df.columns.tolist()}")
            return None
        
        result_df.set_index('Asset Class', inplace=True)
        
        # Convert to float, handling any non-numeric values
        try:
            result_df = result_df.astype(float)
        except ValueError as e:
            logger.error(f"Error converting correlation matrix data to float: {e}")
            return None
        
        return result_df


    def _eigen_values_positive(self) -> bool:
        """ Returns True if the eigen values of a correlation matrix are positive"""
        # ToDo: add teststo ensure that df_corr is a correlation matrix
        try:
            eigen_values = np.linalg.eigvals(self.df_corr.astype(float))
        except np.linalg.LinAlgError:
            return False  # not a viable matrix
        positive_eigen = [eigen >= 0 for eigen in eigen_values ]
        return True if all(positive_eigen) else False




    def _make_random_correlation_matrix(self, labels: [str], seed=None) -> pd.DataFrame:
        """
        Generates a cross correlation matrix with labels as labels for index and columns
        the size of the matrix is NxN where N = len(labels)
        They are generated by the random number generator rng - provided as parameter
        @poram labels: labels
        @param seed: seed for random number generator
        @return cross_correlation matrix
        """
        # self.df_corr = pd.DataFrame(index=labels, columns=labels)
        dimension = len(labels)
        self.df_corr = pd.DataFrame(make_spd_matrix(dimension, random_state=seed), index=labels, columns=labels)
        # Make it a correlation matrix
        diag_inv = [1/np.sqrt(self.df_corr.iloc[i, i]) for i in range(dimension)]
        # for i in range(dimension):
        #     for j in range(dimension):
        #         self.df_corr.iloc[i, j] *= diag_inv[i]
        # for i in range(dimension):
        #     for j in range(dimension):
        #         self.df_corr.iloc[i, j] *= diag_inv[j]

        for idx, ro in enumerate(self.df_corr.index):  # normalize the rows
            self.df_corr.loc[ro] *= diag_inv[idx]
        for idx, cl in enumerate(self.df_corr.columns):  # normalize the rows
            self.df_corr[cl] *= diag_inv[idx]
        if not self._eigen_values_positive():
            logger.error(f"Failed to create a correlation matrix")
            exit(-1)
        return self.df_corr


    def generate_correlated_rvs(self) -> pd.DataFrame:
        """ Generate nb_smpl of Random Variable which are cross-correlated
        df_stat contains 2 columns: "Expected Return" i.e. mean and "Standard Deviation"
        df_corr is the cross-correlation matrix of the variables
        Must have df_stat.index == df_corr.index == df_corr.columns
        nb_smpl is the number of samples that are generated
        Returns an list of  'nb_asset" lists - each list is nb_smpl long
        nb_asset = len(df_stat.index)
        """

        # Compute eigen values to make sure none is negative - otherwise, cholesky will fail
        try:
            eigen_values = np.linalg.eigvals(self.df_corr.astype(float))
        except np.linalg.LinAlgError as e:
            logger.error(f"correlated_rvs could not converge on eigen value decomposition")
            exit(-1)
        negative_eigen = [eigen < 0 for eigen in eigen_values ]
        if any(negative_eigen):
            logger.error(f"correlated_rvs cannot handle matrices with negative eigen values:\n{eigen_values}")
            exit(-1)


        data_series_list = []  # list of nb_asset data series
        return_list = list[float](self.df_stat['Expected Return'].values)
        stddev_list = list[float](self.df_stat['Standard Deviation'].values)
        # Create nb_asset lists of nb_smpl random variables
        for _ in range(len(self.df_stat.index)):
            # ToFix the random_state option is supposed to make the random number generator repeatable - however it
            #  seems to break mean/loc and scale/stddev
            # x1 = norm.rvs(size=nb_smpl, loc=0.0, scale=1.0, random_state=RvsRandomState)  # create a series w/ the desired
            x1 = norm.rvs(size=self.nb_smpl, loc=0.0, scale=1.0)  # create a series w/ the desired stats
            data_series_list.append(x1)  # add data series list to list of lists
        
        c_matrix = cholesky(self.df_corr, lower=True)
        data_df = pd.DataFrame(np.dot(c_matrix, data_series_list), index=self.df_stat.index)
        
        # Assign the desired mean and std_dev to each series
        def mk_lin_interp(mean_val, std_val):
            def f(x):
                return mean_val + std_val * x
            return f
        for rw, mn, stdv in zip(data_df.index, return_list, stddev_list):
            data_df.loc[rw] = data_df.loc[rw].map(mk_lin_interp(mn, stdv))

        if self.validate_cross_correlation_flag:
            logger.info(f'Validating cross-correlation matrix')
            mean_validate, stddev_validate, corr_validate = self._validate_cross_correlation(data_df)
            if  mean_validate and  stddev_validate   and corr_validate:
                logger.info(f"Successfully validated cross-correlation matrix")
            else:
                error_exit(f"Failed to validate cross-correlation matrix:\nMean Validation: {mean_validate}\nStddev Validation: {stddev_validate}\nCorrelation Validation: {corr_validate}")
        else:
            logger.info(f'Skipping Validation of cross-correlation matrix')
        return data_df


    def _validate_cross_correlation(self,data_df: pd.DataFrame) -> [bool]:
        """
        Validate that a data series has the statistics and cross-correlation passed as parameters
        @param data_df: NxM DF - N series of M samples
        @param self.df_stat: Nx2 DF with expected mean and std_dev as columns
        @param self.df_corr: NxN cross-correlation matrix for series
        @param error_margin: margin of error when testing equality
        @return: True if validation passes
        """
        fct_name = sys._getframe().f_code.co_name
        error_margin = self.validation_error_margin
        logger.debug(f'data_df.index: {data_df.index}')
        logger.debug(f'self.df_stat.index: {self.df_stat.index}')
        logger.debug(f'self.df_corr.index: {self.df_corr.index}')
        logger.debug(f'self.df_corr.columns: {self.df_corr.columns}')
        if set(list(data_df.index)) != set(list(self.df_stat.index)):
            logger.error(f'data_df and self.df_stat don\'t have same index')
            logger.error(f'data_df.index: {data_df.index}')
            logger.error(f'self.df_stat.index: {self.df_stat.index}')
            exit(-1)
        if set(list(data_df.index)) != set(list(self.df_corr.index)):
            logger.error(f'data_df and self.df_corr don\'t have same index')
            logger.error(f'data_df.index: {data_df.index}')
            logger.error(f'self.df_corr.index: {self.df_corr.index}')
            exit(-1)
        if set(list(self.df_corr.index)) != set(list(self.df_corr.columns)):
            logger.error(f'self.df_corr index and columns are different')
            logger.error(f'self.df_corr.index: {self.df_corr.index}')
            logger.error(f'self.df_corr.columns: {self.df_corr.columns}')
            exit(-1)
        mean_ser = data_df.mean(axis=1)

        # FYI: the _validate variables are (flag, error) tuples
        mean_validate, _ = matrix_equal(mean_ser, self.df_stat.iloc[:,0], error_margin)  # mean is the first column

        std_ser = data_df.std(axis=1)
        stddev_validate, _ = matrix_equal(std_ser, self.df_stat.iloc[:,1], error_margin)  # std_dev is second column

        # IMPORTANT: astype(float) is needed otherwise, corr() returns empty DF
        data_corr = data_df.T.astype(float).corr()  # Corr works on columns
        corr_validate, _ = matrix_equal(data_corr, self.df_corr, error_margin)

        return mean_validate,  stddev_validate, corr_validate




    def _make_ben_model(alloc_df):
        # Create the list of allocation choices  %stock/%bond
        allocation_choices = []
        for stock in range(20, 81, 5):
            allocation_choices.append(str(stock) + '/' + str(100-stock))
        logger.info(f'Allocation Choices: {allocation_choices}')

        # Create DF for Ben Models
        ben_df = alloc_df[allocation_choices + ['Morningstar Category']].copy(deep=True)
        ben_df.dropna(axis=0, inplace=True)  # drop rows that are either blank or are used as sub-totals
        ben_df.set_index('Morningstar Category', drop=True, inplace=True)
        logger.info(f'Ben Model:\n{ben_df}')

        # Figure out the mapping using BF Categories
        map_df = alloc_df[['Morningstar Category', 'Mapping']].copy(deep=True)
        map_df.reset_index(drop=True, inplace=True)
        map_df.dropna(axis=0, inplace=True)  # drop rows that are blank
        # Create a matrix of mapping weights - Initialize to 0.
        morning_cat = list(alloc_df['BF Morningstar'].dropna())
        map_weights = pd.DataFrame(0.0, index=morning_cat, columns=map_df['Morningstar Category'], dtype=float)
        logger.info(f'Mapping:\n{map_df}')
        logger.info('\n----\n')

        for col, mapg in zip(map_df['Morningstar Category'], map_df['Mapping']):
            # Determine if there is one or more entries
            map_lst = mapg.split('+')
            map_lst = [x.strip(' ') for x in map_lst]  # get rid of ' '
            wgt = 1.0 / len(map_lst)
            for idx in map_lst:
                logger.info(f'{mapg} {idx} {col} {wgt}')
                map_weights.loc[idx, col] = wgt
        logger.info(f'Mapping Weight Matrix:\n{map_weights}')
        logger.info(f'\nSum of weights by column= \n{map_weights.sum(axis=0)}')
        logger.info(f'\nSum of weights by row= \n{map_weights.sum(axis=1)}')
        logger.info(f'\nSum of total weights = {map_weights.sum().sum()}')

        bf_alloc = map_weights.dot(ben_df)
        return bf_alloc



def main_old(argv):
    prog_name = re.sub("\\.py$", "", os.path.relpath(sys.argv[0]))
    plt_file = prog_name + "_out.pdf"  # replace the ".xlsx" extension
    config_file = prog_name + '.toml'
    today, _ = str(dt.datetime.now()).split(' ')

    param_dict = config_param(config_file, argv)
    nb_smpl = param_dict['nb_smpl']
    model_file = param_dict['model_file']
    xl_out_name = param_dict['xl_out_name']
    xl_wr = pd.ExcelWriter(xl_out_name + '_' + today + '.xlsx')

    df_stat, df_corr = get_asset_stats(param_dict['url_stats'], param_dict['url_corr'])
    # FIXME: the morningstar webpage has the wrong column names for the correlation matrix
    df_corr.columns = df_corr.index
    logger.info(f'\nAsset Class Statistics\n{df_stat}')
    logger.info(f'\nAsset Class Correlations\n{df_corr}')
    df_stat.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)
    df_corr.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)
    asset_class = list(df_corr.index)  # List of assets

    y = correlated_rvs(df_stat, df_corr, nb_smpl)
    # Set index to asset_class
    y['AssetClass'] = asset_class
    y.set_index('AssetClass', drop=True, inplace=True)

    logger.info('\nMean')
    logger.info(y.mean(axis=1))
    logger.info('\nMean Delta')
    logger.info(df_stat['Expected Return'] - y.mean(axis=1))
    logger.info('\nStddev')
    logger.info(y.std(axis=1))
    logger.info('\nStddev Delta')
    logger.info(df_stat['Standard Deviation'] - y.std(axis=1))

    # Validate by computing cross-correlation on generated samples
    corr_out = np.corrcoef(y)

    asset_corr_df = pd.DataFrame(corr_out, index=asset_class,columns=asset_class,dtype=float)
    asset_corr_df.to_excel(xl_wr, sheet_name='validation', float_format='%0.2f', header=True, index=True)
    delta = df_corr - asset_corr_df   # this matrix should be all 0
    logger.info(f"\nNorm of Delta Matrix (should be 0.0): {matrix_norm(delta, ord='fro')}\n")
    delta.to_excel(xl_wr, sheet_name='delta', float_format='%0.2f', header=True, index=True)

    # Perform portfolio mapping from Ben's Portfolio
    alloc_df = pd.read_excel(model_file, sheet_name='Models', header=0, engine='openpyxl')
    alloc_df.to_excel(xl_wr, sheet_name='Mappings', float_format='%0.2f', header=True, index=True)
    bf_alloc = make_ben_model(alloc_df)
    logger.info('\n\nPortfolio allocation')
    logger.info(bf_alloc)
    logger.info(f'Sum of weights by column= \n{bf_alloc.sum(axis=0)}')

    # Verify Stock/Bond ratios
    # Remove empty cells and get rid of index
    bf_alloc['Stock/Bond'] = list(alloc_df['Stock/Bond'].dropna())
    bf_alloc.to_excel(xl_wr, sheet_name='Model', float_format='%0.2f', header=True, index=True)
    stock_bond = bf_alloc.groupby('Stock/Bond',  axis=0).sum()
    stock_bond.to_excel(xl_wr, sheet_name='Stock-Bond Ratios', float_format='%0.2f', header=True, index=True)
    logger.info(f'\nStock-Bond Ratio:\n{stock_bond}')
    bf_alloc.drop(['Stock/Bond'], axis=1, inplace=True)  # no longer needed
    # Save the data file
    xl_wr.close()

    # Plot Asset Correlation Matrix
    logger.info('\nPlotting Asset Correlation Matrix:\n')
    legend_dict = dict()
    legend_dict['title'] = "Asset Class Correlation Matrix"
    legend_dict['x_label'] = "Asset Class"
    legend_dict['y_label'] = "Asset Class"
    legend_dict['xticklabels'] = [make_xlabel(ss, XLabelLen) for ss in asset_corr_df.columns]  # Asset Classes, truncated
    legend_dict['yticklabels'] = [ss.replace(' ', '') for ss in asset_corr_df.columns]  # Asset Classes, 'truncated
    pcm(asset_corr_df, legend_dict=legend_dict, vmin_val=-1, vmax_val=1, plot_file=plt_file)  #
    # Alternative plot
    # plt.imshow(asset_corr_df)
    # plt.colorbar()
    # plt.show()

    return

def main(cmd_line: List[str]):
    morningstar_stats = MorningstarStats(cmd_line)
    df_stat, df_corr = morningstar_stats.get_asset_stats()
    logger.info(f'\nAsset Class Statistics\n{df_stat}')
    logger.info(f'\nAsset Class Correlations\n{df_corr}')
    correlated_rvs = morningstar_stats.generate_correlated_rvs()
    logger.info(f'\nCorrelated Random Variables\n{correlated_rvs}')


if __name__ == '__main__':

    main(sys.argv)
    exit(0)
