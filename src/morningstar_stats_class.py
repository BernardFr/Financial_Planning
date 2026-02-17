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
NB_SMPL_DEFAULT = 100000

DEBUG_FLAG = False


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
    denom = len(df_1.index) if df_1.ndim == 1 else len(df_1.index) * len(df_1.columns)  # needed to handle 1-dimensional DF
    err = np.linalg.norm(delta) / denom
    flag = True if abs(err) <= abs(error_margin) else False
    return flag, err



class MorningstarStats:
    """
    Class to manage the Morningstar stats
    """
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        
        self.must_have_param = self.config['must_have_param']
        self.XLabelLen = self.config['XLabelLen']
        self.must_have_param = self.config['must_have_param']
        self.XLabelLen = self.config.get('XLabelLen', XLabelLenDefault)
        self.quick_flag = self.config.get('quick_flag', QuickFlagDefault)
        self.stat_corr_name_map = dict[str, str](self.config['STAT_CORR_NAME_MAP']) # map the list of 2-element lists to a dict
        self.validation_error_margin = self.config['validation_error_margin']
        self.validate_cross_correlation_flag = self.config['validate_cross_correlation_flag']

    def set_nb_smpl(self, nb_smpl: int) -> None:
        """
        Set the number of samples to generate
        @param nb_smpl: number of samples to generate
        """
        self.nb_smpl = nb_smpl
        return

    def set_stat_df_and_corr_df(self, stat_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
        """
        Set the asset statistics and correlation matrix -- needed for when we match stats and holdings
        @param stat_df: asset statistics
        @param corr_df: asset correlation matrix
        """
        self.stat_df = stat_df
        self.corr_df = corr_df
        return

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
        # test if stat_df and corr_df have been created
        if hasattr(self, 'stat_df') and hasattr(self, 'corr_df'):
            return self.stat_df, self.corr_df

        # else, we have to compute them
        self.stat_df = self._get_morningstar_stats()
        self.corr_df = self._get_morningstar_corr()
        # Remap the name of asset classes to match the names in corr_df
        self._remap_names()

        return self.stat_df, self.corr_df


    def match_stats_vs_assets(self, asset_class_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Match the asset statistics and correlation matrix to the holdings
        @param asset_class_df: asset class DataFrame
        @return: (stat_df, corr_df)        
        Check if the asset classes list and the assets in the portfolio are the same
        If asset_class_df does not have all the asset classes in stat_df, corr_df - strip the unused asset classes from stat_df, corr_df
        If stat_df, corr_df do not have all the asset classes in asset_class_df - Error & exit
        Return the stripped stat_df, corr_df - re-indexed to the asset classes in the portfolio
        """
        asset_class_set = set(asset_class_df.index)  
        stats_set = set(self.stat_df.index)
        missing_stats = [x for x in asset_class_set if x not in stats_set]
        if len(missing_stats) > 0:
            # Exit we are missing stats for some asset classes in the portfolio
            error_exit(f"Some asset classes in the portfolio are not in Morningstar asset classes: {missing_stats}")

        # Now drop the stats that are not in the portfolio
        missing_assets = [x for x in stats_set if x not in asset_class_set]
        if len(missing_assets) > 0:  # Some Morningstar stats have assets classes that are not in the portfolio
            logger.info(f"Some Morningstar asset classes are extra (not in the asset_class_df): {missing_assets}")
            logger.info("Taking the extra asset classes out of the stats and correlation dataframes")
            self.stat_df = self.stat_df.drop(index=missing_assets)
            self.corr_df = self.corr_df.drop(index=missing_assets, columns=missing_assets).copy(deep=True)
        
        # Index the stats and correlation matrices to the asset classes in the portfolio
        self.stat_df = self.stat_df.reindex(index=asset_class_df.index)
        self.corr_df = self.corr_df.reindex(index=asset_class_df.index, columns=asset_class_df.index)
        return self.stat_df, self.corr_df


    def _remap_names(self) -> None:
        """
        Remap the names in names according to the mapping of old_names new_names
        """
        # Check that stat_df and corr_df indexes match DF_STAT_NAMES and DF_CORR_NAMES
        error_flag = False
        if set(list(self.stat_df.index)) != set(self.stat_corr_name_map.keys()):
            error_flag = True
            logger.error(f"stat_df index: {list(self.stat_df.index)} does not match STAT_CORR_NAME_MAP keys: {self.stat_corr_name_map.keys()}")
        if set(list(self.corr_df.index)) != set(self.stat_corr_name_map.values()):
            error_flag = True
            logger.error(f"corr_df index: {list(self.corr_df.index)} does not match STAT_CORR_NAME_MAP values: {self.stat_corr_name_map.values()}")
        if error_flag:
            error_exit(f"Error in remapping names")

        # Remap the index of stat_df to match the index of corr_df
        asset_class_names = [self.stat_corr_name_map.get(x, "Error") for x in self.stat_df.index]  # remap
        if "Error" in asset_class_names:
            error_exit(f"Error(s) in remapping names: {self.stat_df.index} -> {asset_class_names}")
        self.stat_df['IDX'] = asset_class_names
        self.stat_df.set_index('IDX', drop=True, inplace=True)
        # Reorder  stat_df index to match corr_df index
        self.stat_df = self.stat_df.reindex(index=self.corr_df.index)

        # FIXME: HACK the Morningstar webpage has the wrong column names for the correlation matrix
        self.corr_df.columns = self.corr_df.index

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
        # ToDo: add teststo ensure that corr_df is a correlation matrix
        try:
            eigen_values = np.linalg.eigvals(self.corr_df.astype(float))
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
        # self.corr_df = pd.DataFrame(index=labels, columns=labels)
        dimension = len(labels)
        self.corr_df = pd.DataFrame(make_spd_matrix(dimension, random_state=seed), index=labels, columns=labels)
        # Make it a correlation matrix
        diag_inv = [1/np.sqrt(self.corr_df.iloc[i, i]) for i in range(dimension)]
        # for i in range(dimension):
        #     for j in range(dimension):
        #         self.corr_df.iloc[i, j] *= diag_inv[i]
        # for i in range(dimension):
        #     for j in range(dimension):
        #         self.corr_df.iloc[i, j] *= diag_inv[j]

        for idx, ro in enumerate(self.corr_df.index):  # normalize the rows
            self.corr_df.loc[ro] *= diag_inv[idx]
        for idx, cl in enumerate(self.corr_df.columns):  # normalize the rows
            self.corr_df[cl] *= diag_inv[idx]
        if not self._eigen_values_positive():
            error_exit(f"Failed to create a correlation matrix")
        return self.corr_df


    def generate_correlated_rvs(self) -> pd.DataFrame:
        """ Generate nb_smpl of Random Variable which are cross-correlated
        stat_df contains 2 columns: "Expected Return" i.e. mean and "Standard Deviation"
        corr_df is the cross-correlation matrix of the variables
        Must have stat_df.index == corr_df.index == corr_df.columns
        nb_smpl is the number of samples that are generated
        Returns a DF with the index as the asset classes and the columns as the samples
        """

        # Compute eigen values to make sure none is negative - otherwise, cholesky will fail
        try:
            eigen_values = np.linalg.eigvals(self.corr_df.astype(float))
        except np.linalg.LinAlgError as e:
            error_exit(f"correlated_rvs could not converge on eigen value decomposition: {e}")
        negative_eigen = [eigen < 0 for eigen in eigen_values ]
        if any(negative_eigen):
            error_exit(f"correlated_rvs cannot handle matrices with negative eigen values:\n{eigen_values}")


        data_series_list = []  # list of nb_asset data series
        return_list = list[float](self.stat_df['Expected Return'].values)
        stddev_list = list[float](self.stat_df['Standard Deviation'].values)
        # Create nb_asset lists of nb_smpl random variables
        for _ in range(len(self.stat_df.index)):
            x1 = norm.rvs(size=self.nb_smpl, loc=0.0, scale=1.0)  # create a series w/ the desired stats
            data_series_list.append(x1)  # add data series list to list of lists
        
        c_matrix = cholesky(self.corr_df, lower=True)
        data_df = pd.DataFrame(np.dot(c_matrix, data_series_list), index=self.stat_df.index)
        
        # Assign the desired mean and std_dev to each row
        def mk_lin_interp(mean_val, std_val):
            def f(x):
                return mean_val + std_val * x
            return f
        rvs_df = pd.DataFrame(index=self.stat_df.index, columns=range(self.nb_smpl))
        for rw, mn, stdv in zip(data_df.index, return_list, stddev_list):
            rvs_df.loc[rw] = data_df.loc[rw].map(mk_lin_interp(mn, stdv))
        if DEBUG_FLAG:
            print(f"\nDEBUG: comparing target stats vs generated stats ")
            for nn,idx in enumerate(self.stat_df.index):
                print(f'{idx}  Expected Return: {return_list[nn]} -  Generated Return: {rvs_df.loc[idx].mean()}')
                print(f'{idx}  Expected Stddev: {stddev_list[nn]} -  Generated Stddev:  {rvs_df.loc[idx].std()}')

        if self.validate_cross_correlation_flag or DEBUG_FLAG:
            logger.info(f'Validating cross-correlation matrix')
            mean_validate, mean_error, stddev_validate, stddev_error, corr_validate, corr_error = self._validate_cross_correlation(rvs_df)
            if  mean_validate and  stddev_validate  and corr_validate:
                logger.info(f"Successfully validated cross-correlation matrix")
            else:
                logger.error(f"Mean Error: {mean_error}\nStddev Error: {stddev_error}\nCorrelation Error: {corr_error}")
                logger.error(f"Failed to validate cross-correlation matrix")
        else:
            logger.info(f'Skipping Validation of cross-correlation matrix')
        # FIXME: Figure out how to clip the rvs_df - like ArrayRandGen class
        return rvs_df


    def _validate_cross_correlation(self,data_df: pd.DataFrame) -> [bool, float, bool, float, bool, float]:
        """
        Validate that a data series has the statistics and cross-correlation passed as parameters
        @param data_df: NxM DF - N series of M samples
        @param self.stat_df: Nx2 DF with expected mean and std_dev as columns
        @param self.corr_df: NxN cross-correlation matrix for series
        @param error_margin: margin of error when testing equality
        @return: True if validation passes
        """
        logger.debug(f'data_df.index: {data_df.index}')
        logger.debug(f'self.stat_df.index: {self.stat_df.index}')
        logger.debug(f'self.corr_df.index: {self.corr_df.index}')
        logger.debug(f'self.corr_df.columns: {self.corr_df.columns}')
        if set(list(data_df.index)) != set(list(self.stat_df.index)):
            logger.error(f'data_df and self.stat_df don\'t have same index')
            logger.error(f'data_df.index: {data_df.index}')
            logger.error(f'self.stat_df.index: {self.stat_df.index}')
            exit(-1)
        if set(list(data_df.index)) != set(list(self.corr_df.index)):
            logger.error(f'data_df and self.corr_df don\'t have same index')
            logger.error(f'data_df.index: {data_df.index}')
            logger.error(f'self.corr_df.index: {self.corr_df.index}')
            exit(-1)
        if set(list(self.corr_df.index)) != set(list(self.corr_df.columns)):
            logger.error(f'self.corr_df index and columns are different')
            logger.error(f'self.corr_df.index: {self.corr_df.index}')
            logger.error(f'self.corr_df.columns: {self.corr_df.columns}')
            exit(-1)

        error_margin = self.validation_error_margin
        # FYI: the _validate variables are (flag, error) tuples
        mean_ser = data_df.mean(axis=1)
        mean_validate, mean_error = matrix_equal(mean_ser, self.stat_df.iloc[:,0], error_margin)  # mean is the first column

        std_ser = data_df.std(axis=1)
        stddev_validate, stddev_error = matrix_equal(std_ser, self.stat_df.iloc[:,1], error_margin)  # std_dev is second column

        # IMPORTANT: astype(float) is needed otherwise, corr() returns empty DF
        data_corr = data_df.T.astype(float).corr()  # Corr works on columns
        corr_validate, corr_error = matrix_equal(data_corr, self.corr_df, error_margin)

        return mean_validate,  mean_error, stddev_validate, stddev_error, corr_validate, corr_error




    def _xx_make_ben_model(alloc_df):
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
    """ FOR REFERENCE ONLY - NOT USED IN THE PROGRAM """
    prog_name = re.sub("\\.py$", "", os.path.relpath(sys.argv[0]))
    plt_file = prog_name + "_out.pdf"  # replace the ".xlsx" extension
    config_file = prog_name + '.toml'
    today, _ = str(dt.datetime.now()).split(' ')

    param_dict = config_param(config_file, argv)
    nb_smpl = param_dict['nb_smpl']
    model_file = param_dict['model_file']
    xl_out_name = param_dict['xl_out_name']
    xl_wr = pd.ExcelWriter(xl_out_name + '_' + today + '.xlsx')

    stat_df, corr_df = get_asset_stats(param_dict['url_stats'], param_dict['url_corr'])
    # FIXME: the morningstar webpage has the wrong column names for the correlation matrix
    corr_df.columns = corr_df.index
    logger.info(f'\nAsset Class Statistics\n{stat_df}')
    logger.info(f'\nAsset Class Correlations\n{corr_df}')
    stat_df.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)
    corr_df.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)
    asset_class = list(corr_df.index)  # List of assets

    y = correlated_rvs(stat_df, corr_df, nb_smpl)
    # Set index to asset_class
    y['AssetClass'] = asset_class
    y.set_index('AssetClass', drop=True, inplace=True)

    logger.info('\nMean')
    logger.info(y.mean(axis=1))
    logger.info('\nMean Delta')
    logger.info(stat_df['Expected Return'] - y.mean(axis=1))
    logger.info('\nStddev')
    logger.info(y.std(axis=1))
    logger.info('\nStddev Delta')
    logger.info(stat_df['Standard Deviation'] - y.std(axis=1))

    # Validate by computing cross-correlation on generated samples
    corr_out = np.corrcoef(y)

    asset_corr_df = pd.DataFrame(corr_out, index=asset_class,columns=asset_class,dtype=float)
    asset_corr_df.to_excel(xl_wr, sheet_name='validation', float_format='%0.2f', header=True, index=True)
    delta = corr_df - asset_corr_df   # this matrix should be all 0
    logger.info(f"\nNorm of Delta Matrix (should be 0.0): {matrix_norm(delta, ord='fro')}\n")
    delta.to_excel(xl_wr, sheet_name='delta', float_format='%0.2f', header=True, index=True)

    # Perform portfolio mapping from Ben's Portfolio
    alloc_df = pd.read_excel(model_file, sheet_name='Models', header=0, engine='openpyxl')
    alloc_df.to_excel(xl_wr, sheet_name='Mappings', float_format='%0.2f', header=True, index=True)
    bf_alloc = self._xx_make_ben_model(alloc_df)
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
    legend_dict['xticklabels'] = [self._make_xlabel(ss, self.XLabelLen) for ss in asset_corr_df.columns]  # Asset Classes, truncated
    legend_dict['yticklabels'] = [ss.replace(' ', '') for ss in asset_corr_df.columns]  # Asset Classes, 'truncated
    pcm(asset_corr_df, legend_dict=legend_dict, vmin_val=-1, vmax_val=1, plot_file=plt_file)  #
    # Alternative plot
    # plt.imshow(asset_corr_df)
    # plt.colorbar()
    # plt.show()

    return

def main(cmd_line: List[str]):
    config_manager = ConfigurationManager(sys.argv)
    morningstar_stats = MorningstarStats(config_manager)
    stat_df, corr_df = morningstar_stats.get_asset_stats()
    logger.info(f'\nAsset Class Statistics\n{stat_df}')
    logger.info(f'\nAsset Class Correlations\n{corr_df}')
    morningstar_stats.set_nb_smpl(NB_SMPL_DEFAULT)
    correlated_rvs = morningstar_stats.generate_correlated_rvs()
    correlated_ror = correlated_rvs.map(lambda x: 1 +x * 0.01)
    # if DEBUG_FLAG:
    #     logger.info(f'\nCorrelated Random Variables\n{correlated_rvs}')
    logger.info(f'\nCorrelated Random Variables\n{correlated_rvs}')
    logger.info(f'\nCorrelated RoR\n{correlated_ror}')
    
    print(f"Confirming that correlated_rvs can be called several times and produce different results ")
    morningstar_stats.set_nb_smpl(10)
    for i in range(5):
        correlated_rvs_2 = morningstar_stats.generate_correlated_rvs()
        logger.info(f'iter: {i} \nCorrelated Random Variables {i}\n{correlated_rvs_2}')

if __name__ == '__main__':

    main(sys.argv)
    exit(0)
