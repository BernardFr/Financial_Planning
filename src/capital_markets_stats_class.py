#!/usr/local/bin/python3

"""
Read the stats (and correlation matrix) for the capital markets stats from input file 
e.g. Morningstar / JP Morgan Capital Market 

Synopsis:
- Determine which capital market model to use (Morningstar, JP Morgan, etc.) based on the configuration file
- Read the stats and correlation matrix from the input file
- Validate that the correlation matrix is positive definite (i.e. all eigen values are positive)
- Validate that the stats and correlation matrix are consistent with each other (i.e. the correlation matrix is consistent with the stats) and consistent with the assets in the portfolio
- Generate correlated random variables based on the stats and correlation matrix

Files are in sub-directory of the master directory. Each capital market model has its directory.
The directory contains one file for Stats and Correlation matrix. The file is in Excel format with 2 sheets: Stats and Correlation. 
The directory also contains the asset allocation of the portfolio in a separate file (and a file to map ETF to Asset Class - not used in this program)

Filename convention - assume Capital Market Model is CMM (Morningstar, JPM, etc.)
Master directory: ./Data
Sub-directory: ./Data/CMM
Stats and Correlation matrix file: ./Data/CMM/CMM_Stats_YYYY_MM_DD.xlsx
Portfolio allocation file: ./Data/CMM/CMM_Portfolio_Allocations_YYYY_MM_DD.xlsx
ETF to Asset Class mapping file: ./Data/CMM/CMM_ETF_Asset_Class_Mapping_YYYY_MM_DD.xlsx (not used)

Note that YYYY_MM_DD are the date of the file creation and will be different for each file. The program will use the latest file based on the date in the filename.

"""

import sys
import os
import re
import datetime as dt

import numpy as np
import pandas as pd
from numpy.linalg import norm as matrix_norm
from numpy.random import default_rng

from scipy import stats
from scipy.stats import norm
from scipy.linalg import cholesky
from sklearn.datasets import make_spd_matrix
import requests
# Use helper functions from Color_Map
sys.path.insert(0, './Color_Map/')  # Add path to the directory containing plot_color_matrix
from find_most_recent import find_most_recent
from plot_color_matrix import plot_color_matrix as pcm
from configuration_manager_class import ConfigurationManager
from logger import logger
from typing import Any, List, Optional, Sequence, Tuple, cast
from utilities import error_exit, clean_excel_text

 
# For testing 
TEST_NB_SMPL = 10
TEST_ITER = 5 
DEBUG_FLAG = False


def matrix_equal(df_1: pd.DataFrame, df_2: pd.DataFrame, error_margin: float) -> Tuple[bool, float]:
    """
    Compares 2 DF which can be 1- or 2-dimensional and returns a tuple (flag, error)
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
    return flag, float(err)



class CapitalMarketsStats:
    """
    Class to manage the Capital Markets stats
    """
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = cast(dict[str, Any], self.config_manager.get_class_config(self.__class__.__name__))
        self.stat_df: Optional[pd.DataFrame] = None
        self.corr_df: Optional[pd.DataFrame] = None
        self.capital_market_model = str(self.config['capital_market_model'])

        valid_model = bool(self.capital_market_model.lower() in [x.lower() for x in self.config['valid_capital_market_models']])
        assert valid_model, f"Invalid capital_market_model: {self.capital_market_model}. Must be one of {self.config['valid_capital_market_models']}"
        self.data_directory = str(self.config['data_directory'])
        self.input_directory = os.path.join(self.data_directory, self.capital_market_model)
        self.cpm_date_format = str(self.config['cpm_date_format'])
        
        self.XLabelLen = int(self.config.get('XLabelLen', self.config['XLabelLenDefault']))
        self.quick_flag = bool(self.config['quick_flag'])
        self.validation_error_margin = float(self.config['validation_error_margin'])
        self.validate_cross_correlation_flag = bool(self.config['validate_cross_correlation_flag'])

    def set_nb_smpl(self, nb_smpl: int) -> None:
        """
        Set the number of samples to generate
        @param nb_smpl: number of samples to generate
        """
        self.nb_smpl = nb_smpl
        return


    @staticmethod
    def _make_xlabel(in_str: str, label_len: int) -> str:
        """
        Shorten the X axis labels to XLabelLen characters
        """
        # remove white space
        in_str = in_str.replace(' ', '')
        return in_str[0:label_len]

    def read_cpm_stats(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        @return: 2 DF for asset statistics and correlation matrix
        Correlation matrix is optional - if not available, return None
        read file based on the capital_market_model and return the stats and correlation matrix
        """
        # Build the file path based on the capital_market_model and the latest file in the directory
        file_prefix = self.capital_market_model.lower() + self.config['cpm_file_prefix']
        cpm_file, _ = find_most_recent(self.input_directory, file_prefix, self.cpm_date_format)
        if cpm_file is None:
            error_exit("No CPM stats file found")
            raise FileNotFoundError("Cannot find CPM Stats file")
        logger.info(f"Reading Capital Market stats from file: {cpm_file}")
        # Check that the file contains the expected sheets: Stats and Correlation
        try:
            stat_df = clean_excel_text(pd.read_excel(cpm_file, sheet_name='Stats', header=0, engine='openpyxl', index_col=0))
            logger.info(f"stat_df =\n{stat_df}")
        except Exception as e:
            logger.error(f"Failed to read Stats sheet from file: {cpm_file}. Error: {e}\nAborting")
            sys.exit("Cannot read CPM Stats")

        try:
            xls = pd.ExcelFile(cpm_file, engine='openpyxl')
            if 'Correlation' in xls.sheet_names:
                corr_df = clean_excel_text(pd.read_excel(cpm_file, sheet_name='Correlation', header=0, engine='openpyxl', index_col=0))
                logger.info(f"corr_df =\n{corr_df}")
            else:
                logger.info(f"Correlation sheet not found in file: {cpm_file}. Correlation matrix will be None")
                corr_df = None
        except Exception as e:
            logger.warning(f"Failed to read Correlation sheet from file: {cpm_file}. Correlation matrix will be None. Error: {e}")
            corr_df = None


        self.stat_df = stat_df
        self.corr_df = corr_df

        return self.stat_df, self.corr_df


    def match_stats_vs_assets(self, asset_class_ser: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Match the asset statistics and correlation matrix to the holdings
        @param asset_class_ser: asset class Series
        @return: (stat_df, corr_df)        
        Check if the asset classes list and the assets in the portfolio are the same
        If asset_class_ser does not have all the asset classes in stat_df, corr_df - strip the unused asset classes from stat_df, corr_df
        If stat_df, corr_df do not have all the asset classes in asset_class_ser - Error & exit
        Return the stripped stat_df, corr_df - re-indexed to the asset classes in the portfolio
        """
        stat_df, corr_df = self._require_stat_and_corr_df()
        asset_class_set = set(asset_class_ser.index)
        stats_set = set(stat_df.index)
        missing_stats = [x for x in asset_class_set if x not in stats_set]
        if len(missing_stats) > 0:
            # Exit we are missing stats for some asset classes in the portfolio
            error_exit(f"Some asset classes in the portfolio are not in Morningstar asset classes: {missing_stats}")

        # Now drop the stats that are not in the portfolio
        missing_assets = [x for x in stats_set if x not in asset_class_set]
        if len(missing_assets) > 0:  # Some Morningstar stats have assets classes that are not in the portfolio
            logger.info(f"Some Morningstar asset classes are extra (not in the asset_class_ser): {missing_assets}")
            logger.info("Taking the extra asset classes out of the stats and correlation dataframes")
            stat_df = stat_df.drop(index=missing_assets)
            corr_df = corr_df.drop(index=missing_assets, columns=missing_assets).copy(deep=True)
        
        # Index the stats and correlation matrices to the asset classes in the portfolio
        stat_df = stat_df.reindex(index=asset_class_ser.index)
        corr_df = corr_df.reindex(index=asset_class_ser.index, columns=asset_class_ser.index)
        self.stat_df = stat_df
        self.corr_df = corr_df
        return self.stat_df, self.corr_df



    def _eigen_values_positive(self) -> bool:
        """ Returns True if the eigen values of a correlation matrix are positive"""
        # ToDo: add teststo ensure that corr_df is a correlation matrix
        _, corr_df = self._require_stat_and_corr_df()
        try:
            eigen_values = np.linalg.eigvals(corr_df.astype(float))
        except np.linalg.LinAlgError:
            return False  # not a viable matrix
        positive_eigen = [eigen >= 0 for eigen in eigen_values ]
        return True if all(positive_eigen) else False


    def _make_random_correlation_matrix(self, labels: List[str], seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generates a cross correlation matrix with labels as labels for index and columns
        the size of the matrix is NxN where N = len(labels)
        They are generated by the random number generator rng - provided as parameter
        @param labels: labels
        @param seed: seed for random number generator
        @return cross_correlation matrix
        """
        # self.corr_df = pd.DataFrame(index=labels, columns=labels)
        dimension = len(labels)
        self.corr_df = pd.DataFrame(make_spd_matrix(dimension, random_state=seed), index=labels, columns=labels, dtype=float)
        corr_df = self.corr_df
        # Make it a correlation matrix
        diag_inv = [1 / np.sqrt(float(corr_df.iloc[i, i])) for i in range(dimension)]
        # for i in range(dimension):
        #     for j in range(dimension):
        #         self.corr_df.iloc[i, j] *= diag_inv[i]
        # for i in range(dimension):
        #     for j in range(dimension):
        #         self.corr_df.iloc[i, j] *= diag_inv[j]

        for idx, ro in enumerate(corr_df.index):  # normalize the rows
            corr_df.loc[ro] *= diag_inv[idx]
        for idx, cl in enumerate(corr_df.columns):  # normalize the rows
            corr_df[cl] *= diag_inv[idx]
        if not self._eigen_values_positive():
            error_exit(f"Failed to create a correlation matrix")
        self.corr_df = corr_df
        return corr_df


    def generate_correlated_ror(self, rng_sequence: list[list[np.random.Generator]]) -> pd.DataFrame:
        """ Generate nb_smpl of Random Variable which are cross-correlated
        stat_df contains 2 columns: "Expected Return" i.e. mean and "Standard Deviation"
        corr_df is the cross-correlation matrix of the variables
        Must have stat_df.index == corr_df.index == corr_df.columns
        nb_smpl is the number of samples that are generated
        Returns a DF with the index as the asset classes and the columns as the samples
        The values are the RoR multipliers - ie. 1.18 (<- 18%) -- 1 + 0.01 * rvs ...  
        """

        stat_df, corr_df = self._require_stat_and_corr_df()

        # Compute eigen values to make sure none is negative - otherwise, cholesky will fail
        try:
            eigen_values = np.linalg.eigvals(corr_df.astype(float))
        except np.linalg.LinAlgError as e:
            error_exit(f"correlated_ror could not converge on eigen value decomposition: {e}")
            raise RuntimeError(f"correlated_ror could not converge on eigen value decomposition: {e}")
        negative_eigen = [eigen < 0 for eigen in eigen_values ]
        if any(negative_eigen):
            error_exit(f"correlated_ror cannot handle matrices with negative eigen values:\n{eigen_values}")


        data_series_list = []  # list of nb_asset data series
        return_list = list[float](stat_df['Expected Return'].values)
        stddev_list = list[float](stat_df['Standard Deviation'].values)
        # FIXME for mult-CPU support
        if rng_sequence is None:
            rng_sequence = np.random.SeedSequence()
        child_seeds = rng_sequence.spawn(len(stat_df.index))
        for child_seed in child_seeds:
            rng = default_rng(child_seed)
            x1 = norm.rvs(size=self.nb_smpl, loc=0.0, scale=1.0, random_state=rng)  # create a series w/ the desired stats
            data_series_list.append(x1)  # add data series list to list of lists
        c_matrix = cholesky(corr_df, lower=True)
        data_df = pd.DataFrame(np.dot(c_matrix, data_series_list), index=stat_df.index)
        # Assign the desired mean and std_dev to each row
        def mk_lin_interp(mean_val: float, std_val: float):
            def f(x: float) -> float:
                return mean_val + std_val * x
            return f
        rvs_df = pd.DataFrame(index=stat_df.index, columns=range(self.nb_smpl))
        for rw, mn, stdv in zip(data_df.index, return_list, stddev_list):
            rvs_df.loc[rw] = data_df.loc[rw].map(mk_lin_interp(mn, stdv))
        if DEBUG_FLAG:
            print(f"\nDEBUG: comparing target stats vs generated stats ")
            for nn,idx in enumerate(stat_df.index):
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
        
        # Transform numbers from % to multipliers 15% -> 1.15, etc.
        ror_df = rvs_df.map(lambda x: 1 + 0.01 * x)
        return ror_df



    def _validate_cross_correlation(self, data_df: pd.DataFrame) -> Tuple[bool, float, bool, float, bool, float]:
        """
        Validate that a data series has the statistics and cross-correlation passed as parameters
        @param data_df: NxM DF - N series of M samples
        @param self.stat_df: Nx2 DF with expected mean and std_dev as columns
        @param self.corr_df: NxN cross-correlation matrix for series
        @param error_margin: margin of error when testing equality
        @return: True if validation passes
        """
        stat_df, corr_df = self._require_stat_and_corr_df()
        logger.debug(f'data_df.index: {data_df.index}')
        logger.debug(f'self.stat_df.index: {stat_df.index}')
        logger.debug(f'self.corr_df.index: {corr_df.index}')
        logger.debug(f'self.corr_df.columns: {corr_df.columns}')
        if set(list(data_df.index)) != set(list(stat_df.index)):
            logger.error(f'data_df and self.stat_df don\'t have same index')
            logger.error(f'data_df.index: {data_df.index}')
            logger.error(f'self.stat_df.index: {stat_df.index}')
            exit(-1)
        if set(list(data_df.index)) != set(list(corr_df.index)):
            logger.error(f'data_df and self.corr_df don\'t have same index')
            logger.error(f'data_df.index: {data_df.index}')
            logger.error(f'self.corr_df.index: {corr_df.index}')
            exit(-1)
        if set(list(corr_df.index)) != set(list(corr_df.columns)):
            logger.error(f'self.corr_df index and columns are different')
            logger.error(f'self.corr_df.index: {corr_df.index}')
            logger.error(f'self.corr_df.columns: {corr_df.columns}')
            exit(-1)

        error_margin = self.validation_error_margin
        # FYI: the _validate variables are (flag, error) tuples
        mean_ser = pd.DataFrame(data_df.mean(axis=1))
        mean_validate, mean_error = matrix_equal(mean_ser, stat_df.iloc[:,0:1], error_margin)  # mean is the first column

        std_ser = pd.DataFrame(data_df.std(axis=1))
        stddev_validate, stddev_error = matrix_equal(std_ser, stat_df.iloc[:,1:2], error_margin)  # std_dev is second column

        # IMPORTANT: astype(float) is needed otherwise, corr() returns empty DF
        data_corr = data_df.T.astype(float).corr()  # Corr works on columns
        corr_validate, corr_error = matrix_equal(data_corr, corr_df, error_margin)

        return mean_validate,  mean_error, stddev_validate, stddev_error, corr_validate, corr_error
    
    def plot_correlation_matrix(self) -> None:
        """
        Plot the correlation matrix using the plot_color_matrix function
        """
        corr_df = self.corr_df
        if corr_df is None: 
            logger.error(f"Correlation matrix is None - cannot plot")
            return
        logger.info('\nPlotting Asset Correlation Matrix:\n')
        legend_dict = dict()
        legend_dict['title'] = "Asset Class Correlation Matrix"
        legend_dict['x_label'] = "Asset Class"
        legend_dict['y_label'] = "Asset Class"
        legend_dict['xticklabels'] = [self._make_xlabel(ss, self.XLabelLen) for ss in corr_df.columns]  # Asset Classes, truncated
        legend_dict['yticklabels'] = [ss.replace(' ', '') for ss in corr_df.columns]  # Asset Classes, 'truncated
        pcm(corr_df, legend_dict=legend_dict, vmin_val=-1, vmax_val=1, plot_file=plt_file)  
        #
        # Alternative plot
        # plt.imshow(asset_corr_df)
        # plt.colorbar()
        # plt.show()



    @staticmethod
    def _xx_make_ben_model(alloc_df: pd.DataFrame) -> pd.DataFrame:
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



def main_old(argv: Sequence[str]) -> None:
    """ FOR REFERENCE ONLY - NOT USED IN THE PROGRAM """
    prog_name = re.sub("\\.py$", "", os.path.relpath(sys.argv[0]))
    plt_file = prog_name + "_out.pdf"  # replace the ".xlsx" extension
    config_file = prog_name + '.toml'
    today, _ = str(dt.datetime.now()).split(' ')

    param_dict = config_param(config_file, argv) # type: ignore 
    nb_smpl = param_dict['nb_smpl']
    model_file = param_dict['model_file']
    xl_out_name = param_dict['xl_out_name']
    xl_wr = pd.ExcelWriter(xl_out_name + '_' + today + '.xlsx')

    stat_df, corr_df = read_cpm_stats(param_dict['url_stats'], param_dict['url_corr']) # type: ignore 
    # FIXME: the morningstar webpage has the wrong column names for the correlation matrix
    corr_df.columns = corr_df.index
    logger.info(f'\nAsset Class Statistics\n{stat_df}')
    logger.info(f'\nAsset Class Correlations\n{corr_df}')
    stat_df.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)
    corr_df.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)
    asset_class = list(corr_df.index)  # List of assets

    y = correlated_rvs(stat_df, corr_df, nb_smpl) # type: ignore 
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
    alloc_df = clean_excel_text(pd.read_excel(model_file, sheet_name='Models', header=0, engine='openpyxl'))
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

def main(cmd_line: List[str]) -> None:
    config_manager = ConfigurationManager(sys.argv)
    cpm_stats_class = CapitalMarketsStats(config_manager)
    stat_df, corr_df = cpm_stats_class.read_cpm_stats()
    logger.info(f'\nAsset Class Statistics\n{stat_df}')
    logger.info(f'\nAsset Class Correlations\n{corr_df}')



    # Testing
    cpm_stats_class.set_nb_smpl(config_manager.config['nb_smpl_default'])
    correlated_ror = cpm_stats_class.generate_correlated_ror()
    logger.info(f'\nCorrelated RoR multipliers\n{correlated_ror}')
    logger.info(f'\nCorrelated RoR\n{correlated_ror}')
    
    print(f"\nConfirming that correlated_rvs can be called several times and produce different results ")
    cpm_stats_class.set_nb_smpl(TEST_NB_SMPL)
    for i in range(TEST_ITER):
        correlated_ror_2 = cpm_stats_class.generate_correlated_ror()
        # print(f"correlated_ror_2.mean(): {100*(-1+correlated_ror_2.mean(axis=1))}")
        # print(f"correlated_ror_2.std(): {100*correlated_ror_2.std(axis=1)}")
        logger.info(f'iter: {i} \nCorrelated RoR multipliers {i}\n{correlated_ror_2}')

if __name__ == '__main__':

    main(sys.argv)
    exit(0)
