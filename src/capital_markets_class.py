#!/usr/local/bin/python3

"""
Capital Markets Operations 
"""

from logging import config
import sys
import os
import re
import datetime as dt

import numpy as np
import pandas as pd
from numpy.linalg import norm as matrix_norm
from numpy.random import default_rng

from scipy.stats import norm
from scipy.linalg import cholesky
from sklearn.datasets import make_spd_matrix
# Use helper functions from Color_Map
sys.path.insert(0, '../Color_Map/')  # Add path to the directory containing plot_color_matrix
from plot_color_matrix import plot_color_matrix as pcm
from configuration_manager_class import ConfigurationManager
from logger import logger
from typing import Any, List, Optional, Sequence, Tuple, cast
from utilities import error_exit, clean_excel_text
from find_most_recent import find_most_recent

XLabelLenDefault = 5  # Length of the X axis labels
QuickFlagDefault = False
# Random state for the random number generator used to generate the random variables to ensure we have the same
# numbers across runs
RvsRandomState = 42
NB_SMPL_DEFAULT = 100000
NB_ITER_DEFAULT = 5
DEBUG_FLAG = False


class CapitalMarkets:
    """
    Class to manage the Capital Markets stats
    """
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = cast(dict[str, Any], self.config_manager.get_class_config(self.__class__.__name__))
        self.stat_df: Optional[pd.DataFrame] = None
        self.corr_df: Optional[pd.DataFrame] = None
        self.XLabelLen = int(self.config.get('XLabelLen', XLabelLenDefault))
        self.quick_flag = bool(self.config.get('quick_flag', QuickFlagDefault))
        if DEBUG_FLAG:
            # set the logging level to DEBUG if the DEBUG_FLAG is set
            logger.setLevel('DEBUG')
            logger.debug(f"CapitalMarkets config: {self.config}")

    def load_capital_markets_stats(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ 
        Load the Capital Markets stats and set the stats_df and corr_df attributes in the capital_markets_stats object
        Use the index of the correlation matrix as the source of truth for the asset classes
        """
        input_dir = self.config['input_directory']
        capital_markets_file_prefix = self.config['capital_markets_file_prefix']
        capital_markets_date_format = self.config['capital_markets_date_format']
        capital_markets_file, _ = find_most_recent(input_dir, capital_markets_file_prefix, capital_markets_date_format)
        logger.info(f"Capital Markets file: {capital_markets_file}")
        self.stats_df = clean_excel_text(pd.read_excel(capital_markets_file, sheet_name='Stats', index_col=0, header=0))
        self.corr_df = clean_excel_text(pd.read_excel(capital_markets_file, sheet_name='Correlation', index_col=0, header=0))
        if "Yield" in self.stats_df.columns:
            self.stats_df.drop( columns=['Yield'], inplace=True) 
        # Make sure that stats_df.index, corr_df.index and corr_df.columns are the same and in the same order
        corr_index = self.corr_df.index   # source of truth
        corr_columns = self.corr_df.columns 
        stats_index = self.stats_df.index
        if not (stats_index.equals(corr_columns) and stats_index.equals(corr_index)):
            error_exit(f"Stats index and Correlation index and columns do not match:\nStats index: {stats_index}\nCorrelation index: {corr_index}\nCorrelation columns: {corr_columns}")
        return self.stats_df, self.corr_df


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
        self.stats_df = stat_df
        self.corr_df = corr_df
        return


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
        if self.stats_df is None or self.corr_df is None:
            self.load_capital_markets_stats()
        assert isinstance(asset_class_ser, pd.Series), f"asset_class_ser must be a pandas Series"
        assert self.stats_df is not None, f"stats_df is None"
        assert self.corr_df is not None, f"corr_df is None"
        stat_df = self.stats_df
        corr_df = self.corr_df
        asset_class_set = set(asset_class_ser.index)
        stats_set = set(stat_df.index)
        missing_stats = [x for x in asset_class_set if x not in stats_set]
        if len(missing_stats) > 0:
            # Exit we are missing stats for some asset classes in the portfolio
            error_exit(f"Some asset classes in the portfolio are not in Capital Markets asset classes: {missing_stats}")

        # Now drop the stats that are not in the portfolio
        missing_assets = [x for x in stats_set if x not in asset_class_set]
        if len(missing_assets) > 0:  # Some Morningstar stats have assets classes that are not in the portfolio
            logger.info(f"Some Capital Markets asset classes are extra (not in the asset_class_ser): {missing_assets}")
            logger.info("Taking the extra asset classes out of the stats and correlation dataframes")
            stat_df = stat_df.drop(index=missing_assets)
            corr_df = corr_df.drop(index=missing_assets, columns=missing_assets).copy(deep=True)
        
        # Index the stats and correlation matrices to the asset classes in the portfolio
        stat_df = stat_df.reindex(index=asset_class_ser.index)
        corr_df = corr_df.reindex(index=asset_class_ser.index, columns=asset_class_ser.index)
        self.stats_df = stat_df
        self.corr_df = corr_df
        return self.stats_df, self.corr_df
    
    @staticmethod
    def _eigen_values_positive(corr_df: pd.DataFrame) -> bool:
        """ Returns True if the eigen values of a correlation matrix are positive"""
        # ToDo: add tests to ensure that corr_df is a correlation matrix
        if corr_df is None:
            logger.error(f"_eigen_values_positive: corr_df is None")   
            return False
        try:
            eigen_values = np.linalg.eigvals(corr_df.astype(float))
        except np.linalg.LinAlgError:
            return False  # not a viable matrix
        positive_eigen = [eigen >= 0 for eigen in eigen_values ]
        return True if all(positive_eigen) else False


    def make_random_correlation_matrix(self, labels: List[str], seed: Optional[int] = None) -> pd.DataFrame:
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
        corr_df = pd.DataFrame(make_spd_matrix(dimension, random_state=seed), index=labels, columns=labels, dtype=float)
        # Make it a correlation matrix
        diag = np.diag(corr_df.to_numpy(dtype=float))  # make linter happy
        diag_inv = [1 / np.sqrt(x) for x in diag]
        for idx, ro in enumerate(corr_df.index):  # normalize the rows
            corr_df.loc[ro] *= diag_inv[idx]
        for idx, cl in enumerate(corr_df.columns):  # normalize the columns
            corr_df[cl] *= diag_inv[idx]
        self.corr_df = corr_df
        if not self._eigen_values_positive(corr_df):
            error_exit(f"make_random_correlation_matrix: Failed to create a correlation matrix")
        return corr_df


    def generate_correlated_ror(self) -> pd.DataFrame:
        """ Generate nb_smpl of Random Variable which are cross-correlated
        stat_df contains 2 columns: "Expected Return" i.e. mean and "Standard Deviation"
        corr_df is the cross-correlation matrix of the variables
        Must have stat_df.index == corr_df.index == corr_df.columns
        nb_smpl is the number of samples that are generated
        Returns a DF with the index as the asset classes and the columns as the samples
        The values are the RoR multipliers - ie. 1.18 (<- 18%) -- 1 + 0.01 * rvs ...  
        """
        if self.stats_df is None or self.corr_df is None:
            self.load_capital_markets_stats()
        stat_df, corr_df = self.stats_df, self.corr_df

        # Compute eigen values to make sure none is negative - otherwise, cholesky will fail
        try:
            assert corr_df is not None, f"corr_df is None"
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
        rng_sequence = np.random.SeedSequence()
        child_seeds = rng_sequence.spawn(len(stat_df.index))
        for child_seed in child_seeds:
            rng = default_rng(child_seed)
            x1 = norm.rvs(size=self.nb_smpl, loc=0.0, scale=1.0, random_state=rng)  # create a series w/ the desired stats
            data_series_list.append(x1)  # add data series list to list of lists
        data_matrix = np.asarray(data_series_list, dtype=float)   # shape: (nb_asset, nb_smpl)
        c_matrix = cholesky(corr_df, lower=True)
        # data_df = pd.DataFrame(np.dot(c_matrix, data_series_list), index=stat_df.index)
        correlated_matrix = c_matrix @ data_matrix                # shape: (nb_asset, nb_smpl)
        data_df = pd.DataFrame(correlated_matrix, index=stat_df.index)

        # Assign the desired mean and std_dev to each row
        def mk_lin_interp(mean_val: float, std_val: float):
            def f(x: float) -> float:
                return mean_val + std_val * x
            return f
        rvs_df = pd.DataFrame(index=stat_df.index, columns=range(self.nb_smpl))
        for rw, mn, stdv in zip(data_df.index, return_list, stddev_list):
            rvs_df.loc[rw] = data_df.loc[rw].map(mk_lin_interp(mn, stdv))
        if DEBUG_FLAG:
            logger.debug(f"\nDEBUG: comparing target stats vs generated stats ")
            for nn,idx in enumerate(stat_df.index):
                logger.debug(f'{idx}  Expected Return: {return_list[nn]} -  Generated Return: {rvs_df.loc[idx].mean()}')
                logger.debug(f'{idx}  Expected Stddev: {stddev_list[nn]} -  Generated Stddev:  {rvs_df.loc[idx].std()}')

        
        # Transform numbers from % to multipliers 15% -> 1.15, etc.
        ror_df = rvs_df.map(lambda x: 1 + 0.01 * x)
        return ror_df


class CapitalMarketsValidation():
    """
    Class to validate the Capital Markets stats
    """
    def __init__(self, config_manager: ConfigurationManager, capital_markets_stats: CapitalMarkets):
        self.config_manager = config_manager
        self.config = cast(dict[str, Any], self.config_manager.get_class_config(self.__class__.__name__))
        self.validate_cross_correlation_flag = self.config.get('validate_cross_correlation_flag', True)
        self.validation_error_margin = float(self.config['validation_error_margin'])
        self.stats_df = capital_markets_stats.stats_df
        self.corr_df = capital_markets_stats.corr_df
        self.generate_correlated_ror = capital_markets_stats.generate_correlated_ror

    @staticmethod
    def _matrix_equal(df_1: pd.DataFrame, df_2: pd.DataFrame, error_margin: float) -> Tuple[bool, float]:
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
        return flag, float(err)



    def _validate_cross_correlation(self, data_df: pd.DataFrame) -> Tuple[bool, float, bool, float, bool, float]:
        """
        Validate that a data series has the statistics and cross-correlation passed as parameters
        @param data_df: NxM DF - N series of M samples
        @param self.stat_df: Nx2 DF with expected mean and std_dev as columns
        @param self.corr_df: NxN cross-correlation matrix for series
        @param error_margin: margin of error when testing equality
        @return: True if validation passes
        """
        stat_df, corr_df = self.stats_df, self.corr_df
        if stat_df is None or corr_df is None:
            logger.error(f'_validate_cross_correlation: stat_df or corr_df is None')
            exit(-1)
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
        mean_ser = data_df.mean(axis=1)
        mean_validate, mean_error = self._matrix_equal(mean_ser.to_frame(), stat_df.iloc[:,0:1], error_margin)  # mean is the first column

        std_ser = data_df.std(axis=1)
        stddev_validate, stddev_error = self._matrix_equal(std_ser.to_frame(), stat_df.iloc[:,1:2], error_margin)  # std_dev is second column

        # IMPORTANT: astype(float) is needed otherwise, corr() returns empty DF
        data_corr = data_df.T.astype(float).corr()  # Corr works on columns
        corr_validate, corr_error = self._matrix_equal(data_corr, corr_df, error_margin)
        return mean_validate,  mean_error, stddev_validate, stddev_error, corr_validate, corr_error


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
    
    def run(self):
        if self.validate_cross_correlation_flag:
            logger.info(f'Validating cross-correlation matrix')
            mean_validate, mean_error, stddev_validate, stddev_error, corr_validate, corr_error = self._validate_cross_correlation(self.generate_correlated_ror())
            if  mean_validate and  stddev_validate  and corr_validate:
                logger.info(f"Successfully validated cross-correlation matrix")
            else:
                logger.error(f"Mean Error: {mean_error}\nStddev Error: {stddev_error}\nCorrelation Error: {corr_error}")
                logger.error(f"Failed to validate cross-correlation matrix")
        else:
            logger.info(f'Skipping Validation of cross-correlation matrix')
        # FIXME: Figure out how to clip the rvs_df - like ArrayRandGen class



def main(cmd_line: List[str]) -> None:
    config_manager = ConfigurationManager(sys.argv)
    capital_markets_stats = CapitalMarkets(config_manager)
    stat_df, corr_df = capital_markets_stats.load_capital_markets_stats()
    logger.info(f'\nAsset Class Statistics\n{stat_df}')
    logger.info(f'\nAsset Class Correlations\n{corr_df}')
    nb_smpl = int(config_manager.get_class_config('CapitalMarkets').get('nb_smpl', NB_SMPL_DEFAULT))
    capital_markets_stats.set_nb_smpl(nb_smpl)
    correlated_ror = capital_markets_stats.generate_correlated_ror()
    logger.info(f'\nCorrelated RoR\n{correlated_ror}')
    
    capital_markets_validation = CapitalMarketsValidation(config_manager, capital_markets_stats)
    if not capital_markets_validation.validate_cross_correlation_flag:  # skip validation if the flag is set to False
        logger.info(f'Skipping validation of cross-correlation matrix')

    logger.info(f"\nConfirming that correlated_rvs can be called several times and produce different results ")
    capital_markets_stats.set_nb_smpl(nb_smpl)
    nb_iter = int(config_manager.get_class_config('CapitalMarkets').get('nb_iter', NB_ITER_DEFAULT))
    for i in range(nb_iter):
        correlated_ror_2 = capital_markets_stats.generate_correlated_ror()
        # logger.info(f"correlated_ror_2.mean(): {100*(-1+correlated_ror_2.mean(axis=1))}")
        # logger.info(f"correlated_ror_2.std(): {100*correlated_ror_2.std(axis=1)}")
        logger.info(f'iter: {i} \nCorrelated RoR multipliers {i}\n{correlated_ror_2}')

if __name__ == '__main__':

    main(sys.argv)
    exit(0)
