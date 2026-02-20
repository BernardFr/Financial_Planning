#!/usr/local/bin/python3

"""
This class is used to run the Montecarlo simulation
MontecarloSimulationDataLoader is a helper class to load the initial data for the Montecarlo simulation: portfolio, cashflow, Morningstar stats, etc.
so that each MC simulation can be initialized quickly with the same data.

"""

from typing import Any
import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from cashflow_class import Cashflow 
from portfolio_class import Portfolio
from morningstar_stats_class import MorningstarStats
from arrayrandgen_class import ArrayRandGen
from utilities import error_exit, display_series, dollar_str
import collections
import sys
DEBUG_FLAG = False
ROR_STATS = [ (3.00, 9.00),(2.60, 7.70), (1.50, 0.90),  (-0.05, 0.0)]  # Stocks, Bonds, TBills, Cash
ASSET_CLASSES = {'Stocks': 800000,
                'Bonds': 400000,
                'TBills': 200000,
                'Cash': 100000}
INITIAL_HOLDINGS = pd.DataFrame({'Market Value': ASSET_CLASSES})
# CASHFLOW_SERIES = pd.Series(index=range(67, 102), data=[50000] * 35, name="Cashflows")

class MontecarloSimulationDataLoader:
    """
    Load the initial data for the Montecarlo simulation
    """
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        #FYI: we have to use the class name "MontecarloSimulation" rather than __class__.__name__ 
        self.config = self.config_manager.get_class_config("MontecarloSimulation")
        self.target_end_funds = self.config['target_end_funds']
        self.target_success_rate = self.config['target_success_rate']
        self.seed = self.config['seed']
        self.run_cnt = self.config['run_cnt']
        self.nb_cpu = self.config['nb_cpu']
        self.re_alloc_error = self.config['re_alloc_error']
        self.Funds_step = self.config['Funds_step']
        self.Discreet_step = self.config['Discreet_step']
        self.Success_threshold = self.config['Success_threshold']
        self.mgt_fee = self.config['mgt_fee']
        self.rebalance_flag = self.config['rebalance_flag']
        self.cross_correlated_rvs_flag = self.config['cross_correlated_rvs_flag']
        # Create a sequence of pseudo-random seeds for the random number generators for each CPU
        self.master_seed_sequence = np.random.SeedSequence(self.seed)    
        self.data_loaded_flag = False

        return None

    def set_cashflow_series(self, cashflow_ser: pd.Series) -> None:
        """Set the cashflow : Series of yearly cashflows by age """
        self.cashflow_ser = cashflow_ser
        self.age_lst = list(map(int, cashflow_ser.index))
        self.start_age = self.age_lst[0]
        self.end_age = self.age_lst[-1]
        self.nb_ages = len(self.age_lst)
        return None


    def set_correlated_ror(self, correlated_rvs: pd.DataFrame) -> None:
        """Set the correlated RoR series when generated from Morningstar stats
        Rows are asset classes and columns are nb_ages * run_cnt 
        Transform numbers from % to multipliers 15% -> 1.15, etc. """
        self.correlated_ror = correlated_rvs.map(lambda x: 1 + 0.01 * x)
        return None


    def load_data(self) -> None:
        """
        Load the initial data:
        - goals
        - generate cashflow series from goals
        - load holdings 
        - map to ETF asset classes and allocate cash to ETF -> initial asset allocation
        - Load Morningstar stats
        - correlated RoR series or ror generators based on cross_correlated_rvs_flag
        """
        # Load the cashflow and set the cashflow series
        cashflow_class = Cashflow(self.config_manager)
        goals_df = cashflow_class.process_goals_file()
        logger.info(f"Goals DataFrame:\n{goals_df}")
        cashflow_df, cashflow_total_ser = cashflow_class.make_cashflow()
        logger.info(f"Cashflow DataFrame:\n{cashflow_df}")
        logger.info(f"Cashflow Total Series:\n{cashflow_total_ser.map(dollar_str)}")
        self.set_cashflow_series(cashflow_total_ser)

        # Load holdings and compute the initial holdings and set the initial holdings
        portfolio = Portfolio(self.config_manager)
        holdings_df, cash_amount = portfolio.load_holdings_data()
        logger.info(f"Holdings DataFrame:\n{holdings_df}")
        logger.info(f"Cash amount: ${cash_amount:,.2f}")
        holdings_df = portfolio.assign_cash_to_etf(holdings_df, cash_amount)
        logger.info(f"Holdings DataFrame after reassigning cash to ETF_for_cash:\n{holdings_df}")
        portfolio.set_holdings_df(holdings_df)
        self.asset_class_df = portfolio.map_etf_to_asset_class()
        self.initial_asset_class_df = self.asset_class_df.copy(deep=True)  # Used when we iterate over starting funds values
        logger.info(f"Asset Class DataFrame:\n{self.asset_class_df}")

        # Get the Morningstar Asset Stats and create the asset allocation model
        morningstar_stats = MorningstarStats(self.config_manager)
        stats_df, corr_df = morningstar_stats.get_asset_stats()
        logger.info(f'\nAsset Class Statistics\n{stats_df}')
        logger.info(f'\nAsset Class Correlations\n{corr_df}')
        # Match the stats and asset classes in the portfolio 
        self.stats_df, self.corr_df = morningstar_stats.match_stats_vs_assets(self.asset_class_df)
        # Update stats_df and corr_df in morningstar_stats
        morningstar_stats.set_stat_df_and_corr_df(self.stats_df, self.corr_df)
        logger.info(f"\nstats_df shape: {self.stats_df.shape} corr_df shape: {self.corr_df.shape}") 

        if self.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
            # FIXME:
            logger.info(f"Using cross-correlated RoR series")
            morningstar_stats.set_nb_smpl(self.run_cnt * self.nb_ages)
            self.correlated_rvs = morningstar_stats.generate_correlated_rvs()
            logger.info(f"Correlated returns Series (%):\n{self.correlated_rvs}")
            # Set the correlated RoR series
            self.set_correlated_ror(self.correlated_rvs)

        else:          # Use Morningstar stats as is  
            logger.info(f"Using Morningstar stats as is")
            # Make list of asset classes stats and correlations
            self.stats_lst = [(name, mean, stdv) for name, mean, stdv in zip(self.stats_df.index, self.stats_df['Expected Return'], self.stats_df['Standard Deviation'])]
            logger.info(f"Asset Classes List:\n{self.stats_lst}")
            logger.info(f"Using Morningstar stats as is")
            # Create and set the list of generators
            if self.nb_cpu == 1:
                self.seed_sequence = self.master_seed_sequence.spawn(len(self.stats_lst))
            else: # FIXME - need to spawn the seed sequence for each CPU
                # self.seed_sequence = self.master_seed_sequence.spawn(len(self.stats_lst))
                error_exit(f"FIXME: Need to spawn the seed sequence for each CPU")
            ror_gen_list = []
            for (name, mean, stdv), seed in zip(self.stats_lst, self.seed_sequence):
                array_rand_gen = ArrayRandGen(self.config_manager, name, mean, stdv, seed)
                ror_gen_list.append(array_rand_gen)
            logger.debug(f"ror_gen_list (names): {[gen.name for gen in ror_gen_list]}")
            self.ror_gen_list = ror_gen_list

        self.data_loaded_flag = True
        return None


class MontecarloSimulation:
    """
    Run Montecarlo simulation
    :param init_asset: (N,) SERIES of assets allocated by asset class
    :param ror_stats: Namedstuple keyed by asset class name with 2-tuple values
    :param cashflow_ser: Series of yearly withdrawals [or income] for each consecutive age
    :param cnt: Nb of iterations to run the simulation
    :return: [N, cnt] DF with ending amount by asset class for each simulation run

    Each process must initialize randon number generator's seed
    https://bbabenko.github.io/multiprocessing-and-seeded-RNGs/
    """


    def __init__(self, config_manager: ConfigurationManager, mc_data_loader: MontecarloSimulationDataLoader) -> None:
        """
        Loads initial holdings - from which we compute the target asset allocation and the starting funds
        Loads the yearly cashflows by age
        Loads a DF of rates of return for each asset class for each year
        Runs the simulation for the number of iterations specified in the configuration
        """
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)

        # load the initial data from the data loader
        self.cross_correlated_rvs_flag = mc_data_loader.cross_correlated_rvs_flag
        self.start_age = mc_data_loader.start_age
        self.end_age = mc_data_loader.end_age
        self.age_lst = mc_data_loader.age_lst
        self.nb_ages = mc_data_loader.nb_ages
        self.run_cnt = mc_data_loader.run_cnt
        self.seed = mc_data_loader.seed
        self.re_alloc_error = mc_data_loader.re_alloc_error
        self.Funds_step = mc_data_loader.Funds_step
        self.Discreet_step = mc_data_loader.Discreet_step
        self.Success_threshold = mc_data_loader.Success_threshold
        self.mgt_fee = mc_data_loader.mgt_fee
        self.rebalance_flag = mc_data_loader.rebalance_flag
        self.cross_correlated_rvs_flag = mc_data_loader.cross_correlated_rvs_flag
        self.busted_ages = []
        self.busted_cnt = 0

        # Load the data from the data loader
        # if the DataLoader has not been loaded, load the data
        if not mc_data_loader.data_loaded_flag:
            mc_data_loader.load_data()
        if self.cross_correlated_rvs_flag:
            self.correlated_ror = mc_data_loader.correlated_rvs
        else:
            self.ror_gen_list = mc_data_loader.ror_gen_list
        print(f"ror_gen_list: {[gen.name for gen in self.ror_gen_list]}")
        self.asset_class_df = mc_data_loader.asset_class_df
        self.initial_asset_class_df = mc_data_loader.initial_asset_class_df
        self.initial_pfolio_value = self.initial_asset_class_df.sum().item()
        self.cashflow_ser = mc_data_loader.cashflow_ser
        return None

    def set_run_cnt(self, run_cnt: int) -> None:
        """Set the number of iterations to run the simulation"""
        self.run_cnt = run_cnt
        return None

    def reinitialize_data(self, assets_multiplier: float) -> None:
        """Reinitialize the data for the next simulation"""
        # Scale the initial asset classes by the assets_multiplier
        self.asset_class_df = self.initial_asset_class_df.mul(assets_multiplier, axis=0)
        logger.info(f"portfolio value after reinitialization: {self.asset_class_df.sum().item():,.2f}")
        self.final_result_df = pd.DataFrame(index=self.asset_class_df.index, columns=range(self.run_cnt))
        self.busted_ages = []
        self.busted_cnt = 0
        return None

    def mk_ror_df(self) -> pd.DataFrame:
        """
        Generate a dataframe where columns are integers in range(start_age, end_age) - where the rates of return are 
        generated by the generators in gen_list
        :return: pd.DataFrame with row as asset classes and columns as ages - indexed to the initial asset_class_df index

        """
        if self.cross_correlated_rvs_flag:  # FIXME
            # Make sure the correlated_rvs dataframe exists has at least nb_ages columns
            if self.correlated_ror.shape[1] < self.nb_ages:
                error_exit(f"correlated_ror dataframe has less than nb_ages columns: {self.correlated_ror.shape[1]} < {self.nb_ages}")
            # return the first nb_ages columns of the correlated_rvs dataframe and strip them from the self.correlated_rvs dataframe
            ror_df = self.correlated_ror.iloc[:, :self.nb_ages]
            self.correlated_ror = self.correlated_ror.iloc[:, self.nb_ages:].copy(deep=True)
            return self.correlated_ror
        else:  # Use ArrayRandGen to generate the RoR series
            ror_df = pd.DataFrame(index=self.initial_asset_class_df.index, columns=self.age_lst)
            for gener in  self.ror_gen_list:
                # generator returns a list which contains the list of values for each age
                name = gener.name  # get the asset class name from the generator
                # generator returns a list of ror values for each age - total nb_ages values
                new_ror_values = list[Any](next(gener))
                logger.debug(f"name: {name}:\n{new_ror_values}")
                ror_df.loc[name] = pd.Series(new_ror_values, index=self.age_lst)  # ror_lst has nb_ages for this asset class
            ror_df = ror_df.apply(lambda x: 1 + 0.01 * x)  # Convert to multipliers
            return ror_df  # DF with row as asset classes and columns as ages



    def run_one_iter(self) -> (pd.DataFrame, bool, int):
        """Run one iteration of the simulation
        Calls mk_ror_df() to generate a new RoR series
        """
        portfolio_ser = self.asset_class_df.copy(deep=True)
        logger.info(f"portfolio value at start of iteration: {portfolio_ser.sum().item():,.2f}")

        busted_flag = False
        busted_age = self.end_age + 1
        # Generate a new array of rate of returns for all ages and each asset class

        if self.cross_correlated_rvs_flag:  # FIXME
            ror_df = self.correlated_ror.copy(deep=True)
        else:
            ror_df = self.mk_ror_df()
        for age, cashflow_val in zip(self.age_lst, self.cashflow_ser):
            ror_lst = list[float](ror_df[age])
            # print(f"age: {age} - ror_lst: {ror_lst} - cashflow_val: {cashflow_val}")
            # print(f"portfolio_ser: {portfolio_ser}")
            new_portfolio_ser, busted_flag = self.run_one_year(portfolio_ser, ror_lst, cashflow_val)
            if busted_flag:
                busted_age = age
                break  # Stop iterating over age when we bust out
            else:
                portfolio_ser = new_portfolio_ser
        return portfolio_ser, busted_flag, busted_age

    def run_one_year(self, portfolio_ser: pd.Series, ror_lst: list[float], cashflow_val: float) -> (pd.Series, bool):
        """Run one year of the simulation
        Note the order of operations is important 

        1. Compute the returns for the year
        2. Compute the management fee based on the portfolio value
        3. Subtract the management fee and the cashflow from the portfolio value - this can make the total portfolio value negative
        4. Determine if the portfolio is busted (negative total) - if so, return the busted flag (true)  
        5. If the portfolio is not busted, rebalance the portfolio based on rebalance flag, and
        return the portfolio and the busted flag (false)

        Note: cashflow_val is positive if it is a cashflow going out, and negative if it is a cashflow coming in
        ... so cashflow_val is expected to be positive almost always:)
        mgt_fee is a positive value that will be subtracted from the portfolio value
        """
        pfolio_ser = portfolio_ser.mul(ror_lst, axis=0)  # add the returns to the portfolio
        pfolio_value = pfolio_ser.sum().item()  # always positive - RoR cannot be > 100%
        logger.info(f"portfolio value: {pfolio_value:,.2f}")
        management_fee_value = pfolio_value * self.mgt_fee
        wdrwl_value = cashflow_val + management_fee_value  # money going out
        adjusted_pfolio_value = pfolio_value - wdrwl_value

        if adjusted_pfolio_value <= 0.0:  # withdrawls are greater than the portfolio value
            pfolio_ser = pfolio_ser * adjusted_pfolio_value / pfolio_value
            return pfolio_ser, True  # Busted - note pfolio_ser is not updated
        else:
            # Rebalance the portfolio based on rebalance flag
            if self.rebalance_flag:
                pfolio_ser = self.initial_asset_class_df.mul(adjusted_pfolio_value / self.initial_pfolio_value, axis=0)
            else:
                pfolio_ser = pfolio_ser.mul(adjusted_pfolio_value / pfolio_value, axis=0)
            return pfolio_ser, False


    def run(self) -> (pd.Series, dict, int):
        # print("run_mc_multi: pid: {}, cnt= {}, seed={}".format(os.getpid(), cnt, seed))
        np.random.seed(self.seed)
        self.busted_ages = []
        self.busted_cnt = 0
        self.final_result_df = pd.DataFrame(index=self.asset_class_df.index, columns=range(self.run_cnt))
        for itr in range(self.run_cnt):
            final_portfolio_df, busted_flag, busted_age = self.run_one_iter()
            self.final_result_df[itr] = final_portfolio_df
            if busted_flag:
                self.busted_cnt += 1
                self.busted_ages.append(busted_age)

        # create a series for the final portfolio values
        final_result_series = self.final_result_df.sum(axis=0)
        logger.debug(f"final_result_series:\n{final_result_series.map(dollar_str)}")
        # compute the average by asset class (rows  )
        average_by_asset_class = self.final_result_df.mean(axis=1)
        logger.info(f"result average_by_asset_class:\n{average_by_asset_class.map(dollar_str)}")

        # Create a dict that counts the number of busted ages
        busted_ages_dict = dict(collections.Counter(self.busted_ages))
        return final_result_series, busted_ages_dict, self.busted_cnt

def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    montecarlo_simulation = MontecarloSimulation(config_manager)

    montecarlo_simulation.load_initial_data()
    final_result_series, busted_ages_dict, busted_cnt = montecarlo_simulation.run()
    logger.info(f"Final Result Series:\n{final_result_series.map(dollar_str)}")
    logger.info(f"Busted Ages Dict:\n{busted_ages_dict}")
    logger.info(f"Busted Count:\n{busted_cnt}")
    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
