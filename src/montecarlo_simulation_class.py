#!/usr/local/bin/python3

import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from cashflow_class import Cashflow 
from holdings_class import Holdings
from morningstar_stats_class import MorningstarStats
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



class ArrayRandGen:
    def __init__(self,config_manager: ConfigurationManager, mean: float, stdv: float):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.run_cnt = self.config['run_cnt']
        self.nb_rv = self.config['end_age'] - self.config['start_age'] + 1  # number of random values to generate
        self.mean = mean
        self.stdv = stdv
        self.ror_floor = self.config['RoR_floor']
        self.ror_ceiling = self.config['Ror_ceiling']

    def __iter__(self):
        return self

    def __next__(self) -> list[float]:
        """
        returns a list of nb random numbers based on mean and stddev
        numbers are: 1 + random_number/100  (interest rates)
        NOTE that the RoR is clipped to the floor and ceiling to account for historical data
        """
        random_values = np.random.normal(self.mean, self.stdv, size=self.nb_rv)
        ror_values = 1 + 0.01 * random_values
        # Clip the final RoR values to historical bounds
        ror_values = np.clip(ror_values, self.ror_floor, self.ror_ceiling)
        return ror_values.tolist()


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


    def __init__(self, config_manager: ConfigurationManager) -> None:
        """
        Loads initial holdings - from which we compute the target asset allocation and the starting funds
        Loads the yearly cashflows by age
        Loads a DF of rates of return for each asset class for each year
        Runs the simulation for the number of iterations specified in the configuration
        """
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.target_end_funds = self.config['target_end_funds']
        self.target_success_rate = self.config['target_success_rate']
        self.seed = self.config['seed']
        self.run_cnt = self.config['run_cnt']
        self.re_alloc_error = self.config['re_alloc_error']
        self.nb_cpu = self.config['nb_cpu']
        self.Funds_step = self.config['Funds_step']
        self.Discret_step = self.config['Discret_step']
        self.Success_threshold = self.config['Success_threshold']
        self.mgt_fee = self.config['mgt_fee']
        self.rebalance_flag = self.config['rebalance_flag']
        self.busted_ages = []
        self.busted_cnt = 0
        self.cross_correlated_rvs_flag = self.config['cross_correlated_rvs_flag']
        return None


    def set_initial_holdings(self, initial_holdings: pd.DataFrame) -> None:
        """Set the initial holdings DataFrame and the target asset allocation"""
        self.initial_holdings = initial_holdings
        # Create a string representation of the initial holdings, with commas and 0 decimal points
        logger.info(f"initial_holdings:\n{self.initial_holdings.map(dollar_str)}")
        self.start_funds = initial_holdings['Market Value'].sum()
        logger.info(f"Starting Funds: ${self.start_funds:,.0f}")
        self.target_asset_allocation = initial_holdings['Market Value'] / self.start_funds
        t_a_a_str = self.target_asset_allocation.map(lambda x: f"{100*x:.2f} %")
        logger.info(f"Target Asset Allocation:\n{t_a_a_str}")
        self.final_result_df = pd.DataFrame(index=self.initial_holdings.index, columns=range(self.run_cnt))  

        return None

    def set_cashflow_series(self, cashflow_ser: pd.Series) -> None:
        """Set the cashflow : Series of yearly cashflows by age """
        self.cashflow_ser = cashflow_ser
        self.age_lst = list(map(int, cashflow_ser.index))
        self.start_age = self.age_lst[0]
        self.end_age = self.age_lst[-1]
        self.nb_ages = len(self.age_lst)
        return None

    def match_stats_vs_holdings(self, df_stat: pd.DataFrame,df_corr: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Check if the asset classes list and the initial holdings are the same
        If Holdings do not have all the asset classes in df_stat, df_corr - strip the unused asset classes from df_stat, df_corr
        If df_stat, df_corr do not have all the asset classes in Holdings - Error & exit
        Return the stripped df_stat, df_corr
        """
        holdings_set = set(self.initial_holdings.index)
        stats_set = set(df_stat.index)
        missing_stats = [x for x in holdings_set if x not in stats_set]
        if len(missing_stats) > 0:
            # Exit we are missing stats for some holdings
            error_exit(f"Some Holding are Not in Morningstart asset classes: {missing_stats}")

        # Now drop the stats that are not in the holdings
        missing_holdings = [x for x in stats_set if x not in holdings_set]
        if len(missing_holdings) > 0:  # Some Morningstar asset classes are not in the holdings
            logger.info(f"Some Morningstar asset classes are not in the holdings: {missing_holdings}")
            logger.info("Taking the extra asset classes out of the stats and correlation dataframes")
            df_stat = df_stat.drop(index=missing_holdings)
            df_corr = df_corr.drop(index=missing_holdings, columns=missing_holdings)
        return df_stat, df_corr


    def set_correlated_rvs(self, correlated_rvs: pd.DataFrame) -> None:
        """Set the correlated RoR series when generated from Morningstar stats
        Rows are asset classes and columns are nb_ages * run_cnt """
        self.correlated_rvs = correlated_rvs
        return None

    def set_ror_generator_list(self, ror_gen_list: list[ArrayRandGen]) -> None:
        """Set the list of generators when using ArrayRandGen to generate the RoR series"""
        self.ror_gen_list = ror_gen_list
        return None


    def mk_ror_df(self) -> pd.DataFrame:
        """
        Generate a dataframe where columns are integers in range(start, end) - where the rates of return are 
        generated by the generators in gen_list
        :param gen_list: list of generators
        :param df_inde: df_inde of DF
        :return: pd.DataFrame with row as asset classes and columns as ages

        """
        if self.cross_correlated_rvs_flag:
            # Make sure the correlated_rvs dataframe exists has at least nb_ages columns
            if self.correlated_rvs.shape[1] < self.nb_ages:
                error_exit(f"correlated_rvs dataframe has less than nb_ages columns: {self.correlated_rvs.shape[1]} < {self.nb_ages}")
            # return the first nb_ages columns of the correlated_rvs dataframe and strip them from the self.correlated_rvs dataframe
            ror_df = self.correlated_rvs.iloc[:, :self.nb_ages]
            self.correlated_rvs = self.correlated_rvs.iloc[:, self.nb_ages:].copy(deep=True)
            return ror_df
        else:  # Use ArrayRandGen to generate the RoR series
            ror_lst_lst = []  # list of lists of RoR values for each asset class. Each has nb_ages * run_cnt values
            for gener in  self.ror_gen_list:
                # generator returns a list which contains the list of values for each age
                new_ror_values = list(next(gener))
                ror_lst_lst.append(new_ror_values)
                if DEBUG_FLAG:
                    logger.debug(f"New RoR values: {new_ror_values}")
                    logger.debug(f"mean: {np.mean(new_ror_values)} - ie. {100*(-1+np.mean(new_ror_values)):.2f}% , stdv: {100*np.std(new_ror_values):.2f} \n")
                # add the new ror values to the dataframe
            # each list becomes a row in the dataframe
            ror_df = pd.DataFrame(ror_lst_lst, index = self.initial_holdings.index, columns = self.age_lst)
        return ror_df  # list of list of RoR values for each age


    def run_one_iter(self) -> (pd.DataFrame, bool, int):
        """Run one iteration of the simulation
        """
        portfolio_ser = self.initial_holdings.copy(deep=True)
        busted_flag = False
        busted_age = self.end_age + 1
        # Generate a new array of rate of returns for all ages and each asset class
        ror_df = self.mk_ror_df()

        for age, ror_lst, cashflow_val in zip(self.age_lst, ror_df, self.cashflow_ser):
            ror_lst = list[float](ror_df[age])
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
        """
        pfolio_ser = portfolio_ser.mul(ror_lst, axis=0)  # add the returns to the portfolio
        pfolio_value = float(pfolio_ser.sum())  # always positive - RoR cannot be > 100%
        management_fee_value = pfolio_value * self.mgt_fee
        wdrwl_value = cashflow_val + management_fee_value  # money going out
        adjusted_pfolio_value = pfolio_value - wdrwl_value

        if adjusted_pfolio_value <= 0.0:  # withdrawls are greater than the portfolio value
            pfolio_ser = pfolio_ser * adjusted_pfolio_value / pfolio_value
            return pfolio_ser, True  # Busted - note pfolio_ser is not updated
        else:
            # Rebalance the portfolio based on rebalance flag
            if self.rebalance_flag:
                pfolio_ser = self.target_asset_allocation * adjusted_pfolio_value
            else:
                pfolio_ser = pfolio_ser * adjusted_pfolio_value / pfolio_value
            return pfolio_ser, False


    def run(self) -> None:
        # print("run_mc_multi: pid: {}, cnt= {}, seed={}".format(os.getpid(), cnt, seed))
        np.random.seed(self.seed)
        self.busted_ages = []
        self.busted_cnt = 0
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
        logger.info(f"average_by_asset_class:\n{average_by_asset_class.map(dollar_str)}")
        # Compute statistics on the final portfolio values
        final_result_series_stats = final_result_series.describe()
        logger.info(f"Final result  stats:\n{final_result_series_stats}")

        # Create a dict that counts the number of busted ages
        busted_ages_dict = dict(collections.Counter(self.busted_ages))
        logger.info(f"Busted ages dict: {busted_ages_dict}")
        logger.info(f"Busted count: {self.busted_cnt}")
        # Compute confidence level
        confidence_level = 100.0 * (self.run_cnt - self.busted_cnt) / self.run_cnt
        logger.info(f"Confidence level: {confidence_level:.2f}%")

        return

if __name__ == "__main__":
    config_manager = ConfigurationManager(sys.argv)
    montecarlo_simulation = MontecarloSimulation(config_manager)

    # Load the cashflow and set the cashflow series
    cashflow_class = Cashflow(config_manager)
    goals_df = cashflow_class.process_goals_file()
    print(f"Goals DataFrame:\n{goals_df}")
    cashflow_df, cashflow_total_ser = cashflow_class.make_cashflow()
    print(f"Cashflow DataFrame:\n{cashflow_df}")
    print(f"Cashflow Total Series:\n{cashflow_total_ser.map(dollar_str)}")
    print(display_series(cashflow_total_ser, 2))
    montecarlo_simulation.set_cashflow_series(cashflow_total_ser)

    # Load holdings and compute the initial holdings and set the initial holdings
    portfolio = Holdings(config_manager)
    holdings_df, cash_amount = portfolio.load_holdings_data()
    print(f"Holdings DataFrame:\n{holdings_df}")
    print(f"Cash amount: ${cash_amount:,.2f}")
    holdings_df = portfolio.assign_cash_to_etf(holdings_df, cash_amount)
    print(f"Holdings DataFrame after reassigning cash to ETF_for_cash:\n{holdings_df}")
    portfolio.set_holdings_df(holdings_df)
    asset_class_df = portfolio.map_etf_to_asset_class()
    print(f"Asset Class DataFrame:\n{asset_class_df}")
    montecarlo_simulation.set_initial_holdings(asset_class_df)

    # Get the Morningstar Asset Stats and create the asset allocation model
    morningstar_stats = MorningstarStats(config_manager)
    df_stat, df_corr = morningstar_stats.get_asset_stats()
    logger.info(f'\nAsset Class Statistics\n{df_stat}')
    logger.info(f'\nAsset Class Correlations\n{df_corr}')
    # Match the stats and holdings
    df_stat, df_corr = montecarlo_simulation.match_stats_vs_holdings(df_stat, df_corr)


    if montecarlo_simulation.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
        logger.info(f"Using cross-correlated RoR series")
        morningstar_stats.set_nb_smpl(montecarlo_simulation.run_cnt * montecarlo_simulation.nb_ages)
        correlated_rvs = morningstar_stats.generate_correlated_rvs()
        logger.info(f"Correlated returns Series (%):\n{correlated_rvs}")
        # Set the correlated RoR series
        montecarlo_simulation.set_correlated_rvs(correlated_rvs)
    else:          # Use Morningstar stats as is  
        logger.info(f"Using Morningstar stats as is")
        # Make list of asset classes stats and correlations
        asset_classes_list = [(mean, stdv) for mean, stdv in zip(df_stat['Expected Return'], df_stat['Standard Deviation'])]
        print(f"Asset Classes List:\n{asset_classes_list}")
        logger.info(f"Using Morningstar stats as is")
        # Create and setthe list of generators
        ror_gen_list = [ArrayRandGen(config_manager, mean, stdv) for mean, stdv in asset_classes_list]
        montecarlo_simulation.set_ror_generator_list(ror_gen_list)

    montecarlo_simulation.run()
    sys.exit("---\nDone!")