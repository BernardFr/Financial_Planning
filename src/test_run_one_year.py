#!/usr/local/bin/python3
"""
Test run of one year of a Monte Carlo simulation

See: test_run_one_year.xlsx for validation

IMPORTANT Parameters:
mgt_fee =  0
rebalance_flag = false
cross_correlated_rvs_flag = false
run_cnt = 10
nb_cpu = 1
seed = 42
"""


import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from montecarlo_simulation_class import MontecarloSimulation
from utilities import error_exit, display_series, dollar_str
import sys
DEBUG_FLAG = False

# ROR_STATS = [ (3.00, 9.00),(2.60, 7.70), (1.50, 0.90),  (-0.05, 0.0)]  # Stocks, Bonds, TBills, Cash
# ASSET_CLASSES = {'Stocks': 800000,
#                 'Bonds': 400000,
#                 'TBills': 200000,
#                 'Cash': 100000}
# INITIAL_HOLDINGS = pd.DataFrame({'Market Value': ASSET_CLASSES})
ROR_STATS = [ (1.1, 0.0),(1.01, 0.0)]  
ASSET_CLASSES = {'Stocks': 1000,
                'Bonds': 1000}
INITIAL_HOLDINGS = pd.DataFrame({'Market Value': ASSET_CLASSES})
ROR_LST = [1.1, 1.01]
CASHFLOW_VAL = 100
RUN_CNT = 1000
BUSTED_THRESHOLD = 0.2  # 80% success rate
CASHFLOW_MULTIPLIER = 1.1
MAX_ITER_CNT = 1000
NB_AGES = 35

class TestRunOneYear:
    def __init__(self, config_manager: ConfigurationManager) -> None:
        """
        Loads initial holdings - from which we compute the target asset allocation and the starting funds
        Loads the yearly cashflows by age
        Loads a DF of rates of return for each asset class for each year
        Runs the simulation for the number of iterations specified in the configuration
        """
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.cross_correlated_rvs_flag = self.config['cross_correlated_rvs_flag']  # determines whether we use Morningstar stats as is (false) or if we create cross-correlated RoR
        self.run_cnt = self.config['run_cnt']
        self.nb_cpu = self.config['nb_cpu']
        self.seed = self.config['seed']
        return None


    def run(self) -> None:
        montecarlo_simulation = MontecarloSimulation(self.config_manager)
        montecarlo_simulation.set_run_cnt(self.config_manager.config['run_cnt'])
        montecarlo_simulation.load_initial_data()  # we override a lot

        ror_lst = ROR_LST
        print(f"ror_lst: {ror_lst}")
        cashflow_val = CASHFLOW_VAL 
        previous_busted_flag = False
        prev_cashflow_val = cashflow_val
        portfolio_ser = INITIAL_HOLDINGS['Market Value'].copy(deep=True)
        print(f"Initial portfolio value: {portfolio_ser.sum():,.0f}")
        """ 
        Note: we only run one iteration for NB_AGES years
        there is no need to run multiple iterations since the RoR are constant (not randomized)
        then we adjust the cashflow_val based on the average final value and the busted ratio
        """ 
        for iter_cnt in range(MAX_ITER_CNT):  # make sure we don't run forever
            portfolio_ser = INITIAL_HOLDINGS['Market Value'].copy(deep=True)
            for yr in range(NB_AGES):
                portfolio_ser, busted_flag = montecarlo_simulation.run_one_year(portfolio_ser, ror_lst, cashflow_val)
                # print(f"Year: {yr} - Busted: {busted_flag} - Portfolio:\n{portfolio_ser.map(dollar_str)} ")
                if busted_flag:
                    break
                # print(f"portfolio_value: {portfolio_value:,.2f} - previous_portfolio_value: {previous_portfolio_value:,.2f}")
                # print(f"ratio: {portfolio_value / (previous_portfolio_value +CASHFLOW_VAL):.6f}")
            final_value = portfolio_ser.sum()
            print(f"Iter: {iter_cnt} -Busted: {busted_flag}  - Final value: {final_value:,.0f}")
            if iter_cnt > 0:  # skip first iteration, so we don't have a previous value to compare to
                if previous_busted_flag != busted_flag:
                    print(f"Done:cashflow_val: {prev_cashflow_val:,.2f}")
                    break
            prev_cashflow_val = cashflow_val
            if busted_flag:  # diminished cashflow_val
                cashflow_val = cashflow_val / CASHFLOW_MULTIPLIER
            else:
                cashflow_val = cashflow_val * CASHFLOW_MULTIPLIER
            print(f"Next cashflow_val: {cashflow_val:,.2f}\n")
            previous_busted_flag = busted_flag
        return None

if __name__ == "__main__":
    config_manager = ConfigurationManager(sys.argv)
    test_run_one_year = TestRunOneYear(config_manager)
    test_run_one_year.run()
    sys.exit("---\nDone!")