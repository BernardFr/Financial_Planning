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
from montecarlo_simulation_class import MontecarloSimulation, MontecarloSimulationDataLoader
from utilities import error_exit, display_series, dollar_str
import sys
DEBUG_FLAG = False

# ROR_STATS = [ (3.00, 9.00),(2.60, 7.70), (1.50, 0.90),  (-0.05, 0.0)]  # Stocks, Bonds, TBills, Cash
# ASSET_CLASSES = {'Stocks': 800000,
#                 'Bonds': 400000,
#                 'TBills': 200000,
#                 'Cash': 100000}
# INITIAL_HOLDINGS = pd.DataFrame({'Market Value': ASSET_CLASSES})
ASSET_CLASSES = {'Stocks': 2000,
                'Bonds': 1000}
INITIAL_HOLDINGS = pd.DataFrame({'Market Value': ASSET_CLASSES})
# ROR_STATS = [ (1.1, 0.5),(1.01, 0.2)]  
ROR_LST = [1.1, 1.01]
CASHFLOW_VAL = 150
NB_YEARS = 10
ROR_STTDEV = 0.1  # Standard deviation of the RoR 

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
        self.data_loader = MontecarloSimulationDataLoader(self.config_manager)
        self.mc_sim = MontecarloSimulation(self.config_manager, self.data_loader, 1) # 1 cpu
        self.mc_sim.set_run_cnt(self.data_loader.run_cnt)
        # Overload the initialization of the MontecarloSimulation class with our test data
        # self.mc_sim.initial_asset_class_ser = INITIAL_HOLDINGS['Market Value'].copy(deep=True)
        # self.mc_sim.initial_pfolio_value = self.mc_sim.initial_asset_class_ser.sum().item()
        # self.mc_sim.asset_allocation__ratio_ser = self.mc_sim.initial_asset_class_ser.div(self.mc_sim.initial_asset_class_ser.sum().item())    
        # self.asset_class_ser = self.mc_sim.initial_asset_class_ser.copy(deep=True)

        return None


    def run(self) -> None:

        # Compute random RORs for each asset class - constants over the years in this test case
        # Value is gau
        ror_lst = [np.random.normal(loc=1.0, scale=ROR_STTDEV) for _ in range(len(self.mc_sim.asset_class_ser.index)) ]
        print(f"ror_lst: {ror_lst}")

        result_index = []
        for asset_class in self.mc_sim.asset_class_ser.index:
            result_index.append(f"{asset_class}")
        for nb, _ in enumerate(ror_lst):
            result_index.append(f"RoR_{nb}")
        result_index.append("Cashflow")
        result_index.append("Mgt_Fee") # will be calculated in Excel
        result_index.append("Computed Portfolio Value")
        result_df = pd.DataFrame(index=result_index, dtype=float)
        # ror_lst = ROR_LST
        # print(f"ror_lst: {ror_lst}")
        cashflow_val = CASHFLOW_VAL 
        # portfolio_ser = INITIAL_HOLDINGS['Market Value'].copy(deep=True)
        portfolio_ser = self.mc_sim.initial_asset_class_ser.copy(deep=True)
        print(f"Initial portfolio value: {portfolio_ser.sum():,.0f}")
        busted_flag = False
        print(f"--\nPortfolio value: {portfolio_ser.sum():,.0f} - cashflow_val: {cashflow_val:,.0f}")
        for yr in range(NB_YEARS):
            print(f"portfolio_ser before year {yr}: {portfolio_ser}")
            print(f"ror_lst: {ror_lst} - cashflow_val: {cashflow_val}")
            value_sub_total = portfolio_ser.sum()
            result_list = list(portfolio_ser.values) + [value_sub_total] + ror_lst
            result_list.append(cashflow_val)
            result_list.append(0.0) # Mgt fee - will be calculated in Excel
            portfolio_ser, busted_flag = self.mc_sim.run_one_year(portfolio_ser, ror_lst, cashflow_val)
            result_list.append(portfolio_ser.sum())
            result_df[yr] = result_list
            print(f"Year: {yr} - Busted: {busted_flag} - Portfolio value: {portfolio_ser.sum():,.0f}")
            if busted_flag:
                break
            # print(f"portfolio_value: {portfolio_value:,.2f} - previous_portfolio_value: {previous_portfolio_value:,.2f}")
            # print(f"ratio: {portfolio_value / (previous_portfolio_value +CASHFLOW_VAL):.6f}")
        final_value = portfolio_ser.sum()
        print(f"Final portfolio value: {final_value:,.2f}")
        result_df.columns = [f"Year_{x}" for x in result_df.columns]
        print(f"result_df:\n{result_df}")
        result_df.to_excel("test_run_one_year_result.xlsx", index=True, header=True)
        return None

if __name__ == "__main__":
    config_manager = ConfigurationManager(sys.argv)
    test_run_one_year = TestRunOneYear(config_manager)
    test_run_one_year.run()
    sys.exit("---\nDone!")