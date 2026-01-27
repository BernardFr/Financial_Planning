#!/usr/local/bin/python3

"""
Runs Monte Carlo simulations to determine the likelihood that a given investment strategy will succeed, based on the
projected lifetime cashflows.
Optionally, iteratively adjusts either starting funds, or discretionary spending, in order to meet the target level of
confidence.
Classes:  Asset_Model, Cashflow and Monte_Carlo
"""

# FYI: CONVENTION: all percentage values are stored as float numbers: e.g. 10% is stored as 0.1 and not 10.0 - the
#  helper function pct_str displays these numbers nicely as a string i.e. 0.1 -> "10.00%"
#  Constants representing percentages are written as 10e-2 -> 10% -> 0.1
# FIXME: enforce the above

# FYI: CONVENTION: all financial amounts are stored as positive / negative values for credit/liability -> this means
#  they are always added (no subtraction)

# FIXME: fix naming conventions - should be:
#  run_{one_year, one_iter}[_one_asset][_make_it_work]

# ToDo: use the same convention for file names for Envision as for Morningstar - update/rename play_envision.py


import sys
import os
import collections
import json
import getopt
import datetime as dt
from pprint import pformat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from logger import logger
from configuration_manager_class import ConfigurationManager
from cashflow_class import Cashflow 
from holdings_class import Holdings
from morningstar_stats_class import MorningstarStats
from montecarlo_simulation_class import MontecarloSimulation, ArrayRandGen
from utilities import error_exit, display_series, dollar_str, pct_str, write_nice_df_2_xl, print_df
import collections

global outf
DEBUG = False
# Configuration
plt.style.use('seaborn-v0_8-deep')
max_iter_to_print_param = 1e4  # Save iteration results if nb_iter_param smaller than this

pd.options.display.float_format = '{:,.2f}'.format  # Display 2 decimal places and commas for floats

class RunMontecarloSimulation():
    """Run the Montecarlo simulation"""
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.cross_correlated_rvs_flag = self.config_manager.get_param('cross_correlated_rvs_flag')
        self.multi_cpu_threshold = config_manager.get_param('multi_cpu_threshold')

        # Decide whether to use multiple CPUs or not
        if self.run_cnt >= self.multi_cpu_threshold:
            self.nb_cpu = self.config['nb_cpu']
        else:
            self.nb_cpu = 1
        self.simul_run_cnt = int(round(self.config['run_cnt'] / self.nb_cpu, 0))  # Number of simulations to run on each CPU
        self.run_cnt = self.simul_run_cnt * self.nb_cpu  # Total number of simulations to run (in case of rounding errors)

        return None

    def run(self) -> None:
        """Run the Montecarlo simulation"""

        # Load the cashflow and set the cashflow series
        cashflow_class = Cashflow(self.config_manager)
        goals_df = cashflow_class.process_goals_file()
        print(f"Goals DataFrame:\n{goals_df}")
        cashflow_df, cashflow_total_ser = cashflow_class.make_cashflow()
        print(f"Cashflow DataFrame:\n{cashflow_df}")
        print(f"Cashflow Total Series:\n{cashflow_total_ser.map(dollar_str)}")
        print(display_series(cashflow_total_ser, 2))
    
        # Load holdings and compute the initial holdings and set the initial holdings# Load holdings and compute the initial holdings and set the initial holdings
        self.portfolio = Holdings(self.config_manager)
        self.holdings_df, self.cash_amount = self.portfolio.load_holdings_data()
        print(f"Holdings DataFrame:\n{holdings_df}")
        print(f"Cash amount: ${cash_amount:,.2f}")
        holdings_df = self.portfolio.assign_cash_to_etf(self.holdings_df, self.cash_amount)
        print(f"Holdings DataFrame after reassigning cash to ETF_for_cash:\n{holdings_df}")
        self.portfolio.set_holdings_df(holdings_df)
        self.asset_class_df = self.portfolio.map_etf_to_asset_class()
        print(f"Asset Class DataFrame:\n{self.asset_class_df}")

        # Get the Morningstar Asset Stats and create the asset allocation model
        self.morningstar_stats = MorningstarStats(self.config_manager)
        df_stat, df_corr = self.morningstar_stats.get_asset_stats()
        logger.info(f'\nAsset Class Statistics\n{df_stat}')
        logger.info(f'\nAsset Class Correlations\n{df_corr}')
        # Match the stats and holdings (updates montecarlo_simulation.df_stat and df_corr)
        self.df_stat, self.df_corr = self.morningstar_stats.match_stats_vs_holdings(self.holdings_df)
        print(f"\nmontecarlo_simulation.df_stat shape: {self.df_stat.shape} montecarlo_simulation.df_corr shape: {self.df_corr.shape}") 

        #FIXME: Figure out the logic below
        #FIXME: handle the random number generator seed / split of ror_df in multi-CPU case
        if self.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
            logger.info(f"Using cross-correlated RoR series")
            self.morningstar_stats.set_nb_smpl(self.run_cnt * self.nb_ages)
            correlated_rvs = self.morningstar_stats.generate_correlated_rvs()
            logger.debug(f"Correlated returns Series (%):\n{correlated_rvs}")
            # Set the correlated RoR series
        else:          # Use Morningstar stats as is  
            logger.info(f"Using Morningstar stats as is")
            # Make list of asset classes stats and correlations
            asset_classes_list = [(mean, stdv) for mean, stdv in zip(df_stat['Expected Return'], df_stat['Standard Deviation'])]
            print(f"Asset Classes List:\n{asset_classes_list}")
            logger.info(f"Using Morningstar stats as is")
            # Create a sequence of pseudo-random seeds for the random number generators for each CPU
            master_seed_sequence = np.random.SeedSequence(self.config_manager.get_param('seed'))
            seed_sequence = master_seed_sequence.spawn(self.nb_cpu)



        # Create the Montecarlo simulation classes based on the number of CPUs
        self.montecarlo_simulation_list = [MontecarloSimulation(self.config_manager) for _ in range(self.nb_cpu)]
        for nb, mc_sim in enumerate(self.montecarlo_simulation_list):
            mc_sim.set_run_cnt(self.simul_run_cnt)
            mc_sim.set_cashflow_series(self.cashflow_series)
            mc_sim.set_holdings(self.holdings)
            mc_sim.set_morningstar_stats(self.morningstar_stats) 
            mc_sim.set_initial_holdings(self.asset_class_df)
            # FIXME: line below needs to change 
            if self.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
                # Assign simul_run_cnt columns to the correlated RoR series
                mc_sim.set_correlated_ror(correlated_rvs.iloc[:, nb*self.simul_run_cnt:(nb+1)*self.simul_run_cnt])
            else:          # Use Morningstar stats as is  
                # Assign the list of generators to the Montecarlo simulation                
                ror_gen_list = [ArrayRandGen(self.config_manager, mean, stdv, seed_sequence[nb]) for mean, stdv in asset_classes_list]
                mc_sim.set_ror_generator_list(ror_gen_list)



        return None 

def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    run_montecarlo_simulation = RunMontecarloSimulation(config_manager)

    
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
    # Match the stats and holdings (updates montecarlo_simulation.df_stat and df_corr)
    df_stat, df_corr = montecarlo_simulation.match_stats_vs_holdings(df_stat, df_corr)
    # Update df_stat and df_corr in morningstar_stats
    morningstar_stats.set_df_stat_and_corr(df_stat, df_corr)
    print(f"\nmontecarlo_simulation.df_stat shape: {montecarlo_simulation.df_stat.shape} montecarlo_simulation.df_corr shape: {montecarlo_simulation.df_corr.shape}") 

    if montecarlo_simulation.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
        logger.info(f"Using cross-correlated RoR series")
        morningstar_stats.set_nb_smpl(montecarlo_simulation.run_cnt * montecarlo_simulation.nb_ages)
        correlated_rvs = morningstar_stats.generate_correlated_rvs()
        logger.info(f"Correlated returns Series (%):\n{correlated_rvs}")
        # Set the correlated RoR series
        montecarlo_simulation.set_correlated_ror(correlated_rvs)
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
    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")