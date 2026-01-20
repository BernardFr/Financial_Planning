#!/usr/local/bin/python3

"""
http://blog.miguelgrinberg.com/post/designing-a-restful-api-using-flask-restful
http://flask.pocoo.org/docs/quickstart/
http://flask-restful.readthedocs.org/en/latest/

Computes the amount of $$ that can be withdrawn during retirement - using Monte-Carlo simulations
All inputs are within the program
A scenario includes a number of phases (e.g. working, semi-retirement, retirement)
Each phase has a portfolio allocation of asset types (stocks, bonds, etc), and a contribution (negative for withdrawal).
If the "ToCompute" flag is set in a phase, then the contribution (in this case withdrawal) will be computed by the MC simulations.
More that one phase can set ToCompute, but all these phases will share the same computed contribution
The MC simulations use the Historical mean,stddev for each asset class - and generate random rate of returns using a
normal distribution based on the mean/stddev of the given asset class
The simulations are run "NbRun" and the contribution amount is returned based on the confidence factor

IMPORTANT
1/ Make sure each process uses a known seed for random number generator
See: https://bbabenko.github.io/multiprocessing-and-seeded-RNGs/
2/ See use of partial

"""

# ToDo: Add Readme - including syntax for TOML file
# ToDo: add loop to find value for retirement_spend that meets goal and/or optimal asset allocation

import sys, os
import traceback
import numpy as np
import pandas as pd
import collections
import datetime as dt
import montecarlo_utilities as mc_ut
import utilities as ut
from pprint import pformat
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, set_start_method
from logger import logger
from configuration_manager_class import ConfigurationManager
from cashflow_class import Cashflow
from morningstar_stats_class import MorningstarStats
from holdings_class import Holdings
from montecarlo_simulation_class import MontecarloSimulation
from holdings_class import Holdings
from typing import List
# import itertools

# from multiprocessing import cpu_count
from functools import partial

# Configuration
plt.style.use("seaborn-v0_8-deep")
Debug = False
plot_flag = False
outf = None  # placeholder for global variable
Max_run_to_save = 250  # Max number of runs for which we save assets at end of each run

"""
From: http://fc.standardandpoors.com/sites/client/generic/axa/axa4/Article.vm?topic=5991&siteContent=8088
Total Returns From 1926-2017*
     Stocks    Bonds    T-Bills
Annualized Return    10.2%    5.6%    3.5%
Best Year    60.0 (1935)    42.1 (1982)    14.0 (1981)
Worst Year    -41.1 (1931)    -12.2 (2009)    0.0 (1940)
Standard Deviation    18.7    7.7    0.9
"""
AssetClasses = collections.namedtuple(
    "AssetClasses", "Stocks Bonds TBills Cash"
)  # Immutable
historical_return_2017 = AssetClasses(
    Stocks=(10.20, 18.70), Bonds=(5.60, 7.70), TBills=(3.50, 0.90), Cash=(0.0, 0.0)
)
default_pfolio_alloc = AssetClasses(Stocks=0.75, Bonds=0.15, TBills=0.05, Cash=0.05)



class MonteCarloMulti:
    """
    Main orchestrator class for MonteCarlo  processing with multiple processors.
        1. Initialize: Load Config
        2. Load goals and compute cashflow by year
        3. Load Morningstar stats
        4 Load holdings - map to Morningstar asset classes -> generate portfolio with assets, %, Morninstar stats
        5. Option: load other stat sources ... and corresponding holding <-> asset class matcher
        6. Run simulation with cashflows and portfolio
            6.1. Option: "make it work": i.e. adjust spending in down years
        7. Show results
    """

    def __init__(self, cmd_line: List[str]):
        """
        1. Load Configuration
        """
        self.config_manager = ConfigurationManager(cmd_line)
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.nb_cpu = self.config['nb_cpu']
        self.run_cnt = self.config['run_cnt']
        self.cashflow_class = Cashflow(self.config_manager)
        self.morningstar_stats = MorningstarStats(self.config_manager)
        self.holdings = Holdings(self.config_manager)
        self.montecarlo_simulation = MontecarloSimulation(self.config_manager)

    def _log_info_tick(self, msg: str, xtra_log_msg = None) -> None:
        """Log info and tick timer"""
        if xtra_log_msg is not None:
            logger.info(f"{msg}\n{xtra_log_msg}")
        else:
            logger.info(msg)
        self.config_manager._tick_timer(msg)
        return

    def run(self):
        """Main processing method.
        2. Load goals and compute cashflow by year
        3. Load Morningstar stats
        4 Load holdings - map to Morningstar asset classes -> generate portfolio with assets, %, Morninstar stats
        5. Option: load other stat sources ... and corresponding holding <-> asset class matcher
        6. Run simulation with cashflows and portfolio
            6.1. Option: "make it work": i.e. adjust spending in down years
        7. Show results
        """
        # Load goals
        goals_df = self.cashflow_class.process_goals_file()
        print(f"Goals DataFrame:\n{goals_df}")
        # Load cashflow
        self.cashflow_df, self.cashflow_total_ser = self.cashflow_class.make_cashflow()
        print(f"Cashflow DataFrame:\n{self.cashflow_df}")
        print(f"Cashflow Total Series:\n{self.cashflow_total_ser}")

        # Load Morningstar stats
        self.df_stat, self.df_corr = self.morningstar_stats.get_asset_stats()
        logger.info(f'\nAsset Class Statistics\n{self.df_stat}')
        logger.info(f'\nAsset Class Correlations\n{self.df_corr}')
        self.correlated_rvs = self.morningstar_stats.generate_correlated_rvs()
        logger.info(f'\nCorrelated Random Variables\n{self.correlated_rvs}')

        # Load holdings
        holdings_df, cash_amount = self.holdings.load_holdings_data()
        print(f"Holdings DataFrame:\n{holdings_df}")
        print(f"Cash amount: ${cash_amount:,.2f}")
        holdings_df = self.holdings.assign_cash_to_etf(holdings_df, cash_amount)
        print(f"Holdings DataFrame after reassigning cash to ETF_for_cash:\n{holdings_df}")
        self.holdings.set_holdings_df(holdings_df)
        self.asset_class_df = self.holdings.map_etf_to_asset_class()
        print(f"Asset Class DataFrame:\n{self.asset_class_df}")

        # Load all the info in the montecarlo simulation
        self.montecarlo_simulation.set_initial_holdings(self.holdings_df)
        self.montecarlo_simulation.set_cashflow_df(self.cashflow_df)
        self.montecarlo_simulation.set_asset_class_df(self.asset_class_df)  # FIXME - holdings vs asset class
        self.montecarlo_simulation.set_ror_df(self.correlated_rvs)   # FIXME


    def cleanup(self, processor_kill_flag: bool = False):
        pass


def main(cmd_line: [str]):
    """Main entry point for MonteCarlo Multi.
    """
    simulation = MonteCarloMulti(cmd_line)
    # Start the timer for the entire program
    start_time = dt.datetime.now()
    try:
        simulation.run()
        logger.info(f"\nDone!")
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Cleaning up...")
        # Cancel all async tasks
    except Exception as e:
        logger.error(f"\nError: {e}")
        # dump stack trace
        logger.error(traceback.format_exc())
        logger.error("\n----\n")
    # finally:

    end_time = dt.datetime.now()
    logger.info(f'End: {str(end_time)} -- Run Time: {str(end_time - start_time)}\n')
    sys.exit(0)

if __name__ == "__main__":
    # bug fix for multiprocessing
    set_start_method("forkserver")  # from multiprocessing

    main(sys.argv)
    sys.exit(0)

