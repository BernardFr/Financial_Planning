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

import getopt
import logging
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



class MonteCarloProcessor:
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
        self.prog_name = ut.get_prog_name()
        self.config_manager = ConfigurationManager(self.prog_name, cmd_line)
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.nb_cpu = self.config['nb_cpu']
        self.run_cnt = self.config['run_cnt']
        self.iter_cnt = self.config['iter_cnt']
        self.opt_type = self.config['opt_type']
        self.cashflow_class = Cashflow(self.config_manager)

        
    def _log_info_tick(self, msg: str, xtra_log_msg = None) -> None:
        """Log info and tick timer"""
        if xtra_log_msg is not None:
            logger.info(f"{msg}\n{xtra_log_msg}")
        else:
            logger.info(msg)
        self.config_manager._tick_timer(msg)
        return


    def run(self):
        """Main processing method."""
        pass

    def cleanup(self, processor_kill_flag: bool = False):
        pass


    def compute_cashflow(self):



def main(cmd_line: [str]):
    """Main entry point for MonteCarlo Multi.
    """
    processor = MonteCarloProcessor(cmd_line)
    # Start the timer for the entire program
    start_time = dt.datetime.now()
    try:
        processor.run()
        logger.info(f"Processing completed")
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

