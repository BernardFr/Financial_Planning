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

# FYI: CONVENTION: all financial amounts are stored as positive / negative values for credit/liability -> this means
#  they are always added (no subtraction)

import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from logger import logger
from configuration_manager_class import ConfigurationManager
from cashflow_class import Cashflow 
from portfolio_class import Portfolio
from morningstar_stats_class import MorningstarStats
from arrayrandgen_class import ArrayRandGen
from montecarlo_simulation_class import MontecarloSimulation, MontecarloSimulationDataLoader
from utilities import error_exit, display_series, dollar_str, my_age
from collections import Counter

def _run_mc_simulation(simulation):
    """ calls the run method of the Montecarlo simulation object - used for parallel processing
    """
    return simulation.run()

DEBUG = False
# Configuration
plt.style.use('seaborn-v0_8-deep')
max_iter_to_print_param = 1e4  # Save iteration results if nb_iter_param smaller than this
DEFAULT_DELTA_ASSETS_MULTIPLIER = 0.05  # +/- 5%
DEFAULT_TARGET_CONFIDENCE_LEVEL = 80  # %
CONFIDENCE_LEVEL_TOLERANCE = 5.0  # +/- 5%
MAX_ITER = 20
pd.options.display.float_format = '{:,.2f}'.format  # Display 2 decimal places and commas for floats

class RunMontecarloSimulation:
    """Run the Montecarlo simulation"""
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.goals_df = pd.DataFrame()
        self.data_loader = MontecarloSimulationDataLoader(config_manager)
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.done_flag = False
        self.cross_correlated_rvs_flag = self.config_manager.config['cross_correlated_rvs_flag']
        self.multi_cpu_threshold = config_manager.config['multi_cpu_threshold']
        self.run_cnt = self.config['run_cnt']
        self.nb_cpu = self.config['nb_cpu']
        self.start_age = my_age(self.config['BDAY'])
        self.end_age = self.config['End_age']
        self.nb_ages = self.end_age - self.start_age + 1
        self.assets_multiplier = 1.0
        self.delta_assets_multiplier = DEFAULT_DELTA_ASSETS_MULTIPLIER  # +/- 5%
        self.target_confidence_level = DEFAULT_TARGET_CONFIDENCE_LEVEL  # %
        self.prev_confidence_level = 0.0
        self.montecarlo_simulation_list = []
        if self.nb_cpu > 1:
            self.simul_run_cnt = int(round(self.run_cnt / self.nb_cpu, 0))  # number of simulations to run on each CPU
            self.run_cnt = self.simul_run_cnt * self.nb_cpu  # total number of simulations to run
            for cpu_idx in range(self.nb_cpu):
                mc_sim = MontecarloSimulation(self.config_manager, self.data_loader, cpu_idx)
                mc_sim.set_run_cnt(self.simul_run_cnt)
                self.montecarlo_simulation_list.append(mc_sim)
            self.initial_pfolio_value = self.montecarlo_simulation_list[0].initial_pfolio_value
        else: # 1 CPU case
            self.mc_sim = MontecarloSimulation(self.config_manager, self.data_loader, 0)
            self.simul_run_cnt = self.run_cnt
            self.initial_pfolio_value = self.mc_sim.initial_pfolio_value
        logger.info(f"Run count: {self.run_cnt} - Number of CPUs: {self.nb_cpu} - Number of simulations to run on each CPU: {self.simul_run_cnt}")

        return


    def reinitialize_data(self, assets_multiplier: float) -> None:
        self.assets_multiplier = assets_multiplier
        if self.nb_cpu == 1:
            print(f"Reinitializing data for single CPU case: assets_multiplier: {assets_multiplier}")
            self.mc_sim.initialize_data(assets_multiplier)
        else: # Multi-CPU case
            for mc_sim in self.montecarlo_simulation_list:
                print(f"Reinitializing data for CPU {mc_sim.cpu_idx}: assets_multiplier: {assets_multiplier}")
                mc_sim.initialize_data(assets_multiplier)
        return None

    def run(self) -> None:
        """Run the Montecarlo simulation"""

        if self.nb_cpu == 1:    # Single CPU case
            self.final_result_series, self.busted_ages_dict, self.busted_cnt = self.mc_sim.run()
        else:    # Multi-CPU case
            with Pool(processes=self.nb_cpu) as pool:
                # Run each of the Montecarlo simulations on a different CPU in parallel
                result_obj = pool.map_async(_run_mc_simulation, self.montecarlo_simulation_list)
                result_list = result_obj.get()
                # Each simulation run produces (1) a [m, N] DF of resulting assets DF and (2) a [N,] series of final ages
                final_result_series_list = [x[0] for x in result_list]
                busted_ages_dict_list = [x[1] for x in result_list]
                busted_cnt_list = [x[2] for x in result_list]
                # Concatenate the final result series into a single series
                self.final_result_series = pd.concat(final_result_series_list, ignore_index=True)
                # Aggregate the busted ages dict into a single dict
                self.busted_ages_dict = sum((Counter(d) for d in busted_ages_dict_list), Counter())
                self.busted_cnt = sum(busted_cnt_list)

        return None

    def analyze_results(self) -> tuple[bool, float]:
        """Analyze the results of the Montecarlo simulation and adjust the assets multiplier"""
        confidence_level = 100.0 *  (self.run_cnt - self.busted_cnt) / self.run_cnt
        delta_confidence_level = confidence_level - self.target_confidence_level
        logger.info(f"Busted Count: {self.busted_cnt} - Confidence Level: {confidence_level:.2f}%")
        if self.busted_cnt > 0:
            logger.info(f"Busted Ages Dict:\n{self.busted_ages_dict}")
        logger.info(f"Final result series MEAN: ${self.final_result_series.mean():,.2f}")
        if confidence_level <= self.target_confidence_level or abs(delta_confidence_level) <= CONFIDENCE_LEVEL_TOLERANCE:
            keep_running = False
        elif self.prev_confidence_level * confidence_level < 0: # Sign of confidence level has changed - and is not zero (first iteration)
            keep_running = False
        else:
            self.prev_confidence_level = confidence_level
            # Delta positive -> reduce assets multiplier
            self.assets_multiplier = self.assets_multiplier - np.sign(delta_confidence_level) * self.delta_assets_multiplier
            if self.assets_multiplier < 0.0:
                keep_running = False
                error_exit(f"Assets multiplier is less than zero: {self.assets_multiplier:.2f}")
            else:
                keep_running = True
            logger.info(f"Delta confidence level: {delta_confidence_level:.2f} - New Assets multiplier: {self.assets_multiplier:.2f}\n")

        return keep_running, self.assets_multiplier

    def display_results(self) -> None:
        """Display the results of the Montecarlo simulation"""
        logger.info(f"--\n Final Assets multiplier: {self.assets_multiplier:.2f}")
        confidence_level = 100.0 *  (self.run_cnt - self.busted_cnt) / self.run_cnt
        logger.info(f"Confidence Level: {confidence_level:.2f}% - Busted Count: {self.busted_cnt} - Busted Ages Dict:\n{self.busted_ages_dict}")
        logger.info(f"Adjusted starting portfolio value: ${self.initial_pfolio_value * self.assets_multiplier:,.0f}")
        logger.info(f"Final result series MEAN: ${self.final_result_series.mean():,.2f}")
        logger.info(f"Final result  stats:\n{self.final_result_series.describe()}")
        return None

def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    # change logger level to WARNING
    logger.setLevel(logging.WARNING)
    simulation_runner = RunMontecarloSimulation(config_manager)
    logger.setLevel(logging.INFO)
    logger.info(f"Initialization done: Initial portfolio value: ${simulation_runner.initial_pfolio_value:,.0f}")

    nb_iter = 0
    keep_running = True
    assets_multiplier = 1.0
    while keep_running:
        simulation_runner.run()
        keep_running, assets_multiplier = simulation_runner.analyze_results()
        logger.info(f"Assets multiplier: {assets_multiplier:.2f}")
        nb_iter += 1
        if nb_iter >= MAX_ITER:
            keep_running = False
        simulation_runner.reinitialize_data(assets_multiplier)

    # Done
    simulation_runner.display_results()

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")