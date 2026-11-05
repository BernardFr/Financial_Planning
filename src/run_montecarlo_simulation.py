#!/usr/local/bin/python3

"""
Runs Monte Carlo simulations to determine the likelihood that a given investment strategy will succeed, based on the
projected lifetime cashflows.
Optionally, iteratively adjusts either starting funds, or discretionary spending, in order to meet the target level of
confidence.
Classes:  Asset_Model, Cashflow and Monte_Carlo

Inputs - in ./Data: use find_most_recent helper function to find the most recent file 
- Portfolio: portfolio_by_Asset_class_YYYY_mm_dd.xlsx
- Cashflows: WFA_life_plan_YYYY_MM_DD.xlsx
- Asset model parameters (e.g. expected return, volatility, etc.)
- Monte Carlo simulation parameters (e.g. number of simulations, confidence level, etc.)
"""

# FYI: CONVENTION: all percentage values are stored as float numbers: e.g. 10% is stored as 0.1 and not 10.0 - the
#  helper function pct_str displays these numbers nicely as a string i.e. 0.1 -> "10.00%"
#  Constants representing percentages are written as 10e-2 -> 10% -> 0.1

# FYI: CONVENTION: all financial amounts are stored as positive / negative values for credit/liability -> this means
#  they are always added (no subtraction)

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from logger import logger
from configuration_manager_class import ConfigurationManager
from montecarlo_simulation_class import MontecarloSimulation, MontecarloSimulationDataLoader
from simulation_worker_class import SimulationWorker
from utilities import error_exit
from collections import Counter

DEBUG = False
# Configuration
plt.style.use('seaborn-v0_8-deep')
max_iter_to_print_param = 1e4  # Save iteration results if nb_iter_param smaller than this
pd.options.display.float_format = '{:,.2f}'.format  # Display 2 decimal places and commas for floats

class RunMontecarloSimulation:
    """Run the Montecarlo simulation"""
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.goals_df = pd.DataFrame()
        self.data_loader = MontecarloSimulationDataLoader(config_manager)
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.done_flag = False
        self.cross_correlated_rvs_flag = self.config['cross_correlated_rvs_flag']
        print(f"config: {self.config_manager.config}")
        self.multi_cpu_threshold = self.config['multi_cpu_threshold']
        self.run_cnt = self.config['run_cnt']
        self.nb_cpu = self.config['nb_cpu']
        self.assets_multiplier = 1.0
        self.delta_assets_multiplier = self.config['delta_assets_multiplier']
        self.target_confidence_level = self.config['target_confidence_level'] 
        self.confidence_level_tolerance = self.config['confidence_level_tolerance']
        self.prev_confidence_level = 0.0
        self.worker_list: list[SimulationWorker] = []
        self.simul_run_cnt = int(round(self.run_cnt / self.nb_cpu, 0))  # number of simulations to run on each CPU
        self.run_cnt = self.simul_run_cnt * self.nb_cpu  # total number of simulations to run
        for cpu_idx in range(self.nb_cpu):
            mc_sim = MontecarloSimulation(self.config_manager, self.data_loader, cpu_idx)
            mc_sim.set_run_cnt(self.simul_run_cnt)
            self.initial_pfolio_value = mc_sim.initial_pfolio_value
            # One long-lived process per CPU (also when nb_cpu == 1) - keeps its simulation (and RoR table) across
            # iterations, and starts every iteration from the same random-number state
            self.worker_list.append(SimulationWorker(mc_sim))
        logger.info(f"Run count: {self.run_cnt} - Number of CPUs: {self.nb_cpu} - Number of simulations to run on each CPU: {self.simul_run_cnt}")

        return


    def reinitialize_data(self, new_assets_multiplier: float) -> None:
        # Each worker applies the multiplier at the start of its next iteration
        self.assets_multiplier = new_assets_multiplier
        logger.debug(f"Reinitializing data for {self.nb_cpu} CPUs: assets_multiplier: {self.assets_multiplier:,.3f}")
        return None

    def close(self) -> None:
        """Stop the worker processes"""
        for worker in self.worker_list:
            worker.close()
        return None

    def run(self) -> None:
        """Run the Montecarlo simulation"""

        # Start all workers first so they run in parallel, then collect their results
        for worker in self.worker_list:
            worker.submit(self.assets_multiplier)
        result_list = [worker.result() for worker in self.worker_list]
        # Each simulation run produces (1) a [m, N] DF of resulting assets DF and (2) a [N,] series of final ages
        final_result_series_list = [x[0] for x in result_list]
        busted_years_dict_list = [x[1] for x in result_list]
        busted_cnt_list = [x[2] for x in result_list]
        # Concatenate the final result series into a single series
        self.final_result_series = pd.concat(final_result_series_list, ignore_index=True)
        # Aggregate the busted ages dict into a single dict
        self.busted_years_dict = sum((Counter(d) for d in busted_years_dict_list), Counter())
        self.busted_cnt = sum(busted_cnt_list)

        return None

    def analyze_results(self) -> tuple[bool, float]:
        """Analyze the results of the Montecarlo simulation and adjust the assets multiplier"""
        self.display_results()
        confidence_level = 100.0 *  (self.run_cnt - self.busted_cnt) / self.run_cnt
        delta_confidence_level = confidence_level - self.target_confidence_level
        if self.busted_cnt > 0:
            self.busted_years_dict = dict(sorted(self.busted_years_dict.items(), key=lambda item: item[0])) # Sort the dict by year
            logger.info(f"Busted Years Dict:\n{self.busted_years_dict}")
        if abs(delta_confidence_level) <= self.confidence_level_tolerance:
            keep_running = False
        elif self.prev_confidence_level * confidence_level < 0: # Sign of confidence level has changed - and is not zero (first iteration)
            keep_running = False
        else:
            keep_running = True

        if keep_running:
            self.prev_confidence_level = confidence_level
            # Delta positive -> reduce assets multiplier
            self.assets_multiplier = self.assets_multiplier - np.sign(delta_confidence_level) * self.delta_assets_multiplier
            if self.assets_multiplier < 0.0:
                keep_running = False
                error_exit(f"Assets multiplier is less than zero: {self.assets_multiplier:.3f}")
            else:
                keep_running = True
            logger.info(f"Delta confidence level: {delta_confidence_level:.2f} - New Assets multiplier: {self.assets_multiplier:.2f}\n")

        return keep_running, self.assets_multiplier

    def display_results(self) -> None:
        """Display the results of the Montecarlo simulation"""
        logger.info(f"Assets multiplier: {self.assets_multiplier:.3f} -- "
                    f"Adjusted starting portfolio value: ${self.initial_pfolio_value * self.assets_multiplier:,.0f}")
        confidence_level = 100.0 *  (self.run_cnt - self.busted_cnt) / self.run_cnt
        # sort busted years dict by year
        self.busted_years_dict = dict(sorted(self.busted_years_dict.items(), key=lambda item: item[0]))
        logger.info(f"Confidence Level: {confidence_level:.2f}% - Busted Count: {self.busted_cnt} - Busted Years Dict:\n{self.busted_years_dict}")
        logger.info(f"Result stats:\n{self.final_result_series.describe()}")
        return None

def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    logger.setLevel(logging.INFO)
    simulation_runner = RunMontecarloSimulation(config_manager)
    logger.info(f"Initialization done: Initial portfolio value: ${simulation_runner.initial_pfolio_value:,.0f}")

    nb_iter = 0
    keep_running = True
    assets_multiplier = 1.0
    try:
        while keep_running:
            logger.info(f"----\nStarting iteration {nb_iter + 1}")
            simulation_runner.run()
            keep_running, assets_multiplier = simulation_runner.analyze_results()
            logger.info(f"Assets multiplier: {assets_multiplier:.3f}")
            nb_iter += 1
            if nb_iter >= config_manager.config['max_iter']:
                keep_running = False
            simulation_runner.reinitialize_data(assets_multiplier)
    finally:
        simulation_runner.close()

    # Done
    logger.info(f"----\nFinal results: after {nb_iter} iterations")
    simulation_runner.display_results()

if __name__ == "__main__":
    main(sys.argv)
    print("---\nDone!")