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

import sys
import datetime as dt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from logger import logger
from configuration_manager_class import ConfigurationManager
from cashflow_class import Cashflow 
from portfolio_class import Portfolio
import montecarlo_simulation_class
from morningstar_stats_class import MorningstarStats
from arrayrandgen_class import ArrayRandGen
from montecarlo_simulation_class import MontecarloSimulation, MontecarloSimulationDataLoader
from utilities import error_exit, display_series, dollar_str, pct_str, write_nice_df_2_xl, my_age
from collections import Counter
from functools import reduce

global outf
DEBUG = False
# Configuration
plt.style.use('seaborn-v0_8-deep')
max_iter_to_print_param = 1e4  # Save iteration results if nb_iter_param smaller than this
DEFAULT_DELTA_ASSETS_MULTIPLIER = 0.05  # +/- 5%
DEFAULT_TARGET_CONFIDENCE_LEVEL = 80  # %
CONFIDENCE_LEVEL_TOLERANCE = 5.0  # +/- 5%
MAX_ITER = 20
pd.options.display.float_format = '{:,.2f}'.format  # Display 2 decimal places and commas for floats

class RunMontecarloSimulation():
    """Run the Montecarlo simulation"""
    def __init__(self, config_manager: ConfigurationManager) -> None:
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
        # Decide whether to use multiple CPUs or not
        # FIXME: Enable auto cpu when multi-cpu works
        # if self.run_cnt >= self.multi_cpu_threshold:
        #     self.nb_cpu = self.config['nb_cpu']
        # else:
        #     self.nb_cpu = 1
        if self.nb_cpu > 1:
            self.simul_run_cnt = int(round(self.config['run_cnt'] / self.nb_cpu, 0))  # Number of simulations to run on each CPU
            self.run_cnt = self.simul_run_cnt * self.nb_cpu  # Total number of simulations to run (in case of rounding errors)
        else:
            self.simul_run_cnt = self.run_cnt
        logger.info(f"Run count: {self.run_cnt} - Number of CPUs: {self.nb_cpu} - Number of simulations to run on each CPU: {self.simul_run_cnt}")
        mc_data_loader = MontecarloSimulationDataLoader(self.config_manager)
        mc_data_loader.load_data()
        self.montecarlo_simulation_list = [MontecarloSimulation(self.config_manager, mc_data_loader) for _ in range(self.nb_cpu)]
        return None

        return None


    def load_initial_data(self) -> None:
        """Load the initial data for the Montecarlo simulation 
        - Goals -> Cashflow
        - Holdings -> Asset Class DataFrame
        - Morningstar Stats -> Asset Class Statistics and Correlations
        - RoR series -> Correlated RoR series or ror generators based on cross_correlated_rvs_flag
        - Initialize the Montecarlo simulation classes based on the number of CPUs
        """

        # Load the cashflow and set the cashflow series
        cashflow_class = Cashflow(self.config_manager)
        self.goals_df = cashflow_class.process_goals_file()
        logger.info(f"Goals DataFrame:\n{self.goals_df}")
        self.cashflow_df, self.cashflow_series = cashflow_class.make_cashflow()
        logger.info(f"Cashflow DataFrame:\n{self.cashflow_df}")
        logger.info(display_series(self.cashflow_series, 2))
    
        # Load holdings and compute the initial holdings and set the initial holdings# Load holdings and compute the initial holdings and set the initial holdings
        self.portfolio = Portfolio(self.config_manager)

        self.holdings_df, self.cash_amount = self.portfolio.load_holdings_data()
        logger.info(f"Holdings DataFrame:\n{self.holdings_df}")
        logger.info(f"Cash amount: ${self.cash_amount:,.2f}")

        self.holdings_df = self.portfolio.assign_cash_to_etf(self.holdings_df, self.cash_amount)
        logger.info(f"Holdings DataFrame after reassigning cash to ETF_for_cash:\n{self.holdings_df}")
        self.portfolio.set_holdings_df(self.holdings_df)
        self.asset_class_df = self.portfolio.map_etf_to_asset_class()
        logger.info(f"Asset Class DataFrame:\n{self.asset_class_df}")

        # Apply the assets multiplier to the asset class DataFrame
        self.asset_class_df = self.asset_class_df * self.assets_multiplier
        logger.info(f"Asset Class DataFrame after multiplier:\n{self.asset_class_df}")
        logger.info(f"Total Portfolio Market Value after multiplier: ${self.asset_class_df['Market Value'].sum():,.2f}")


        # Get the Morningstar Asset Stats and create the asset allocation model
        self.morningstar_stats = MorningstarStats(self.config_manager)
        df_stat, df_corr = self.morningstar_stats.get_asset_stats()
        logger.info(f'\nAsset Class Statistics\n{df_stat}')
        logger.info(f'\nAsset Class Correlations\n{df_corr}')
        # Match the stats and holdings (updates montecarlo_simulation.df_stat and df_corr)
        self.df_stat, self.df_corr = self.morningstar_stats.match_stats_vs_assets(self.asset_class_df)
        logger.info(f"montecarlo_simulation.df_stat shape: {self.df_stat.shape} montecarlo_simulation.df_corr shape: {self.df_corr.shape}") 
        self.morningstar_stats.set_df_stat_and_corr(self.df_stat, self.df_corr)

        # Generate the RoR series for the Montecarlo simulations
        if self.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
            logger.info(f"Using cross-correlated RoR series")
            self.morningstar_stats.set_nb_smpl(self.run_cnt * self.nb_ages)
            correlated_rvs = self.morningstar_stats.generate_correlated_rvs()
            logger.debug(f"Correlated returns Series (%):\n{correlated_rvs}")
            # Set the correlated RoR series
        else:          # Use Morningstar stats as is  
            logger.info(f"Using Morningstar stats as is")
            # Make list of asset classes stats and correlations
            asset_classes_list = [(mean, stdv) for mean, stdv in zip(self.df_stat['Expected Return'], self.df_stat['Standard Deviation'])]
            logger.info(f"Asset Classes List:\n{asset_classes_list}")
            logger.info(f"Using Morningstar stats as is")
            # Create a sequence of pseudo-random seeds for the random number generators for each CPU
            master_seed_sequence = np.random.SeedSequence(self.config_manager.config['seed'])
            seed_sequence = master_seed_sequence.spawn(self.nb_cpu)
        
        # FIXME: handle multi-cpu case
        # Create the Montecarlo simulation classes based on the number of CPUs
        self.data_loader = MontecarloSimulationDataLoader(self.config_manager)
        if self.nb_cpu == 1:    
            self.montecarlo_simulation_list = [MontecarloSimulation(self.config_manager,self.data_loader)]
        else:
            error_exit(f"Multi-CPU case not implemented yet")
        for nb, mc_sim in enumerate(self.montecarlo_simulation_list):
            mc_sim.set_run_cnt(self.simul_run_cnt)
            mc_sim.set_cashflow_series(self.cashflow_series)
            mc_sim.set_initial_holdings(self.asset_class_df)
            # FIXME: line below needs to change 
            if self.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
                # Assign simul_run_cnt*nb_ages columns to the correlated RoR series
                mc_sim.set_correlated_ror(correlated_rvs.iloc[ :(1+nb)*self.nb_ages*self.simul_run_cnt])
            else:          # Use Morningstar stats as is  
                # Assign the list of generators to the Montecarlo simulation                
                ror_gen_list = [ArrayRandGen(self.config_manager, mean, stdv, seed_sequence[nb]) for mean, stdv in asset_classes_list]
                mc_sim.set_ror_generator_list(ror_gen_list)
        
        return None

    def reinitialize_data(self) -> None:
        # FIXME: handle add seed sequence as a parameter
        for mc_sim in self.montecarlo_simulation_list:
            mc_sim.reinitialize_data(self.assets_multiplier)
        return None

    def run(self) -> None:
        """Run the Montecarlo simulation"""

        if self.nb_cpu == 1:    # Single CPU case
            self.final_result_series, self.busted_ages_dict, self.busted_cnt = self.montecarlo_simulation_list[0].run()
        else:    # Multi-CPU case
            # FIXME: handle multi-cpu case
            error_exit(f"Multi-CPU case not implemented yet")
            # Create a pool of workers
            with Pool(processes=self.nb_cpu) as pool:
                result_obj = pool.map_async(self.montecarlo_simulation_list[nb].run, range(self.simul_run_cnt))
                result_list = result_obj.get()
                # Each simulation run produces (1) a [m, N] DF of resulting assets DF and (2) a [N,] series of final ages
                final_result_series = [x[0] for x in result_list]   
                busted_ages_dict_list = [x[1] for x in result_list]
                busted_cnt = [x[2] for x in result_list]
                # Concatenate the final result series into a single series
                self.final_result_series = pd.concat(final_result_series, ignore_index=True)
                # Aggregate the busted ages dict into a single dict
                self.busted_ages_dict = sum((Counter(d) for d in busted_ages_dict_list), Counter())
                self.busted_cnt = sum(busted_cnt)

        return None

    def analyze_results(self) -> None:
        """Analyze the results of the Montecarlo simulation and adjust the assets multiplier"""
        confidence_level = 100.0 *  (self.run_cnt - self.busted_cnt) / self.run_cnt
        delta_confidence_level = confidence_level - self.target_confidence_level
        logger.info(f"Busted Count: {self.busted_cnt} - Confidence Level: {confidence_level:.2f}%")
        if self.busted_cnt > 0:
            logger.info(f"Busted Ages Dict:\n{self.busted_ages_dict}")
        logger.info(f"Final result  stats:\n{self.final_result_series.describe()}")
        logger.info(f"Final result series MEAN: ${self.final_result_series.mean():,.2f}")
        if abs(delta_confidence_level) <= CONFIDENCE_LEVEL_TOLERANCE:
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

    def reset_data(self, assets_multiplier: float) -> None:
        """Reset the data for the next simulation"""
        self.assets_multiplier = assets_multiplier
        self.montecarlo_simulation.reinitialize_data(self.assets_multiplier)
        return None

    def display_results(self) -> None:
        """Display the results of the Montecarlo simulation"""
        logger.info(f"Assets multiplier: {self.assets_multiplier:.2f}")
        logger.info(f"Final result series:\n{self.final_result_series.map(dollar_str)}")
        logger.info(f"Busted Ages Dict:\n{self.busted_ages_dict}")
        logger.info(f"Busted Count: {self.busted_cnt}")
        return None

def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    simulation_runner = RunMontecarloSimulation(config_manager)
    # FIXME: Move this to RunMontecarloSimulation class / load intial data
    # simulation_runner.load_initial_data()

    nb_iter = 0
    while True:
        simulation_runner.run()
        keep_running, assets_multiplier = simulation_runner.analyze_results()
        if not keep_running:
            break
        else:
            logger.info(f"Assets multiplier: {assets_multiplier:.2f}")
            simulation_runner.reinitialize_data()
            nb_iter += 1
            if nb_iter >= MAX_ITER:
                break
    # Done
    simulation_runner.display_results()

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")