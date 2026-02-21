#!/usr/local/bin/python3
"""
Test run of making a Morningstar RoR DataFrame


IMPORTANT Parameters:
End_age = 70 - in .toml file
Death_age = 68 - in .toml file
cross_correlated_rvs_flag = false
run_cnt = 10
nb_cpu = 1
seed = 42
"""

import logging
import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from montecarlo_simulation_class import MontecarloSimulation, MontecarloSimulationDataLoader
from utilities import error_exit, display_series, dollar_str
import sys
DEBUG_FLAG = False

RUN_CNT = 10
CROSS_CORRELATED_RVS_FLAG = False
START_AGE = 67
ITER_CNT = 1000

logger.setLevel(logging.WARNING) 

class TestMkRorDf:
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
        # Run the test for uncorrelated RoR
        # Need to initialize the config manager and MontecarloSimulation class with the relevant rvs_flag
        for rvs_flag in [False, True]:
            self.config_manager.config['cross_correlated_rvs_flag'] = rvs_flag
            mc_data_loader = MontecarloSimulationDataLoader(self.config_manager)
            mc_data_loader.load_data()
            montecarlo_simulation = MontecarloSimulation(self.config_manager, mc_data_loader)
            montecarlo_simulation.set_run_cnt(RUN_CNT)

            # Modify data for this test
            montecarlo_simulation.cross_correlated_rvs_flag = rvs_flag
            logger.warning(f"\ncorrelated_rvs_flag: {montecarlo_simulation.cross_correlated_rvs_flag} - run_cnt: {montecarlo_simulation.run_cnt} - nb_ages: {montecarlo_simulation.nb_ages}")
            all_ror_df = pd.DataFrame()
            for _ in range(RUN_CNT):
                ror_df = montecarlo_simulation.mk_ror_df()
                # compute the average ror for each asset class as % and add to the dataframe
                all_ror_df = pd.concat([all_ror_df, ror_df], axis=1)

            if not rvs_flag:
                avg_regular = all_ror_df.apply(lambda x: 100*(x.mean() -1), axis=1)
                std_regular = all_ror_df.apply(lambda x: 100*np.std(x), axis=1)
            else:
                avg_cross_correlated = all_ror_df.apply(lambda x: 100*(x.mean() -1), axis=1)
                std_cross_correlated = all_ror_df.apply(lambda x: 100*np.std(x), axis=1)
        # Make a DF with 3 columns: Expected Return from Morningstar, avg regular, avg cross-correlated - making sure the index is the asset class
        result_index = all_ror_df.index
        # apply the index to the avg_regular and avg_cross_correlated
        avg_regular = avg_regular.reindex(result_index)
        avg_cross_correlated = avg_cross_correlated.reindex(result_index)
        result_df = pd.DataFrame(index=result_index)
        result_df['Expected Return'] = mc_data_loader.stats_df.loc[result_index, 'Expected Return']
        result_df['Avg Regular'] = avg_regular
        result_df['Avg Cross-Correlated'] = avg_cross_correlated
        result_df['Standard Deviation'] = mc_data_loader.stats_df.loc[result_index, 'Standard Deviation']
        result_df['Std Regular'] = std_regular
        result_df['Std Cross-Correlated'] = std_cross_correlated

        return result_df

if __name__ == "__main__":
    config_manager = ConfigurationManager(sys.argv)
    test_mk_ror = TestMkRorDf(config_manager)
    result_df = test_mk_ror.run()
    logger.warning(f"result_df: {result_df}")
    sys.exit("---\nDone!")