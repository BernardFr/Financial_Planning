#!/usr/local/bin/python3
"""
Compare the performance of spipy.stats.norm.rvs vs np.random.default_rng.normal
"""


import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from utilities import error_exit, display_series, dollar_str
from scipy.stats import norm

import sys
DEBUG_FLAG = False

MEAN = 22.0
STDV = 9.5

class RngVsRvsClass:
    """
    Compare the performance of the default RNG vs the RVs
    Parameters:
    config_manager: ConfigurationManager object
    Returns:
    None
    """

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.seed = self.config['seed']
        self.run_cnt = self.config['run_cnt']
        self.rng = np.random.default_rng(self.seed)

        return None

    def generate_with_norm_rvs(self) -> np.array:
        """
        Generate random variables using the default RNG
        Parameters:
        mean: float
        stdv: float
        seed: int
        Returns:
        np.array
        """
        result_array = norm.rvs(size=self.run_cnt, loc=MEAN, scale=STDV, random_state=self.seed)  # create a series w/ the desired stats
        return result_array

    def generate_rvs_with_default_rng(self) -> np.array:
        """
        Generate random variables using the Default_RNG class
        Parameters:
        mean: float
        stdv: float
        seed: int
        Returns:
        np.array
        """
        random_values = self.rng.normal(MEAN, STDV, size=self.run_cnt)
        return np.array(random_values)

    def run(self) -> None:
        """
        Run the simulation
        Parameters:
        None
        Returns:
        None
        """
        norm_rvs_array = self.generate_with_norm_rvs()
        norm_rvs_array_with_default_rng = self.generate_rvs_with_default_rng()
        # store mean and stdv of the arrays in a dataframe
        result_df = pd.DataFrame(columns=['Target', 'NORM_RVS', 'Delta NORM_RVS', 'Default_RNG', 'Delta Default_RNG'], index=['mean', 'stdv'])
        result_df['Target'] = [MEAN, STDV]
        norm_rvs_mean = np.mean(norm_rvs_array)
        rvs_stdv = np.std(norm_rvs_array)
        delta_norm_rvs_mean = norm_rvs_mean - MEAN
        delta_norm_rvs_stdv = rvs_stdv - STDV
        result_df['NORM_RVS'] = [norm_rvs_mean, rvs_stdv]
        result_df['Delta NORM_RVS'] = [delta_norm_rvs_mean,   delta_norm_rvs_stdv]
        rvs_mean_with_default_rng = np.mean(norm_rvs_array_with_default_rng)
        rvs_stdv_with_default_rng = np.std(norm_rvs_array_with_default_rng)
        delta_rvs_mean_with_default_rng = rvs_mean_with_default_rng - MEAN
        delta_rvs_stdv_with_default_rng = rvs_stdv_with_default_rng - STDV
        result_df['Default_RNG'] = [rvs_mean_with_default_rng, rvs_stdv_with_default_rng]
        result_df['Delta Default_RNG'] = [delta_rvs_mean_with_default_rng, delta_rvs_stdv_with_default_rng]

        if abs(delta_norm_rvs_mean) < abs(delta_rvs_mean_with_default_rng) and abs(delta_norm_rvs_stdv) < abs(delta_rvs_stdv_with_default_rng):
            logger.info("NORM_RVS is better than Default_RNG")
        elif abs(delta_norm_rvs_mean) > abs(delta_rvs_mean_with_default_rng) and abs(delta_norm_rvs_stdv) > abs(delta_rvs_stdv_with_default_rng):
            logger.info("Default_RNG is better than NORM_RVS")
        else:
            logger.info("Default NORM_RVS and Default_RNG are about the same")

        return result_df

def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    rng_vs_rvs_class = RngVsRvsClass(config_manager)

    result_df = rng_vs_rvs_class.run()
    logger.info(result_df)
    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
