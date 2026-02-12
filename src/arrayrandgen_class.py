#!/usr/local/bin/python3

"""
Use test_rng.py to test the ArrayRandGen class
"""


import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from utilities import error_exit, display_series, dollar_str
import sys
DEBUG_FLAG = False

# For Testing
ROR_STATS = [ (3.00, 9.00),(2.60, 7.70), (1.50, 0.90),  (-0.05, 0.0)]  # Stocks, Bonds, TBills, Cash
ASSET_CLASSES = {'Stocks': 800000,
                'Bonds': 400000,
                'TBills': 200000,
                'Cash': 100000}
MASTER_SEED = 42
NB_CPU = 4
RUN_CNT = 1000

class ArrayRandGen:
    """
    This class is used to generate a list of random numbers based on a mean and stddev
    The numbers are: 1 + random_number/100  (interest rates)
    NOTE that the RoR is clipped to the floor and ceiling to account for historical data

    Parameters:
    config_manager: ConfigurationManager object
    name: str - the name of the asset class
    mean: float - the mean of the random numbers
    stdv: float - the stddev of the random numbers
    seed: int - the seed for the random number generator
    clip_flag: bool - whether to clip the random numbers to the floor and ceiling ... set it to False for testing
    """
    def __init__(self,config_manager: ConfigurationManager, name: str, mean: float, stdv: float, seed) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.nb_rv = self.config['end_age'] - self.config['start_age'] + 1  # number of random values to generate
        self.mean = mean
        self.stdv = stdv
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.name = name
        self.clip_flag = self.config['clip_flag']
        if self.clip_flag:
            self.ror_floor = self.config['RoR_floor']
            self.ror_ceiling = self.config['Ror_ceiling']
        else:
            self.ror_floor = None
            self.ror_ceiling = None

    def __iter__(self):
        return self

    def __next__(self) -> list[float]:
        """
        returns a list of nb random numbers based on mean and stddev
        numbers are: 1 + random_number/100  (interest rates)
        NOTE that the RoR is clipped to the floor and ceiling to account for historical data
        """
        random_values = self.rng.normal(self.mean, self.stdv, size=self.nb_rv)
        ror_values = 1 + 0.01 * random_values
        # Clip the final RoR values to historical bounds
        if self.clip_flag:
            ror_values = np.clip(ror_values, self.ror_floor, self.ror_ceiling)
        return ror_values.tolist()


def main(cmd_line: list[str]) -> None:
    print("Use test_rng.py to test the ArrayRandGen class")
    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit()
