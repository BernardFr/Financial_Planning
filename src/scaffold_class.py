#!/usr/local/bin/python3
"""
Scaffolding for creating a class 
"""


import numpy as np
import pandas as pd
from logger import logger
from configuration_manager_class import ConfigurationManager
from utilities import error_exit, display_series, dollar_str
import sys
DEBUG_FLAG = False

class ScaffoldClass:
    """
    Blabla
    """


    def __init__(self, config_manager: ConfigurationManager) -> None:
        """
        Loads initial holdings - from which we compute the target asset allocation and the starting funds
        Loads the yearly cashflows by age
        Loads a DF of rates of return for each asset class for each year
        Runs the simulation for the number of iterations specified in the configuration
        """
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.var1 = self.config['var1']
        return None


    def run(self) -> None:
       pass


def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    scaffold_class = ScaffoldClass(config_manager)

    scaffold_class.run()
    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")
