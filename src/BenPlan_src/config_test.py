#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple
import os
import sys
import tomli
import matplotlib.pyplot as plt
from tick_timer_class import TickTimer
from utils import get_program_name


class ConfigurationManager:
    """Manages configuration settings and constants for BenPlan.
    The ConfigurationManager config has the full config dict
    get_class_config() returns the config for the given class name
    The config file is in the same directory as the program with the same name
    but with a .toml extension
    the .toml is organized by class name
    [class_name]
    key = value
    key2 = value2
    ...
    """
    def __init__(self):
        self.prog_name = get_program_name()
        self.class_name = self.__class__.__name__
        self.config = self._load_config_from_file()
        return

    def _load_config_from_file(self) -> dict:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        # the name of the config file is the name of the program with .toml extension
        config_file_name = self.prog_name + '.toml'
        # print(f"config_file_name: {config_file_name}")
        config_file = os.path.join(this_dir, config_file_name)
        with open(config_file, mode="rb") as fp:
            config = tomli.load(fp)
        # print(f"config: {config}")
        return config

    def get_class_config(self, class_name) -> dict:
        return self.config.get(class_name, {})


class TestClass:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        return


def main(args: list[str]):
    config_manager = ConfigurationManager()
    test_class = TestClass(config_manager)
    print(f"test_class.config: {test_class.config}")

    return

if __name__ == '__main__':
    main(sys.argv)