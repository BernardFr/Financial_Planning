from typing import Dict, List, Optional, Tuple
import os
import re
import string
import sys
import getopt
import tomli
import matplotlib.pyplot as plt
from tick_timer_class import TickTimer
from logger import logger

class ConfigurationManager:
    """Manages configuration settings and constants for BenPlan."""
    def __init__(self, prog_name: str, cmd_line: List[str]):
        self.prog_name = prog_name
        self.full_config = self._get_config(prog_name, cmd_line)
        self.config = self.get_class_config(self.__class__.__name__)
        self.home_dir = self.config['HOME_DIR']
        self.data_dir = self.home_dir + self.config['DATA_DIR']

        self.current_dir = self.data_dir
        self.output_paths = self._get_output_paths()
        # Initialize and start tick timer
        self.tick_timer = TickTimer(name=self.prog_name)
        self._start_timer()

        # Set matplotlib style
        plt.style.use(self.config['PLT_STYLE'])

        return

    def _start_timer(self) -> None:
        """Start the tick timer."""
        self.tick_timer.start()

    def _stop_timer(self) -> None:
        """Stop the tick timer."""
        self.tick_timer.stop()

    def _tick_timer(self, message: str) -> None:
        """Add a tick to the timer with a message."""
        self.tick_timer.tick(message)

    def _get_output_paths(self) -> Dict[str, str]:
        """Get all output file paths for the current directory."""
        base_path = self.data_dir 
        paths_dict =  {'xl_out_filename': base_path + self.prog_name  + '.xlsx',
                'out_file': base_path + self.prog_name + '_out.txt'}
        return paths_dict


    def  _read_config_file(self) -> dict:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(this_dir, self.prog_name + '.toml')
        with open(config_file, mode="rb") as fp:
            config = tomli.load(fp)
        return config


    def _get_config(self, prog_name: str, cmd_args: list[str]) -> dict:
        config_dict = self._read_config_file()
        default_config = config_dict['default']

        # Check if any of the parameters are overridden by command-line flags
        Usage = f"Usage: {prog_name} -h -v -p\n"
        cmd_line_param = {}
        try:
            opts, args = getopt.getopt(cmd_args, "hpv")
        except getopt.GetoptError: 
            print(f"{Usage}\n{__doc__}")
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print(f"{Usage}\n{__doc__}")
                sys.exit(2)
            else:
                print(f'Error: Unrecognized option: {opt}')
                print(f"{Usage}\n{__doc__}")
                sys.exit(2)

        # Override the default parameters with the ones from the command line
        for key in cmd_line_param.keys():
                default_config[key] = cmd_line_param[key]

        # Add content of default_config to config_dict
        config_dict.update(default_config)

        return config_dict

    def get_class_config(self, class_name: str) -> dict:
        """ 
        Get the config for the given class name 
        It combines the default section and the section under the class name 
        """
        # get the default config
        config = self.full_config.get('default', {})
        # get the class config
        class_config = self.full_config.get(class_name, {})
        # combine the default and class configs
        config.update(class_config)    
        # return the combined config
        return config
