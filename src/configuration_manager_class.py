from typing import Dict, List, Optional, Tuple
import os
import re
import string
from pathlib import Path
import sys
import getopt
import tomli
import datetime as dt
import matplotlib.pyplot as plt
from tick_timer_class import TickTimer

DEFAULT_NB_CPU = 1
DEFAULT_RUN_CNT = 100


class ConfigurationManager:
    """Manages configuration settings and constants 
    The config file is a TOML file with the following sections:
    [default]
    [ConfigurationManager]
    [ClassName_1]
    [ClassName_2]
    ...
    The default section is the default values for the configuration - values that can be overridden by the command line arguments.
    The ConfigurationManager section is the configuration for the ConfigurationManager class - values that are common to all classes.
    The ClassName_X section is the configuration for the ClassName_X class - values that are specific to the ClassName_X class.
    ...
    Each class will load:  
    - the default section - values that can be overridden by the command line arguments
    - the section under the class name - values that are specific to the class
    - the section under the ConfigurationManager class - values that are common to all classes
    The sections are merged together to form the final configuration.

    get_start_end_ages is utility function that returns the start and end ages
    """
   
    def __init__(self,  cmd_line: List[str]):
        self.prog_name = Path(cmd_line[0]).stem
        self.full_config = self._get_config(cmd_line[1:])
        self.config = self.get_class_config(self.__class__.__name__)
        self.start_age = self._compute_age_today()
        self.end_age = self.config['End_age']
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

    def _compute_age_today(self) -> int:
        """
        Compute my age today based on my DoB
        @return: age today

        Assume dob has format mm/dd/YYYY
        """
        dob = self.config['BF_BDAY']
        dob_mo, dob_dy, dob_yr = map(int, dob.split('/'))  # extract month, day, year as int
        dob_dt = dt.datetime(dob_yr, dob_mo, dob_dy)
        today_dt =dt.datetime.now()
        age_today_dt = today_dt - dob_dt
        age_today = int(age_today_dt.days/365)
        return age_today


    def get_start_end_ages(self):
        """
        utility function that returns the start and end ages.
        """
        return self.start_age, self.end_age

    def _get_output_paths(self) -> Dict[str, str]:
        """Get all output file paths for the current directory."""
        base_path = self.data_dir + self.current_dir + '/'
        paths_dict =  {'xl_out_filename': base_path + self.prog_name + '_' + self.current_dir + '.xlsx',
                'out_file': base_path + self.prog_name + '_' + self.current_dir + '_out.txt',
                'plt_file': self.prog_name + '_plots_' + self.current_dir + '.pdf',
                'quick_dir': base_path + self.quick_dir}
        return paths_dict


    def _read_config_file(self) -> dict:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(this_dir, self.prog_name + '.toml')
        with open(config_file, mode="rb") as fp:
            config = tomli.load(fp)
        return config


    def _get_config(self, cmd_line: List[str]) -> dict:
        """Get the config for the given program name
        Updates the default config with the command line arguments
        """
        config_dict = self._read_config_file()
        default_config = config_dict['default']
        prog_name = self.prog_name

        # Check if any of the parameters are overridden by command-line flags
        cmd_line_param = {}
        # Handle command line options
        Usage = f"{prog_name} -p plot_flag -u nb_cpu -c run_count -o {'s','d'}\n"
        try:
            opts, args = getopt.getopt(cmd_line, "hpc:u:c:o:")
        except getopt.GetoptError as err:
            # print help information and exit:
            print(err)  # will print something like "option -a not recognized"
            print(Usage)
            print(__doc__)
            sys.exit(2)
        for opt, arg in opts:
            if opt == "-h":
                assert False, Usage
            elif opt in "-p":  # show plots on display
                cmd_line_param['plot_flag'] = True
            elif opt in "-u":  # number of processes
                cmd_line_param['nb_cpu'] = int(arg)
            elif opt in "-c":  # set number of runs
                cmd_line_param['run_cnt'] = int(arg)
            elif opt in "-o":
                if arg not in ['s', 'd']:
                    assert False, Usage
                elif arg == 's':  # adjusting start_funds
                    cmd_line_param['opt_type'] = "start_funds"
                elif arg == 'd':  # adjust discretionary income
                    cmd_line_param['opt_type'] = 'discretionary'
                else:
                    assert False, Usage
            else:
                print(Usage)
                print(__doc__)
                assert False, f"unhandled option: {opt}"

        # Override the default parameters with the ones from the command line
        default_config.update(cmd_line_param)
        # Add content of default_config to config_dict
        config_dict.update(default_config)
        # Add default values for nb_cpu and run_cnt if not present in the config_dict
        if 'nb_cpu' not in config_dict.keys():   
            config_dict['nb_cpu'] = DEFAULT_NB_CPU
        if 'run_cnt' not in config_dict.keys():  
            config_dict['run_cnt'] = DEFAULT_RUN_CNT

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
