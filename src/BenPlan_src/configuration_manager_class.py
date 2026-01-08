from typing import Dict, List, Optional, Tuple
import os
import re
import string
import sys
import getopt
import tomli
import matplotlib.pyplot as plt
from tick_timer_class import TickTimer


class ConfigurationManager:
    """Manages configuration settings and constants for BenPlan."""
    def __init__(self, prog_name: str, cmd_line: List[str]):
        self.prog_name = prog_name
        self.full_config = self._get_config(prog_name, cmd_line)
        self.config = self.get_class_config(self.__class__.__name__)
        self.quick_dir = self.config['QUICK_DIR']
        self.home_dir = self.config['HOME_DIR']
        self.master_cat_file = self.home_dir + self.config['BENPLAN_MAP_FILE']
        self.data_dir = self.home_dir + self.config['DATA_DIR']

        self.a_z = list(string.ascii_uppercase)
        self.current_dir = self.get_current_directory()
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

    def get_current_directory(self) -> str:
        """Get the most recent data directory."""
        d_files = []
        with os.scandir(self.data_dir) as dir_list:
            for file_or_dir in dir_list:
                if file_or_dir.is_dir():
                    d_files.append(file_or_dir.name)

        dir_files = [d for d in d_files if re.fullmatch(self.config['dirNamePattern'], d)]
        return max(dir_files)

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
            elif opt in "-p":  # show plots on display
                cmd_line_param['PLOT_FLAG'] = True
            elif opt in "-v":  # skip Venmo
                cmd_line_param['SKIP_VENMO'] = True
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
