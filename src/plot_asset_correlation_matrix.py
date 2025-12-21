#!/usr/bin/env python

"""
Creates a colored plot of the correlation of the asset classes
Reference for colormap: https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
"""

import sys
import os
import re
import getopt
from pathlib import Path
import pandas as pd
sys.path.insert(0, '../Color_Map/')  # Add path to the directory containing plot_color_matrix
from plot_color_matrix import plot_color_matrix as pcm


ProgName = ''  # placeholder for program name
DEBUG = False
NB_COLORS = 20   # Number of colors in the colormap
NamePattern = 'Morningstar_asset_stats_\d{4}-\d{2}-\d{2}.xlsx'  # e.g 'Morningstar_asset_stats_2023-04-11.xlsx'
XLabelLen = 5  # Length of the X axis labels


def make_xlabel(in_str: str, label_len) -> str:
    """
    Shorten the X axis labels to XLabelLen characters
    """
    # remove white space
    in_str = in_str.replace(' ', '')
    return in_str[0:label_len]

def get_most_recent_stats_file() -> str:
    """
    Returns the name of the most recent Morningstar Stats file in the directory
    """
    # Get the list of all files in current directory
    all_files = []
    with os.scandir('.') as file_list:
        for file_or_dir in file_list:
            if file_or_dir.is_file():
                all_files += [file_or_dir.name]
    # fullmatch returns None if there is No match
    all_files = [f for f in all_files if re.fullmatch(NamePattern, f)]

    # Get the most recent file - i.e. the max as computed on the string
    return max(all_files)


def main(argv: [str]) -> None:
    """
    Plots a color matrix representation of a Morningstart Stats file
    """
    global ProgName

    in_xl_file = get_most_recent_stats_file()  # default input file - the most recent one in the directory

    usage_str = f"Usage: {ProgName} -f Morningstar_asset_stats_YYYY_MM_DD.xlsx\n"
    # Check if any of the parameters are overridden by command-line flags
    try:
        opts, args = getopt.getopt(argv[1:], "hf:")
    except getopt.GetoptError:
        print(f'Error: Unrecognized option in: {" ".join(argv)}')
        print(f"{usage_str}\n{__doc__}")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(f"{usage_str}\n{__doc__}")
            sys.exit(2)
        elif opt in "-f":  # Input file - optional
            in_xl_file = str(arg)
        else:
            print(f'Error: Unrecognized option: {opt}')
            print(f"{usage_str}\n{__doc__}")
            sys.exit(2)
    if DEBUG:
        print(f"Input file: {in_xl_file}")
    plt_file = in_xl_file.replace(".xlsx", "_out.pdf")  # replace the ".xlsx" extension

    in_df = pd.read_excel(in_xl_file, sheet_name='validation', header=0, index_col=0)
    legend_dict = dict()
    legend_dict['title'] = "Asset Class Correlation Matrix"
    legend_dict['x_label'] = "Asset Class"
    legend_dict['y_label'] = "Asset Class"
    legend_dict['xticklabels'] = [make_xlabel(ss, XLabelLen) for ss in in_df.columns]  # Asset Classes, truncated
    legend_dict['yticklabels'] = [ss.replace(' ', '') for ss in in_df.columns]  # Asset Classes, 'truncated
    pcm(in_df, legend_dict=legend_dict, vmin_val=-1, vmax_val=1, plot_file=plt_file)  #
    return


if __name__ == "__main__":
    # execute only if run as a script
    ProgName = Path(sys.argv[0]).stem
    main(sys.argv)
    sys.exit("All Done")
