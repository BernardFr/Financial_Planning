#!/usr/bin/env python

"""
Creates a colored plot of a DF
Reference for colormap: https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import getopt
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ProgName = ''  # placeholder for program name
DEBUG = False
NB_COLORS = 20   # Number of colors in the colormap


def plot_color_matrix(matrix: pd.DataFrame, vmin_val: float = None, vmax_val: float = None, legend_dict: dict = None,
                      plot_file: str = None) -> None:
    """
    Helper function to plot data with associated colormap.
    """
    # print(f"matplotlib verion: {mpl.__version__}")
    # Figure out the range of the data set
    matrix_min = matrix.values.min()
    matrix_max = matrix.values.max()
    if vmin_val is None or vmax_val is None:
        vmin_v = matrix_min
        vmax_v = matrix_max
        print(f"Missing range arguments - Assigning: vmin: {vmin_v} - vmax: {vmax_v}")
    elif matrix_min < vmin_val or matrix_max > vmax_val:
        vmin_v = matrix_min
        vmax_v = matrix_max
        print(f"Matrix values exceed range arguments - Assigning: vmin: {vmin_v} - vmax: {vmax_v}")
    else:
        vmin_v = vmin_val
        vmax_v = vmax_val

    # Build nice colormap
    top_cmap = mpl.colormaps['Oranges_r'].resampled(NB_COLORS)
    bottom_cmap = mpl.colormaps['Blues'].resampled(NB_COLORS)
    # print(f"size of colormap: {top_cmap.N}")  # Colormaps have 256 samples by default
    # Take the first 128 samples of each colormap - and combine them into new colormap
    new_colors = np.vstack((top_cmap(np.linspace(0, 1, int(NB_COLORS/2))),
                            bottom_cmap(np.linspace(0, 1,int(NB_COLORS/2)))))
    newcmap = ListedColormap(new_colors, name='OrangeBlue')

    # set defaults values of title and axis labels
    title = ''
    x_lbl = ''
    y_lbl = ''
    if legend_dict is not None:  # set values passed as arguments
        title = legend_dict.setdefault('title', '')
        x_lbl = legend_dict.setdefault('x_label', '')
        y_lbl = legend_dict.setdefault('y_label', '')

    figr = plt.figure(figsize=[11, 8.5], num=title, clear=True)
    ax = figr.add_subplot(1, 1, 1)  # 1 row , 1 column, index: 1
    ax.set_title(title)
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    if legend_dict is not None:  # set values passed as arguments
        if 'xticklabels' in legend_dict.keys():
            x_label = legend_dict['xticklabels']
            ax.set_xticks(list(np.arange(0.5, len(x_label), 1.0)))  # center the ticks on the square
            ax.set_xticklabels(x_label)
        if 'yticklabels' in legend_dict.keys():
            y_label = legend_dict['yticklabels']
            ax.set_yticks(list(np.arange(0.5, len(y_label), 1.0)))  # center the ticks on the square
            ax.set_yticklabels(y_label)
            ax.set_yticklabels(legend_dict['yticklabels'])
    # Map data to colors
    # data range is vmin_v - vmax_v
    psm = ax.pcolormesh(matrix, cmap=newcmap, vmin=vmin_v, vmax=vmax_v)
    figr.colorbar(psm, ax=ax)  # Add colorbar legend
    if plot_file is not None:
        pdf_file = PdfPages(plot_file)
        plt.savefig(pdf_file, format="pdf", dpi=300, bbox_inches="tight")
        pdf_file.close()
    plt.show()
    return


def main(argv: [str]) -> None:
    """
    Plots a color matrix representation of a DF read from an input file
    """

    usage_str = f"Usage: {ProgName} -f excel_file\n"
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
        elif opt in "-f":  # show plots on display
            in_xl_file = str(arg)
        else:
            print(f'Error: Unrecognized option: {opt}')
            print(f"{usage_str}\n{__doc__}")
            sys.exit(2)

    try:
        in_xl_file
    except NameError:
        in_xl_file = ProgName + "_in.xlsx"

    plt_file = in_xl_file.replace("_in.xlsx", "_out.pdf")
    in_df = pd.read_excel(in_xl_file, header=0, index_col=0)
    print(f"in_df head:\n{in_df.head()}")
    legend_dict = dict()
    legend_dict['title'] = "Configence Level by Strategy and Starting Funds"
    legend_dict['x_label'] = "Strategy"
    legend_dict['y_label'] = "Starting Funds ($M)"
    legend_dict['xticklabels'] = in_df.columns  # Strategies
    legend_dict['yticklabels'] = [f"{x / 1e6:,.2f}" for x in in_df.index]  # Starting funds in $M
    plot_color_matrix(in_df, legend_dict=legend_dict, vmin_val=0, vmax_val=100, plot_file=plt_file)  #
    return


if __name__ == "__main__":
    # execute only if run as a script
    ProgName = Path(sys.argv[0]).stem
    main(sys.argv)
    sys.exit("All Done")
