#!/usr/bin/env python

"""
Reference: https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
and downloaded code example: colormap-manipulation.py
Tick Placement Reference: https://www.tutorialspoint.com/matplotlib/matplotlib_setting_ticks_and_tick_labels.htm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import sys
import string


def plot_matrix(matrix: pd.DataFrame, vmin_val: float = None, vmax_val: float = None) -> None:
    """
    Helper function to plot data with associated colormap.
    """
    print(f"matplotlib verion: {mpl.__version__}")
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
    top_cmap = mpl.colormaps['Oranges_r']
    bottom_cmap = mpl.colormaps['Blues']
    # print(f"size of colormap: {top_cmap.N}")  # Colormaps have 256 samples by default
    # Take the first 128 samples of each colormap - and combine them into new colormap
    new_colors = np.vstack((top_cmap(np.linspace(0, 1, 128)),
                           bottom_cmap(np.linspace(0, 1, 128))))
    newcmap = ListedColormap(new_colors, name='OrangeBlue')
    figr = plt.figure(figsize=[11, 8.5], num=f"Confidence Level by Starting Funds", clear=True)
    ax = figr.add_subplot(1, 1, 1)  # 1 row , 1 column, index: 1
    # Map data to colors
    # im = ax.imshow(matrix, cmap=newcmap, vmin=vmin_v, vmax=vmax_v)  # data range is vmin_v - vmax_v
    psm = ax.pcolormesh(matrix, cmap=newcmap, vmin=vmin_v, vmax=vmax_v)  # data range is vmin_v - vmax_v
    # Show all ticks and label them with the respective list entries
    upr_ltrs = string.ascii_uppercase
    x_label = [c for _, c in zip(range(20), upr_ltrs)]
    ax.set_xticks(list(np.arange(0.5, 20, 1.0)))  # center the ticks on the squqre
    ax.set_xticklabels(x_label)

    # Create a list of 10 labels for Y axis
    y_min = round(vmin_v/10) * 10
    y_max = round(vmax_v/10 + 0.5) * 10
    step = (y_max - y_min) / 20
    y_label = list(np.arange(y_min, y_max, step))
    print(f"y_label: {y_label}")
    # FIXME - Want to have labels at 0, 5, ..., 100 ... i.e at the edge not middle
    # FIXME - see: https://www.tutorialspoint.com/matplotlib/matplotlib_setting_ticks_and_tick_labels.htm
    # ax.set_yticks(np.arange(len(y_label)), labels=y_label)
    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels([0, 10, 20])
    # figr.colorbar(im, ax=ax)  # Add colorbar legend
    figr.colorbar(psm, ax=ax)  # Add colorbar legend
    plt.show()
    return


def main(min_val, max_val, dimx):
    np.random.seed(19680801)
    # data = pd.DataFrame(np.random.randn(30, 30))
    df = pd.DataFrame(np.random.randint(min_val, max_val, size=dimx*dimx).reshape(dimx, dimx))  # Reshape to 20x20
    print(f"data\n{df.head()}")
    plot_matrix(df, vmin_val=min_val,  vmax_val=max_val)
    return


if __name__ == "__main__":
    # execute only if run as a script
    main(0, 100, 20)  # to simulate percentages - array size: dimx x dimx
    sys.exit("All Done")


