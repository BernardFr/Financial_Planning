import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from logger import logger
MAX_X_LABELS_LENGTH = 24  # Max number of labels written out on X axis of a plot

def tick_format(x, pos) -> str:  # pylint: disable=[W0613]
    """
    Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points
    Note that the unused argument 'pos' is required by FuncFormatter
    """
    # Reference:
    # https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return f'${x:,.0f}'


def lineplot_df(df: pd.DataFrame, title=None, plt_file=None, show_mean=None):
        """ Line plot of a DataFrame - each column is a line """
        max_x_labels_length = MAX_X_LABELS_LENGTH
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        labels = []
        cat_list = list(df.index)
        x_tick = range(df.shape[1])  # 0 ... nb_columns-1
        text_label = ''
        if show_mean:
            show_mean_flag = True
            if show_mean == 'Mean':  # show overall mean
                text_label = 'Means:\n'
            elif show_mean == 'Last12Mo':  # show average of 12 most recent months
                text_label = 'Last 12 months average:'
            else:
                try:
                    logger.warning(f"WARNING: Unknown option for lineplot_df show_mean: {show_mean}")
                except Exception:
                    print(f"WARNING: Unknown option for lineplot_df show_mean: {show_mean}")
                text_label = ''
        else:
            show_mean_flag = False

        # Plot each column in the array and add category name to list of labels
        for i in range(df.shape[0]):
            # Convert to Numpy array to make matplotlib happy
            plt.plot(x_tick, np.array(df.iloc[i]))
            labels.append(' ' + cat_list[i])
            mean = 0.  # make syntax check happy
            if show_mean:
                if show_mean == 'Mean':  # show overall mean
                    mean = df.iloc[i].mean()
                    ax.axhline(y=mean, color='darkgrey', ls='dashed', alpha=0.75)
                    labels.append(' ' + cat_list[i] + '_mean')
                elif show_mean == 'Last12Mo':  # show average of 12 most recent months
                    # last 12 months average - i.e. sum of last 12
                    mean = df.iloc[i, -12:].mean()
                    ax.axhline(y=mean, color='darkgrey', ls='dashed', alpha=0.75)
                    labels.append(' ' + cat_list[i] + '_Last12Mo')
                text_label += f'\n{cat_list[i]}: ${mean:,.0f}'
                text_label += f'\n    (${12.0 * mean:,.0f} / yr)'

        if show_mean_flag:  # Add a box with the actual values of the means
            # plt.text(0, mx, text_label, bbox=dict(edgecolor='red', facecolor='white'), ha='left',
            # va='top')
            plt.text(0.8, 0.85, text_label, bbox=dict(edgecolor='grey', facecolor='white'),
                     ha='left', va='top', transform=plt.gcf().transFigure)

        # TODO - see bpy for code
        if len(df.columns) > max_x_labels_length:
            ratio = int(len(df.columns) / max_x_labels_length)
            x_labels = [''] * len(df.columns)  # create a list of empty strings
            # fill in the labels with every ratio-th label
            actual_labels = list(df.columns[::ratio])
            for i in range(len(actual_labels)):
                x_labels[i * ratio] = str(actual_labels[i])
            plt.xticks(x_tick, x_labels, rotation=270)
            plt.legend(labels, ncol=1, loc='center left',
                       # place the center left anchor 100% right, and 50% down, i.e. center
                       bbox_to_anchor=[1, 0.5], columnspacing=1.0, labelspacing=0.0,
                       handletextpad=0.0, handlelength=1.5, fancybox=True, shadow=True)
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
        ax.set_facecolor('#E6E6E6')
        # Shrink current axis by 15% to make room for legend
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width * 0.85, box.height))
        plt.grid(axis='y', which='both', color='black', linestyle='-', linewidth=1, alpha=0.5)
        if title:
            plt.title(title)
        if plt_file:
            # bbox_inches -> makes the legend fit
            plt.savefig(plt_file, format='pdf', dpi=300, bbox_inches='tight')
        # plt.show()
        return

def plot_with_rails(df: pd.DataFrame, title=None, plt_file=None) -> None:
    """ Single line plot with rails: Average and +/- std dev"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    x_tick = range(df.shape[0])
    # Convert to Numpy array to make matplotlib happy
    plt.plot(x_tick, np.array(df), color='steelblue')
    # Plot lines for mean & rails for stddev
    # Build array of containing the mean value for all entries - as a Reference
    df_mean = float(df.mean())  # to ensure the result is not a Series
    df_stddev = float(df.std())
    # Create horizontal lines showing mean, +/- stddev
    ax.axhline(y=df_mean, color='red', ls='solid', alpha=0.5)
    ax.axhline(y=df_mean + df_stddev, color='red', ls='dashed', alpha=0.5)
    ax.axhline(y=df_mean - df_stddev, color='red', ls='dashed', alpha=0.5)

    # Decorate plot w/ labels, title etc
    plt.xticks(x_tick, list(df.index), rotation=270)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor('#E6E6E6')
    box = ax.get_position()
    ax.set_position((box.x0, box.y0, box.width, box.height))
    plt.grid(axis='y', which='both', color='darkgreen', linestyle='-', linewidth=1, alpha=0.5)
    if title:
        plt.title(title)
    if plt_file:
        # bbox_inches -> makes the legend fit
        plt.savefig(plt_file, format='pdf', dpi=300, bbox_inches='tight')
    return
