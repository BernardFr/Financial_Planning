import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from logger import logger

Money_format = '$#,##0.00_);[Red]($#,##0.00)'  # Excel cell number format for dollar values
Zoom_level = 110
XL_col_width = 11  # Width of a column for $ in Excel
XL_col_1_width = 18  # Width of 1st column



Money_format = '$#,##0.00_);[Red]($#,##0.00)'  # Excel cell number format for dollar values
Zoom_level = 110
XL_col_width = 11  # Width of a column for $ in Excel
XL_col_1_width = 18  # Width of 1st column

def error_exit(error_message: str) -> None:
    """ Exit the program with an error message """
    logger.error(error_message)
    exit(-1)

def make_date_string() -> str:
    return dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d")


def mv_last_n_to_front(lst, n):
    """ Moves the last n elements of a list to first positions (in the same order """
    return lst[len(lst)-n:]+lst[:-n]


def mv_last_n_col_to_front(df, n):
    """ Moves the last n columns of a DF to first positions (in the same order """
    col = list(df.columns)
    return df[mv_last_n_to_front(col, n)]


def linear_transform_on_matrix(col, v, w):
    """
    Linear transformation on a matrix
    Given a matrix M, and vector X with columns ['a', 'b'], perform y = a*x + b for each cell x
    R[i,j] = M[i,j] * X['a'][j] +  X['b'][j]  - by row
    R[i,j] = M[i,j] * X['a'][i] +  X['b'][i]  - by column

    # FYI: Note that the index/column values need to be reset after the apply operation
    """
    return pd.Series([float(x * y + z) for x, y, z in zip(col.values, v.values, w.values)])


def print_df(df: pd.DataFrame, msg: str = None) -> None:
    """ Print troubleshooting info on a DF """
    if msg is not None:
        print(f"{msg}\n{len(df.index)} rows")
    print(f"Columns: {df.columns}")
    print(f"Index: {df.index}")
    print(df.head())
    print('/* ... */')
    print(df.tail())
    print('/* --------- */')
    return


def print_out(outfile, msg='\n'):
    """ Prints message to both outfile and terminal
    :rtype: None
    """
    print(msg)
    outfile.write(msg)
    return


def write_nice_df_2_xl(xlw: pd.ExcelWriter, df: pd.DataFrame, sheet: str, index: bool = False, mv_last: int = None) -> None:
    """
    Writes to an Excel worksheet and formats nicely a DF whose values represent $
    """
    if mv_last:  # Move last columns to front
        df = mv_last_n_col_to_front(df, mv_last)
    df.to_excel(xlw, sheet_name=sheet, float_format='%.2f', header=True, index=index)
    # Format the data in the worksheet
    workbook = xlw.book
    money_fmt = workbook.add_format({'num_format': Money_format})
    # bg_color does not seem to work
    header_row_fmt = workbook.add_format({'bold': True, 'align': 'center'})
    header_col_fmt = workbook.add_format({'bold': True, 'align': 'left'})
    worksheet = xlw.sheets[sheet]
    worksheet.set_zoom(Zoom_level)
    if mv_last:
        worksheet.freeze_panes(1, 1+mv_last)
    else:
        worksheet.freeze_panes(1, 1)
    # Format header row, header col and data cells by column
    worksheet.set_row(0, None, header_row_fmt)
    worksheet.set_column(0, 0, XL_col_1_width, header_col_fmt)
    worksheet.set_column(1, len(df.columns), XL_col_width, money_fmt)
    return


def to_dollar(x):
    """Converts a number to a string prepended with '$", and with commas rounded 2 digits"""
    return "${:,.2f}".format(x)


def tick_format(x, pos):
    """
    Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points
    Note: The unused 'pos' argument is necessary for FuncFormatter (see below)
    """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return "{:,.0f}".format(x)


def tick_format_w_dollar(x, pos):
    """
    Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points
    Note: The unused 'pos' argument is necessary for FuncFormatter (see below)
    """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return "${:,.0f}".format(x)


def plot_with_rails(df: pd.DataFrame, title: str = None, plt_file: str = None, rails: bool = False, dollar: bool = False, plot_flag: bool = False) -> None:
    """ Single line plot with rails: Average and +/- std dev"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    x_tick = range(df.shape[0])
    plt.plot(x_tick, list(df.values), color="steelblue")
    if rails:  # Plot lines for mean & rails for stddev
        # Build array of containing the mean value for all entries - as a Reference
        df_mean = float(df.mean())  # to ensure the result is not a Series
        df_stddev = float(df.std())
        # Create horizontal lines showing mean, +/- stddev
        ax.axhline(y=df_mean, color="red", ls="solid", alpha=0.5)
        ax.axhline(y=df_mean + df_stddev, color="red", ls="dashed", alpha=0.5)
        ax.axhline(y=df_mean - df_stddev, color="red", ls="dashed", alpha=0.5)

    # Decorate plot w/ labels, title etc
    plt.xticks(x_tick, list(df.index), rotation=0)
    if dollar:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format_w_dollar))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor("#E6E6E6")
    box = ax.get_position()
    ax.set_position([box.x0 + 0.05, box.y0, box.width, box.height])
    plt.grid(
        axis="y", which="both", color="darkgreen", linestyle="-", linewidth=1, alpha=0.5
    )
    if title is not None:
        plt.title(title)
    if plt_file is not None:
        # bbox_inches -> makes the legend fit
        plt.savefig(plt_file, format="pdf", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()
    return

