"""
Get TOML configuration files
Reference:
https://towardsdatascience.com/from-novice-to-expert-how-to-write-a-configuration-file-in-python-273e171a8eb3
https://realpython.com/python-toml/#configurations-and-configuration-files
"""

import os
import re
import sys
import inspect
import pandas as pd
import datetime as dt
import string

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from plot_functions import lineplot_df
from logger import logger

AGG_COLS = ['YearlyAvgAll', 'Last12Mo']
MY_BD = '04-09-1958'
OTHER_PCT = 0.2  # 20%

A_Z = list(string.ascii_uppercase)  # ['A', .., 'Z']


def get_prog_name() -> str:
    """ Get the program name from the path """
    basename = os.path.basename(sys.argv[0])
    prog_name = re.sub("\\.py$", "", basename)

    return prog_name


def this_function():
    """ returns the name of the function that called this_function """
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return "unknown"
    return frame.f_back.f_code.co_name
    

def my_age() -> int:
    """Returns my age"""
    today = dt.datetime.now().strftime("%m-%d-%Y")
    my_mo, my_dy, my_yr = [int(x) for x in MY_BD.split('-')]
    today_mo, today_dy, today_yr = [int(x) for x in today.split('-')]
    if today_mo > my_mo or (today_mo == my_mo and today_dy >= my_dy):
        return today_yr - my_yr
    else:
        return today_yr - my_yr - 1


def print_out(outfile_name: str, msg='\n'):
    """ Prints message to both outfile and terminal
    :rtype: None
    """
    print(msg + '\n')
    with open(outfile_name, 'a') as f:
        f.write(msg)
    return


def to_dollar_str(z) -> str:
    """ Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points   i.e $xx,xxx.xx """
    return z if isinstance(z, str) else f'${float(z):,.2f}'


def make_dollar_pretty(df: pd.DataFrame) -> str:
    """
    Converts numbers in the array to dollar format i.e $xx,xxx.xx
    df  must be a dataframe - so use df[['a']] if single column of a bigger DF
    """
    # return repr(df.applymap(to_dollar_str))  # make it pretty and convert it to a string
    return repr(df.map(to_dollar_str))  # make it pretty and convert it to a string


def now_str() -> str:
    """ Returns the current date and time as a string """
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def month_2_year(mm: str) -> str:
    """ YYYY-MM -> YYYY """
    return mm.split('-')[0]


def month_2_age(mm: str) -> int:
    """ Convert YYYY-MM to age based on MY_BD (04-09-1958) """
    yr, mo = mm.split('-')
    yr = int(yr)
    mo = int(mo)
    my_bd_mo, _, my_bd_yr = MY_BD.split('-')
    my_bd_mo = int(my_bd_mo)
    my_bd_yr = int(my_bd_yr)
    return yr - my_bd_yr  if mo > my_bd_mo else yr - my_bd_yr - 1


def mv_last_n_to_front(lst, n):
    """ Moves the last n elements of a list to first positions (in the same order """
    return lst[len(lst) - n:] + lst[:-n]


def mv_last_n_col_to_front(df, n):
    """ Moves the last n columns of a DF to first positions (in the same order """
    col = list(df.columns)
    return df[mv_last_n_to_front(col, n)]


def index_2_ltr(idx: int) -> str:
    """ Map N to N'th letter of alphabet  3 -> 'C' """
    assert idx < len(A_Z), f"idx too large: {idx} - index_2_ltr only works with numbers < 26"
    return A_Z[idx]


def col_name_2_ltr(col_list: list, start_name, finish_name=None) -> str:
    """
    Maps the names of 2 columns to the corresponding letter range in Excel worksheet
    E.g: col_list = ['X', 'Y', 1]
    start_name = 'Y'
    finish_name = 1
    return -> 'B:C"
    """
    start_idx = col_list.index(start_name)
    if finish_name:
        finish_idx = col_list.index(finish_name)
    else:
        finish_idx = start_idx
    if start_idx > finish_idx:  # In case the order is wrong, swap the 2 indices
        start_idx, finish_idx = finish_idx, start_idx
    return index_2_ltr(start_idx) + ':' + index_2_ltr(finish_idx)


def expand_dimensions(dim_str):
    """
    :param dim_str: format 'A1:XYddd' where ddd is a number and XY a set of letters
    :return:
    """
    t_l, b_r = dim_str.split(':')
    assert t_l == "A1", f"Whaaat??? top left {t_l} is not A1"
    # figure out the label bottom right cell
    last_col = ''
    digits = []
    for c in b_r:
        if c.isalpha():
            last_col += c
        elif c.isnumeric():
            digits += [c]
        else:
            print(f"Wow we have an unexpected character: {c}")
    # print(f"digits: {digits}")
    # Convert the list of digits into an actual number
    digits.reverse()  # list digits in increasing power of 10
    powr_10 = 1
    last_row = 0
    for d in digits:
        last_row += int(d) * powr_10
        powr_10 *= 10

    # Make a list of all the letter labels of the columns
    assert len(last_col) <= 2, (f"FixMe: I can only deal with 1-2 letter column labels not "
                                f"{len(last_col)} - \
        last_col = {last_col}")
    col_range = []
    # Start with single letter labels
    for ll in string.ascii_uppercase:
        col_range += [ll]
        if ll == last_col:
            break
    if len(last_col) == 2:  # 2-letter column labels - if needed
        for l1 in string.ascii_uppercase:
            for l2 in string.ascii_uppercase:
                ll = l1 + l2
                col_range += [ll]
                if ll == last_col:
                    break
    return col_range, last_row

def mk_trailing_df(df: pd.DataFrame, lag: int = 12) -> pd.DataFrame:
    """Create trailing analysis DataFrame."""
    col_len = len(df.columns)
    assert lag > 0, 'mk_trailing_df lag parameter must be > 0'
    assert col_len >= lag, 'mk_trailing_df must have at least lag columns'
    lag_df = pd.DataFrame(index=df.index)  # output DF
    # Use rolling window - use .T because axis=1 is deprecated
    tmp = df.T.rolling(window=lag, min_periods=lag).sum()
    tmp = pd.DataFrame(tmp.T)
    # Remove the first lag-1 columns
    lag_df = tmp.iloc[:, lag - 1:].copy(deep=True)
    
    return lag_df




def get_month_list(df: pd.DataFrame) -> list[str]:
    """ Get a list of months excluding the aggregate columns """
    month_list = list(df.columns)
    # Exclude the aggregate columns
    month_list = [x for x in month_list if x not in AGG_COLS]
    return month_list


def top_n_summary(pivot: pd.DataFrame, label: str, window: int, plt_file: PdfPages) -> None:
    """
    Create a new summary array with top N category and a N+1 "Other" summing up the rest - for
    last 12 months
    """
    # need deeop copy, so that the drop works
    tmp_df = make_dollar_pretty(pd.DataFrame(pivot[['YearlyAvgAll', 'Last12Mo']]))
    logger.info(f'\n{label} - Averages for Categories (Annualized)\n' + tmp_df)
    avg_series = pivot['Last12Mo'].copy(deep=True)

    # Keep categories that contribute to 80% of expenses
    total_spend = avg_series.sum()
    logger.info(f'{label} - Last 12 Months Annualized Total Spend: ' + to_dollar_str(total_spend))
    # Find the number of categories that cover 80% of expenses: other_idx
    for other_idx in range(len(avg_series)):
        other_spend = avg_series.iloc[other_idx:].sum()  # Accumulate the tail
        if other_spend / total_spend <= OTHER_PCT:
            break
    # Relabel the last entry and assign it the cumulative tail spend
    avg_series.index.values[other_idx] = 'Other'
    avg_series.iloc[other_idx] = other_spend
    avg_series = avg_series.iloc[0:other_idx + 1]
    avg_pct = avg_series / total_spend
    # Make it pretty by formatting the values
    logger.info(f'{label} - TopN Categories Average (annualized) (Last 12 Mo)\n' + repr(
            avg_series.apply(to_dollar_str)))

    # Add-up the smaller categories into 'Other'
    pivot.drop(['YearlyAvgAll', 'Last12Mo'], axis=1, inplace=True)  # only keep the data
    pivot.iloc[other_idx] = pivot.iloc[other_idx:].sum()
    # truncate to only keep Top N - including Other
    pivot = pivot.iloc[0:other_idx + 1]

    # Plot pie chart for top N average overall
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.title(f'{label} - Monthly Expenses for TopN Categories - % - last 12 Mo')
    # plt.pie(avg_pct.values, labels=avg_pct.index.values, autopct='%1.1f%%', normalize=True)
    plt.pie(avg_pct.values, labels=avg_pct.index.values, autopct='%1.1f%%')
    ax.set_facecolor('#E6E6E6')
    plt.savefig(plt_file, format='pdf', dpi=300)
    lineplot_df(pivot, title='Monthly Expenses for TopN Categories', plt_file=plt_file)

    # compare with using the pandas rolling sum
    tmp = pivot.T.rolling(window=window, min_periods=window).sum()
    tmp = pd.DataFrame(tmp.T)
    # remove the first window-1 columns
    run_avg = tmp.iloc[:, window - 1:].copy(deep=True)

    lineplot_df(run_avg, title=f'Trailing 12-month Yearly Expenses for TopN {label} Categories',
                plt_file=plt_file)
    return

def quarterly_expenses(pivot: pd.DataFrame, plt_file: PdfPages) -> pd.DataFrame:
    qtrly = pd.DataFrame(pivot.index)
    qtrly.set_index('Category', drop=True, inplace=True)  # same index as pivot
    col = list(pivot.columns)
    # quarters = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
    quarter = {3: 'Q1', 6: 'Q2', 9: 'Q3', 12: 'Q4'}
    q_list = []
    m_cnt = 0
    for m in col:
        m_nb = int(m.split('-')[1])  # get month as a number
        q_list.append(m)
        m_cnt += 1
        # end of quarter or last month in series
        if m_nb in quarter.keys() or m == col[len(col) - 1]:  # end of quarter - add it up
            yr = m.split('-')[0]  # get the year string
            # Handle the special case of last month of series to find the quarter
            while m_nb not in quarter.keys():
                m_nb += 1
            qtr = yr + '-' + quarter[m_nb]  # Build column name: yyyy-Qx
            # Add values for months available in that quarter - and scale to
            # Yearly
            qtrly[qtr] = pivot[q_list].sum(axis=1) * 12 / m_cnt
            # Reset the counters
            q_list = []
            m_cnt = 0  # sum by row the 12 columns, where the last one is month m
    # ut.print_df(qtrly, msg='Quarterly  Expenses (scaled to yearly)', verbose=True)
    lineplot_df(qtrly, title='Quarterly Expenses for TopN Categories (annualized))',
                plt_file=plt_file)
    return qtrly


def print_df(df, msg=None, verbose=False):
    """ Print troubleshooting info on a DF """
    if msg is None:
        print(msg, len(df.index), "rows")
    if verbose:
        print('Columns:', df.columns)
        print('Index', df.index)
        print(df.head())
        print('/* ... */')
        print(df.tail())
        print('/* --------- */')
    return



def get_program_name():
    """ Get the program name from the path """
    return re.sub("\\.py$", "", os.path.relpath(sys.argv[0]))


def remap_date(val):
    """ Remap from 2018-03-21 00:00:00 to 03/21/2018"""
    in_date, _ = str(val).split(' ')
    yy, mm, dd = in_date.split('-')
    return mm + '/' + dd + '/' + yy


