#!/usr/local/bin/python3
""" Common utilities for MonteCarlo  """

import re
import sys
import os

import pandas
import pandas as pd
import datetime as dt

DEBUG = False


def process_goals_file(goals_filename: str, dob: str, death_age: int, end_age: int, default_inflation_rate: float) ->\
        pd.DataFrame:
    """
    Reads the file containing goals, cleans it up to handle Death, End strings as well as inflation
    @param goals_filename: name of goals file
    @param dob: my DoB
    @param death_age: my target death age
    @param end_age: age that I would have when Dinna passes away
    @param default_inflation_rate: value for inflation when labeled 'Default' in the file
    @return: DF: rows are cashflows column are (1) age at which cashflow starts (2) cashflow ends (3) inflation rate
    """
    df_in = pd.read_excel(goals_filename, sheet_name='Goals', index_col=0)
    age_today = compute_age_today(dob)  # compute my current age
    # Replace strings Death and End with numeric values
    goals_df_age = df_in[['Start_age', 'End_age']]
    goals_df_age = goals_df_age.map(lambda x: death_age if x == "Death" else x)
    goals_df_age = goals_df_age.map(lambda x: end_age if x == "End" else x)
    goals_df_start = goals_df_age['Start_age'].apply(lambda x: age_today if x < age_today else x)
    goals_df_end = goals_df_age['End_age'].apply(lambda x: end_age if x > end_age else x)
    # Handle inflation
    goals_df_inflation = df_in['Inflation_pct'].fillna(0.0)  # empty values mean 0% implations
    goals_df_inflation = goals_df_inflation.apply(lambda x: default_inflation_rate if x == 'Default' else x)
    goals_df = pd.concat([df_in[['Amount','Discretionary']], goals_df_start, goals_df_end, goals_df_inflation], axis=1)
    goals_df = goals_df[goals_df['End_age']>= age_today].copy(deep=True)  # get ride of items that have expired
    return goals_df


def lineitem_cashflow(lineitem: pd.Series, age_list: [int]) -> pd.Series:
    amount = lineitem['Amount']
    inflation_nb = 1 + lineitem['Inflation_pct']
    nb_ages = len(age_list)  # number of ages
    # Create an array of inflation multipliers
    if inflation_nb == 1.0:
        inflation_list = [1.0] * nb_ages
    else:
        inflation_list = []
        mult = 1.0
        for _ in age_list:
            inflation_list += [mult]
            mult *= inflation_nb
    # Create list of amounts which are 0 before start_age and after end_age of that line_item
    amount_list = [0]*(lineitem['Start_age']-age_list[0])
    amount_list += [amount]*(lineitem['End_age']-lineitem['Start_age']+1)
    amount_list += [0]*(age_list[-1] -lineitem['End_age'])
    if DEBUG:
        print(f"lineitem: {lineitem.name} - #Anounts: {len(amount_list)} - #Inflation: {len(inflation_list)}")

    cashflow_lst = [x*y for x,y in zip(amount_list, inflation_list)]
    cashflow_ser = pd.Series(cashflow_lst, index=age_list)
    return cashflow_ser


def make_cashflow(goals_df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # determine the age range
    start_age = min(goals_df['Start_age'])
    end_age = max(goals_df['End_age'])
    age_lst = list(range(start_age, end_age + 1))  # include end_age
    # Create a list of cashflow Series for each line item in goals_df and aggregate into a DF
    cashflow_list = [lineitem_cashflow(goals_df.loc[lineitem], age_lst) for lineitem in goals_df.index]
    cashflow_df = pd.DataFrame(cashflow_list, index=goals_df.index)
    cashflow_df = pd.concat([cashflow_df, goals_df['Discretionary']], axis=1)

    # Compute total cashflows for each age
    cashflow_df.loc['Total'] = cashflow_df.sum(axis=0)
    # Nullify the Discretionary column for the Total row
    cashflow_df.loc['Total', 'Discretionary'] = ''
    cashflow_total_ser = pd.Series(cashflow_df.loc['Total'], index=age_lst, name="Cashflows")

    return cashflow_df, cashflow_total_ser


def adjust_goals(goals_df: pd.DataFrame, discret_mult: float) -> pd.DataFrame:
    """ Multiply the discretionary amounts in goals_df by the discret_mult param
    @param goals_df: goals file with Discretionary column 'Y/N'
    @param discret_mult: multiplier for amounts that are discretionary
    @return: new goals_df
    """
    def conditional_mult(row, mult):
        """Conditionally multiply the first argument based on Y/N of 2nd argument"""
        return row[0]*mult if row[1] == 'Y' else row[0]
    new_amount_ser = goals_df[['Amount','Discretionary']].apply(lambda x: conditional_mult(x, discret_mult), axis=1)
    new_amount_ser.name = 'Amount'
    # replace the Amount column with the newly computed values
    return pd.concat([goals_df.drop('Amount', axis=1), new_amount_ser], axis=1)


def display_series(in_seri: pd.Series, dcml: int = 2) -> str:
    """
    Compact display of a series as a string in the form of ... index: value; ...
    @param in_seri: input Series
    @param dcml: (optional) decimal - default to 2
    @return: string  "index[0]: rounded(in_seri[0], dcml); index[1]: rounded(in_seri[1], dcml); ..."
    """
    if not dcml:
        dcml = 2
    if dcml == 0:
        return "; ".join([f"{idx}: {round(x,dcml):,.0f}" for idx, x in zip(in_seri.index, in_seri)])
    elif dcml <= 2:
        return "; ".join([f"{idx}: {round(x,dcml):,.2f}" for idx, x in zip(in_seri.index, in_seri)])
    elif dcml <= 4:
        return "; ".join([f"{idx}: {round(x,dcml):,.4f}" for idx, x in zip(in_seri.index, in_seri)])
    else:
        return "; ".join([f"{idx}: {round(x,dcml):,.f}" for idx, x in zip(in_seri.index, in_seri)])


def linear_transform_fastest(M_in, slope, intercept):
    M = M_in.copy(deep=True)   # avoid modifying the input matrix
    for i in M.index:
        M.loc[i, :] *= slope[i]
        M.loc[i, :] += intercept[i]
    return M

def non_zero_dict(dd):
    """
    Returns a subset of the input dict dd - by removing the items whose value is 0.0
    @param dd:
    @rtype: dict
    @return: subset where all values are non-zero
    """
    return dict([x for x in dd.items() if x[1] != 0.0])  # x is a (key,value) tuple

def to_dollar(x):
    """Converts a number to a string prepended with '$", and with commas rounded 2 digits"""
    return f'${x:,.2f}'


def today_dollar(tomrw, nb_yr, inflation_rate):
    """ Convert tomorrow's dollars into today's dollars based on inflation rate and number of years"""
    # ToDo: handle nb_yr <= 0
    return tomrw / pow(inflation_rate, nb_yr)


def tick_format(x, pos):
    """
    Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points
    Note: The unused 'pos' argument is necessary for FuncFormatter (see below)
    """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return f'{x:,.0f}'


def tick_format_w_dollar(x, pos):
    """
    Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points
    Note: The unused 'pos' argument is necessary for FuncFormatter (see below)
    """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return f'${x:,.0f}'


def cashflow_item_2_string(lineitem, item_dict, age_dict):
    """
    Prints a cashflow line item nicely - 1 line per phase of each cashflow:
    line_item: <start age> - <end_age> -> <amount>
    @param lineitem: line item name (string)
    @param item_dict: dictionary of cashflow items
    @param age_dict: name - value pairs of significant ages
    @return: string ready for print out
    """
    start_age = age_dict['start_age']
    end_age = age_dict['end_age']
    bf_end = age_dict['bf_end']
    out_str = ''

    if 'start' in list(item_dict.keys()):  # we have a single phase
        dict_list = [item_dict]
    else:  # we have multiple phases
        # Make sure keys are sorted - Must convert to int to sort correctly - ky is int
        dict_list = [item_dict[str(ky)] for ky in sorted(map(int, item_dict.keys()))]

    # iterate over each phase and add the amounts
    # We don't check if the phases overlap or are contiguous
    end_idx = start_age  # initialize for when start == '"
    for dd in dict_list:
        if dd['start'] == '':
            start_idx = 1 + end_idx  # 1 year after the end of previous segment
        else:
            start_idx = start_age if dd['start'] == 'start' else int(dd['start'])
        if dd['end'] == 'bf_end':
            end_idx = bf_end
        else:
            end_idx = end_age if dd['end'] == 'end' else int(dd['end'])
        if start_idx < start_age or end_idx > end_age:
            ut.print_out(outf, f'Data error - ages must be in start/end range: {start_age} - {end_age} ')
            ut.print_out(outf, str(dd))
            ut.print_out(outf, f'start_idx = {start_idx} - end_idx = {end_idx}')
        amount = dd['amount']
        out_str += f'{lineitem}: {start_idx} -> {end_idx}: {to_dollar(amount)}\n'
    return out_str


def cashflow_2_string(in_dict, age_dict):
    """
    Prints a cashflow stream nicely - 1 line per phase of each cashflow:
    line_item: <start age> - <end_age> -> <amount>
    @param in_dict: dictionary of cashflow items
    @param age_dict: name - value pairs of significant ages
    @return: string ready for print out
    """
    out_str = '\n'

    # in_dict contains a series of line items - each line item can have 1 or more phases
    for lineitem in in_dict.keys():
        # Figure out if we have a single entry of multiple
        out_str += cashflow_item_2_string(lineitem, in_dict[lineitem], age_dict)

    return out_str


def plot_with_rails(df, title=None, plt_file=None, rails=False, dollar=False):
    """ Single line plot with rails: Average and +/- std dev"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    x_tick = range(df.shape[0])
    plt.plot(x_tick, df, color='steelblue')
    if rails:  # Plot lines for mean & rails for stddev
        # Build array of containing the mean value for all entries - as a Reference
        df_mean = float(df.mean())  # to ensure the result is not a Series
        df_stddev = float(df.std())
        # Create horizontal lines showing mean, +/- stddev
        ax.axhline(y=df_mean, color='red', ls='solid', alpha=0.5)
        ax.axhline(y=df_mean + df_stddev, color='red', ls='dashed', alpha=0.5)
        ax.axhline(y=df_mean - df_stddev, color='red', ls='dashed', alpha=0.5)

    # Decorate plot w/ labels, title etc
    plt.xticks(x_tick, list(df.index), rotation=0)
    if dollar:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format_w_dollar))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor('#E6E6E6')
    box = ax.get_position()
    ax.set_position([box.x0 + 0.05, box.y0, box.width, box.height])
    plt.grid(axis='y', which='both', color='darkgreen', linestyle='-', linewidth=1, alpha=0.5)
    if title:
        plt.title(title)
    if plt_file:
        # bbox_inches -> makes the legend fit
        plt.savefig(plt_file, format='pdf', dpi=300, bbox_inches='tight')
    if plot_flag:
        plt.show()
    return


