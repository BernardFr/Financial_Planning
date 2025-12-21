#!/usr/local/bin/python3
""" Common utilities for MonteCarlo  """

import re
import sys
import os

import pandas
import pandas as pd
import datetime as dt

DEBUG = False
Money_format = '$#,##0.00_);[Red]($#,##0.00)'  # Excel cell number format for dollar values
Zoom_level = 110
XL_col_width = 11  # Width of a column for $ in Excel
XL_col_1_width = 18  # Width of 1st column


def compute_age_today(dob: str) -> int:
    """
    Compute my age today based on my DoB
    @param dob: DoB
    @return: age today

    Assume dob has format mm/dd/YYYY
    """
    dob_mo, dob_dy, dob_yr = map(int, dob.split('/'))  # extract month, day, year as int
    dob_dt = dt.datetime(dob_yr, dob_mo, dob_dy)
    today_dt =dt.datetime.now()
    age_today_dt = today_dt - dob_dt
    age_today = int(age_today_dt.days/365)
    return age_today


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
    goals_df_age = goals_df_age.applymap(lambda x: death_age if x == "Death" else x)
    goals_df_age = goals_df_age.applymap(lambda x: end_age if x == "End" else x)
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


def get_program_name():
    return re.sub("\.py$", "", os.path.relpath(sys.argv[0]))


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


def print_df(df, msg=None, verbose=False):
    """ Print troubleshooting info on a DF """
    if msg is None:
        print(f"{msg}: {len(df.index)} rows")
    if verbose:
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


def write_nice_df_2_xl(xlw: pandas.ExcelWriter, df: pandas.DataFrame, sheet: str, index=False, mv_last=None) -> None:
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


