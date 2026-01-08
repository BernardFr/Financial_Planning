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

