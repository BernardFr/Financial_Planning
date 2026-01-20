#!/usr/local/bin/python3

"""
Compute cashflow by year based on goals file
"""

from typing import Any
import sys
import datetime as dt


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm as matrix_norm
from asset_stats_util import get_asset_stats, correlated_rvs, make_ben_model
from configuration_manager_class import ConfigurationManager
from utilities import my_age, display_series
# Use helper functions from Color_Map
sys.path.insert(0, '../Color_Map/')  # Add path to the directory containing plot_color_matrix
from plot_color_matrix import plot_color_matrix as pcm
from logger import logger

XLabelLenDefault = 5  # Length of the X axis labels
QuickFlagDefault = False
DEBUG_FLAG = True

class Cashflow:
    """
    Class to manage the Morningstar stats
    """
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.goals_file = self.config['Goals_file']
        self.dob = self.config['BF_BDAY']
        self.death_age = int(self.config['Death_age'])
        self.end_age = int(self.config['End_age'])
        self.default_inflation_rate = float(self.config['Default_inflation_rate'])
        self.age_today = my_age(self.dob)


    def process_goals_file(self) -> pd.DataFrame:
        """
        Reads the file containing goals, cleans it up to handle Death, End strings as well as inflation
        @return: DF: rows are cashflows column are (1) age at which cashflow starts (2) cashflow ends (3) inflation rate
        """
        df_in = pd.read_excel(self.goals_file, sheet_name='Goals', index_col=0)
        logger.info(f"Read goals file: {self.goals_file}")
        logger.info(f"My age today: {self.age_today}")
        # Replace strings Death and End with numeric values
        goals_df_age = df_in[['Start_age', 'End_age']]
        goals_df_age = goals_df_age.map(lambda x: self.death_age if x == "Death" else x)
        goals_df_age = goals_df_age.map(lambda x: self.end_age if x == "End" else x)
        goals_df_start = goals_df_age['Start_age'].apply(lambda x: self.age_today if x < self.age_today else x)
        goals_df_end = goals_df_age['End_age'].apply(lambda x: self.end_age if x > self.end_age else x)
        # Handle inflation
        goals_df_inflation = df_in['Inflation_pct'].fillna(0.0)  # empty values mean 0% implations
        goals_df_inflation = goals_df_inflation.apply(lambda x: self.default_inflation_rate if x == 'Default' else x)
        goals_df = pd.concat([df_in[['Amount','Discretionary']], goals_df_start, goals_df_end, goals_df_inflation], axis=1)
        goals_df = goals_df[goals_df['End_age']>= self.age_today].copy(deep=True)  # get ride of items that have expired
        self.goals_df = goals_df.copy(deep=True)
        self.original_goals_df = goals_df.copy(deep=True)   # Original goals - goals_df may be adjusted
        return goals_df


    def _lineitem_cashflow(self, lineitem: pd.Series) -> pd.Series:
        """
        Compute the cashflow for a given line item
        @param lineitem: pd.Series containing the line item data
        @param age_lst: list of ages
        @return: pd.Series containing the cashflow for the line item
        """
        age_lst = self.age_lst
        amount = lineitem['Amount']
        inflation_nb = 1 + lineitem['Inflation_pct']
        nb_ages = len(age_lst)  # number of ages
        # Create an array of inflation multipliers
        if inflation_nb == 1.0:
            inflation_list = [1.0] * nb_ages
        else:
            inflation_list = []
            mult = 1.0
            for _ in age_lst:
                inflation_list += [mult]
                mult *= inflation_nb
        # Create list of amounts which are 0 before start_age and after end_age of that line_item
        amount_list = [0]*(lineitem['Start_age']-age_lst[0])
        amount_list += [amount]*(lineitem['End_age']-lineitem['Start_age']+1)
        amount_list += [0]*(age_lst[-1] -lineitem['End_age'])
        logger.debug(f"lineitem: {lineitem.name} - #Amounts: {len(amount_list)} - #Inflation: {len(inflation_list)}")

        cashflow_lst = [x*y for x,y in zip(amount_list, inflation_list)]
        cashflow_ser = pd.Series(cashflow_lst, index=age_lst)
        return cashflow_ser


    def make_cashflow(self) -> (pd.DataFrame, pd.Series):
        # determine the age range
        start_age = min(self.goals_df['Start_age'])
        end_age = max(self.goals_df['End_age'])
        self.age_lst = list(range(start_age, end_age + 1))  # include end_age
        # Create a list of cashflow Series for each line item in goals_df and aggregate into a DF
        cashflow_list = [self._lineitem_cashflow(self.goals_df.loc[lineitem]) for lineitem in self.goals_df.index]
        cashflow_df = pd.DataFrame(cashflow_list, index=self.goals_df.index)
        cashflow_df = pd.concat([cashflow_df, self.goals_df['Discretionary']], axis=1)

        # Compute total cashflows for each age
        cashflow_df.loc['Total'] = cashflow_df.sum(axis=0)
        # Nullify the Discretionary column for the Total row
        cashflow_df.loc['Total', 'Discretionary'] = ''
        cashflow_total_ser = pd.Series(cashflow_df.loc['Total'], index=self.age_lst, name="Cashflows")

        return cashflow_df, cashflow_total_ser


    def adjust_goals(self, discret_mult: float) -> pd.DataFrame:
        """ Multiply the discretionary amounts in goals_df by the discret_mult param
        @param goals_df: goals file with Discretionary column 'Y/N'
        @param discret_mult: multiplier for amounts that are discretionary
        @return: new goals_df
        """
        def conditional_mult(row, mult):
            """Conditionally multiply the first argument based on Y/N of 2nd argument"""
            return row[0]*mult if row[1] == 'Y' else row[0]
        new_amount_ser = self.goals_df[['Amount','Discretionary']].apply(lambda x: conditional_mult(x, discret_mult), axis=1)
        new_amount_ser.name = 'Amount'
        # replace the Amount column with the newly computed values
        return pd.concat([self. goals_df.drop('Amount', axis=1), new_amount_ser], axis=1)




def linear_transform_fastest(M_in, slope, intercept):
    M = M_in.copy(deep=True)   # avoid modifying the input matrix
    for i in M.index:
        M.loc[i, :] *= slope[i]
        M.loc[i, :] += intercept[i]
    return M



def main(argv):
    config_manager = ConfigurationManager(argv)
    cashflow_class = Cashflow(config_manager)
    goals_df = cashflow_class.process_goals_file()
    print(f"Goals DataFrame:\n{goals_df}")
    cashflow_df, cashflow_total_ser = cashflow_class.make_cashflow()
    print(f"Cashflow DataFrame:\n{cashflow_df}")
    print(f"Cashflow Total Series:\n{cashflow_total_ser}")
    print(display_series(cashflow_total_ser, 2))
    return


if __name__ == '__main__':

    main(sys.argv)
    exit(0)
