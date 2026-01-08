#!/usr/local/bin/python3

"""
Compute cashflow by year based on goals file
"""

import sys
import os
import re
import datetime as dt


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm as matrix_norm
from asset_stats_util import get_asset_stats, correlated_rvs, make_ben_model

# Use helper functions from Color_Map
sys.path.insert(0, '../Color_Map/')  # Add path to the directory containing plot_color_matrix
from plot_color_matrix import plot_color_matrix as pcm

XLabelLenDefault = 5  # Length of the X axis labels
QuickFlagDefault = False

class Cashflow:
    """
    Class to manage the Morningstar stats
    """
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.must_have_param = self.config['must_have_param']
        self.XLabelLen = self.config['XLabelLen']
        self.must_have_param = self.config['must_have_param']
        self.XLabelLen = self.config.get('XLabelLen', XLabelLenDefault)
        self.quick_flag = self.config.get('quick_flag', QuickFlagDefault)


# --------------------------


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




def main(argv):
    prog_name = re.sub("\\.py$", "", os.path.relpath(sys.argv[0]))
    plt_file = prog_name + "_out.pdf"  # replace the ".xlsx" extension
    config_file = prog_name + '.toml'
    today, _ = str(dt.datetime.now()).split(' ')

    param_dict = config_param(config_file, argv)
    nb_smpl = param_dict['nb_smpl']
    model_file = param_dict['model_file']
    xl_out_name = param_dict['xl_out_name']
    xl_wr = pd.ExcelWriter(xl_out_name + '_' + today + '.xlsx')

    df_stat, df_corr = get_asset_stats(param_dict['url_stats'], param_dict['url_corr'])
    # FIXME: the morningstar webpage has the wrong column names for the correlation matrix
    df_corr.columns = df_corr.index
    print(f'\nAsset Class Statistics\n{df_stat}')
    print(f'\nAsset Class Correlations\n{df_corr}')
    df_stat.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)
    df_corr.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)
    asset_class = list(df_corr.index)  # List of assets

    y = correlated_rvs(df_stat, df_corr, nb_smpl)
    # Set index to asset_class
    y['AssetClass'] = asset_class
    y.set_index('AssetClass', drop=True, inplace=True)

    print('\nMean')
    print(y.mean(axis=1))
    print('\nMean Delta')
    print(df_stat['Expected Return'] - y.mean(axis=1))
    print('\nStddev')
    print(y.std(axis=1))
    print('\nStddev Delta')
    print(df_stat['Standard Deviation'] - y.std(axis=1))

    # Validate by computing cross-correlation on generated samples
    corr_out = np.corrcoef(y)

    asset_corr_df = pd.DataFrame(corr_out, index=asset_class,columns=asset_class,dtype=float)
    asset_corr_df.to_excel(xl_wr, sheet_name='validation', float_format='%0.2f', header=True, index=True)
    delta = df_corr - asset_corr_df   # this matrix should be all 0
    print("\nNorm of Delta Matrix (should be 0.0):", matrix_norm(delta, ord='fro'),'\n')
    delta.to_excel(xl_wr, sheet_name='delta', float_format='%0.2f', header=True, index=True)

    # Perform portfolio mapping from Ben's Portfolio
    alloc_df = pd.read_excel(model_file, sheet_name='Models', header=0, engine='openpyxl')
    alloc_df.to_excel(xl_wr, sheet_name='Mappings', float_format='%0.2f', header=True, index=True)
    bf_alloc = make_ben_model(alloc_df)
    print('\n\nPortfolio allocation')
    print(bf_alloc)
    print('Sum of weights by column= \n', bf_alloc.sum(axis=0))

    # Verify Stock/Bond ratios
    # Remove empty cells and get rid of index
    bf_alloc['Stock/Bond'] = list(alloc_df['Stock/Bond'].dropna())
    bf_alloc.to_excel(xl_wr, sheet_name='Model', float_format='%0.2f', header=True, index=True)
    stock_bond = bf_alloc.groupby('Stock/Bond',  axis=0).sum()
    stock_bond.to_excel(xl_wr, sheet_name='Stock-Bond Ratios', float_format='%0.2f', header=True, index=True)
    print('\nStock-Bond Ratio:\n', stock_bond)
    bf_alloc.drop(['Stock/Bond'], axis=1, inplace=True)  # no longer needed
    # Save the data file
    xl_wr.close()

    # Plot Asset Correlation Matrix
    print('\nPlotting Asset Correlation Matrix:\n')
    legend_dict = dict()
    legend_dict['title'] = "Asset Class Correlation Matrix"
    legend_dict['x_label'] = "Asset Class"
    legend_dict['y_label'] = "Asset Class"
    legend_dict['xticklabels'] = [make_xlabel(ss, XLabelLen) for ss in asset_corr_df.columns]  # Asset Classes, truncated
    legend_dict['yticklabels'] = [ss.replace(' ', '') for ss in asset_corr_df.columns]  # Asset Classes, 'truncated
    pcm(asset_corr_df, legend_dict=legend_dict, vmin_val=-1, vmax_val=1, plot_file=plt_file)  #
    # Alternative plot
    # plt.imshow(asset_corr_df)
    # plt.colorbar()
    # plt.show()

    return


if __name__ == '__main__':

    main(sys.argv)
    exit(0)
