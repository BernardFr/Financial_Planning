#!/usr/bin/env python

"""
Implements algorithm to adapt discretionary spending in order to always have money at the end
# ToDo: reconcile START_FUNDS with "initial" Assets in Goals DF ... should be only 1
"""

import matplotlib.pyplot as plt
import sys
import getopt
import os
import collections
import datetime as dt
from pprint import pformat
from pathlib import Path
import tomli
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import cholesky
from scipy.linalg import norm as matrix_norm
import openpyxl
import mc_new_utilities as UT
import plot_color_matrix as plot_c_m

plt.style.use('seaborn-deep')
ProgName = ''  # Place holder for program name
goals_col_2_keep = ['Amount', 'Start_age', 'End_age', 'Inflation', 'Discretionary']
DEBUG = False


def get_params(cmd_args: [str]) -> dict:
    usage_str = f"Usage: {ProgName} -p  -u nb_cpu -n nb_iter -t toml_file -s start_funds\n"
    # Check if any of the parameters are overridden by command-line flags
    cmd_line_param_dict = dict()
    try:
        opts, args = getopt.getopt(cmd_args[1:], "hpu:n:t:s:")
    except getopt.GetoptError:
        print(f'Error: Unrecognized option in: {" ".join(cmd_args)}')
        print(f"{usage_str}\n{__doc__}")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(f"{usage_str}\n{__doc__}")
            sys.exit(2)
        elif opt in "-p":  # show plots on display
            cmd_line_param_dict['PLOT_FLAG'] = True
        elif opt in "-u":  # nb cpus
            cmd_line_param_dict['NB_CPU'] = int(arg)
        elif opt in "-n":
            cmd_line_param_dict['NB_ITER'] = int(arg)
        elif opt in "-t":  # different config file
            cmd_line_param_dict['TOML_FILE'] = arg
        elif opt in "-s":  # specify the starting funds
            cmd_line_param_dict['START_FUNDS'] = float(arg)
        else:
            print(f'Error: Unrecognized option: {opt}')
            print(f"{usage_str}\n{__doc__}")
            sys.exit(2)

    # Read parameters from JSON file
    # print_out(outf, f"Using JSON file: {config_file}")
    config_file = ProgName + ".toml"
    config_file = cmd_line_param_dict.get("TOML_FILE", config_file)  # use config_file from command line, if specified
    print(f"Using TOML file: {config_file}")
    with open(config_file, mode='rb') as fp:
        param_dict = tomli.load(fp)

    # Override, or add, the params from config file with values from command line
    if 'START_FUNDS' in cmd_line_param_dict.keys():
        param_dict['user']['START_FUNDS'] = cmd_line_param_dict['START_FUNDS']
        # delete START_FUNDS from cmd_line_param_dict
        del cmd_line_param_dict['START_FUNDS']
    for ky in cmd_line_param_dict.keys():  # copy the rest of the cmd_line_param_dict to param_dict
        param_dict['config'][ky] = cmd_line_param_dict[ky]

    return param_dict


def max_and_index(in_ser: pd.Series) -> (float, int):
    """
    Returns  the max value and its index of a series
    @param in_ser: input series
    @return: tuple: max value of series and its index value in the series
    """
    lst = list(in_ser)
    mx = max(lst)
    idx = lst.index(mx)
    return [mx, in_ser.index[idx]]


def compute_age_today(dob: str) -> int:
    """
    Compute my age today based on my DoB
    @param dob: DoB - Assume dob has format mm/dd/YYYY
    @return: age today
    """
    dob_mo, dob_dy, dob_yr = map(int, dob.split('/'))  # extract month, day, year as int
    dob_dt = dt.datetime(dob_yr, dob_mo, dob_dy)
    today_dt = dt.datetime.now()
    age_today_dt = today_dt - dob_dt
    age_today = int(age_today_dt.days / 365)
    return age_today


def compound_rate(age: int, age_0: int, rr: float) -> float:
    """
    Compute the compound rate
    @param age:
    @param age_0:
    @param rr:  rr is a number e.g 2.5 for 2.5%
    @return:
    """
    assert age >= age_0, "compound_rate: age must be larger than initial age"
    return pow(1 + 0.01 * rr, age - age_0)  # rr is a number e.g 2.5 for 2.5%


def compound_value(amount: float, age: int, age_0: int, rr: float) -> float:
    """
    Compute the compound value of amount
    @param amount: in dollars at age_0
    @param age: current age
    @param age_0: age at which amount is known
    @param rr:  rr is a number e.g 2.5 for 2.5%
    @return: amount compounded by the interest between age and age_0
    """
    assert age >= age_0, "compound_rate: age must be larger than initial age"
    return amount * pow(1 + 0.01 * rr, age - age_0)  # rr is a number e.g 2.5 for 2.5%


class AssetModel:
    """
    Class that encapsulates assets and models to compute returns of these assets
    """

    def __init__(self, params: dict, user_params: dict) -> None:
        # Get the Morningstar statistics and the Model data
        cwd = os.getcwd()
        morningstar_path = cwd + '/' + params["MORNINGSTAR_DIR"]
        file_lst = [f for f in os.listdir(morningstar_path) if os.path.isfile(os.path.join(morningstar_path, f))]
        ms_file_lst = [f for f in file_lst if f.startswith(params['MORNINGSTAR_ROOT']) and f.endswith('.xlsx')]
        ms_file_lst.sort(reverse=True)
        morningstar_file = f"./{params['MORNINGSTAR_DIR']}/{ms_file_lst[0]}"
        print(f"Morningstar file: {morningstar_file}")
        self.asset_stats = pd.read_excel(morningstar_file, sheet_name='Stats', header=0, index_col=0, engine='openpyxl')
        self.asset_corr = pd.read_excel(morningstar_file, sheet_name='Correlation', header=0, index_col=0,
                                        engine='openpyxl')
        self.alloc_model = pd.read_excel(morningstar_file, sheet_name='Model', header=0, index_col=0, engine='openpyxl')
        self.asset_alloc = self.alloc_model[user_params["STRATEGY"]].copy(deep=True)
        self.blended_return = 0.01 * self.asset_stats['Expected Return'].dot(self.asset_alloc)

        self.goals_df = pd.DataFrame()
        self.goals_df = self.process_goals_file(params, user_params)
        self.end_asset_df = pd.DataFrame()

        return

    def make_pretty_goal_df(self, row: pd.Series) -> pd.Series:
        return pd.Series([f"${row['Amount']:,.0f}", str(row['Start_age']), str(row['End_age']),
                          f"{UT.float_2_pct(row['Inflation_pct'])}", str(row['Discretionary'])],
                         index=self.goals_df.columns)

    def display(self, verbose_flag=True) -> None:
        UT.print_df(self.asset_stats, msg="Morningstar Stats", verbose=verbose_flag)
        UT.print_df(self.asset_corr, msg="Morningstar Correlations", verbose=verbose_flag)
        UT.print_df(self.alloc_model, msg="Allocation Model", verbose=verbose_flag)
        UT.print_df(self.asset_alloc, msg="Selected Asset Allocation", verbose=verbose_flag)
        # UT.print_df(self.goals_df, msg="Goals", verbose=verbose_flag)
        print(f"\n***Spending Goals:\n{self.goals_df.apply(self.make_pretty_goal_df, axis=1)}\n")
        return

    def to_excel(self, xl_file: str, a_w_mode: str = 'w') -> None:
        """ Write all data to Excel file
        - Morningstar expected returns
        - Morningstar cross-correlations
        - Allocation Model
        - Selected Asset Allocation
        - Goals
        """
        with pd.ExcelWriter(xl_file, mode=a_w_mode, engine="openpyxl") as xl_wr:
            # Save data for financial models
            self.asset_stats.to_excel(xl_wr, sheet_name="Morningstar Stats", float_format='%.2f', header=True,
                                      index=True)
            self.asset_corr.to_excel(xl_wr, sheet_name="Morningstar Correlationss", float_format='%.2f', header=True,
                                     index=True)
            self.alloc_model.to_excel(xl_wr, sheet_name="Allocation Model", float_format='%.2f', header=True,
                                      index=True)
            self.asset_alloc.to_excel(xl_wr, sheet_name="Selected Asset Allocatio", float_format='%.2f', header=True,
                                      index=True)
            pretty_goals_df = self.goals_df.apply(self.make_pretty_goal_df, axis=1)
            pretty_goals_df.to_excel(xl_wr, sheet_name="Spending Goals", header=True, index=True)
            return

    def get_asset_alloc_data(self):
        return self.asset_alloc, self.blended_return

    def process_goals_file(self, params: dict, user_params: dict) -> pd.DataFrame:
        """
        Reads the file containing goals, cleans it up to handle Death, End strings as well as inflation
        @param params: dict of misc configuration parameters
        @return: DF: rows are cashflows column are (1) age at which cashflow starts (2) cashflow ends (3) inflation rate
        """
        goals_filename = params['GOALS_FILE']
        df_in = pd.read_excel(goals_filename, sheet_name='Goals', index_col=0, engine='openpyxl')
        age_today = compute_age_today(user_params['BDAY'])  # compute my current age
        death_age = user_params['DEATH_AGE']
        end_age = user_params['END_AGE']
        default_inflation_rate = params['DEFAULT_INFLATION_RATE']
        # Replace strings Death and End with numeric values
        goals_df_age = df_in[['Start_age', 'End_age']]
        goals_df_age = goals_df_age.applymap(lambda x: death_age if x == "Death" else x)
        goals_df_age = goals_df_age.applymap(lambda x: end_age if x == "End" else x)
        goals_df_start = goals_df_age['Start_age'].apply(lambda x: age_today if x < age_today else x)
        goals_df_end = goals_df_age['End_age'].apply(lambda x: end_age if x > end_age else x)
        # Handle inflation
        goals_df_inflation = df_in['Inflation_pct'].fillna(0.0)  # empty values mean 0% implations
        goals_df_inflation = goals_df_inflation.apply(lambda x: default_inflation_rate if x == 'Default' else x)
        goals_df = pd.concat([df_in['Amount'], goals_df_start, goals_df_end, goals_df_inflation,
                              df_in['Discretionary']], axis=1)
        goals_df = goals_df[goals_df['End_age'] >= age_today].copy(deep=True)  # get rid of items that have expired
        # Important so that the rest of the spreadsheet can be used to manipulate the goal data manually
        col_2_drop = [col for col in self.goals_df.columns if col not in goals_col_2_keep]
        self.goals_df.drop(col_2_drop, inplace=True, axis=1)
        return goals_df

    def generate_correlated_rvs(self, nb_sampl: int, seed=42, validate_flag=False) -> pd.DataFrame:
        """
        Generate nb_smpl of Random Variable which are cross-correlated

        @param nb_sampl:
        @param seed: initialization seed
        @param validate_flag: if True, runs extra computations to confirm that the resulting series meet:
        mean, stddev and cross-correlation passed as arguments
        @rtype: DataFrame
        @return: Cross-correlated series (rows)  shape: nb_asset, nb_smpl

        nb_asset = len(df_stat.index)
        Must have df_stat.index == df_corr.index == df_corr.columns

        Steps:
        1/ Generate "nb_asset" pseudo-random series Norm(0.0, 1.0) of length nb_smpl
        2/ Perform the Cholesky decomposition
        3/ Generate cross-correlated series
        4/ Update the series to have the specified mean and stddev

        References:
        - https://en.wikipedia.org/wiki/Covariance_matrix#Definition - Note difference between
        covariance matrix and cross-correlation
        - Cholesky: https://en.wikipedia.org/wiki/Cholesky_decomposition
        """

        np.random.seed(seed)  # Initialize for consistent results
        # Create  N series of cross-correlated samples
        nb_asset = self.asset_corr.shape[0]  # square matrix
        x = []  # list of nb_asset data series
        for _ in range(nb_asset):  # generate nb_asset time series
            x1 = norm.rvs(size=nb_sampl, loc=0.0, scale=1.0)  # create a series w/ the desired stats
            x.append(x1)  # add data series list to list of lists

        # Use Cholesky method to decompose the correlation matrix
        L_matrix = cholesky(self.asset_corr, lower=True)  # get the lower triangular matrix
        # Apply the cross-correlation
        y = np.dot(L_matrix, x)  # y is the result
        # print(f'\nY SHAPE: {y.shape}\n')
        # Get data series with the desired mean and stddev
        y = pd.DataFrame(y, columns=range(y.shape[1]), index=self.asset_stats.index)  # tranform from Array to DF
        y = UT.linear_transform_fastest(y, self.asset_stats["Standard Deviation"], self.asset_stats["Expected Return"])

        if validate_flag:  # Recompute the original stats from the generated series and compare
            print("\nVALIDATING")
            asset_class = list(self.asset_corr.columns)  # List of assets
            # Validate by computing cross-correlation on generated samples
            corr_out = np.corrcoef(y)
            df_corr_out = pd.DataFrame(corr_out, index=asset_class, columns=asset_class, dtype=float)
            # df2.to_excel(xl_wr, sheet_name='validation', float_format='%0.2f', header=True, index=True)
            delta = self.asset_corr - df_corr_out  # this matrix should be all 0
            stats_df = pd.DataFrame(index=self.asset_stats.index)
            stats_df['Expected Return'] = self.asset_stats["Expected Return"]
            stats_df['Computed Mean'] = y.mean(axis=1)
            stats_df['Mean Delta'] = stats_df['Expected Return'] - stats_df['Computed Mean']
            stats_df['Standard Deviation'] = self.asset_stats["Standard Deviation"]
            stats_df['Computed StdDev'] = y.std(axis=1)
            stats_df['Stddev Delta'] = stats_df['Standard Deviation'] - stats_df['Computed StdDev']
            print(f"Original vs computed stats:\n{stats_df}")
            print("\nSUMMARY: (The following 3 norms should be 0.0)")
            print("Norm of Mean Delta")
            print(matrix_norm(self.asset_stats["Expected Return"] - y.mean(axis=1)))
            print("Norm of Stddev Delta")
            print(matrix_norm(self.asset_stats["Standard Deviation"] - y.std(axis=1)))
            print("Norm of Delta Matrix:", matrix_norm(delta, ord="fro"), "\n")

        y *= 0.01  # all numbers before where % -- 18% -> 0.18
        return y  # DataFrame


class MonteCarloBase:
    """
    Base class for Monte Carlo Simulation
    """

    def __init__(self, params: dict, user_params: dict, goals_df: pd.DataFrame, mkitwk_params: dict = None) -> None:
        self.confidence_level = 0
        self.ror_increment_init = params['ROR_INCREMENT']
        self.strategy = user_params['STRATEGY']
        self.max_iter = params['MAX_ITER']
        self.nb_iter = params['NB_ITER']
        self.age_bin = params['AGE_BIN']
        self.min_delta = params['MIN_DELTA']
        self.delta_norm = params['DELTA_NORM']
        self.management_fee = params['MANAGEMENT_FEE']
        self.seed = params['SEED']
        self.goals_df = goals_df.copy(deep=True)
        # Convert the Discretionary column to boolean
        self.goals_df['Discretionary'] = self.goals_df['Discretionary'].apply(lambda x: True if x == 'Y' else False)
        assert 'START_FUNDS' in user_params.keys(), "Goals must include initial assets"
        self.start_asset = user_params['START_FUNDS']
        self.end_age = user_params['END_AGE']
        # my current age in years
        self.now = int((dt.datetime.now() - dt.datetime.strptime(user_params['BDAY'], "%m/%d/%Y")).days / 365)
        self.age_lst = list(range(self.now, self.end_age + 1))  # list of all age values
        self.nb_ages = len(self.age_lst)
        self.discretionary_ratio_by_age = pd.Series(dtype=float)
        self.end_assets_df = pd.DataFrame(dtype=float)

        # Compute Cashflow
        # yearly total cashflow by age based on goals
        self.init_flow_ser = pd.Series(0, index=self.age_lst, name="Total Cashflows")
        # yearly total cashflow by age - adjusted over time
        self.flow_ser = pd.Series(0, index=self.age_lst, name="Total Cashflows")
        self.cashflow_df = pd.DataFrame()
        self.make_cashflow()
        # print(f"\n***Cashflow:\n{UT.print_df_dollar(self.cashflow_df)})\n")
        # If we provide algorithm arguments, then we run with make_it_work option
        self.make_it_work_flag = True if mkitwk_params is not None else False
        if self.make_it_work_flag:
            self.asset_delta_pos = mkitwk_params['asset_delta_pos']
            self.asset_delta_neg = mkitwk_params['asset_delta_neg']
            self.disc_adjust = mkitwk_params['disc_adjust']
            self.min_disc_pct = mkitwk_params['min_disc_pct']
            assert self.asset_delta_pos >= self.asset_delta_neg, \
                f"Need ASSET_DELTA_POS ({self.asset_delta_pos}) >= ASSET_DELTA_NEG ({self.asset_delta_neg})"
            assert self.min_disc_pct < 1.0, f"MIN_DISC_PCT ({self.min_disc_pct}) must be < 1.0"

        return

    def make_cashflow_row(self, in_row: pd.Series) -> pd.Series:
        """
        Note the use of pd.Series() in the return value - this allows each value in the list to be stored in a
        different column - if we returned a list, the result would be a single column containing the whole list
        @param in_row:
        @return:
        """
        amount, start_age, end_age, rate, _ = in_row.values  # last value is "Discretionary"
        # print(f"cashflow_row: amount: {amount:,.0f} - start_age: {start_age} - end_age: {end_age} - rate: {rate:,.2f}")
        if end_age < self.now:
            return pd.Series(0.0, index=self.age_lst)
        if start_age < self.now:
            start_age = self.now
        cashflow_dict = {age: 0.0 for age in self.age_lst}  # initialize to 0.0
        # compute the values for ages when it is active
        values_dict = {age: compound_value(amount, age, self.now, rate) for age in range(start_age, end_age + 1)}
        cashflow_dict.update(values_dict)  # replace 0 with actual values where relevant
        # print(f"cashflow_dict : {cashflow_dict}")
        return pd.Series(cashflow_dict)

    def make_cashflow(self) -> None:
        """
        Compute:
        self.cashflow_df: DF of cashflow by category by year
        self.flow_ser: Series of aggregate cashflow by year
        @return: None
        """
        self.cashflow_df = self.goals_df.apply(self.make_cashflow_row, axis=1)
        if DEBUG:
            print(f"Make_cashflow: {print(UT.print_ser_float(self.flow_ser, nb_decimal=0))}")
        self.flow_ser = self.cashflow_df.sum(axis=0)  # add up all cashflows by age
        self.init_flow_ser = self.flow_ser.copy(deep=True)  # NEED deep copy
        # Also compute the ratio of total to discretionary expenses
        # self.compute_discretionary_ratio_by_age()
        if DEBUG:
            print(f"Make_cashflow: {UT.print_ser_float(self.flow_ser, nb_decimal=0)}")
        return

    def reset_cashflow(self) -> None:
        """
        To run at each iteration: compute cashflow if it has not been computed already, or re-initialize flow_ser
        which may have been changed by "make_it_work"
        @return: None
        """
        if not hasattr(self, "cashflow_df"):  # compute the cashflow if it has not already been done
            self.make_cashflow()
        else:  # otherwise re-initialize flow_ser
            self.flow_ser = self.init_flow_ser.copy(deep=True)  # NEED deep copy
        return

    def compute_confidence_level(self) -> float:
        """" Compute confidence level """
        self.confidence_level = (self.nb_iter - self.end_assets_df['Busted'].sum()) / self.nb_iter
        return self.confidence_level

    def end_summary(self, xl_file: str = None, a_w_mode: str = 'a', prefix_lbl: str = None) -> None:
        """
        Prints statistics on results of simulation
        @return: None
        """
        end_assets_df = self.end_assets_df
        nb_iter = self.nb_iter
        age_bin = self.age_bin
        print(f"\n*** End Asset Stats: Initial Assets: ${self.start_asset:,.0f} - Strategy: {self.strategy}")
        conf_lvl = self.compute_confidence_level()
        print(f"-> Confidence Level = {UT.float_2_pct(conf_lvl)}%\n")
        print(f"{pd.Series(end_assets_df['Assets'].apply(float)).describe(percentiles=[.25, .5, .75])}")
        # Statistics on Busted Ages - Only keep the Busted row
        busted_df = end_assets_df[end_assets_df['Busted'] == True]
        busted_ages = dict(collections.Counter(busted_df['Age']))
        # Sort by age
        busted_ages = dict(sorted([(k, v) for k, v in busted_ages.items()], key=lambda x: x[0]))
        print(f"Busted Ages: {busted_ages}")
        busted_ages_df = pd.DataFrame.from_dict(busted_ages, orient='index', columns=['Count'])
        busted_ages_df['Age_Bin'] = busted_ages_df.index.map(lambda x: age_bin * int(x / age_bin))
        busted_by_bin = busted_ages_df.groupby(['Age_Bin']).sum()
        busted_by_bin.sort_index(axis=0, inplace=True)
        busted_by_bin['Pct'] = busted_by_bin['Count'].map(lambda x: 100 * x / nb_iter)
        print(f"Busted Ages in % by {age_bin}-year bins:\n{busted_by_bin}")
        # Save results
        if xl_file:  # Write all data to Excel file
            # Save portfolio assumptions -if they have not already been saved
            wb = openpyxl.load_workbook(xl_file)
            if "Cashflow" not in wb.sheetnames:
                cashflow_with_total = self.cashflow_df.copy(deep=True)
                pd.concat([cashflow_with_total, self.init_flow_ser], ignore_index=True, axis=0)  # add cashflow totals
                # Need to do it separately because if needs its own writer
                UT.write_dollar_df_2_xl(xl_file, cashflow_with_total, "Cashflow", a_w_mode='a', index=True)
            with pd.ExcelWriter(xl_file, mode=a_w_mode, engine="openpyxl") as xl_wr:
                # Build and save Summary
                summary_df = pd.DataFrame(columns=["A", "B", "C"])
                summary_df.loc[len(summary_df)] = ['Initial Assets', f"${self.start_asset:,.0f}", ""]
                summary_df.loc[len(summary_df)] = ['Strategy', f"{self.strategy}", ""]
                summary_df.loc[len(summary_df)] = ['Confidence Level', f"{UT.float_2_pct(self.confidence_level)}%", ""]
                end_assets_ser = pd.Series(self.end_assets_df['Assets'].apply(float)).describe(
                        percentiles=[.25, .5, .75])
                summary_df.loc[len(summary_df)] = ['End Assets Stats', "", ""]
                # summary_df.loc[len(summary_df)] = [["", k, v] for k, v in end_assets_ser.items()]
                for k, v in end_assets_ser.items():
                    summary_df.loc[len(summary_df)] = ["", k, v]
                # print(f"Busted Ages: {busted_ages}")
                summary_df.loc[len(summary_df)] = [f"Busted Ages in % by {age_bin}-year bins", "", ""]
                # Make the bin names a column
                busted_by_bin['Age_Bin'] = busted_by_bin.index
                # Make Age_Bin the first column
                new_busted_by_bin_columns = ['Age_Bin'] + list(busted_by_bin.columns)[0:-1]
                busted_by_bin = busted_by_bin[new_busted_by_bin_columns]  # change the order of the columns
                summary_df.loc[len(summary_df)] = list(busted_by_bin.columns)  # add the columns to the summary
                busted_by_bin.columns = summary_df.columns  # so that we can append it
                summary_df = pd.concat([summary_df, busted_by_bin], axis=0)
                sht_name = f"{prefix_lbl} - Summary" if prefix_lbl else "Summary"
                summary_df.to_excel(xl_wr, sheet_name=sht_name, header=False, index=False)
        return


class MonteCarloOneAsset(MonteCarloBase):
    def __init__(self, params: dict, user_params: dict, goals_df: pd.DataFrame, mkitwk_params: dict = None) -> None:
        super().__init__(params, user_params, goals_df, mkitwk_params)
        return

    def generate_ror(self, mean: float, stddev: float, nb_iter: int = 1) -> pd.Series:
        """
        Generates a series of pseudo-random Rates of Return for each age, based on mean and stddev arguments
        @param mean: mean of the pseudo-random series
        @param stddev: stddev of the pseudo-random series
        @param nb_iter: number iterations for which to generate samples - defaults to 1
        @return: generated series of size nb_iter * self.nb_ages
        """
        nb_samples = nb_iter * self.nb_ages
        # create a series w/ the desired mean and ste dev
        result_ser = pd.Series(np.random.normal(loc=mean, scale=stddev, size=nb_samples))
        if DEBUG:
            print(f"ror: {dict(result_ser.items())}")
        return result_ser

    def run_one_iter(self, return_ser: pd.Series) -> [pd.Series, int, bool]:
        """ Run the simulation for one full iteration and return the asset value for each age in the iteration """

        self.reset_cashflow()
        if DEBUG:
            print(f"-- cashflow: {UT.print_ser_float(self.flow_ser, nb_decimal=0)}")
        # FYI: initialize to 0, so that value is 0 for ages > busted_age
        asset_by_age = pd.Series(0, index=self.age_lst)
        asset_value = self.start_asset  # starting $amount
        busted_age = self.end_age + 1
        busted_flag = False
        for age in self.age_lst:
            busted_flag, asset_value = self.run_one_year(asset_value, return_ser[age], age)
            asset_by_age[age] = asset_value
            if busted_flag:
                busted_age = age
                break
        return asset_by_age, busted_age, busted_flag

    def run_one_year(self, asset_now: float, ror: float, age: int) -> (bool, float):
        """
        Because we rebalance every year, we can simplify and use only the total asset value, and compute a blended
        Rate of Return which is the weighted average of the RoR for that age by the (static) asset allocation
        @param asset_now:
        @param ror:
        @param age:
        @return: busted_flag, asset_eoy
        """
        # compute the change in total asset values: investment income, cashflows, management fee
        # print(f"Age: {age} - investment_income: {investment_income:,.2f} - cashflow: {self.flow_ser[age]:,.2f}")
        # Change in asset value is: investment income - cashflow - management fee
        delta_asset = asset_now * ror + self.flow_ser[age] - asset_now * self.management_fee
        asset_eoy = asset_now + delta_asset
        # if DEBUG:
        #     print(
        #         f"Age: {age} - asset in: {asset_now:,.2f} - asset out: {asset_eoy:,.2f} - delta: {delta_asset:,.2f}")
        busted_flag = True if asset_eoy <= 0.0 else False  # we're busted if total assets are negative
        return busted_flag, asset_eoy

    def find_min_return(self, init_ror: float) -> (bool, float, pd.Series):
        """
        Iterates to find the minimum rate of return that achieves the objectives spec'd in the config
        @param init_ror:
        @return:
        """
        ror_ser = pd.Series(init_ror, index=self.age_lst)
        prev_ror = init_ror
        current_ror = init_ror
        ror_increment = self.ror_increment_init
        if DEBUG:
            print(f"ror_increment: {ror_increment}")
        keep_running = True
        success_flag = False
        iter_cnt = 0
        while keep_running:
            asset_by_age, busted_age, busted_flag = self.run_one_iter(ror_ser)
            if DEBUG:
                print(f"find_min_return: iter #: {iter_cnt} - current_ror = {100 * current_ror:,.2f}% - busted_flag ="
                      f" {busted_flag}")
                print(f"-- asset_by_age: {UT.print_ser_float(asset_by_age, nb_decimal=0)}")
                # print(f"asset_by_age: {dict(display_asset)}")
            # Now end_asset is $0 by definition. end_asset is entered in goals
            # delta = asset_by_age[monte_carlo_one.end_age] - monte_carlo_one.end_asset
            delta = asset_by_age[self.end_age]
            if iter_cnt > self.max_iter:
                print(f"find_min_return: Stopping Iterations iter: {iter_cnt} >= Limit {self.max_iter}")
                success_flag = False
                keep_running = False
            elif busted_flag or delta < 0.0:
                current_ror += ror_increment
                ror_ser = ror_ser * current_ror / prev_ror
                prev_ror = current_ror
            elif delta > self.min_delta:
                ror_increment = self.ror_increment_init * delta / self.delta_norm
                current_ror -= ror_increment
                ror_ser = ror_ser * current_ror / prev_ror
                prev_ror = current_ror
            elif 0.0 <= delta <= self.min_delta:
                success_flag = True
                keep_running = False
                print(f"find_min_return: SUCCESS after #Iterations: {iter_cnt}")
            iter_cnt += 1
        asset_by_age.name = "Assets by Age"
        return success_flag, current_ror, asset_by_age

    def run(self, nb_iter: int, ror_val: float, stddev_val: float) -> pd.DataFrame:
        """ Run nb_iter simulations
         return: DF with iterations as rows and end_asset, busted_age, busted_flag as columns where end_asset is
         asset value at busted_age if busted, or asset value and end_age if no bust
         busted age is end_age +1 if busted_age is False
         """
        end_assets_df = pd.DataFrame(data=None, columns=['Assets', 'Age', 'Busted'], index=range(nb_iter))

        # Generate pseudo-random rates of return
        big_ror_ser = self.generate_ror(ror_val, stddev_val, nb_iter)
        start_index = 0
        for itr in range(nb_iter):
            ror_ser = big_ror_ser.iloc[start_index:start_index + self.nb_ages].copy(deep=True)
            ror_ser.index = self.age_lst
            asset_by_age, busted_age, busted_flag = self.run_one_iter(ror_ser)
            if busted_flag:
                end_asset = asset_by_age[busted_age]
            else:
                end_asset = asset_by_age[self.end_age]
            end_assets_df.loc[itr] = [end_asset, busted_age, busted_flag]
            start_index += self.nb_ages

        return end_assets_df


class MonteCarloMultipleAssets(MonteCarloBase):
    def __init__(self, params: dict, user_params: dict, goals_df: pd.DataFrame, asset_alloc: pd.Series,
                 mkitwk_params: dict = None) -> None:
        super().__init__(params, user_params, goals_df, mkitwk_params)
        self.asset_alloc = asset_alloc.copy(deep=True)
        return

    def compute_ror_this_year(self, return_ser: pd.Series) -> float:
        """ Compute the blended RoR  across assets weighted by their respective RoR """
        return self.asset_alloc.dot(return_ser)

    def run_one_year(self, asset_now: float, return_ser: pd.Series, age: int) -> (bool, float):
        """
        Because we rebalance every year, we can simplify and use only the total asset value, and compute a blended
        Rate of Return which is the weighted average of the RoR for that age by the (static) asset allocation
        @param asset_now:
        @param return_ser:
        @param age:
        @return:
        """
        investment_income = asset_now * self.compute_ror_this_year(return_ser)
        return self.update_one_year(asset_now, investment_income, age)

    def update_one_year(self, asset_now: float, investment_income: float, age: int) -> (bool, float):
        """
        Because we rebalance every year, we can simplify and use only the total asset value, and compute a blended
        Rate of Return which is the weighted average of the RoR for that age by the (static) asset allocation
        @param asset_now:
        @param investment_income:
        @param age:
        @return:
        """
        global DEBUG

        # compute the change in total asset values: investment income, cashflows, management fee
        # print(f"Age: {age} - investment_income: {investment_income:,.2f} - cashflow: {self.flow_ser[age]:,.2f}")
        delta_asset = investment_income + self.flow_ser[age] - asset_now * self.management_fee
        asset_eoy = asset_now + delta_asset
        # if DEBUG:
        #     print(
        #         f"Age: {age} - asset in: {asset_now:,.2f} - asset out: {asset_eoy:,.2f} - delta: {delta_asset:,.2f}")
        busted_flag = True if asset_eoy <= 0.0 else False  # we're busted if total assets are negative
        return busted_flag, asset_eoy

    def run(self, nb_iter: int, asset_model: AssetModel) -> pd.DataFrame:
        """ Run nb_iter simulations """
        global DEBUG

        end_assets_df = pd.DataFrame(data=None, columns=['Assets', 'Age', 'Busted'], index=range(nb_iter))

        # Generate pseudo-random rates of return
        big_ror_df = asset_model.generate_correlated_rvs(self.nb_ages * nb_iter, seed=self.seed, validate_flag=False)
        if DEBUG:
            big_ror_df_stats = pd.DataFrame(index=big_ror_df.index)
            big_ror_df_stats['Mean'] = big_ror_df.mean(axis=1)
            big_ror_df_stats['StdDev'] = big_ror_df.std(axis=1)
            print(f"big_ror_df  Stats with {nb_iter:,d} iterations")
            print(big_ror_df_stats)

        start_index = 0
        for itr in range(nb_iter):
            ror_df = big_ror_df.iloc[:, start_index:start_index + self.nb_ages]
            ror_df.columns = self.age_lst
            end_assets_df.iloc[itr] = self.run_one_iter(ror_df)  # add assets, busted age and busted flag
            start_index += self.nb_ages
        return end_assets_df

    def run_one_iter(self, ror_df) -> (float, int, bool):
        """
        Run the simulation for one full iteration

        IMPORTANT: ror_df is either a DF (for MultiAsset) or a Series for (OneAsset)
        """

        self.reset_cashflow()
        asset_value = self.start_asset  # starting $amount
        busted_flag = False
        busted_age = self.end_age + 1
        for age in self.age_lst:
            busted_flag, asset_value = self.run_one_year(asset_value, ror_df[age], age)
            if busted_flag:
                busted_age = age
                break  # Stop iterating over age when we bust out

        # return total assets at the end, busted_age (in case we busted), flag for busted
        return asset_value, busted_age, busted_flag


def main(argv: [str]) -> None:
    """
    Read the parameters, financial goals (including inflation projection) - the financial model and get the Stock
    Statistics
    Run the simulation
    Print the results
    @param argv: command line arguments
    @return: None
    """
    start_time = dt.datetime.now()

    # Get configuration parameters
    param_dict = get_params(argv)
    print(f"Parameters:\n{pformat(param_dict)}")
    nb_iter = param_dict['config']['NB_ITER']
    xl_file = ProgName + "_out.xlsx"
    plt_file = ProgName + "_out.pdf"
    plot_flag = param_dict['config']['PLOT_FLAG']

    # Compute Asset Models
    my_assets = AssetModel(param_dict['config'], param_dict['user'])
    asset_alloc, blended_return = my_assets.get_asset_alloc_data()
    print(f"\nInitial Funds: ${param_dict['user']['START_FUNDS']:,.0f}")
    print(f"\nBlended Expected Returns = {100 * blended_return:,.2f}%\n")
    print(f"Asset Allocation Strategy: {param_dict['user']['STRATEGY']}\n{asset_alloc}")
    my_assets.display(verbose_flag=False)
    my_assets.to_excel(xl_file, a_w_mode='w')  # 'w' first time we write to file
    # Set up a one-asset simulation
    monte_carlo_one = MonteCarloOneAsset(param_dict['config'], param_dict['user'], my_assets.goals_df, param_dict[
        'make_it_work'])
    success_flag, min_blended_return, asset_by_age = monte_carlo_one.find_min_return(blended_return)
    print(f'----\nInitial Assets: ${monte_carlo_one.start_asset:,.0f} - Strategy: {monte_carlo_one.strategy}: Blended '
          f'Return = {UT.float_2_pct(blended_return)} - Minimum Blended Return: {UT.float_2_pct(min_blended_return)}')

    # --- Run the MC Simulation
    # Use S&P 500 statistics
    ror_mean = param_dict['config']['SP500_MEAN']
    ror_stddev = param_dict['config']['SP500_STDDEV']
    print(f"\n--- Running {nb_iter} iteration with S&P 500 ror: {UT.float_2_pct(ror_mean)} - stddev: "
          f"{UT.float_2_pct(ror_stddev)} - Initial Assets: ${monte_carlo_one.start_asset:,.0f}")
    monte_carlo_one.end_assets_df = monte_carlo_one.run(nb_iter, ror_mean, ror_stddev)
    monte_carlo_one.end_summary(xl_file, a_w_mode='a', prefix_lbl="One")  # 'a' Not first time we write to file

    # ---- Multi-asset Simulation
    monte_carlo_multi = MonteCarloMultipleAssets(param_dict['config'], param_dict['user'],my_assets.goals_df,
                                                 asset_alloc)  # No make_it_work
    monte_carlo_multi.end_assets_df = monte_carlo_multi.run(nb_iter, my_assets)
    monte_carlo_multi.end_summary(xl_file, a_w_mode='a', prefix_lbl="Multi")  # 'a' Not first time we write to file

    # ---- Compute confidence level for range of starting funds and range of strategies
    if param_dict['options']['RUN_ALL_STRATEGIES'] and "start_funds" in param_dict.keys():
        # Create list of all allocation strategies
        strategy_lst = list(my_assets.alloc_model.columns)
        if "Stock/Bond" in strategy_lst:
            strategy_lst.remove("Stock/Bond")
        print(f"\n--- Confidence Level by Strategy - AND - Starting Funds\n{strategy_lst}")
        min_start_funds = int(param_dict['start_funds']['min_start_funds'])
        max_start_funds = int(param_dict['start_funds']['max_start_funds'])
        step_start_funds = int(param_dict['start_funds']['step_start_funds'])
        out_indx = range(min_start_funds, max_start_funds + 1, step_start_funds)
        out_columns = strategy_lst
        out_df = pd.DataFrame(index=out_indx, columns=out_columns, dtype=float)
        for start_funds in range(min_start_funds, max_start_funds + 1, step_start_funds):
            for strategy in strategy_lst:
                param_dict['user']['START_FUNDS'] = start_funds
                param_dict['user']['STRATEGY'] = strategy
                my_assets = AssetModel(param_dict['config'])
                asset_alloc, blended_return = my_assets.get_asset_alloc_data()
                # print(f"\nStrategy: {param_dict['user']['STRATEGY']} - start funds: "
                #       f"{param_dict['user']['START_FUNDS']} - Blended Expected Returns ={100 * blended_return:,.2f}%\n")
                # print(f"Asset Allocation Strategy: {param_dict['user']['STRATEGY']}\n{asset_alloc}")
                monte_carlo_multi = MonteCarloMultipleAssets(param_dict['config'], param_dict['user'],
                                                             my_assets.goals_df, asset_alloc)
                monte_carlo_multi.end_assets_df = monte_carlo_multi.run(nb_iter, my_assets)
                conf_lvl = monte_carlo_multi.compute_confidence_level()
                out_df.loc[start_funds, strategy] = conf_lvl * 100.0
                print(f"Strategy {strategy} - Starting funds:$M {start_funds / 1e6:,.2f} -  - Blended Expected "
                      f"Returns ={100 * blended_return:,.2f}% - Confidence Level:{out_df.loc[start_funds, strategy]:,.2f}%")
        print(f"\n--- Confidence Level by Start Funds & Strategy Results\n{out_df.head()}")
        best_strategy_df = pd.DataFrame(columns=["Confidence Level", "Strategy"])
        best_strategy_df.index.name = "Starting Funds"
        out_df['Max Conf Lvl'], out_df['Best Strategy'] = zip(*out_df.apply(max_and_index, axis=1))
        best_strategy_df = out_df[['Max Conf Lvl', 'Best Strategy']].copy(deep=True)
        out_df.drop(columns=['Max Conf Lvl', 'Best Strategy'], inplace=True)
        print(f"Best Strategy by Starting Fund:\n{best_strategy_df}")
        with pd.ExcelWriter(xl_file, mode='a', engine="openpyxl") as xl_wr:
            out_df.to_excel(xl_wr, sheet_name="Start Funds & Strategy", header=True, index=True)

        # Plot
        if plot_flag:
            legend_dict = dict()
            legend_dict['title'] = "Configence Level by Strategy and Starting Funds"
            legend_dict['x_label'] = "Strategy"
            legend_dict['y_label'] = "Starting Funds ($M)"
            legend_dict['xticklabels'] = out_columns  # Strategies
            legend_dict['yticklabels'] = [f"{x/1e6:,.2f}" for x in out_indx]  # Starting funds in $M
            plot_c_m.plot_color_matrix(out_df, vmin_val=0, vmax_val=100, legend_dict=legend_dict,
                                       plot_file=plt_file)  # Percentages

    # ---- Compute confidence level for range of initial assets
    elif param_dict['options']['RUN_START_FUNDS'] and param_dict['start_funds']:
        # we iterate over initial funds values to gather level of confidence
        min_start_funds = int(param_dict['start_funds']['min_start_funds'])
        max_start_funds = int(param_dict['start_funds']['max_start_funds'])
        step_start_funds = int(param_dict['start_funds']['step_start_funds'])
        strategy = param_dict['user']['STRATEGY']
        print(f"\nComputing Confidence Level by Starting Funds with:")
        print(f"MIN: {min_start_funds} - MAX: {max_start_funds} - Step:{step_start_funds}")
        confidence_dict = dict()
        for init_funds in range(min_start_funds, max_start_funds + 1, step_start_funds):
            param_dict['user']['START_FUNDS'] = init_funds
            monte_carlo_multi = MonteCarloMultipleAssets(param_dict['config'], param_dict['user'],
                                                         my_assets.goals_df, asset_alloc)
            monte_carlo_multi.end_assets_df = monte_carlo_multi.run(nb_iter, my_assets)
            ky = round(init_funds * 1e-6, 2)  # Make readable key for dict, given that funds are in $Millions
            conf_lvl = monte_carlo_multi.compute_confidence_level()
            confidence_dict[ky] = conf_lvl * 100.0
            print(f"Strategy {strategy} - Starting funds:$M {ky:,.2f} - Confidence Level:"
                  f" {confidence_dict[ky]}%")
        confidence_dict_out = dict(sorted(confidence_dict.items(), key=lambda x: x[0], reverse=False))
        print(f"\n--- Confidence Level by Starting Funds\n{pformat(confidence_dict_out)}")
        with pd.ExcelWriter(xl_file, mode='a', engine="openpyxl") as xl_wr:
            out_df = pd.DataFrame(confidence_dict_out.values(), index=confidence_dict_out.keys(),
                                  columns=["Confidence Level"])
            out_df.to_excel(xl_wr, sheet_name="Starting Funds", header=True, index=True)
        # Plot
        if plot_flag:
            plt.figure(figsize=[11, 8.5], num=f"Confidence Level by Starting Funds", clear=True)
            plt.plot(list(confidence_dict_out.keys()), list(confidence_dict_out.values()))
            plt.ylim(0, 100)  # keep the scale constant for all iterations 0%-100%
            target = param_dict['config']['MC_TARGET'] * 100.0
            plt.hlines(target, min(confidence_dict_out.keys()), max(confidence_dict_out.keys()), colors='red',
                       linestyles='solid')
            plt.xlabel("Starting Funds ($M)")
            plt.ylabel("Confidence Level (%)")
            plt.savefig(plt_file, format="pdf", dpi=300, bbox_inches="tight")
            plt.show()  # FYI - Use backend=WebAgg in matplotlibrc

    # ---- Compute confidence level for range of strategies
    elif param_dict['options']['RUN_ALL_STRATEGIES']:
        # Create list of all allocation strategies
        strategy_lst = list(my_assets.alloc_model.columns)
        if "Stock/Bond" in strategy_lst:
            strategy_lst.remove("Stock/Bond")
        print(f"\n--- Confidence Level by Strategy\n{strategy_lst}")
        start_funds = param_dict['user']['START_FUNDS']
        confidence_dict = dict()
        for strategy in strategy_lst:
            param_dict['user']['STRATEGY'] = strategy
            my_assets = AssetModel(param_dict['config'], param_dict['user'])
            asset_alloc, blended_return = my_assets.get_asset_alloc_data()
            print(f"\nStrategy: {param_dict['user']['STRATEGY']} - Blended Expected Returns ="
                  f" {100 * blended_return:,.2f}%\n")
            print(f"Asset Allocation Strategy: {param_dict['user']['STRATEGY']}\n{asset_alloc}")
            monte_carlo_multi = MonteCarloMultipleAssets(param_dict['config'], param_dict['user'],
                                                         my_assets.goals_df, asset_alloc)
            monte_carlo_multi.end_assets_df = monte_carlo_multi.run(nb_iter, my_assets)
            ky = strategy  # Make readable key for dict, given that funds are in $Millions
            conf_lvl = monte_carlo_multi.compute_confidence_level()
            confidence_dict[ky] = conf_lvl * 100.0
            print(f"Strategy {ky} - Starting funds:$M {start_funds / 1e6:,.2f} - Confidence Level:"
                  f" {confidence_dict[ky]}%")
        confidence_dict_out = dict(sorted(confidence_dict.items(), key=lambda x: x[0], reverse=False))
        confidence_dict_out = {k: f"{v:,.2f}%" for k, v in confidence_dict_out.items()}
        print(f"\n--- Confidence Level by Strategy\n{pformat(confidence_dict_out)}")
        # ToDo: For each starting fund value, find the strategy that results in highest confidence level
        with pd.ExcelWriter(xl_file, mode='a', engine="openpyxl") as xl_wr:
            out_df = pd.DataFrame(confidence_dict_out.values(), index=confidence_dict_out.keys(),
                                  columns=["Confidence Level"])
            out_df.to_excel(xl_wr, sheet_name="Strategy", header=True, index=True)
        # Plot
        if plot_flag:
            plt.figure(figsize=[11, 8.5], num=f"Confidence Level by Strategies", clear=True)
            confidence_dict_out = dict(sorted(confidence_dict.items(), key=lambda x: x[0], reverse=False))
            plt.plot(list(confidence_dict_out.keys()), list(confidence_dict_out.values()))
            plt.ylim(0, 100)  # keep the scale constant for all iterations 0%-100%
            target = param_dict['config']['MC_TARGET'] * 100.0
            plt.hlines(target, min(confidence_dict_out.keys()), max(confidence_dict_out.keys()), colors='red',
                       linestyles='solid')
            plt.xlabel("Strategies")
            plt.ylabel("Confidence Level (%)")
            plt.savefig(plt_file, format="pdf", dpi=300, bbox_inches="tight")
            plt.show()  # FYI - Use backend=WebAgg in matplotlibrc

    end_time = dt.datetime.now()
    print(f"\nEnd: {str(end_time)}")
    print(f"Run Time: {str(end_time - start_time)}\n")
    return


if __name__ == "__main__":
    # execute only if run as a script
    ProgName = Path(sys.argv[0]).stem
    main(sys.argv)
    sys.exit("All Done")
