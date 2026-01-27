#!/usr/local/bin/python3

"""
Runs Monte Carlo simulations to determine the likelihood that a given investment strategy will succeed, based on the
projected lifetime cashflows.
Optionally, iteratively adjusts either starting funds, or discretionary spending, in order to meet the target level of
confidence.
Classes:  Asset_Model, Cashflow and Monte_Carlo
"""

# FYI: CONVENTION: all percentage values are stored as float numbers: e.g. 10% is stored as 0.1 and not 10.0 - the
#  helper function pct_str displays these numbers nicely as a string i.e. 0.1 -> "10.00%"
#  Constants representing percentages are written as 10e-2 -> 10% -> 0.1
# FIXME: enforce the above

# FYI: CONVENTION: all financial amounts are stored as positive / negative values for credit/liability -> this means
#  they are always added (no subtraction)

# FIXME: fix naming conventions - should be:
#  run_{one_year, one_iter}[_one_asset][_make_it_work]

# ToDo: use the same convention for file names for Envision as for Morningstar - update/rename play_envision.py


import sys
import os
import collections
import json
import getopt
import datetime as dt
from pprint import pformat
import MonteCarlo_utilities as UT
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import cholesky
from scipy.linalg import norm as matrix_norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

global outf
DEBUG = False
# Configuration
plt.style.use('seaborn-v0_8-deep')
max_iter_to_print_param = 1e4  # Save iteration results if nb_iter_param smaller than this


def config_param():
    global outf

    prog_name = UT.get_program_name()
    config_file = prog_name + ".json"
    param_dict = dict()

    # Check if any of the parameters are overridden by command-line flags
    usage_str = f"usage_str: {prog_name} -p plot_flag -u nb_cpu -n nb_iter -j json_file -s starting_funds " \
                f"-o {{'s','d'}}\n"

    def print_info_and_exit(error_msg=None) -> None:
        if error_msg is not None:
            print(error_msg)  # will print something like "option -a not recognized"
        print(usage_str)
        print(__doc__)
        if error_msg is not None:
            sys.exit(2)
        else:
            sys.exit(0)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hqpn:j:u:o:s:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print_info_and_exit(err)  # will print something like "option -a not recognized"
    for opt, arg in opts:
        if opt == '-h':
            print_info_and_exit()
        elif opt in "-j":  # use selected json file
            config_file = str(arg)  # Override default JSON file
        elif opt in "-p":  # show plots on display
            param_dict['PLOT_FLAG'] = True
        elif opt in "-q":  # Quick run - don't iterate
            param_dict['QUICK_FLAG'] = True
        elif opt in "-u":  # number of processes
            param_dict['NB_CPU'] = int(arg)
        elif opt in "-n":  # set number of runs
            param_dict['nb_iter_param'] = int(arg)
        elif opt in "-s":  # override Starting Funds
            param_dict['START_FUNDS'] = float(arg)
        elif opt in "-u":  # number of processes
            param_dict["NB_CPU"] = int(arg)
        elif opt in "-o":
            if arg not in ['s', 'd']:
                print_info_and_exit("Unknown option")
            elif arg == 's':  # adjusting start_funds
                param_dict['OPT_TYPE'] = "START_FUNDS"
            elif arg == 'd':  # adjust discretionary income
                param_dict['OPT_TYPE'] = "DISCRETIONARY"
            else:
                print_info_and_exit("Unknown option")
        else:
            print_info_and_exit("Unknown option")

    # Read parameters from JSON file
    UT.print_out(outf, f"Using JSON file: {config_file}")
    with open(config_file, 'r') as f:
        in_data = json.load(f)

    # Get other config data
    # Make list of parameters set by commmand line and do not update from JSON file
    cmd_line_param_key = list(param_dict.keys())
    user_data = in_data["user_data"]
    for json_key in user_data.keys():
        if json_key not in cmd_line_param_key:
            param_dict[json_key] = user_data[json_key]
    default_values = in_data["default_values"]
    for json_key in default_values.keys():
        if json_key not in cmd_line_param_key:
            param_dict[json_key] = default_values[json_key]
    UT.print_out(outf, "Parameters:")
    UT.print_out(outf, pformat(param_dict))
    return param_dict


def pct_str(pct: float, decml: int = 2) -> str:
    """ Convert a decimal number that represents a percentage to a printable string
     e.g.  pct_s(0.2) -> "20.00%"
     pct(0.2423436, 4) -> "24.2344%"
     @param pct: float number representing a percent 01. for 10%
     @param decml: number of decimals in the string - default 2
     """
    return f"{round(100 * pct, decml):,f}%"


def generate_ror(nb_sampl: int, mean: float, stddev: float, seed: float = None, new_index: pd.Series = None) -> \
        pd.Series:
    """
    Generates a series of pseudo-random numbers
    @param nb_sampl: number of samples in the series
    @param mean: mean of the pseudo-random series
    @param stddev: stddev of the pseudo-random series
    @param seed: Seed of the random number generator
    @param new_index: Index of the output series
    @return: generated series
    """
    if seed:
        np.random.seed(seed)  # Initialize for consistent results
    result_ser = pd.Series(np.random.normal(loc=mean, scale=stddev, size=nb_sampl))  # create a series w/ the desired
    # stats
    if new_index is not None:
        result_ser.index = new_index
    return result_ser


def compound_rate(age: int, age_0: int, age_1: int, age_2: int, rr: float) -> float:
    """
    Compute the compound rate
    @param age:
    @param age_0:
    @param age_1:
    @param age_2:
    @param rr:
    @return:
    """
    if age_1 <= age <= age_2:
        return pow(1 + 0.01 * rr, age - age_0)  # rr is a number e.g 2.5 for 2.5%
    else:
        return 0.0


def make_cashflow_row(in_row: pd.Series, age_col: [int]) -> pd.Series:
    """
    Note the use of pd.Series() in the return value - this allows each value in the list to be stored in a
    different column - if we returned a list, the result would be a single column containing the whole list
    @param age_col:
    @param in_row:
    @return:
    """
    age_0 = age_col[0]
    amount, start_age, end_age, rate, _ = in_row.values  # last value is "Discretionary"
    return pd.Series([amount * compound_rate(age, age_0, start_age, end_age, rate) for age in age_col], index=age_col)


class AssetModel:
    def __init__(self, param_dict: dict) -> None:
        # Get the Morningstar statistics and the Model data
        cwd = os.getcwd()
        morningstar_path = cwd + '/' + param_dict["MORNINGSTAR_DIR"]
        file_lst = [f for f in os.listdir(morningstar_path) if os.path.isfile(os.path.join(morningstar_path, f))]
        ms_file_lst = [f for f in file_lst if f.startswith(param_dict['MORNINGSTAR_ROOT'])]
        ms_file_lst.sort(reverse=True)
        morningstar_file = f"./{param_dict['MORNINGSTAR_DIR']}/{ms_file_lst[0]}"
        print(f"Morningstar file: {morningstar_file}")
        self.asset_stats = pd.read_excel(morningstar_file, sheet_name='Stats', index_col=0)
        self.asset_corr = pd.read_excel(morningstar_file, sheet_name='Correlation', index_col=0)
        self.alloc_model = pd.read_excel(morningstar_file, sheet_name='Model', index_col=0)
        self.asset_alloc = self.alloc_model[param_dict["STRATEGY"]].copy(deep=True)
        self.blended_return = 0.01 * self.asset_stats['Expected Return'].dot(self.asset_alloc)
        return

    def display(self) -> None:
        UT.print_df(self.asset_stats, msg="Stats", verbose=True)
        UT.print_df(self.asset_corr, msg="Correlations", verbose=True)
        UT.print_df(self.alloc_model, msg="Model", verbose=True)
        return

    def get_data(self):
        return self.asset_stats, self.asset_alloc, self.blended_return

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
    def __init__(self, params: dict, goals_df: pd.DataFrame, mkitwk_params: dict = None) -> None:
        """
        @param params: configuration parameters
        @param goals_df: Goals for income and expenses over the years (from Envision)
        @param mkitwk_params: If present, we run the make_it_work option - i.e. we iterate over starting funds or
        discretionary expenses
        """
        self.management_fee = params['MANAGEMENT_FEE']
        self.seed = params['SEED']
        self.goals_df = goals_df.copy(deep=True)
        # Convert the Discretionary column to boolean
        self.goals_df['Discretionary'] = self.goals_df['Discretionary'].apply(lambda x: True if x == 'Y' else False)

        assert 'START_FUNDS' in params.keys(), "Goals must include initial assets"
        self.start_asset = params['START_FUNDS']

        self.end_age = params['END_AGE']
        self.now = int((dt.datetime.now() - dt.datetime.strptime(params["BF_BDAY"], "%m/%d/%Y")).days / 365)  # my
        # current
        # age
        # in years
        self.age_col = list(range(self.now, self.end_age + 1))
        self.nb_ages = len(self.age_col)
        self.init_flow_ser = pd.Series(0, index=self.age_col)  # yearly total cashflow by age based on goals
        self.flow_ser = pd.Series(0, index=self.age_col)  # yearly total cashflow by age - adjusted over time
        self.cashflow_df = pd.DataFrame()  # DF of cashflow by category by year
        self.discretionary_ratio_by_age = pd.Series(dtype=float)

        # If we provide algorithm arguments, then we run with make_it_work option
        if mkitwk_params is not None:
            self.make_it_work = True
            self.asset_delta_pos = mkitwk_params['asset_delta_pos']
            self.asset_delta_neg = mkitwk_params['asset_delta_neg']
            self.disc_adjust = mkitwk_params['disc_adjust']
            self.min_disc_pct = mkitwk_params['min_disc_pct']
            assert self.asset_delta_pos >= self.asset_delta_neg, \
                f"Need ASSET_DELTA_POS ({self.asset_delta_pos}) >= ASSET_DELTA_NEG ({self.asset_delta_neg})"
            assert self.min_disc_pct < 1.0, f"MIN_DISC_PCT ({self.min_disc_pct}) must be < 1.0"
        else:
            self.make_it_work = False
        return

    def find_min_return(self, params: dict, init_ror: float) -> (bool, float, pd.Series):
        ror_ser = generate_ror(self.nb_ages, init_ror, stddev=0, seed=self.seed, new_index=pd.Series(self.age_col))
        prev_ror = init_ror
        current_ror = init_ror
        ror_increment = params["ALPHA"]
        keep_running = True
        success_flag = False
        iter_cnt = 0
        while keep_running:
            asset_by_age, busted_age, busted_flag = self.run_one_iter_asset_by_age(ror_ser)
            # Now end_asset is $0 by definition. end_asset is entered in goals
            # delta = asset_by_age[monte_carlo_one.end_age] - monte_carlo_one.end_asset
            delta = asset_by_age[self.end_age]
            if iter_cnt > params["MAX_ITER"]:
                print(f"find_min_return: Stopping Iterations iter: {iter_cnt} >= Limit {params['MAX_ITER']}")
                success_flag = False
                keep_running = False
            elif busted_flag or delta < 0.0:
                current_ror += ror_increment
                ror_ser = ror_ser * current_ror / prev_ror
                prev_ror = current_ror
            elif delta > params["MIN_DELTA"]:
                ror_increment = params['ALPHA'] * delta / params['DELTA_NORM']
                current_ror -= ror_increment
                ror_ser = ror_ser * current_ror / prev_ror
                prev_ror = current_ror
            elif 0.0 <= delta <= params['MIN_DELTA']:
                success_flag = True
                keep_running = False
            iter_cnt += 1
        return success_flag, current_ror, asset_by_age

    def update_disc_pct_new(self, disc_pct: float, asset_value: float, reference_asset_val: float) -> float:
        """
        Adjust percentage of discretionary spending if asset_value is too far from asset_by_age[age]
        @param disc_pct:
        @param asset_value:
        @param reference_asset_val:
        @return:
        """
        asset_ratio = asset_value / reference_asset_val

        if asset_ratio >= self.asset_delta_pos:  # Doing well, increase spending
            out_val = disc_pct + self.disc_adjust
        elif asset_ratio >= self.asset_delta_neg:  # doing OK - make no change
            out_val = disc_pct
        else:  # Need to decrease spending - MIN_DISC_PCT is the floor for disc_pct
            out_val = disc_pct - self.disc_adjust
        out_val = max(self.min_disc_pct, out_val)  # out_val does not go below min_disc_pct
        # print(f"asset_ratio: {asset_ratio} - out_val: {out_val}")

        # print(f"age: {age} - asset_value = {asset_value:,.2f} - _by_age: {asset_by_age[age]:,.2f} - delta: "
        #       f"{asset_delta:,.2f} - out_val = {out_val:,.2f}")
        return out_val

    def make_cashflow(self) -> None:
        global DEBUG

        """
        Compute:
        self.cashflow_df: DF of cashflow by category by year
        self.flow_ser: Series of aggregate cashflow by year
        @return: None
        """
        self.cashflow_df = self.goals_df.apply(lambda x: make_cashflow_row(x, self.age_col), axis=1)
        self.flow_ser = self.cashflow_df.sum(axis=0)  # add up all cashflows by age
        self.init_flow_ser = self.flow_ser.copy(deep=True)  # NEED deep copy
        if DEBUG:
            print(f"init_flow_ser: {UT.display_series(self.init_flow_ser, 0)}")
        # Also compute the ratio of total to discretionary expenses
        self.compute_discretionary_ratio_by_age()
        return

    def recompute_cashflow(self, age: int, disc_pct: float) -> None:
        """
        Modify discretionary spending by disc_pct (> or < 100%) starting at AGE+1
        @param age:
        @param disc_pct:
        @return:
        """
        # FYI: cannot use age as index to age_col since it does not start at 0
        new_age_lst = [i for i in self.age_col if i > age]  # only modify future cash flows
        new_age_lst.sort()  # just in case

        # Apply discount to future ages
        for age_idx in new_age_lst:
            self.flow_ser[age_idx] = self.init_flow_ser[age_idx] * disc_pct
        return

    def reset_cashflow(self) -> None:
        """
        To run at each iteration: compute cashflow if it has not been computed already, or re-initialize flow_ser
        which may have been changed by "make_it_work"
        @return: None
        """
        if not hasattr(self, "flow_df"):  # compute the cashflow if it has not already been done
            self.make_cashflow()
        else:  # otherwise re-initialize flow_ser
            self.flow_ser = self.init_flow_ser.copy(deep=True)  # NEED deep copy
        return

    def compute_discretionary_ratio_by_age(self) -> None:
        """
        Compute - by age - the ratio total_cashflow / discretionary_cashflow
        If total cashflow is adjusted by R  then discretionary cashflow needs to be adjusted by coef Q
        so that cashflow_$$ * (1-R) = discretionary_$$ * (1-Q) -- ratio = cashflow_$$ / discretionary_$$
        Q = 1 - ratio * (1-R)
        e.g. R = 0.9, ratio = 2.5  => discretionay_cashflow *= Q = 0.75
        @return:
        """
        self.cashflow_df['Discretionary'] = self.goals_df['Discretionary']
        # Total cashflow by year
        yearly_cashflow_df = pd.DataFrame(columns=['Total Cashflow', 'Discretionary Cashflow', "Discretionary Ratio"])
        yearly_cashflow_df['Total Cashflow'] = self.cashflow_df.sum(axis=0)
        # Total Discretionary cashflow by year
        yearly_cashflow_df['Discretionary Cashflow'] = self.cashflow_df[self.cashflow_df['Discretionary']].sum(axis=0)
        yearly_cashflow_df.drop(index='Discretionary', inplace=True)

        yearly_cashflow_df['Discretionary Ratio'] = \
            yearly_cashflow_df['Total Cashflow'] / yearly_cashflow_df['Discretionary Cashflow']
        self.discretionary_ratio_by_age = yearly_cashflow_df['Discretionary Ratio'].copy(deep=True)
        self.discretionary_ratio_by_age.index.rename("Age", inplace=True)
        # print(f"self.discretionary_ratio_by_age:\n{UT.display_series(self.discretionary_ratio_by_age, 2)}")
        return

    def compute_discretionary_pct(self, disc_ser: pd.Series) -> pd.Series:
        """
        Based on the total cashflow discount percentage required, compute the corresponding discount percentage for
        discretionary cashflow
        See description of: compute_discretionary_ratio_by_age
        Note that the Discretionary cashflow discount could be negative - meaning that we are busted
        @param disc_ser: cashflow discount percentage required
        @return: Discretionary cashflow discount percentage required
        """
        discretionary_ser = pd.Series(index=disc_ser.index)
        for idx in disc_ser.index:
            discretionary_ser[idx] = 1.0 - self.discretionary_ratio_by_age[idx] * (1 - disc_ser[idx])
        return discretionary_ser

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

    def run_one_year(self, *args: object, **kwargs: object) -> (bool, float):
        pass

    def run_one_iter(self, ror_df) -> (float, int, bool):
        """
        Run the simulation for one full iteration

        IMPORTANT: ror_df is either a DF (for MultiAsset) or a Series for (OneAsset)
        """

        self.reset_cashflow()
        asset_value = self.start_asset  # starting $amount
        busted_flag = False
        busted_age = self.end_age + 1
        for age in self.age_col:
            busted_flag, asset_value = self.run_one_year(asset_value, ror_df[age], age)
            if busted_flag:
                busted_age = age
                break  # Stop iterating over age when we bust out

        # return total assets at the end, busted_age (in case we busted), flag for busted
        return asset_value, busted_age, busted_flag

    def run_one_iter_asset_by_age(self, return_ser: pd.Series) -> [pd.Series, int, bool]:
        """ Run the simulation for one full iteration and return the asset value for each age in the iteration """

        self.reset_cashflow()
        # FYI: initialize to 0, so that value is 0 for ages >= busted_age
        asset_by_age = pd.Series(0, index=self.age_col)
        asset_value = self.start_asset  # starting $amount
        busted_age = self.end_age + 1
        busted_flag = False
        for age in self.age_col:
            busted_flag, asset_value = self.run_one_year(asset_value, return_ser[age], age)
            if busted_flag:
                busted_age = age
                break
            else:
                asset_by_age[age] = asset_value
        return asset_by_age, busted_age, busted_flag

    def run_one_iter_make_it_work(self, *args, **kwargs) -> (float, int, bool, [float]):
        pass

    def run(self, *args, **kwargs) -> pd.DataFrame:
        pass

    def print_results(self, details=True) -> None:
        if details:
            print(f"Monte Carlo Variables:\n{pformat(self.__dict__)}")
            UT.print_df(self.goals_df, "Goals", verbose=True)
            UT.print_df(self.flow_ser, "Cashflow", verbose=True)
            # UT.print_df(self.assets_df, "Assets", verbose=True)
        return

    def to_excel(self, xl_wr: pd.ExcelWriter) -> None:
        """ Write all data to Excel file """
        self.goals_df.to_excel(xl_wr, sheet_name="Goals", float_format='%.2f', header=True, index=True)
        if hasattr(self, "asset_alloc"):  # only for multiAsset
            self.asset_alloc.to_excel(xl_wr, sheet_name="Model", float_format='%.2f', header=True, index=True)
        tot_cashflow_df = pd.DataFrame(self.init_flow_ser, columns=["Annual Cashflow"])
        UT.write_nice_df_2_xl(xl_wr, tot_cashflow_df, "Annual Cashflow", index=True)
        UT.write_nice_df_2_xl(xl_wr, self.cashflow_df, "ALL Cashflow", index=True)
        return


class MonteCarloOneAsset(MonteCarloBase):
    def __init__(self, params: dict, goals_df: pd.DataFrame, mkitwk_params: dict = None) -> None:
        super().__init__(params, goals_df, mkitwk_params)
        return

    def run_one_year(self, asset_now: float, ror: float, age: int) -> (bool, float):
        """
        @param ror:
        @param asset_now:
        @param ror:
        @param age:
        @return:
        """
        investment_income = asset_now * ror
        return self.update_one_year(asset_now, investment_income, age)

    def run_one_iter_make_it_work(self, return_ser: pd.Series, asset_by_age: pd.Series) -> \
            (float, int, bool, [float]):
        """
        Run the simulation for one full iteration and return the asset value for each age in the iteration
        FYI: disc_pct_lst has fewer than nb_ages elements if the iteration busts
        """
        self.reset_cashflow()
        discretionary_pct = 1.0
        disc_pct_lst = []  # list of discounts for each year
        prev_discretionary_pct = discretionary_pct
        asset_value = self.start_asset  # starting $amount
        busted_flag = False
        busted_age = self.end_age + 1
        for age in self.age_col:
            disc_pct_lst.append(discretionary_pct)
            busted_flag, asset_value = self.run_one_year(asset_value, return_ser[age], age)
            if busted_flag:
                busted_age = age
                break  # Stop iterating over age when we bust out
            # Not busted - let's see if we need to adjust discretionary spending
            discretionary_pct = self.update_disc_pct_new(discretionary_pct, asset_value, asset_by_age[age])
            # Recompute the cashflow, if we changed  discretionary spending
            # ToDo: log the series of discretionary_pct for all iterations plus busted_flag and busted_age
            if discretionary_pct != prev_discretionary_pct:
                self.recompute_cashflow(age, discretionary_pct)  # FYI: only impacts cashflow starting at age+1
                prev_discretionary_pct = discretionary_pct

        # if busted_flag:  # we busted - fill out the list with 0.0 for ages post bust
        #     disc_pct_lst += [0.0] * (self.nb_ages - len(disc_pct_lst))

        # return total assets at the end, age (in case we busted), flag for busted
        return asset_value, busted_age, busted_flag, disc_pct_lst

    def run(self, ror_val: float, stddev_val: float, nb_iter: int) -> pd.DataFrame:
        """ Run nb_iter simulations """
        end_asset_df = pd.DataFrame(data=None, columns=['Assets', 'Age', 'Busted'], index=range(nb_iter))

        # Generate pseudo-random rates of return
        big_ror_ser = generate_ror(self.nb_ages * nb_iter, ror_val, stddev_val, self.seed)
        start_index = 0
        for itr in range(nb_iter):
            ror_ser = big_ror_ser.iloc[start_index:start_index + self.nb_ages]
            ror_ser.index = self.age_col
            end_asset_df.loc[itr] = self.run_one_iter(ror_ser)
            start_index += self.nb_ages

        return end_asset_df

    def run_make_it_work(self, ror_val: float, stddev_val: float, nb_iter: int, asset_by_age: pd.Series,
                         show_disc: bool = False) -> (pd.DataFrame, pd.Series):
        """ Run nb_iter simulations
        if show_disc: compute the average discount by age. Otherwise, return an empty series
        """
        end_asset_col = ['Assets', 'Age', 'Busted', 'Min_Discretionary_%', 'Avg_Discretionary_%']
        end_asset_df = pd.DataFrame(data=None, columns=end_asset_col, index=range(nb_iter))

        # Generate pseudo-random rates of return
        big_ror_ser = generate_ror(self.nb_ages * nb_iter, ror_val, stddev_val, seed=self.seed)
        if show_disc:
            disc_pct_df = pd.DataFrame(data=None, columns=self.age_col, index=range(nb_iter))
        else:
            disc_pct_ser = pd.Series(data=None, index=self.age_col)

        start_index = 0
        for itr in range(nb_iter):
            ror_ser = big_ror_ser.iloc[start_index:start_index + self.nb_ages]
            ror_ser.index = self.age_col
            asset_value, busted_age, busted_flag, disc_pct_lst = self.run_one_iter_make_it_work(ror_ser, asset_by_age)
            sumry_tuple = asset_value, busted_age, busted_flag, min(disc_pct_lst), np.mean(disc_pct_lst)
            end_asset_df.loc[itr] = pd.Series(sumry_tuple, index=end_asset_col)
            if show_disc:
                # fill the list w/ 0s if we busted early
                disc_pct_lst += [0.0] * (self.nb_ages - len(disc_pct_lst))
                disc_pct_df.loc[iter] = pd.Series(disc_pct_lst, index=self.age_col)
            start_index += self.nb_ages

        if show_disc:
            disc_pct_ser = disc_pct_df.mean(axis=1)  # otherwise, it's just empty
        return end_asset_df, disc_pct_ser


class MonteCarloMultipleAssets(MonteCarloBase):
    def __init__(self, params: dict, goals_df: pd.DataFrame, asset_alloc: pd.Series, mkitwk_params: dict = None) -> None:
        super().__init__(params, goals_df, mkitwk_params)
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

    def run_one_iter_make_it_work(self, ror_df: pd.DataFrame, asset_by_age: pd.Series) -> \
            (float, int, bool, [float]):
        """
        Run the simulation for one full iteration - adjust discretionary spending to make it work
        FYI: disc_pct_lst has fewer than nb_ages elements if the iteration busts
        """
        self.reset_cashflow()
        discretionary_pct = 1.0
        disc_pct_lst = []  # list of discounts for each year
        prev_discretionary_pct = discretionary_pct
        asset_value = self.start_asset  # starting $amount
        busted_flag = False
        busted_age = self.end_age + 1
        for age in self.age_col:
            disc_pct_lst.append(discretionary_pct)
            busted_flag, asset_value = self.run_one_year(asset_value, ror_df[age], age)
            if busted_flag:
                busted_age = age
                break  # Stop iterating over age when we bust out
            # Not busted - let's see if we need to adjust discretionary spending
            discretionary_pct = self.update_disc_pct_new(discretionary_pct, asset_value, asset_by_age[age])
            # Recompute the cashflow, if we changed  discretionary spending
            # ToDo: log the series of discretionary_pct for all iterations plus busted_flag and busted_age
            if discretionary_pct != prev_discretionary_pct:
                self.recompute_cashflow(age, discretionary_pct)  # FYI: only impacts cashflow starting at age+1
                prev_discretionary_pct = discretionary_pct

        # if busted_flag:  # we busted - fill out the list with 0.0 for ages post bust
        #     disc_pct_lst += [0.0] * (self.nb_ages - len(disc_pct_lst))

        # return total assets at the end, age (in case we busted), flag for busted
        return asset_value, busted_age, busted_flag, disc_pct_lst

    def run(self, asset_model: AssetModel, nb_iter: int) -> pd.DataFrame:
        """ Run nb_iter simulations """
        global DEBUG

        end_asset_df = pd.DataFrame(data=None, columns=['Assets', 'Age', 'Busted'], index=range(nb_iter))

        # Generate pseudo-random rates of return
        validation_flag = True if DEBUG else False
        big_ror_df = asset_model.generate_correlated_rvs(self.nb_ages * nb_iter, seed=self.seed, validate_flag=validation_flag)
        if DEBUG:
            big_ror_df_stats = pd.DataFrame(index=big_ror_df.index)
            big_ror_df_stats['Mean'] = big_ror_df.mean(axis=1)
            big_ror_df_stats['StdDev'] = big_ror_df.std(axis=1)
            print(f"big_ror_df  Stats with {nb_iter:,d} iterations")
            print(big_ror_df_stats)

        start_index = 0
        for itr in range(nb_iter):
            ror_df = big_ror_df.iloc[:, start_index:start_index + self.nb_ages]
            ror_df.columns = self.age_col
            end_asset_df.iloc[itr] = self.run_one_iter(ror_df)
            start_index += self.nb_ages
        return end_asset_df

    def run_make_it_work(self, asset_model: AssetModel, nb_iter: int, asset_by_age: pd.Series,
                         show_disc: bool = False) -> (pd.DataFrame, pd.Series):

        """ Run nb_iter simulations
        if show_disc: compute the average discount by age. Otherwise, return an empty series
        """
        end_asset_col = ['Assets', 'Age', 'Busted', 'Min_Discretionary_%', 'Avg_Discretionary_%']
        end_asset_df = pd.DataFrame(data=None, columns=end_asset_col, index=range(nb_iter))

        # Generate pseudo-random rates of return
        big_ror_df = asset_model.generate_correlated_rvs(self.nb_ages * nb_iter, seed=self.seed, validate_flag=False)
        if show_disc:
            disc_pct_df = pd.DataFrame(data=None, columns=self.age_col, index=range(nb_iter))
        else:
            disc_pct_ser = pd.Series(data=None, index=self.age_col)

        start_index = 0
        for itr in range(nb_iter):
            ror_df = big_ror_df.iloc[:, start_index:start_index + self.nb_ages]
            ror_df.columns = self.age_col
            asset_value, busted_age, busted_flag, disc_pct_lst = self.run_one_iter_make_it_work(ror_df, asset_by_age)
            end_asset_df.loc[itr] = asset_value, busted_age, busted_flag, min(disc_pct_lst), np.mean(disc_pct_lst)
            if show_disc:
                # fill the list w/ 0s if we busted early
                # FYI: Adding 0's may cause min() or mean() to fall below self.min_disc_pct
                disc_pct_lst += [0.0] * (self.nb_ages - len(disc_pct_lst))
                disc_pct_df.loc[itr] = pd.Series(disc_pct_lst, index=self.age_col)
            start_index += self.nb_ages

        if show_disc:
            disc_pct_ser = disc_pct_df.mean(axis=0)  # otherwise, it's just empty
        return end_asset_df, disc_pct_ser


def process_goals_file(param_dict: dict) -> pd.DataFrame:
    """
    Reads the file containing goals, cleans it up to handle Death, End strings as well as inflation
    goals_filename: name of goals file
    dob: my DoB
    death_age: my target death age
    end_age: age that I would have when Dinna passes away
    default_inflation_rate: value for inflation when labeled 'Default' in the file
    @return: DF: rows are cashflows column are (1) age at which cashflow starts (2) cashflow ends (3) inflation rate
    """
    age_today = UT.compute_age_today(param_dict['BF_BDAY'])

    in_df = pd.read_excel(param_dict['GOALS_FILE'], sheet_name='Goals', index_col=0)
    in_df_age = in_df[['Start_age', 'End_age']]
    in_df_age = in_df_age.applymap(lambda x: param_dict['DEATH_AGE'] if x == "Death" else x)
    in_df_age = in_df_age.applymap(lambda x: param_dict['END_AGE'] if x == "End" else x)
    in_df_start = in_df_age['Start_age'].apply(lambda x: age_today if x < age_today else x)
    in_df_end = in_df_age['End_age'].apply(lambda x: param_dict['END_AGE'] if x > param_dict['END_AGE'] else x)
    # Handle inflation
    in_df_inflation = in_df['Inflation_pct'].fillna(0.0)  # empty values mean 0% implations
    in_df_inflation = in_df_inflation.apply(lambda x: param_dict['DEFAULT_INFLATION_RATE'] if x == 'Default' else x)
    in_df_inflation.rename('Inflation', inplace=True)
    goals_df = pd.concat([in_df['Amount'], in_df_start, in_df_end, in_df_inflation, in_df['Discretionary']],
                         axis=1)
    # get rid of items that have expired
    goals_df = goals_df[goals_df['End_age'] >= age_today].copy(deep=True)
    return goals_df


def main(argv: [str]) -> None:
    """
    Read the paramaters, financial goals (including inflation projection) - the financial model and get the Stock
    Statistics
    Run the simulation
    Print the results
    @param argv: command line arguments
    @return: None
    """

    global DEBUG, outf
    start_time = dt.datetime.now()
    prog_name = UT.get_program_name()
    out_file = prog_name + "_out.txt"
    outf = open(out_file, "w")
    xl_wr = pd.ExcelWriter(f"{prog_name}_out_{UT.make_date_string()}.xlsx", engine='xlsxwriter')
    plt_file = PdfPages(prog_name + ".pdf")

    param_dict = config_param()
    if DEBUG:
        print(f"Parameters:\n{pformat(param_dict)}")
    # Frequently used parameters
    nb_iter_param = param_dict['NB_ITER']
    mc_threshold_param = param_dict["MC_THRESHOLD"]
    max_iter_param = param_dict["MAX_ITER"]
    age_bin_param = param_dict["AGE_BIN"]

    # Compute Asset Models
    asset_model = AssetModel(param_dict)
    asset_stats, asset_alloc, blended_return = asset_model.get_data()
    print(f"\nBlended Expected Returns = {100 * blended_return:,.2f}%\n")
    print(f"Asset Allocation Strategy: {param_dict['STRATEGY']}\n{asset_alloc}")

    # Figure out the spending goals
    goals_df = process_goals_file(param_dict)
    print(f"\nInitial Funds: ${param_dict['START_FUNDS']:,.0f}")

    def make_pretty_row(row):
        return pd.Series([f"${row[0]:,.0f}", str(row[1]), str(row[2]), f"{100 * row[3]:.2f}%", str(row[4])],
                         index=goals_df.columns)

    print(f"Spending Goals:\n{goals_df.apply(make_pretty_row, axis=1)}\n")

    monte_carlo_one = MonteCarloOneAsset(param_dict, goals_df)

    # Compute asset value for each age, assuming the ROR is equal to the blended return every year
    init_ror = blended_return
    current_ror = init_ror
    stddev_val = 0.0  # No risk
    alpha = 0.01  # 1 %
    # Run nb_iter_param Monte Carlo simulations - adjust RoR until success rate is close to target
    keep_running = True
    iter_cnt = 1
    prev_delta = None
    while keep_running:
        ror_blended_ser = generate_ror(monte_carlo_one.nb_ages * nb_iter_param, current_ror, stddev_val,
                                       seed=monte_carlo_one.seed)
        asset_by_age, _, _ = monte_carlo_one.run_one_iter_asset_by_age(ror_blended_ser)
        delta = asset_by_age[monte_carlo_one.end_age]
        # if delta and prev_delta have opposite signs, reduce alpha (skip the first time)
        if prev_delta and delta * prev_delta < 0.0:
            alpha *= 0.5
        print(f"alpha = {100 * alpha}% - prev_delta = {prev_delta:,.0f}  - delta = {delta:,.0f}")
        prev_delta = delta
        # Adjust the mean value of the rate of returns series
        ror_increment = - alpha if delta > mc_threshold_param else alpha
        current_ror += ror_increment
        if True:
            print(f"Iteration: {iter_cnt} - Current Ror = {100 * current_ror:,.2f}% - stddev_val = "
                  f"{100 * stddev_val:,.2f}% - big_ror_ser mean: {100 * ror_blended_ser.mean():,.4f}%")
        # stop if we have converged - or - exceeded the number of iterations
        if abs(delta) < mc_threshold_param or iter_cnt >= max_iter_param:
            if abs(delta) < mc_threshold_param:  # converged - Success
                print(f"Success - Iteration: {iter_cnt} - Current Ror = {100 * current_ror:,.2f}% - stddev_val = "
                      f"{100 * stddev_val:,.2f}% - big_ror_ser mean / stddev: "
                      f"{100 * ror_blended_ser.mean():,.4f}% / {100 * ror_blended_ser.std():,.4f}%")
            else:  # Exceed the max number of iterations
                print(f"Stopping Iterations iter: {iter_cnt} >= Limit {max_iter_param} -  Current Ror = "
                      f"{100 * current_ror:,.2f}% - stddev_val = {100 * stddev_val:,.2f}% - "
                      f"success_rate = {100.0 * ror_blended_ser:,.2f}% - mean / stddev:"
                      f" {100 * ror_blended_ser.mean():,.4f}% / {100 * ror_blended_ser.std():,.4f}%")
            keep_running = False  # in either case, stop
        iter_cnt += 1
    min_blended_return = current_ror

    print(f"Assets by Age with Blended Return {100 * min_blended_return:,.2f}%:\n{UT.display_series(asset_by_age, 2)}")
    # Plot
    plt.figure(figsize=[11, 8.5])
    plt.plot(list(asset_by_age.index), list(asset_by_age.values))
    plt.xlabel("Age")
    plt.ylabel("Assets")
    plt.title(f"Assets by Age with Blended Return= {100 * min_blended_return:.2f}%")
    # FIXME : next line causes segmentation fault
    # plt.show()
    if plt_file:
        plt.savefig(plt_file, format='pdf', dpi=300, bbox_inches='tight')

    monte_carlo_multi = MonteCarloMultipleAssets(param_dict, goals_df, asset_alloc)
    end_asset_df = monte_carlo_multi.run(asset_model, nb_iter_param)
    print(end_asset_df['Assets'].describe())
    if nb_iter_param <= max_iter_to_print_param:
        end_asset_df.to_excel(xl_wr, sheet_name="End Asset", float_format='%.2f', header=True, index=True)
    monte_carlo_multi.to_excel(xl_wr)  # must run after simulation

    # Display results
    # monte_carlo_multi.print_results(details=True)
    # monte_carlo_multi.to_excel(xl_wr)
    print(f"\nEnd Asset Stats:\n"
          f"{pd.Series(end_asset_df['Assets'].apply(float)).describe(percentiles=[.25, .5, .75])}")
    confidence_level = 100.0 * (nb_iter_param - end_asset_df['Busted'].sum()) / nb_iter_param
    print(f"\n>> Confidence Level = {confidence_level:.2f}%\n")
    busted_ages = dict(collections.Counter(end_asset_df['Age']))
    busted_ages_percent = {age: 100 * cnt / nb_iter_param for age, cnt in busted_ages.items()}
    # Sort by age
    busted_ages_percent = dict(sorted([(k, v) for k, v in busted_ages_percent.items()], key=lambda x: x[0]))
    print(f"Busted Ages in %: {busted_ages_percent}")
    busted_ages_df = pd.DataFrame.from_dict(busted_ages_percent, orient='index', columns=['Age'])
    busted_ages_df['Bin'] = busted_ages_df.index.map(lambda x: age_bin_param * int(x / age_bin_param))
    busted_by_bin = busted_ages_df.groupby(['Bin']).sum()
    busted_by_bin.sort_index(axis=0, inplace=True)
    print(f"Busted Ages in % by {age_bin_param}-year bins:\n{busted_by_bin}")
    assert abs(busted_by_bin['Age'].sum() - 100) < 1e-6, \
        f"Error: Busted Ages should add up to 100% vs {busted_by_bin['Age'].sum()}"

    # Plot
    plt.clf()
    x_tick = range(len(busted_by_bin))
    plt.bar(x_tick, busted_by_bin['Age'].values, 0.5)
    plt.xticks(x_tick, busted_by_bin.index)
    plt.ylabel("% of Busts by Age bin)")
    plt.ylim(0, 100)  # range 0-100%
    plt.title(f"Age Busts by {age_bin_param}-year Bin\nConfidence Level: {confidence_level:.2f}%")
    # plt.show()
    if plt_file:
        plt.savefig(plt_file, format='pdf', dpi=300, bbox_inches='tight')

    # Wrap up
    xl_wr.save()
    if plt_file:
        plt_file.close()
    end_time = dt.datetime.now()
    print(f"Run Time: {str(end_time - start_time)}\n")
    return


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
    sys.exit("All Done")
