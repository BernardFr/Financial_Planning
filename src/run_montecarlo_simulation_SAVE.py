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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from logger import logger
from configuration_manager_class import ConfigurationManager
from cashflow_class import Cashflow 
from holdings_class import Holdings
from morningstar_stats_class import MorningstarStats
from montecarlo_simulation_class import MontecarloSimulation, ArrayRandGen
from utilities import error_exit, display_series, dollar_str, pct_str, write_nice_df_2_xl, print_df
import collections

global outf
DEBUG = False
# Configuration
plt.style.use('seaborn-v0_8-deep')
max_iter_to_print_param = 1e4  # Save iteration results if nb_iter_param smaller than this




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



class xx_MonteCarloBase:
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
            print(f"init_flow_ser: {display_series(self.init_flow_ser, 0)}")
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
        # print(f"self.discretionary_ratio_by_age:\n{display_series(self.discretionary_ratio_by_age, 2)}")
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
            print_df(self.goals_df, "Goals", verbose=True)
            print_df(self.flow_ser, "Cashflow", verbose=True)
            # print_df(self.assets_df, "Assets", verbose=True)
        return

    def to_excel(self, xl_wr: pd.ExcelWriter) -> None:
        """ Write all data to Excel file """
        self.goals_df.to_excel(xl_wr, sheet_name="Goals", float_format='%.2f', header=True, index=True)
        if hasattr(self, "asset_alloc"):  # only for multiAsset
            self.asset_alloc.to_excel(xl_wr, sheet_name="Model", float_format='%.2f', header=True, index=True)
        tot_cashflow_df = pd.DataFrame(self.init_flow_ser, columns=["Annual Cashflow"])
        write_nice_df_2_xl(xl_wr, tot_cashflow_df, "Annual Cashflow", index=True)
        write_nice_df_2_xl(xl_wr, self.cashflow_df, "ALL Cashflow", index=True)
        return


class xx_MonteCarloOneAsset(MonteCarloBase):
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


class xx_MonteCarloMultipleAssets(MonteCarloBase):
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
    age_today = compute_age_today(param_dict['BF_BDAY'])

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


def xx_main(argv: [str]) -> None:
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
    prog_name = get_program_name()
    out_file = prog_name + "_out.txt"
    outf = open(out_file, "w")
    xl_wr = pd.ExcelWriter(f"{prog_name}_out_{make_date_string()}.xlsx", engine='xlsxwriter')
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

    print(f"Assets by Age with Blended Return {100 * min_blended_return:,.2f}%:\n{display_series(asset_by_age, 2)}")
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



def main(cmd_line: list[str]) -> None:
    config_manager = ConfigurationManager(cmd_line)
    montecarlo_simulation = MontecarloSimulation(config_manager)

    # Load the cashflow and set the cashflow series
    cashflow_class = Cashflow(config_manager)
    goals_df = cashflow_class.process_goals_file()
    print(f"Goals DataFrame:\n{goals_df}")
    cashflow_df, cashflow_total_ser = cashflow_class.make_cashflow()
    print(f"Cashflow DataFrame:\n{cashflow_df}")
    print(f"Cashflow Total Series:\n{cashflow_total_ser.map(dollar_str)}")
    print(display_series(cashflow_total_ser, 2))
    montecarlo_simulation.set_cashflow_series(cashflow_total_ser)

    # Load holdings and compute the initial holdings and set the initial holdings
    portfolio = Holdings(config_manager)
    holdings_df, cash_amount = portfolio.load_holdings_data()
    print(f"Holdings DataFrame:\n{holdings_df}")
    print(f"Cash amount: ${cash_amount:,.2f}")
    holdings_df = portfolio.assign_cash_to_etf(holdings_df, cash_amount)
    print(f"Holdings DataFrame after reassigning cash to ETF_for_cash:\n{holdings_df}")
    portfolio.set_holdings_df(holdings_df)
    asset_class_df = portfolio.map_etf_to_asset_class()
    print(f"Asset Class DataFrame:\n{asset_class_df}")
    montecarlo_simulation.set_initial_holdings(asset_class_df)

    # Get the Morningstar Asset Stats and create the asset allocation model
    morningstar_stats = MorningstarStats(config_manager)
    df_stat, df_corr = morningstar_stats.get_asset_stats()
    logger.info(f'\nAsset Class Statistics\n{df_stat}')
    logger.info(f'\nAsset Class Correlations\n{df_corr}')
    # Match the stats and holdings (updates montecarlo_simulation.df_stat and df_corr)
    df_stat, df_corr = montecarlo_simulation.match_stats_vs_holdings(df_stat, df_corr)
    # Update df_stat and df_corr in morningstar_stats
    morningstar_stats.set_df_stat_and_corr(df_stat, df_corr)
    print(f"\nmontecarlo_simulation.df_stat shape: {montecarlo_simulation.df_stat.shape} montecarlo_simulation.df_corr shape: {montecarlo_simulation.df_corr.shape}") 

    if montecarlo_simulation.cross_correlated_rvs_flag: # Create the cross-correlated RoR series
        logger.info(f"Using cross-correlated RoR series")
        morningstar_stats.set_nb_smpl(montecarlo_simulation.run_cnt * montecarlo_simulation.nb_ages)
        correlated_rvs = morningstar_stats.generate_correlated_rvs()
        logger.info(f"Correlated returns Series (%):\n{correlated_rvs}")
        # Set the correlated RoR series
        montecarlo_simulation.set_correlated_ror(correlated_rvs)
    else:          # Use Morningstar stats as is  
        logger.info(f"Using Morningstar stats as is")
        # Make list of asset classes stats and correlations
        asset_classes_list = [(mean, stdv) for mean, stdv in zip(df_stat['Expected Return'], df_stat['Standard Deviation'])]
        print(f"Asset Classes List:\n{asset_classes_list}")
        logger.info(f"Using Morningstar stats as is")
        # Create and setthe list of generators
        ror_gen_list = [ArrayRandGen(config_manager, mean, stdv) for mean, stdv in asset_classes_list]
        montecarlo_simulation.set_ror_generator_list(ror_gen_list)

    montecarlo_simulation.run()
    return None

if __name__ == "__main__":
    main(sys.argv)
    sys.exit("---\nDone!")