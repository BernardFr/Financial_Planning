#!/usr/bin/env python

"""
http://blog.miguelgrinberg.com/post/designing-a-restful-api-using-flask-restful
http://flask.pocoo.org/docs/quickstart/
http://flask-restful.readthedocs.org/en/latest/

Computes the amount of $$ that can be withdrawn during retirement - using Monte-Carlo simulations
All inputs are within the program
A scenario includes a number of phases (e.g. working, semi-retirement, retirement)
Each phase has a portfolio allocation of asset types (stocks, bonds, etc), and a contribution (negative for withdrawal).
If the "ToCompute" flag is set in a phase, then the contribution (in this case withdrawal) will be computed by the MC simulations.
More that one phase can set ToCompute, but all these phases will share the same computed contribution
The MC simulations use the Historical mean,stddev for each asset class - and generate random rate of returns using a
normal distribution based on the mean/stddev of the given asset class
The simulations are run "NbRun" and the contribution amount is returned based on the confidence factor

IMPORTANT
1/ Make sure each process uses a known seed for random number generator
See: https://bbabenko.github.io/multiprocessing-and-seeded-RNGs/
2/ See use of partial
"""

# ToDo: Add Readme - including syntax for JSON file
# ToDo: start with a small number of simulation runs and increase when < 1%
# ToDo: Add Loeper strategy
# ToDo: Add more asset classes per Ben

import getopt
import logging
import sys
import os
import numpy as np
import pandas as pd
import collections
import datetime as dt
import MonteCarlo_utilities as ut
import json
from pprint import pformat
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, set_start_method
from functools import partial
import asset_stats_util as asst
# import itertools
# from multiprocessing import cpu_count
# from asset_stats_util import *
# from pprint import pprint

# Configuration
plt.style.use('seaborn-deep')
outf = None  # placeholder for global variable
# List of configuration parameters
must_have_param = ['nb_cpu', 'rebalance_flag', 'plot_flag', 'quick_flag', 'run_cnt', 'seed', 'portfolio_file',
                   'epsilon', 'model_file', 'url_corr', 'url_stats', 'mc_model_file']

DEBUG = False
global prog_name, plot_flag, rebalance_flag

"""
From: https://fc.standardandpoors.com/sites/client/generic/axa/axa4/Article.vm?topic=5991&siteContent=8088
Total Returns From 1926-2017*
     Stocks    Bonds    T-Bills
Annualized Return    10.2%    5.6%    3.5%
Best Year    60.0 (1935)    42.1 (1982)    14.0 (1981)
Worst Year    -41.1 (1931)    -12.2 (2009)    0.0 (1940)
Standard Deviation    18.7    7.7    0.9
"""


# AssetClasses = collections.namedtuple('AssetClasses', "Stocks Bonds TBills Cash")  # Immutable
# historical_return_2017 = AssetClasses(Stocks=(10.20, 18.70), Bonds=(5.60, 7.70), TBills=(3.50, 0.90), Cash=(0.0, 0.0))
# default_pfolio_alloc = AssetClasses(Stocks=0.75, Bonds=.15, TBills=.05, Cash=.05)

def non_zero_dict(dd):
    """
    Returns a subset of the input dict dd - by removing the items whose value is 0.0
    @param dd:
    @rtype: dict
    @return: subset where all values are non-zero
    """
    return dict([x for x in dd.items() if x[1] != 0.0])  # x is a (key,value) tuple


def config_param(config_file, argv):
    with open(config_file, 'r') as f:
        param_dict = json.load(f)

    # Make sure that all parameters have been configured - i.e. have a default value
    if sorted(must_have_param) != sorted(param_dict.keys()):
        print("Error: missing configuration parameters")
        for p in param_dict.keys():
            if p not in must_have_param:
                print("Warning: Unexpected parameter: ", p)
        for p in must_have_param:
            missing_cnt = 0
            if p not in param_dict.keys():
                print("Must have parameter missing: ", p)
                missing_cnt += 1
            if missing_cnt > 0:
                sys.exit(-1)

    # Check if any of the parameters are overridden by command-line flags
    try:
        opts, args = getopt.getopt(argv[1:], "hqpc:u:j:r:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print(__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt in "-p":  # show plots on display
            param_dict['plot_flag'] = True
        elif opt in "-q":  # Quick run - don't iterate
            param_dict['quick_flag'] = True
        elif opt in "-r":  # Select rebalancing
            if str(arg).lower() == 'yes':
                param_dict['rebalance_flag'] = True
            elif str(arg).lower() == 'no':
                param_dict['rebalance_flag'] = False
        elif opt in "-u":  # number of processes
            param_dict['nb_cpu'] = int(arg)
        elif opt in "-c":  # set number of runs
            param_dict['run_cnt'] = int(arg)
        elif opt in "-j":  # use selected json file
            param_dict['json_file'] = str(arg)
        else:
            print(__doc__)
            assert False, "unhandled option"
    return param_dict


class ArrayRandGen:
    def __init__(self, ages, stats):
        self.nb = ages[1] - ages[0] + 1
        self.mean = stats[0]
        self.stdv = stats[1]

    def __iter__(self):
        return self

    def __next__(self):
        # returns a list of nb random numbers based on mean and stddev
        # numbers are: 1 + random_number/100  (interest rates)
        # yield [1 + 0.01 * random.gauss(self.mean, self.stdv) for x in range(self.nb)]
        yield [1 + 0.01 * np.random.normal(self.mean, self.stdv) for _ in range(self.nb)]


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


def get_stats_and_model(mc_model_file, url_stats, url_corr, model_file):
    """
    Get asset statistics and asset allocation model from file - or recreate it if file is old
    :param mc_model_file: file containing this data
    :return: df_stat, df_corr, alloc_scenarios
    - Assets rates of return
    - Assets cross correlations
    - Asset allocation models
    """

    # Get morningstar stats and create portfolio models
    if os.path.isfile(mc_model_file):  # if the file already exists, just read from it
        df_stat = pd.read_excel(mc_model_file, sheet_name='Stats', header=0, index_col=0)
        df_corr = pd.read_excel(mc_model_file, sheet_name='Correlation', header=0, index_col=0)
        alloc_scenarios = pd.read_excel(mc_model_file, sheet_name='Allocation Scenarios', header=0)
    else:  # Get morningstar stats and create portfolio models
        mc_xl_wr = pd.ExcelWriter(mc_model_file)
        # Get portfolio mapping from Ben's Portfolio
        alloc_scenarios = asst.make_ben_model(model_file)
        alloc_scenarios.to_excel(mc_xl_wr, sheet_name='Allocation Scenarios', float_format='%0.2f', header=True,
                                 index=True)

        # Generate array of correlated run_cnt rates of return for each asset
        df_stat, df_corr = asst.get_asset_stats(url_stats, url_corr)

        df_stat.to_excel(mc_xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)
        df_corr.to_excel(mc_xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)
        mc_xl_wr.save()  # close the file

    return df_stat, df_corr, alloc_scenarios


def mk_ror_df(big_ror_df, nb_iter, ages, offset):
    """
    Extract o subset of big_ror_df - take the first end-start+1 columns
    :param big_ror_df - contains a DF of rates of return for a list of assets (index) and a number of iterations
    :param nb_iter: the iteration count - is used to offset which columns we start from
    :param ages: 2-tuple with start and end age
    :return: pd.DataFrame with end-start columns
    """
    start = ages[0]
    end = ages[1]
    nb_ages = end - start + 1  # we keep both start and end age
    start_col = offset + nb_iter * nb_ages  # Offset for nb_iter_th iteration
    end_col = start_col + nb_ages  # Last column we need + 1
    df = big_ror_df.iloc[:, start_col:end_col].copy(deep=True)  # grab end-start+1 columns
    df.columns = list(range(start, end + 1))
    return df


def rebalance(asset, alloc_pct):
    """
    Re-balannce assets based on desired asset allocation "alloc_pct"
    :param asset: Series of assets
    :param alloc_pct: Desired allocation
    :return: new values of asset
    """
    tot_asset = asset.sum()
    return alloc_pct * tot_asset


def lineitem_cashflow(in_dict, inflation_dict, age_dict):
    """
    Creates a Series - indexed by age_list - of cashflow amounts
    :param in_dict: is either a single cashflow dict (keys = start/end/amounts or
        a dictionary of cashflow dicts indexed by an int - ints must be sorted by start age
    :param age_dict: dict of ages
    :return: cashflow series indexed by age

    ASSUMPTIONS:
    - age_list is sorted and contiguous
    - valid entries for start/end are int (with age_list) | 'start'/'end' | "
    """
    start_age = age_dict['start_age']
    end_age = age_dict['end_age']
    bf_end = age_dict['bf_end']
    age_lst = list(range(start_age, end_age + 1))
    # Initialize cashflowdict with age_list as keys and 0.0 for values
    cf_dict = dict.fromkeys(age_lst, 0.0)

    # Figure out if we have a single entry of multiple
    # Todo: find a nicer way to test
    if 'start' in list(in_dict.keys()):  # we have a single phase
        dict_list = [in_dict]
    else:  # we have multiple phases
        # Make sure keys are sorted - Must convert to int to sort correctly - ky is int
        dict_list = [in_dict[str(ky)] for ky in sorted(map(int, in_dict.keys()))]

    # Compute inflation dict if line item has custom inflation spec
    if 'inflation' in list(in_dict.keys()):  # Use it instead of normal inflation
        infl = float(in_dict['inflation'])
        infl_dict = dict([(age, pow(infl, age - start_age)) for age in range(start_age, end_age + 1)])
    else:
        infl_dict = inflation_dict  # Use default

    # iterate over each phase and add the amounts
    # We don't check if the phases overlap or are contiguous
    end_idx = start_age  # initialize for when start == '"
    for dd in dict_list:
        # ToDo: add data validation
        if dd['start'] == '':
            start_idx = 1 + end_idx  # 1 year after the end of previous segment
        else:
            start_idx = start_age if dd['start'] == 'start' else int(dd['start'])
        if dd['end'] == 'bf_end':
            end_idx = bf_end
        else:
            end_idx = end_age if dd['end'] == 'end' else int(dd['end'])
        if start_idx < start_age or end_idx > end_age:
            ut.print_out(outf, 'Data error - ages must be in start/end range: {} - {} '.format(start_age, end_age))
            ut.print_out(outf, str(dd))
            ut.print_out(outf, 'start_idx = {} - end_idx = {}'.format(start_idx, end_idx))
        amount = dd['amount']
        # Add the amounts for the ages start_idx until end_idx included
        for age in range(start_idx, 1 + end_idx):
            # print('Age: {} - amount: {:.2f}'.format(i,amount))
            cf_dict[age] += amount * infl_dict[age]

    if DEBUG:
        ut.print_out(outf, pformat(in_dict))
        ut.print_out(outf, pformat(cf_dict))

    return pd.DataFrame.from_dict(cf_dict, orient='index')  # returns a DF with the right indices


def make_cashflow(cashflow_data, infl_dict, age_dict, xl_file=None):
    """
    Creates a 1-D dataframe of cashflows based on a dict of cashflow items, inflation dict, and age dict
    @param cashflow_data: dict describing cashflows
    @param infl_dict:
    @param age_dict:
    @param xl_file:
    @return:
    """
    age_lst = list(range(age_dict['start_age'], age_dict['end_age'] + 1))
    # Initialize the cashflows to 0.0
    cashflow_df = pd.DataFrame([0.0] * len(age_lst), index=age_lst, dtype=float)
    all_df = pd.DataFrame(columns=list(cashflow_data.keys()), index=age_lst, dtype=float)
    all_df['Inflation'] = pd.DataFrame.from_dict(infl_dict, orient='index')
    for lineitem in cashflow_data.keys():
        all_df[lineitem] = lineitem_cashflow(cashflow_data[lineitem], infl_dict, age_dict)
        cashflow_df += lineitem_cashflow(cashflow_data[lineitem], infl_dict, age_dict)
    all_df = all_df.T
    if xl_file:
        with pd.ExcelWriter(xl_file) as xl_writer:
            all_df.to_excel(xl_writer, sheet_name="Cashflow", float_format='%.2f', header=True)

    return cashflow_df


def update_asset_val_w_rebalance(asset_val, ror_df, cashflow, asset_alloc, mgt_fee):
    # Compute returns of existing assets and allocate year's withdrawal weighted to available assets
    asset_val += asset_val * ror_df  # Add returns
    # Check if we have more money than the planned withdrawals

    # Apply distributions
    asset_val = rebalance(asset_val, asset_alloc)  # Rebalance
    asset_val += asset_alloc * cashflow  # distributions are done in same % as asset values
    asset_val *= (1 - mgt_fee)  # Subtract mgt fee
    return asset_val


def update_asset_val_no_rebalance(asset_val, ror_df, cashflow, asset_alloc, mgt_fee):
    # Compute returns of existing assets and allocate year's withdrawal weighted to available assets
    asset_val += asset_val * ror_df  # Add returns
    # Check if we have more money than the planned withdrawals

    # Apply distributions
    wdrwl_alloc = asset_val / asset_val.sum()
    asset_val += wdrwl_alloc * cashflow
    asset_val *= (1 - mgt_fee)  # Subtract mgt fee
    return asset_val


def update_asset_val_straight(asset_val, ror_df, cashflow, asset_alloc, mgt_fee):
    # Compute returns of existing assets and allocate year's withdrawal weighted to available assets
    asset_val += asset_val * ror_df  # Add returns
    # Check if we have more money than the planned withdrawals

    # Apply distributions
    asset_val += asset_alloc * cashflow  # distributions are done in same % as asset values
    asset_val *= (1 - mgt_fee)  # Subtract mgt fee
    return asset_val


def run_mc_multi(init_asset, big_ror_df, cashflow_df, mgt_fee, rebal_flag, cnt, offset, test_param=None):
    """
    Run Montecarlo simulation
    @param init_asset: (N,) SERIES of assets allocated by asset class
    @param big_ror_df: Large array of rates of returns (ror) for each kind of asset - pseudo-random based on each asset's stats
    @param cashflow_df: Series of yearly withdrawals [or income] for each consecutive age
    @param cnt: Nb of iterations to run the simulation
    @param offset: offset in big_ror_df from which to extract ror for each iteration - used in parallel processing
    @return: [N, cnt] DF with 2-uple (DF of ending amount by asset class for each simulation run, end age]
    """
    # print('run_mc_multi: pid: {}, cnt= {}, offset={}'.format(os.getpid(), cnt, offset))
    age_col = list(map(int, cashflow_df.index))
    ages = (int(age_col[0]), int(age_col[-1]))  # start and end age
    nb_ages = ages[1] - ages[0] + 1  # We keep both start and end age
    idx_name = init_asset.index
    # Create dataframe for results - rows are asset classes, columns are iterations
    asset_df = pd.DataFrame(index=idx_name, dtype=float, columns=range(cnt))
    # Create final_age series [N] - initialized to 0
    final_age_ser = pd.Series([0] * cnt, index=range(cnt), dtype=int)
    # Compute asset allocation - and use it for rebalancing
    asset_alloc = init_asset / init_asset.sum()  # Fraction
    # assert asset_alloc.sum() == 1.0, f'Asset allocation (= {asset_alloc.sum():,.6f}) must add up to 1.0'
    assert abs(asset_alloc.sum() - 1.0) < 1e-6, f'Asset allocation (= {asset_alloc.sum():,.6f}) must add up to 1.0'
    # Convert cashflow_df from DF with 1 column of label 0 to Series
    cashflow_ser = pd.Series(cashflow_df[0])

    # set straight_distribution_flag if we are in test mode
    straight_distribution_flag = False
    if test_param:
        if test_param['straight_distribution']:
            straight_distribution_flag = True

    # Select the function for asset update
    if not straight_distribution_flag:
        if rebal_flag:
            update_asset_val = update_asset_val_w_rebalance
        else:
            update_asset_val = update_asset_val_no_rebalance
    else:  # straight_distribution - NO rebalance
        update_asset_val = update_asset_val_straight

    # Run nb_iter simulations
    for nb_iter in range(cnt):
        asset_val = init_asset.copy(deep=True)  # initialize
        # Generate a new array of rate of returns for all ages and each asset class
        ror_df = mk_ror_df(big_ror_df, nb_iter, ages, offset)

        # Update asset values at each age
        busted_age = 200
        for age in age_col:
            asset_val = update_asset_val(asset_val, ror_df[age], cashflow_ser[age], asset_alloc, mgt_fee)
            if asset_val.sum() < 0.0:  # Busted
                # print('Busted! @iter: {:d} - age {:d}'.format(nb_iter, int(age)))
                busted_age = age
                break  # Stop iterating over age
        # store final result of this simulation run
        asset_df[nb_iter] = asset_val
        final_age_ser[nb_iter] = busted_age

    return asset_df, final_age_ser


def main(argv):
    global prog_name, outf, plot_flag, rebalance_flag

    start_time = dt.datetime.now()
    prog_name = argv[0].lstrip('./').rstrip('.py')
    out_file = prog_name + '_out.txt'
    cfg_file = prog_name + '_cfg.json'
    outf = open(out_file, 'w')
    xl_file = prog_name + '_out.xls'
    plot_file = prog_name + '.pdf'
    plt_file = PdfPages(plot_file)
    today, _ = str(dt.datetime.now()).split(' ')

    # logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(prog_name + '.log')
    handler.setLevel(logging.DEBUG)
    # create a logging format
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.debug('Start: {}\n'.format(str(start_time)))
    logger.debug('Debug')

    # Load configuration parameters
    param_dict = config_param(cfg_file, argv)
    ut.print_out(outf, 'Configuration Parameters:')
    ut.print_out(outf, pformat(param_dict))

    run_cnt = param_dict['run_cnt']
    nb_cpu = param_dict['nb_cpu']
    plot_flag = param_dict['plot_flag']
    quick_flag = param_dict['quick_flag']
    rebalance_flag = param_dict['rebalance_flag']
    pfolio_file = param_dict['portfolio_file']
    mc_model_file = param_dict['mc_model_file'] + '_' + today + '.xlsx'  # file to store Morningstar stats and model

    # Read parameters from JSON file
    ut.print_out(outf, 'Using JSON file: {}'.format(pfolio_file))
    with open(pfolio_file, 'r') as f:
        in_data = json.load(f)

    user_data = in_data['user_data']
    start_funds = user_data['start_funds']
    age_dict = user_data['ages']
    inflation_rate = user_data['inflation_rate']
    target_end_funds = user_data['target_end_funds']
    mgt_fee = user_data['mgt_fee']
    # stock_val_list = in_data['stock_val_list']
    pfolio_alloc = in_data['pfolio_alloc']
    cashflow_data = in_data['cashflow']

    # get asset statistics and asset allocation model from file
    # or recreate it if file is old
    df_stat, df_corr, alloc_scenarios = get_stats_and_model(mc_model_file, param_dict['url_stats'],
                                                            param_dict['url_corr'], param_dict['model_file'])
    ut.print_out(outf, '\nAsset Returns:\n' + str(df_stat))
    ut.print_out(outf, '\nAsset Correlations:\n' + str(df_corr))
    ut.print_out(outf, '\n\nPortfolio Allocation Scenarios' + str(alloc_scenarios))
    ut.print_out(outf, 'Sum of weights by scenario= \n' + str(alloc_scenarios.sum(axis=0)))

    alloc_options = list(alloc_scenarios)
    assert pfolio_alloc in alloc_options, 'Incorrect portfolio selection: {}'.format(pfolio_alloc)
    ut.print_out(outf, 'allocation options: ' + str(alloc_options))
    ut.print_out(outf, f'Initial Portfolio Allocation: {pfolio_alloc}')
    asset_list = list(alloc_scenarios.index)
    ut.print_out(outf, f'Asset List: {asset_list}')
    start_asset = pd.Series([start_funds * x for x in list(alloc_scenarios[pfolio_alloc])], index=asset_list)

    # Create dict of inflation values
    start_age = age_dict['start_age']
    end_age = age_dict['end_age']
    nb_ages = end_age - start_age + 1  # We keep both start and end age
    infl_dict = dict([(age, pow(inflation_rate, age - start_age)) for age in range(start_age, end_age + 1)])

    asset_class = list(df_corr.columns)  # List of assets
    # Create DF of rates of return for each asset class for run_cnt iterations
    # This will be split up (a) by process and (2) by iteration
    # We need nb_ages samples for each of the run_cnt iterations
    # Move from % to numbers  i.e. 3% ror -> 0.03
    df_stat *= 0.01
    big_ror_df = asst.correlated_rvs(df_stat, df_corr, nb_ages * run_cnt, param_dict['seed'])
    ut.print_out(outf, f'Generated big_ror_df of dimensions: {big_ror_df.shape}')
    print('big_ror_df')
    print(big_ror_df.loc[:, 0:9])
    print('big_ror_df  MEAN')
    print(big_ror_df.mean(axis=1))
    print('big_ror_df  StdDev')
    print(big_ror_df.std(axis=1))

    # run MonteCarlo - Multi-processor mode
    ns = int(run_cnt / nb_cpu)  # number of samples for each process
    run_cnt = nb_cpu * ns  # in case we have rounding above

    # Iterate to find the maximum values of Retirement_spend and Travel
    # assume this initial value of retirement spend and Travel are too high and needs to be reduced
    rtmt_spnd = cashflow_data['Retirement_spend']
    trvl_spnd = cashflow_data['Travel']
    success_flag = False
    epsilon = param_dict['epsilon']  # 5%
    prev_sign = -1.0
    iter = 0
    cumul_adjust = 1.0
    while not success_flag:
        iter += 1
        ut.print_out(outf, f'\nIteration: {iter:,d}')
        ut.print_out(outf, cashflow_item_2_string('Retirement_spend', rtmt_spnd, age_dict))
        ut.print_out(outf, cashflow_item_2_string('Travel', trvl_spnd, age_dict))
        # cashflow_data.update(rtmt_spnd)
        cashflow_data['Retirement_spend'] = rtmt_spnd
        cashflow_data['Travel'] = trvl_spnd
        # Make DF of withdrawals
        cashflow_df = make_cashflow(cashflow_data, infl_dict, age_dict, xl_file)
        # cashflow_out = cashflow_df.iloc[:, 0].apply(to_dollar)
        # ut.print_out(outf, "Withdrawals:\n" + pformat(cashflow_out, indent=4))

        if nb_cpu == 1:  # Skip multi-processing
            assets_at_end, final_age_ser = run_mc_multi(start_asset, big_ror_df, cashflow_df, mgt_fee, rebalance_flag,
                                                        ns, 0)
        else:  # do multi-processing
            nb_smpl_per_process = nb_ages * ns  # each iteration needs nb_ages samples
            # param_list = [ [ns, 0], [ns, N], ... , [ns, (C-1)*N]] when N is nb_smpl_per_process and C is nb_cpu
            param_list = zip([ns] * nb_cpu, range(0, nb_smpl_per_process * nb_cpu, nb_smpl_per_process))
            # partial “freezes” some the first arguments of run_mc_multi and appends values of run_cnt_list as it
            # iterates https://docs.python.org/3.6/library/functools.html
            pool = Pool(processes=nb_cpu)
            result_obj = pool.starmap_async(
                partial(run_mc_multi, start_asset, big_ror_df, cashflow_df, mgt_fee, rebalance_flag), param_list)
            pool.close()
            pool.join()
            result_list = result_obj.get()
            # Each simulation run produces (1) a [m, N] DF of resulting assets DF and (2) a [N,] series of final ages
            result_asset = [x[0] for x in result_list]
            result_age = [x[1] for x in result_list]
            # Concatenate result_age into a single series
            # https://www.quora.com/How-can-I-convert-the-list-1-2-3-into-1-2-3-in-Python-Basically-I-want-the-list-to-be-flattened
            assets_at_end = pd.concat(result_asset, axis=1, ignore_index=True)  # Merge into single DF
            final_age_ser = pd.concat(result_age, ignore_index=True)  # Merge into single series

        busted_age_ser = final_age_ser[final_age_ser < end_age]
        if len(busted_age_ser) > 0:  # we busted once or more
            busted_cnt = collections.Counter(list(busted_age_ser))  # dict: {age, nb_busted_at_this_age}
            # for proper plotting, make sure all ages are in the dict, fill in with 0s
            min_age = min(busted_cnt.keys())
            max_age = max(busted_cnt.keys())
            missing_key = [k for k in range(min_age, max_age + 1) if
                           k not in busted_cnt.keys()]  # ages not in busted_cnt
            missing_dict = {k: 0 for k in missing_key}  # fill w/ 0 (count)
            busted_cnt.update(missing_dict)
            busted_counter = {k: busted_cnt[k] for k in sorted(busted_cnt.keys())}  # sort by age
        else:  # 100% success
            busted_counter = {k: 0 for k in range(start_age, end_age + 1)}  # fill w/ 0 (count)

        assets_at_end.loc['Total_infl'] = assets_at_end.sum(axis=0)  # these totals are in tomorrow's dollars
        # print('Assets at end:\n', assets_at_end.head())

        # Bring the result_asset to today's dollars -> use the final age to adjust back for inflation
        # Add age as a row to assets_at_end
        assets_at_end.loc['age'] = final_age_ser
        # Convert dollars at the end into today's dollars - based on the age
        assets_at_end.loc['Total'] = np.vectorize(today_dollar)(assets_at_end.loc['Total_infl'],
                                                                assets_at_end.loc['age'] - start_age, inflation_rate)
        total_ser = assets_at_end.loc['Total'].copy(deep=True)  # so that we can sort
        total_ser.sort_values(inplace=True, ascending=True)
        total_ser.reset_index(inplace=True, drop=True)  # needed after sort
        # Note that nb_bust and nb_success don't necessarily add up to 100% - 3rd category: 0< end_assets <
        # target_end_funds
        nb_bust = sum([x for x in busted_counter.values()])
        bust_pct = 100.0 * nb_bust / run_cnt
        nb_success = len(total_ser[total_ser >= target_end_funds])
        success_pct = 100.0 * nb_success / run_cnt
        ut.print_out(outf, 'Busted {:,d} times out of {:,d} -> {:.2f}%'.format(nb_bust, run_cnt, bust_pct))
        ut.print_out(outf, 'SUCCESS {:,d} times out of {:,d} -> {:,.2f}%'.format(nb_success, run_cnt,
                                                                                 100.0 * nb_success / run_cnt))
        # Check success
        if quick_flag or abs(success_pct - 80.0) <= 1.0:  # Success
            success_flag = True
        else:  # reduce retirement spend
            # Update adjustment
            cur_sign = np.sign(success_pct - 80.0)
            if prev_sign * cur_sign < 0:  # deltas have different sign - decrease adjustment
                epsilon *= -0.5  # also change sign
            adjust = 1.0 + epsilon
            cumul_adjust *= adjust
            prev_sign = cur_sign
            # for each dict, reduce spending amount
            print('Success_pct = {}% - Cumulative adjustment = {:,.2f}% - new epsilon = {:,.4f}'.format(success_pct,
                                                                                                        100 * (
                                                                                                                    cumul_adjust - 1),
                                                                                                        epsilon))
            for dd in [rtmt_spnd, trvl_spnd]:
                for k in dd.keys():
                    dd[k]['amount'] *= adjust

    # For final result ...
    print('\nFINAL @ Iteration: {:,d} - Portfolio Allocation = {} -> Success_pct = {}% - Cumulative adjustment = {:,.2f}%'.format(
            iter, pfolio_alloc, success_pct, 100.0 * (cumul_adjust - 1)))
    ut.print_out(outf, 'Busted Ages (Count by Age):\n' + str(non_zero_dict(busted_counter)))
    # Plot the busted ages
    busted_for_plot = pd.DataFrame.from_dict(busted_counter, orient='index')
    Busted_title = 'Count of Busted Ages - ({:,.2f}%)'.format(100.0 * nb_bust / run_cnt)
    plot_with_rails(busted_for_plot, title=Busted_title, plt_file=plt_file)

    # Compute & plot deciles for assets
    if run_cnt >= 100:
        final_asset_dict = dict()
        final_asset_dict['Min'] = total_ser.loc[0]
        final_asset_dict['Max'] = total_ser.loc[run_cnt - 1]
        final_asset_dict['Avg'] = total_ser.mean()
        final_asset_dict['Stddev'] = total_ser.std()
        dict_out = {k: to_dollar(v) for k, v in final_asset_dict.items()}
        ut.print_out(outf, "Final Asset Statistics: " + str(dict_out))

        delta = int(run_cnt / 10)
        final_asset_dict = dict()
        for decl, idx in enumerate(range(0, run_cnt, delta)):  # Average over the decile
            final_asset_dict[decl] = total_ser.loc[idx:idx + delta].mean()
        dict_out = {k: to_dollar(v) for k, v in final_asset_dict.items()}
        ut.print_out(outf, "Final Asset Deciles: " + str(dict_out))
        final_for_plot = pd.Series([final_asset_dict[decl] for decl in range(0, 10)])
        plot_with_rails(final_for_plot, title='Decile Results', rails=True, dollar=True, plt_file=plt_file)
    else:  # not enough values
        ut.print_out(outf, str(total_ser))
    # if run_cnt <= 1000: #
    #     with pd.ExcelWriter(xl_file) as xl_writer:
    #         assets_at_end.to_excel(xl_writer, sheet_name="Iterations", float_format='%.2f', header=True)

    ut.print_out(outf, 'Final Results:')
    ut.print_out(outf, cashflow_item_2_string('Retirement_spend', rtmt_spnd, age_dict))
    ut.print_out(outf, cashflow_item_2_string('Travel', trvl_spnd, age_dict))

    # Wrap up
    plt_file.close()
    end_time = dt.datetime.now()
    logger.debug('\nEnd: {}'.format(str(end_time)))
    logger.debug('Run Time: {}\n'.format(str(end_time - start_time)))
    outf.close()
    return


if __name__ == '__main__':
    # bug fix for multiprocessing
    set_start_method('forkserver')  # from multiprocessing

    main(sys.argv)
    exit(0)
