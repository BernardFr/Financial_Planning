#!/usr/bin/env python



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
