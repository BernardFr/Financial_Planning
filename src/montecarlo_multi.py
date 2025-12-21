#!/usr/local/bin/python3


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

# ToDo: Add Readme - including syntax for TOML file
# ToDo: add loop to find value for retirement_spend that meets goal and/or optimal asset allocation

import getopt
import logging
import sys, os
import numpy as np
import pandas as pd
import collections
import datetime as dt
import montecarlo_utilities as ut
import tomli
from pprint import pformat
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, set_start_method
# import itertools

# from multiprocessing import cpu_count
from functools import partial

# Configuration
plt.style.use("seaborn-v0_8-deep")
Debug = False
plot_flag = False
outf = None  # placeholder for global variable
BF_BDAY = '04/09/1958'
Max_run_to_save = 250  # Max number of runs for which we save assets at end of each run

"""
From: http://fc.standardandpoors.com/sites/client/generic/axa/axa4/Article.vm?topic=5991&siteContent=8088
Total Returns From 1926-2017*
     Stocks    Bonds    T-Bills
Annualized Return    10.2%    5.6%    3.5%
Best Year    60.0 (1935)    42.1 (1982)    14.0 (1981)
Worst Year    -41.1 (1931)    -12.2 (2009)    0.0 (1940)
Standard Deviation    18.7    7.7    0.9
"""
AssetClasses = collections.namedtuple(
    "AssetClasses", "Stocks Bonds TBills Cash"
)  # Immutable
historical_return_2017 = AssetClasses(
    Stocks=(10.20, 18.70), Bonds=(5.60, 7.70), TBills=(3.50, 0.90), Cash=(0.0, 0.0)
)
default_pfolio_alloc = AssetClasses(Stocks=0.75, Bonds=0.15, TBills=0.05, Cash=0.05)


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
        yield [
            1 + 0.01 * np.random.normal(self.mean, self.stdv) for _ in range(self.nb)
        ]


def to_dollar(x):
    """Converts a number to a string prepended with '$", and with commas rounded 2 digits"""
    return "${:,.2f}".format(x)


def tick_format(x, pos):
    """
    Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points
    Note: The unused 'pos' argument is necessary for FuncFormatter (see below)
    """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return "{:,.0f}".format(x)


def tick_format_w_dollar(x, pos):
    """
    Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points
    Note: The unused 'pos' argument is necessary for FuncFormatter (see below)
    """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return "${:,.0f}".format(x)


def plot_with_rails(df, title=None, plt_file=None, rails=False, dollar=False):
    """ Single line plot with rails: Average and +/- std dev"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    x_tick = range(df.shape[0])
    plt.plot(x_tick, list(df.values), color="steelblue")
    if rails:  # Plot lines for mean & rails for stddev
        # Build array of containing the mean value for all entries - as a Reference
        df_mean = float(df.mean())  # to ensure the result is not a Series
        df_stddev = float(df.std())
        # Create horizontal lines showing mean, +/- stddev
        ax.axhline(y=df_mean, color="red", ls="solid", alpha=0.5)
        ax.axhline(y=df_mean + df_stddev, color="red", ls="dashed", alpha=0.5)
        ax.axhline(y=df_mean - df_stddev, color="red", ls="dashed", alpha=0.5)

    # Decorate plot w/ labels, title etc
    plt.xticks(x_tick, list(df.index), rotation=0)
    if dollar:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format_w_dollar))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor("#E6E6E6")
    box = ax.get_position()
    ax.set_position([box.x0 + 0.05, box.y0, box.width, box.height])
    plt.grid(
        axis="y", which="both", color="darkgreen", linestyle="-", linewidth=1, alpha=0.5
    )
    if title is not None:
        plt.title(title)
    if plt_file is not None:
        # bbox_inches -> makes the legend fit
        plt.savefig(plt_file, format="pdf", dpi=300, bbox_inches="tight")
    if plot_flag:
        plt.show()
    return


def mk_ror_df(gen_list, index, ages):
    """
    Generate a dataframe where columns are integers in range(start, end) - where the rates of return are 
    generated by the generators in gen_list
    :param gen_list: list of generators
    :param index: index of DF
    :param ages: 2-tuple with start and end age
    :return: pd.DataFrame with end-start columns
    """
    start = ages[0]
    end = ages[1]
    df = pd.DataFrame(
        data=None, index=index, columns=list(range(start, end + 1)), dtype=float
    )
    for idx, gener in zip(list(index), gen_list):
        # generator returns a 1-element list which contains the list of values
        df.loc[idx] = list(next(gener))[0]
    return df


def re_allocate(asset, alloc, re_alloc_error):
    """
    Re-allocate assets based on desired asset allocation "alloc" - when asset amounts are negative
    :param asset: Series of assets
    :param alloc: Target allocation
    :return: new values of asset
    """
    if all(asset >= 0.0):  # all good, all amounts are positive or zero
        return asset, alloc  # done
    if asset.sum() <= 0.0:  # Busted - sum of amounts are negative or 0 -> return all 0s
        return pd.Series(0, index=np.arange(len(asset))), alloc

    # Need to make adjustments
    n_asset = asset.copy(deep=True)
    n_alloc = alloc.copy(deep=True)
    # Need to iterate until all asset classes are positive or zero
    neg_flag = n_asset < 0.0
    while any(neg_flag):  # While one asset category is negative
        # Some = but not all - assets have negative amount
        to_re_allocate = n_asset[neg_flag].sum()  # sum of all negative amounts
        n_asset[neg_flag] = 0.0  # zero out assets that have no money
        n_alloc[neg_flag] = 0.0  # zero out allocation of asset that have no money
        tot = n_alloc.sum()  # will be < 1.0
        n_alloc = (n_alloc / tot)  # rebalance allocations for classes that have positive amount
        assert (abs(n_alloc.sum() - 1.0) < re_alloc_error),\
            "n_alloc adds up to {:,.4f} - not 1.0 - n_alloc:\n".format(n_alloc.sum()) + str(n_alloc)
        adjust = (n_alloc * to_re_allocate)  # distribute amount to re-allocate among assets still active
        n_asset += adjust
        neg_flag = (n_asset < 0.0)  # Re-test - some asset classes may have become negative after rebalancing
        if abs(asset.sum() - n_asset.sum()) > re_alloc_error:
            ut.print_out(outf,
                         "Asset {:,.4f} and re-allocated assets {:,.4f} don't add up - delta = {:,.8f}".format(asset.sum(),
                            n_asset.sum(), asset.sum() - n_asset.sum()))
    return n_asset, n_alloc


def run_mc_multi(init_asset, ror_stats, cashflow_ser, re_alloc_error, cnt, seed):
    """
    Run Montecarlo simulation
    :param init_asset: (N,) SERIES of assets allocated by asset class
    :param ror_stats: Namedstuple keyed by asset class name with 2-tuple values
    :param cashflow_ser: Series of yearly withdrawals [or income] for each consecutive age
    :param cnt: Nb of iterations to run the simulation
    :return: [N, cnt] DF with ending amount by asset class for each simulation run

    Each process must initialize randon number generator's seed
    https://bbabenko.github.io/multiprocessing-and-seeded-RNGs/
    """
    print("run_mc_multi: pid: {}, cnt= {}, seed={}".format(os.getpid(), cnt, seed))
    np.random.seed(seed)
    age_col = list(cashflow_ser.index)
    ages = (int(age_col[0]), int(age_col[-1]))  # start and end age
    idx_name = init_asset.index
    ror_gen_list = [ArrayRandGen(ages, a) for a in ror_stats]  # List of generators
    # Create dataframe for results - rows are asset classes, columns are iterations
    asset_df = pd.DataFrame(index=idx_name, dtype=float)  # Empty DF with the right index
    # Compute initial asset allocation - and use it for rebalancing
    init_alloc = init_asset / init_asset.sum()
    busted_ages = []
    for itr in range(cnt):
        asset_tmp = init_asset.copy(deep=True)  # initialize
        wdrwl_alloc = init_alloc  # Initial allocation of withdrawals
        # Generate a new array of rate of returns for all ages and each asset class
        ror_df = mk_ror_df(ror_gen_list, idx_name, ages)
        for age in age_col:
            # Compute returns of existing assets and allocate year's withdrawal weighted to available assets
            asset_tmp = asset_tmp * ror_df[age] + wdrwl_alloc * cashflow_ser[age]
            if asset_tmp.sum() <= 0.0:  # Busted: no more money
                # ut.print_out(outf, 'Busted! @iter: {:d} - age {:d}'.format(iter, int(age)))
                busted_ages += [age]
                break  # Stop iterating over age
            else:
                n_asset, wdrwl_alloc = re_allocate(asset_tmp, wdrwl_alloc, re_alloc_error)
                assert (
                    abs(asset_tmp.sum() - n_asset.sum()) < re_alloc_error
                ), "Asset {:,.4f} and re-allocated assets {:,.4f} don't add up - delta = {:,.8f}\n, iter={:d} - age={:d}".format(
                    asset_tmp.sum(), n_asset.sum(), asset_tmp.sum() - n_asset.sum(), itr, age,)
                asset_tmp = n_asset
        # asset_df[iter] = asset_tmp  # store final result of this simulation run
        asset_tmp.name = itr
        asset_df = pd.concat([asset_df, asset_tmp], axis=1)  # store final result of this simulation run
    return asset_df, busted_ages


def main(argv):
    global outf, plot_flag

    start_time = dt.datetime.now()
    prog_name = argv[0].lstrip("./").rstrip(".py")
    out_file = prog_name + "_out.txt"
    outf = open(out_file, "w")
    xl_file = prog_name + "_out.xlsx"
    toml_file = prog_name + ".toml"
    plot_file = prog_name + ".pdf"
    plt_file = PdfPages(plot_file)

    # logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(prog_name + ".log")
    handler.setLevel(logging.DEBUG)
    # create a logging format
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.debug("Start: {}\n".format(str(start_time)))
    logger.debug("Debug")

    # Handle command line options
    run_opt = False
    cpu_opt = False
    Usage = f"{prog_name} -p plot_flag -u nb_cpu -c run_count -t toml_file -o {'s','d'}\n"
    try:
        opts, args = getopt.getopt(argv[1:], "hpc:u:j:o:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print(Usage)
        print(__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(Usage)
            sys.exit()
        elif opt in "-p":  # show plots on display
            plot_flag = True
        elif opt in "-u":  # number of processes
            cpu_opt = True
            nb_cpu = int(arg)
        elif opt in "-c":  # set number of runs
            run_opt = True
            run_cnt = int(arg)
        elif opt in "-t":  # use selected toml file
            toml_file = str(arg)
        elif opt in "-o":
            if arg not in ['s', 'd']:
                print(Usage)
                assert False, Usage
            elif arg == 's':  # adjusting start_funds
                opt_type = "start_funds"
            elif arg == 'd':  # adjust discretionary income
                opt_type = 'discretionary'
            else:
                print(Usage)
                assert False, Usage
        else:
            print(Usage)
            print(__doc__)
            assert False, "unhandled option"

    # Read parameters from TOML file
    ut.print_out(outf, "Using TOML file: {}".format(toml_file))
    with open(toml_file, mode='rb') as f:
        in_data = tomli.load(f)
    ut.print_out(outf, "Parameters:")
    ut.print_out(outf, pformat((in_data)))
    # Get other config data
    user_data = in_data["user_data"]
    initial_start_funds = user_data["start_funds"]
    target_end_funds = user_data["target_end_funds"]
    default_inflation_rate = user_data["default_inflation_rate"]
    target_success_rate = user_data["target_success_rate"]
    goals_file = user_data["goals_file"]
    Death_age = user_data["Death_age"]
    End_age = user_data["End_age"]
    default_values = in_data["default_values"]
    Funds_step = default_values["Funds_step"]
    Discret_step = default_values["Discret_step"]
    Success_threshold = default_values["Success_threshold"]
    Max_iter = default_values["Max_iter"]
    seed = default_values["seed"]
    re_alloc_error = default_values["re_alloc_error"]
    if not run_opt:  # Run count was Not overriden on command line
        run_cnt = default_values["run_cnt"]
    if not cpu_opt:  # CPU count was Not overriden on command line
        nb_cpu = default_values["nb_cpu"]
    pfolio_alloc = in_data["pfolio_alloc"]

    # Process the Goals file
    initial_goals_df = ut.process_goals_file(goals_file, BF_BDAY, Death_age, End_age, default_inflation_rate)
    print(f"initial_goals_df:\n{initial_goals_df}")
    start_age = min(initial_goals_df['Start_age'])
    end_age = max(initial_goals_df['End_age'])
    age_lst = list(range(start_age, end_age + 1))  # include end_age

   # Create names for index and columns of dataframes
    index_name = list(pfolio_alloc.keys())
    # Make sure Portfolio Allocation adds up to 100% - if not resize
    if sum(pfolio_alloc.values()) != 1.0:  # Resize commmesurately
        resize = sum(pfolio_alloc.values())
        ut.print_out(outf, "Portfolio allocation is not 100%: {:,.2f} -> Resizing".format(100.0 * resize),)
        if resize == 0.0:  # problem
            ut.print_out(outf, "ERROR: Portfolio allocations are missing or equal to 0 - Stopping",)
            exit(-1)
        else:
            for k in pfolio_alloc.keys():
                pfolio_alloc[k] *= 1.0 / resize
    ut.print_out(outf, "Portfolio Allocations:\n" + pformat(pfolio_alloc))
    ror_stats = historical_return_2017

    start_funds = initial_start_funds
    goals_df = initial_goals_df.copy(deep=True)
    discretionary_mult = 1.0
    not_success = True
    prev_iter_success = False  # does not matter
    iter_cnt = 0
    while not_success:
        cashflow_df, cashflow_total_ser = ut.make_cashflow(goals_df)
        print(f"\ncashflow_df:\n{cashflow_df.head()}")
        # run MonteCarlo - Multi-processor mode
        pool = Pool(processes=nb_cpu)
        # Adjust Start Assets based on new start_funds
        start_assets = pd.Series([start_funds * x for x in pfolio_alloc.values()], index=index_name)
        assets_df = pd.DataFrame(start_assets)
        assets_df.columns = [start_age]  # single value
        ut.print_out(outf, "Start Assets:\n" + str(assets_df))

        # partial “freezes” some the first arguments of run_mc_multi and appends values of run_cnt_list as it iterates
        # https://docs.python.org/3.6/library/functools.html
        # Generate the seeds for each process - for repeatability
        ns = int(run_cnt / nb_cpu)  # number of samples for each process
        run_cnt = nb_cpu * ns  # in case we have rounding above
        np.random.seed(seed)  # Set seed for repeatability
        seed_list = np.random.randint(1, pow(2, 32) - 1, size=nb_cpu)  # Generate random seeds for each process
        param_list = zip([ns] * nb_cpu, seed_list)  # [[ns, seed1], [ns, seed2], ...]
        # ToDo Use chunksize
        result_obj = pool.starmap_async(
            partial(run_mc_multi, start_assets, ror_stats, cashflow_total_ser, re_alloc_error), param_list,)
        pool.close()
        pool.join()
        result_list = result_obj.get()
        result_asset = [x[0] for x in result_list]
        result_busted = [x[1] for x in result_list]
        # Flatten result_busted into a single list
        busted_age_lst = [x for y in result_busted for x in y]
        # Organize the busted ages
        busted_cnt = collections.Counter(busted_age_lst)
        busted_counter = {k: busted_cnt[k] for k in sorted(busted_cnt.keys())}
        assets_at_end = pd.concat(result_asset, axis=1, ignore_index=True)  # Merge into single DF
        print(f"col of assets_at_end: {assets_at_end.columns}")
        assets_at_end.loc["Total"] = assets_at_end.sum(axis=0)
        total_ser = assets_at_end.loc["Total"].copy(deep=True)  # so that we can sort
        total_ser.sort_values(inplace=True, ascending=True)
        total_ser.reset_index(inplace=True, drop=True)  # needed after sort
        nb_bust = sum([x for x in busted_counter.values()])
        ut.print_out(outf, "#1 Busted {:,d} times out of {:,d} -> {:.2f}%".format(
                nb_bust, run_cnt, 100.0 * nb_bust / run_cnt),)
        # Note that nb_bust and nb_success don't necessarily add up to 100% - 3rd category: 0< end_assets < target_end_funds
        success_rate = 100 * len(total_ser[total_ser >= target_end_funds]) / run_cnt
        success_fail = "Success" if success_rate >= target_success_rate else "Failure"
        success_string = f"{success_fail}: Start Funds: ${start_funds:,.0f} - Discretionary Mult: " \
                         f"{discretionary_mult:.3f} - " +\
                         f"Success rate: {success_rate}% vs target: {target_success_rate}%"
        ut.print_out(outf, success_string)
        # Declare success if we are not optimizing or success rate is close to goal
        if abs(success_rate - target_success_rate) < Success_threshold or 'opt_type' not in locals():
            not_success = False  # i.e yes, success
        else:
            # Adjust based on optimization strategy
            this_iter_succcess = True if success_rate > target_success_rate else False
            if opt_type == 'start_funds':
                # Adjust starting funds and repeat
                increase_decrease = -1 if this_iter_succcess else 1  # add or substract funds?
                start_funds += increase_decrease * Funds_step
            elif opt_type == 'discretionary':
                if iter_cnt >= 1:  # see if we need to change the Step
                    if this_iter_succcess != prev_iter_success:
                        Discret_step *= 0.5  # reduce the step
                prev_iter_success = this_iter_succcess
                if this_iter_succcess:  # increase discretionary spend
                   discretionary_mult += Discret_step
                else:  # decrease discretionary spend
                   discretionary_mult -= Discret_step
                goals_df = ut.adjust_goals(initial_goals_df, discretionary_mult)
        iter_cnt += 1
        if iter_cnt >= Max_iter:  # don't run for ever
            break
    # Repeat final results
    ut.print_out(outf, success_string)

    # Plot the busted ages
    print(f"busted_counter: {busted_counter}")
    busted_for_plot = pd.DataFrame.from_dict(busted_counter, orient="index")
    Busted_title = "Count of Busted Ages - ({:,.2f}%)".format(100.0 * nb_bust / run_cnt)
    plot_with_rails(busted_for_plot, title=Busted_title, plt_file=plt_file)

    # Compute & plot deciles for assets
    if run_cnt >= 100:
        final_asset_dict = dict()
        final_asset_dict["Min"] = total_ser.loc[0]
        final_asset_dict["Max"] = total_ser.loc[run_cnt - 1]
        final_asset_dict["Avg"] = total_ser.mean()
        delta = int(run_cnt / 10)
        for idx in range(0, run_cnt, delta):  # Average over the decile
            final_asset_dict[idx] = total_ser.loc[idx : idx + delta].mean()
        dict_out = {k: to_dollar(v) for k, v in final_asset_dict.items()}
        ut.print_out(outf, "Final Asset Deciles: ")
        ut.print_out(outf, str(dict_out))
        final_for_plot = pd.Series([final_asset_dict[idx * delta] for idx in range(0, 10)])
        print(f"final_for_plot: {final_for_plot}")
        plot_with_rails(final_for_plot, title="Decile Results", rails=True, dollar=True, plt_file=plt_file,)
    else:  # not enough values
        ut.print_out(outf, str(total_ser))

    # Save critical data
    with pd.ExcelWriter(xl_file) as xl_writer:
        goals_df.to_excel(xl_writer, sheet_name="Goals", float_format="%.2f", header=True)
        cashflow_df.to_excel(xl_writer, sheet_name="Cashflow", float_format="%.2f", header=True)
        if run_cnt <= Max_run_to_save:  #
                assets_at_end.to_excel(xl_writer, sheet_name="Iterations", float_format="%.2f", header=True)

    # Wrap up
    outf.close()
    plt_file.close()
    end_time = dt.datetime.now()
    logger.debug("\nEnd: {}".format(str(end_time)))
    logger.debug("Run Time: {}\n".format(str(end_time - start_time)))
    return


if __name__ == "__main__":
    # bug fix for multiprocessing
    set_start_method("forkserver")  # from multiprocessing

    main(sys.argv)
    exit(0)
