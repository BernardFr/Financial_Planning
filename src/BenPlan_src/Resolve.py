#!/usr/bin/env python

# import csv
import os
import sys
import time
import getopt
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import expenses_utilities as ut
import logging
import datetime as dt

"""
Attempt to resolve Transfer and Ignore transactions: each category should add up to $0
"""

plt.style.use('seaborn-deep')

# Configure the settings
dirDateFormat = ut.Settings['dirDateFormat']
qDateFormat = ut.Settings['qDateFormat']
firstDate = ut.Settings['firstDate']
benplan_cat_file = ut.Settings['benplan_cat_file']
csvDir = ut.Settings['csvDir']
dataDir = ut.Settings['dataDir']
closed_dir = ut.Settings['closedDir']
dirNamePattern = ut.Settings['dirNamePattern']
col2Keep = ut.Settings['col2Keep']
PLOT_FLAG = False
quick_flag = False
PROG_NAME = ''  # will be defined in main()
cat_to_keep = ['Ignore', 'Transfer']
match_col = ['Day', 'Amount']

# Threshold under which we accumulate the remaining categories into a single one
other_pct = 0.2  # 20%
birth_month = 4  # April
birth_year = 1958


# MAIN ENTITIES
# cumul_df: clean dataframe of all transactions
# agg: dataframe of Amounts groupd be 'Month', 'Category', 'MasterCat', 'BenPlan'


def to_dollar_str(z):
    """ Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points """
    return '${:,.2f}'.format(z)


def tick_format(x, pos):
    """ Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return '${:,.0f}'.format(x)


def remap_sub_cat(df, master_cat_file, outf):
    """ remap Categories/Subcategory to a new Category based on MasterCat file """
    # df['Category'] may be 'Category' or 'Category:Subcategory'

    # Read the Master Categories files
    master_cat_df = pd.read_csv('../../' + master_cat_file, sep=',', header=0)
    # Remove useless columns
    master_cat_df = master_cat_df[['Category', 'MappedCat', 'MasterCat', 'BenPlan']]
    # When the Mapped Category is not specified (blank)
    # we use 'Category'
    # Test if MappedCat exists
    master_cat_df['flag'] = master_cat_df['MappedCat'].apply(pd.notnull)

    # NewCat = MappedCat if MappedCat is present, else Category
    def cat_map(row):
        return row['MappedCat'] if row['flag'] else row['Category']

    master_cat_df['newCat'] = master_cat_df.apply(cat_map, axis=1)
    # Create 2 dict to perform the mapping from Category to newCat and MasterCat
    cat_dict = dict(zip(master_cat_df['Category'], master_cat_df['newCat']))
    master_dict = dict(zip(master_cat_df['Category'], master_cat_df['MasterCat']))
    ben_dict = dict(zip(master_cat_df['Category'], master_cat_df['BenPlan']))

    # Identify if we have any new/bad categories
    def bad_category(strg):
        """ Bad catogry is not in the list or Uncategorized """
        flg_1 = True if strg not in master_dict.keys() else False
        flg_2 = True if strg == "Uncategorized" else False
        return flg_1 | flg_2

    bad_cat = df[df['Category'].apply(bad_category) == 1]
    if len(bad_cat.index) > 0:
        ut.print_out(outf, "ERROR: Bad Categories")
        ut.print_out(outf, repr(bad_cat))
        exit(-1)

    df['MasterCat'] = df['Category'].apply(lambda x: master_dict[x])
    df['BenPlan'] = df['Category'].apply(lambda x: ben_dict[x])
    # !!! IMPORTANT: since we are remapping Category - it has to be done last
    # Otherwise the mapping to masterCat won't work
    df['Category'] = df['Category'].apply(lambda x: cat_dict[x])
    del master_cat_df
    return df


def lineplot_df(df, title=None, plt_file=None):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    labels = []
    cat_list = list(df.index)
    x_tick = range(df.shape[1])  # 0 ... nb_columns-1
    # Plot each column in the array and add category name to list of labels
    for i in range(df.shape[0]):
        plt.plot(x_tick, df.iloc[i])
        labels.append(' ' + cat_list[i])

    plt.xticks(x_tick, list(df.columns), rotation=270)
    plt.legend(labels, ncol=1, loc='center left',  # place the center left anchor 100% right, and 50% down, i.e. center
               bbox_to_anchor=[1, 0.5], columnspacing=1.0, labelspacing=0.0, handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor('#E6E6E6')
    # Shrink current axis by 15% to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    plt.grid(axis='y', which='both', color='darkgreen', linestyle='-', linewidth=1, alpha=0.5)
    if title:
        plt.title(title)
    if plt_file:
        # bbox_inches -> makes the legend fit
        plt.savefig(plt_file, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    return


def plot_with_rails(df, title=None, plt_file=None):
    """ Single line plot with rails: Average and +/- std dev"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    x_tick = range(df.shape[0])
    plt.plot(x_tick, df, color='steelblue')
    # Plot lines for mean & rails for stddev
    # Build array of containing the mean value for all entries - as a Reference
    df_mean = float(df.mean())  # to ensure the result is not a Series
    df_stddev = float(df.std())
    # Create horizontal lines showing mean, +/- stddev
    ax.axhline(y=df_mean, color='red', ls='solid', alpha=0.5)
    ax.axhline(y=df_mean + df_stddev, color='red', ls='dashed', alpha=0.5)
    ax.axhline(y=df_mean - df_stddev, color='red', ls='dashed', alpha=0.5)

    # Decorate plot w/ labels, title etc
    plt.xticks(x_tick, list(df.index), rotation=270)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor('#E6E6E6')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.grid(axis='y', which='both', color='darkgreen', linestyle='-', linewidth=1, alpha=0.5)
    if title:
        plt.title(title)
    if plt_file:
        # bbox_inches -> makes the legend fit
        plt.savefig(plt_file, format='pdf', dpi=300, bbox_inches='tight')
    return


def month_2_year(mm):
    """ YYYY-MM -> YYYY """
    return mm.split('-')[0]


def month_2_age(mm):
    """ Convert YYYY-MM to age based on BIRTH_MONTH and BIRTH_YEAR"""
    yr, mo = mm.split('-')
    yr = int(yr)
    mo = int(mo)
    return yr - birth_year - 1 if mo < birth_month else yr - birth_year


def make_val2test(row):
    """ Combine returns a 2-tuple from 2 columns"""
    return (row['Day'], row['Amount'])


def find_match(neg_df, pos_df, day_offset, neg_lst, pos_lst):
    """ Find transactions that match with day_offset days between neg_df and pos_df
    Lists neg_lst & pos_lst keep the respective matching indices"""
    neg_df['Day'] = neg_df['Date'].apply(ut.date2day)  # Need to recompute
    neg_df['Day'] += day_offset
    neg_df['Val2test'] = neg_df.apply(make_val2test, axis=1)
    # skip the indices where we already have a match
    idx_list = [x for x in neg_df.index if x not in neg_lst]
    for idx in idx_list:
        # get a list in indices of pos_def that match the value neg_df.loc[idx, 'Val2test'
        match_lst = pos_df.index[pos_df['Val2test'] == neg_df.loc[idx, 'Val2test']].tolist()
        match_lst = [x for x in match_lst if x not in pos_lst]  # eliminate dupes
        if match_lst:  # we have a match
            neg_lst += [idx]
            pos_lst += [match_lst[0]]  # take the first one
            # print('Match: {} & {}'.format(idx, match_lst[0]))
            # print(str(rslv_df.loc[idx]))
            # print(str(rslv_df.loc[match_lst[0]]))
    return neg_lst, pos_lst


# def find_match_old(df, ref_row):
#     """ Find 1st row in dataframe df equal to ref_row
#     Returns index of matching transaction if found, None if not """
#     # flag is True for rows that match ref_row
#     # flag_ser = df.apply(lambda x: x.equals(ref_row), axis=1)
#     # df2 = df[flag_ser]
#     print(df.head())
#     print(df.dtypes)
#     print(ref_row)
#     print(ref_row.dtypes)
#     ref_row['Day'] += 1
#     print(ref_row)
#     print(ref_row.dtypes)
#     df['flag'] = df.apply(lambda x: x.equals(ref_row), axis=1)
#     print(df.head())
#     ref_row2 = df.iloc[0]
#     ref_row2['Day'] += 1
#     ref_row2['Amount'] *= -1.0
#     print(ref_row2)
#     print(ref_row2.dtypes)
#     df['flag'] = df.apply(lambda x: x.equals(ref_row2), axis=1)
#     print(df.head())
#     exit(0)
#     df2 = df[df['flag'] == True]
#     # If lst contains 1 or more indices that match, return the first one
#     lst = list(df2.index)
#     return lst if len(lst) > 0 else None


# ---- MAIN
def main(argv):
    global PROG_NAME, other_spend, other_idx
    global PLOT_FLAG, quick_flag

    start_time = dt.datetime.now()
    prog_name = argv[0].lstrip('./').rstrip('.py')

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

    try:
        #   opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
        opts, args = getopt.getopt(argv[1:], "hpq")
    except getopt.GetoptError:
        print('{} -q'.format(prog_name))
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('{} -q'.format(prog_name))
            sys.exit()
        elif opt in "-p":  # show plots on display
            plot_flag = True
        elif opt in "-q":  # quick run - one set of values for hyperparameters
            quick_flag = True
        else:
            print('Error: Unrecognized option: {}'.format(opt))
            print('Syntax: {} -p'.format(prog_name))
            sys.exit(2)

    # Get the list of all directories in dataDir
    d_files = [d.name for d in os.scandir(dataDir) if d.is_dir() is True]
    # fullmatch returns None if there is No match
    dir_files = [d for d in d_files if re.fullmatch(dirNamePattern, d)]

    # Get the most recent directory
    current_dir = max(dir_files)
    out_file = 'Data/' + prog_name + '-' + current_dir + '_out.txt'
    outf = open(out_file, 'w')
    ut.print_out(outf, 'Processing files from: ' + current_dir)
    agg_file = 'aggregate-' + current_dir + '.csv'
    comb_file = 'combined-' + current_dir + '.csv'
    avg_file = 'average-' + current_dir + '.csv'
    pivot_file = 'monthly-' + current_dir + '.csv'
    pivot_master_file = 'monthly-master-' + current_dir + '.csv'
    ben_plan_file = 'benPlan-' + current_dir + '.csv'
    xl_out_file = prog_name + '-' + current_dir + '.xlsx'
    # Ensure the program does not attempt to read these files as data files
    suppress_list = [agg_file, comb_file, avg_file, pivot_file, pivot_master_file, ben_plan_file]
    # For some reason, need to add the directory to the path for plot_file
    plot_file = './Data/' + current_dir + '/' + prog_name + '-plots-' + current_dir + '.pdf'
    plt_file = PdfPages(plot_file)
    xl_writer = pd.ExcelWriter(xl_out_file)

    # Change directory to the most recent
    os.chdir(dataDir + current_dir)
    # Get a list of the CSV files - only
    csv_files = [f.name for f in os.scandir('.') if f.name.split('.')[1] == 'csv']

    csv_files = [f for f in csv_files if f not in suppress_list]
    # Add CLOSED files that no longer change: e.g Citibank
    closed_files = [f.name for f in os.scandir(closed_dir) if f.name.split('.')[1] == 'csv']
    closed_files = [closed_dir + f for f in closed_files]  # Use the full path for these files
    csv_files += closed_files
    ut.print_out(outf, "Files processed:\n" + repr(csv_files))

    # cumul_df is the main array of all transactions
    cumul_df = pd.DataFrame()
    for file in csv_files:
        df = ut.process_file(file)
        # Add to the main array
        cumul_df = cumul_df.append(df)

    # Create date label
    cumul_df['Month'] = cumul_df['Date'].apply(lambda x: ut.date2month(x, qDateFormat))
    z = cumul_df[cumul_df.Month == 'xxxx-xx']
    if len(z.index) > 0:
        ut.print_out(outf, 'rows with weird dates')
        ut.print_out(outf, repr(z))

    # Keep the transactions in specified date range
    first_mo = ut.date2month(firstDate, qDateFormat)
    # Cheeky to use comparison on strings, but the label Month was built for this purpose (also for plot)
    # Would have to use a time function on Date
    # lastMo is in format yyyy-mo-dd - so we need to remove '-dd' without making assumptions on the length of mo or dd
    last_mo = ut.date2month(current_dir, dirDateFormat)
    ut.print_out(outf, 'First Mo: {} - Last Mo: {}'.format(first_mo, last_mo))
    cumul_df = cumul_df[cumul_df.Month >= first_mo]
    # Note the < ... we want to exclude the last month, since it is incomplete
    cumul_df = cumul_df[cumul_df.Month < last_mo]
    # ut.print_df(cumul_df, msg="Good Dates", verbose=False)

    # Convert Amount to Float so that it can be added up - Note that none of the 2 approaches below work
    # E.g Converting to float pukes on commas (e.g 1,599.60)!!
    # cumul_df['Amount'] =cumul_df.Amount.apply(lambda x: x.strip(','))
    # cumul_df['Amount'] =cumul_df['Amount'].astype(str).replace(',','').astype(float)
    cumul_df['Amount'] = cumul_df['Amount'].apply(lambda x: float(str(x).replace(',', '')))

    # Remove sub-categories
    cumul_df = remap_sub_cat(cumul_df, benplan_cat_file, outf)
    cumul_df['Day'] = cumul_df['Date'].apply(ut.date2day)
    cumul_df.sort_values(by='Day', ascending=True, inplace=True, na_position='last')
    cumul_df.to_excel(xl_writer, sheet_name="Combined", float_format='%.2f', header=True)
    ut.print_out(outf, 'Categories:\n')
    ut.print_out(outf, repr(sorted(set(cumul_df['Category']))))

    # ---- For each category in cat_to_keep - strip out transaction pairs that balance out
    cumul_df.drop(['Month', 'BenPlan'], axis=1, inplace=True)
    for ctgry in cat_to_keep:
        rslv_df = cumul_df[cumul_df['MasterCat'] == ctgry].copy(deep=True)
        # Sort by date
        rslv_df.sort_values(by=match_col, ascending=True, inplace=True, na_position='last')
        rslv_df.reset_index(drop=True, inplace=True)
        # print(rslv_df.head(10))
        # Create 2 arrays: (1) negative (2)positive amounts
        neg_df = rslv_df[rslv_df['Amount'] < 0.0].copy(deep=True)
        pos_df = rslv_df[rslv_df['Amount'] >= 0.0].copy(deep=True)
        neg_df['Amount'] *= -1.0  # Make Amount positive to test equality
        pos_df['Val2test'] = pos_df.apply(make_val2test, axis=1)
        # Note neg_def and pos_def keep indices of rslv_df
        neg_lst = []
        pos_lst = []
        for day_offset in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
            # Transfers have positive offsets, credit cards negative ones
            neg_lst, pos_lst = find_match(neg_df, pos_df, day_offset, neg_lst, pos_lst)

        # for idx in neg_df.index:
        #     # get a list in indices of pos_def that match the value
        #     match_lst = pos_df.index[pos_df['Val2test'] == neg_df.loc[idx,'Val2test']].tolist()
        #     match_lst = [x for x in match_lst if x not in pos_lst]
        #     if match_lst:  # we have a match
        #         neg_lst += [idx]
        #         pos_lst += [match_lst[0]]  # take the first one
        #         # print('Match: {} & {}'.format(idx, match_lst[0]))
        #         # print(str(rslv_df.loc[idx]))
        #         # print(str(rslv_df.loc[match_lst[0]]))

        # # Handle cases when credit is 1 day after debit
        # neg_df['Day'] += 1
        # neg_df['Val2test'] = neg_df.apply(make_val2test, axis=1)
        # # skip the indices where we already have a match
        # idx_list = [x for x in neg_df.index if x not in neg_lst]
        # for idx in idx_list:
        #     # get a list in indices of pos_def that match the value
        #     match_lst = pos_df.index[pos_df['Val2test'] == neg_df.loc[idx,'Val2test']].tolist()
        #     match_lst = [x for x in match_lst if x not in pos_lst]
        #     if match_lst:  # we have a match
        #         neg_lst += [idx]
        #         pos_lst += [match_lst[0]]  # take the first one
        #         # print('Match: {} & {}'.format(idx, match_lst[0]))
        #         # print(str(rslv_df.loc[idx]))
        #         # print(str(rslv_df.loc[match_lst[0]]))
        #
        # # Handle cases when credit is 2 days after debit
        # neg_df['Day'] += 1
        # neg_df['Val2test'] = neg_df.apply(make_val2test, axis=1)
        # # skip the indices where we already have a match
        # idx_list = [x for x in neg_df.index if x not in neg_lst]
        # for idx in idx_list:
        #     # get a list in indices of pos_def that match the value
        #     match_lst = pos_df.index[pos_df['Val2test'] == neg_df.loc[idx,'Val2test']].tolist()
        #     match_lst = [x for x in match_lst if x not in pos_lst]
        #     if match_lst:  # we have a match
        #         neg_lst += [idx]
        #         pos_lst += [match_lst[0]]  # take the first one
        #         print('Match: {} & {}'.format(idx, match_lst[0]))
        #         print(str(rslv_df.loc[idx]))
        #         print(str(rslv_df.loc[match_lst[0]]))
        #
        # # Handle cases when credit is 3 days after debit
        # neg_df['Day'] += 1
        # neg_df['Val2test'] = neg_df.apply(make_val2test, axis=1)
        # # skip the indices where we already have a match
        # idx_list = [x for x in neg_df.index if x not in neg_lst]
        # for idx in idx_list:
        #     # get a list in indices of pos_def that match the value
        #     match_lst = pos_df.index[pos_df['Val2test'] == neg_df.loc[idx,'Val2test']].tolist()
        #     match_lst = [x for x in match_lst if x not in pos_lst]
        #     if match_lst:  # we have a match
        #         neg_lst += [idx]
        #         pos_lst += [match_lst[0]]  # take the first one
        #         print('Match: {} & {}'.format(idx, match_lst[0]))
        #         print(str(rslv_df.loc[idx]))
        #         print(str(rslv_df.loc[match_lst[0]]))
        #
        # # Handle cases when credit is 1 days BEFORE debit
        # neg_df['Day'] = neg_df['Date'].apply(ut.date2day)
        # neg_df['Day'] -= 1
        # neg_df['Val2test'] = neg_df.apply(make_val2test, axis=1)
        # # skip the indices where we already have a match
        # idx_list = [x for x in neg_df.index if x not in neg_lst]
        # for idx in idx_list:
        #     # get a list in indices of pos_def that match the value
        #     match_lst = pos_df.index[pos_df['Val2test'] == neg_df.loc[idx,'Val2test']].tolist()
        #     match_lst = [x for x in match_lst if x not in pos_lst]
        #     if match_lst:  # we have a match
        #         neg_lst += [idx]
        #         pos_lst += [match_lst[0]]  # take the first one
        #         print('Match: {} & {}'.format(idx, match_lst[0]))
        #         print(str(rslv_df.loc[idx]))
        #         print(str(rslv_df.loc[match_lst[0]]))

        match_list = neg_lst + pos_lst
        if len(match_list) != len(list(set(match_list))):
            print("ERROR: match_list has duplicates")
            print(match_list)
        rslv_df.drop(rslv_df.index[match_list], inplace=True)
        # print(rslv_df.head(10))
        ut.print_out(outf, 'Category: {} -> Matched {} Transaction Pairs - {} UNmatched remain\n'.format(ctgry,
                                                                                         len(neg_df), rslv_df.shape[0]))
        rslv_df.to_excel(xl_writer, sheet_name=ctgry, float_format='%.2f', header=True)

    # Wrap up
    xl_writer.save()
    plt_file.close()
    outf.close()

    end_time = dt.datetime.now()
    logger.debug('\nEnd: {}\n'.format(str(end_time)))
    logger.debug('Run Time: {}\n'.format(str(end_time - start_time)))
    return


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
    exit(0)
