#!/usr/bin/env python

# import csv
import os
import sys
import getopt
import re
import numpy as np
import pandas as pd
# import time
import matplotlib.pyplot as plt
# from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import expenses_utilities as ut
import logging
import datetime as dt

plt.style.use('seaborn-deep')

# Configure the settings
dirDateFormat = ut.Settings['dirDateFormat']
qDateFormat = ut.Settings['qDateFormat']
firstDate = ut.Settings['firstDate']
masterCatFile = ut.Settings['masterCatFile']
csvDir = ut.Settings['csvDir']
dataDir = ut.Settings['dataDir']
dirNamePattern = ut.Settings['dirNamePattern']
col2Keep = ut.Settings['col2Keep']
PLOT_FLAG = False
quick_flag = False
PROG_NAME = ''  # will be defined in main()

# ---- RULE TO LABEL TRANSACTIONS
# Any account to account transaction is labeled Transfer
# Only transactions coming directly from an external payer (e.g. Investment Acct)
# or transactions going directly to payee are labeled differently than Transfer
# E.g paying for Julia College
# $$ from Credit Line -> WF checking is XTRA
# Check from WF to Chase is Transfer
# Payment to WUSTL is College
# ----


# To FIX -----------------------
# Use Master_Categories file to
# (a) check list of categories (see nk_list_category.py)
# (b) remap some categories and
# (c) use Master Categories
# Add 4-column MonthAvg, YrAvg (12x), TrailingMonthAvg, Trailing12Month
# Figure out why mortgage is not topN - plot pie chart with $$
# Figure out the big spikes
# Review College/Education/Julia/Kayla transactions
# Review Tax expenses classification - Family, Dad, conulting // Fed/CA//Annise
# When reviewing XTRA - subtract Education -> Nex XTRA
# Address Uncategorized
# Review the categoriesToRemove
# Consolidate the *.Reimb categories
# Analyze income -> determine how much is drawn from LT savings
# Detect abnormal spending amounts automatically
# ---- MASTER CATEGORIES FOR EXPENSES
# The master category file has 2 purposes:
# 1/ Map sub-categories to a broader bucket (e.g. credict cards) when it makes sense
# 2/ Add a Master Category tag as follows
# - AdHocIncomeCat: Non-regular income - mostly transfers from investment/savings
# - AdHocExpenseCat: Non-regular expenses, mostly one-time like big purchase/repair
# - OngoingExpenseCat: Planned/regular, typically monthly, expenses (can be yearly - e.g. insurance)
# - IncomeCat: Income - salary/Consulting
# - OneTimeCat: Out of the ordinay one time expenses
# - IgnoreCat: Internal categories to ignore - e.g. Payments - Transfer
# - RentalCat: Combined income & expenses from PPP & Loft
# ---------------

# Threshold under which we accumulate the remaining categories into a single one
other_pct = 0.2 # 20%


def to_dollar_str(z):
    """ Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points """
    return '${:,.2f}'.format(z)


def tick_format(x, pos):
    """ Convert a floating number to a string showing a dollar value
    Prepend with '$', add commas, 2 decimal points """
    # Reference: https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FuncFormatter
    return '${:,.0f}'.format(x)
# ------------------
# def stripSubCat(str):
# 	""" Strip the sub-categories from Categories """
# 	"""
# 	str may have the form: 'Category' or 'Category:Subcategory'
# 	In either case we want to return 'Category'
# 	"""
# 	idx = str.find(':')
# 	if idx == -1:  # no subcategory
# 		return(str)
# 	else:
# 		return(str[:idx]) # return only what's before the ':'


def remap_sub_cat(df, master_cat_file, outf):
    """ remap Categories/Subcategory to a new Category based on MasterCat file """
    # df['Category'] may be 'Category' or 'Category:Subcategory'

    # Read the Master Categories files
    masterCatDF = pd.read_csv('../../' + master_cat_file, sep=',', header=0)
    # Remove useless columns
    masterCatDF = masterCatDF[['Category', 'MappedCat', 'MasterCat', 'BenPlan']]
    # When the Mapped Category is not specified (blank)
    # we use 'Category'
    # Test if MappedCat exists
    masterCatDF['flag'] = masterCatDF['MappedCat'].apply(pd.notnull)
    # NewCat = MappedCat if MappedCat is present, else Category
    def catMap(row):
        return  row['MappedCat'] if row['flag'] else row['Category']
    masterCatDF['newCat'] = masterCatDF.apply(catMap, axis =1)
    # Create 2 dict to perform the mapping from Category to newCat and MasterCat
    catDict = dict(zip(masterCatDF['Category'], masterCatDF['newCat']))
    masterDict = dict(zip(masterCatDF['Category'], masterCatDF['MasterCat']))
    benDict = dict(zip(masterCatDF['Category'], masterCatDF['BenPlan']))
    # Identify if we have any new/bad categories
    def bad_category(str):
        return 1 if str not in masterDict.keys() else 0
    bad_cat = df[df['Category'].apply(bad_category) == 1]
    if len(bad_cat.index) > 0:
        print_out(outf, "ERROR: Bad Categories")
        print_out(outf, repr(bad_cat))
        exit(-1)

    df['MasterCat'] = df['Category'].apply(lambda x: masterDict[x])
    df['BenPlan'] = df['Category'].apply(lambda x: benDict[x])
    # !!! IMPORTANT: since we are remapping Category - it has to be done last
    # Otherwise the mapping to masterCat won't work
    df['Category'] = df['Category'].apply(lambda x: catDict[x])
    del masterCatDF
    return df


# ------------------
def lineplotDF(df, title=None, pF= None):
    plt.figure(figsize=(8,6))
    ax=plt.gca()
    labels = []
    catList = list(df.index)
    x_tick = range(len(df.columns))
    for i in range(len(df.index)):
        plt.plot(x_tick, df.iloc[i])
        labels.append(' ' + catList[i])

    plt.xticks(x_tick, list(df.columns), rotation=270)
    plt.legend(labels, ncol=1, loc='center left',
        # place the center left anchor 100% right, and 50% down, i.e. center
        bbox_to_anchor=[1,0.5],
        columnspacing=1.0, labelspacing=0.0,
        handletextpad=0.0, handlelength=1.5,
        fancybox=True, shadow=True)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor('#E6E6E6')
    # Shrink current axis by 15% to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    plt.grid(axis='y', which='both', color='darkgreen', linestyle='-', linewidth=1, alpha=0.5)
    if title:
        plt.title(title)
    if pF:
        # bbox_inches -> makes the legend fit
        plt.savefig(pF, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    return


def print_out(outfile, msg='\n'):
    """ Prints message to both outfile and terminal
    :rtype: None
    """
    print(msg)
    outfile.write(msg)
    return


def main(argv):
    global PROG_NAME
    global PLOT_FLAG
    global quick_flag

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
        elif opt in ("-p"):  # show plots on display
            plot_flag = True
        elif opt in ("-q"):  # quick run - one set of values for hyperparameters
            quick_flag = True
        else:
            print('Error: Unrecognized option: {}'.format(opt))
            print('Syntax: {} -p'.format(prog_name))
            sys.exit(2)

    # Get the list of all directories in dataDir
    dFiles = [d.name for d in os.scandir(dataDir) if d.is_dir() == True]
    # fullmatch returns None if there is No match
    dirFiles = [d for d in dFiles if re.fullmatch(dirNamePattern,d)]

    # Get the most recent directory
    currentDir = max(dirFiles)
    outFile = 'Data/' + prog_name  + '-' + currentDir + '_out.txt'
    outf = open(outFile, 'w')
    print_out(outf, 'Processing files from: '+ currentDir)
    aggFile = 'aggregate-' + currentDir + '.csv'
    combFile = 'combined-' + currentDir + '.csv'
    avgFile = 'average-' + currentDir + '.csv'
    pivotFile = 'monthly-' + currentDir + '.csv'
    pivotMasterFile = 'monthly-master-' + currentDir + '.csv'
    benPlanFile = 'benPlan-' + currentDir + '.csv'
    xl_outFile = prog_name  + '-' + currentDir + '.xlsx'
    # Ensure the program does not attempt to read these files as data files
    suppressList = [aggFile, combFile, avgFile, pivotFile, pivotMasterFile, benPlanFile]
    # For some reason, need to add the directory to the path for plotFile
    plotFile = './Data/' + currentDir + '/' + prog_name + '-plots-' + currentDir + '.pdf'
    pF = PdfPages(plotFile)
    xl_writer = pd.ExcelWriter(xl_outFile)

    # Change directory to the most recent
    os.chdir(dataDir + currentDir)
    # Get a list of the CSV files - only
    csvFiles = [f.name for f in os.scandir('.') if f.name.split('.')[1] == 'csv']

    csvFiles = [f for f in csvFiles if f not in suppressList]
    # Add OLD files that no longer change: e.g Citibank
    old_files = [f.name for f in os.scandir('../OLD') if f.name.split('.')[1] == 'csv']
    csvFiles += old_files
    print_out(outf, "Files processed:\n" + repr(csvFiles))

    # cumul_df is the main array of all transactions
    cumul_df = pd.DataFrame()
    for file in csvFiles:
        df = ut.processFile(file)
        # Add to the main array
        cumul_df = cumul_df.append(df)
        # ut.printDF(cumul_df, msg="Cumul")

    # Create date label
    # cumul_df['Month'] = cumul_df['Date'].apply(lambda x: ut.date2month(x))
    # cumul_df['Month'] = cumul_df.Date.apply(lambda x: x.split('/')[2])
    # z = cumul_df['Date'][1:].apply(lambda x: ut.date2month(x))

    cumul_df['Month'] = cumul_df['Date'].apply(lambda x: ut.date2month(x,qDateFormat))
    z = cumul_df[cumul_df.Month == 'xxxx-xx']
    if len(z.index) > 0:
        print_out(outf,'rows with weird dates')
        print_out(outf,z)
    # cumul_df['Month'] = cumul_df['Date'][1:].apply(lambda x: ut.date2month(x))
    # ut.printDF(cumul_df, "Cumul")
    # ut.printDF(cumul_df, msg="Final DF", verbose=True)


    # Keep the transactions in specified date range
    firstMo = ut.date2month(firstDate, qDateFormat)
    # Cheeky to use comparison on strings, but the label Month was built for this purpose (also for plot)
    # Would have to use a time function on Date
    # lastMo is in format yyyy-mo-dd - so we need to remove '-dd' without making assumptions on the length of mo or dd
    lastMo = ut.date2month(currentDir, dirDateFormat)
    print_out(outf, 'First Mo: {} - Last Mo: {}'.format(firstMo, lastMo))
    cumul_df = cumul_df[cumul_df.Month >= firstMo]
    # Note the < ... we want to exclude the last month, since it is incomplete
    cumul_df = cumul_df[cumul_df.Month < lastMo]
    # ut.printDF(cumul_df, msg="Good Dates", verbose=False)


    # Convert Amount to Float so that it can be added up - Note that none of the 2 approaches below work
    # E.g Converting to float pukes on commas (e.g 1,599.60)!!
    # cumul_df['Amount'] =cumul_df.Amount.apply(lambda x: x.strip(','))
    # cumul_df['Amount'] =cumul_df['Amount'].astype(str).replace(',','').astype(float)
    cumul_df['Amount'] = cumul_df['Amount'].apply(lambda x: float(str(x).replace(',', '')))

    # Flag Uncategorized transactions
    tmpdf = cumul_df[cumul_df.Category == 'Uncategorized']
    if len(tmpdf.index) > 0:
        print_out(outf, str(len(tmpdf.index)) + ' Uncategorized Transactions:\n' + repr(tmpdf)+'\n')

    # Remove sub-categories
    cumul_df = remap_sub_cat(cumul_df, masterCatFile, outf)
    # cumul_df.to_csv(combFile, float_format='%.2f', header=True)
    cumul_df.to_excel(xl_writer, sheet_name="Combined", float_format='%.2f', header=True)
    print_out(outf, 'Categories:\n')
    print_out(outf, repr(sorted(set(cumul_df['Category']))))
    # ut.printDF(cumul_df, msg='cumul_df', verbose=True)

    # Group / aggregate transactions by month & category
    # Use 'as_index=False' so that we get a dataframe rather than a series
    agg = cumul_df.groupby(['Month', 'Category', 'MasterCat', 'BenPlan'],as_index=False)['Amount'].sum()
    # ut.printDF(agg, msg='agg', verbose=True)
    # agg.sort_values(['Amount'], axis=1, ascending=False, inplace=True, na_position='last')
    # agg =  agg[np.argsort(-agg)] # Trick to sort in reverse order, which sort does not support
    # agg.to_csv(aggFile, float_format='%.2f', header=True)
    agg.to_excel(xl_writer, sheet_name="Aggregate", float_format='%.2f', header=True)


    # Create Pivot Table for Master Categories
    pivot = pd.pivot_table(agg, index=['MasterCat'], values =['Amount'], columns=['Month'])
    # Fill in NA with $0
    pivot.fillna(0.0, inplace=True)
    # Get rid of the multi-level index: pivot.columns.get_level_values(0) = ['Amount', 'Amount', ...]
    pivot.columns = pivot.columns.get_level_values(1)
    # ut.printDF(pivot, msg='pivot', verbose=True)
    # Compute the overall average Yearly expense
    pivot['YrAvg'] = 12.0 * pivot.mean(axis=1)
    # last 12 months average - i.e. sum of last 12
    pivot['12moAvg'] = pivot.iloc[:,-13:-1].sum(axis=1)  # Skip YrAvg column
    pivot['Yearly Average'] =  pivot['YrAvg'].apply(to_dollar_str) # make it pretty
    pivot['Last 12 months Average'] =  pivot['12moAvg'].apply(to_dollar_str)
    print_out(outf, 'Master Categories Averages\n')
    print_out(outf, repr(pivot[['Yearly Average','Last 12 months Average']]))
    pivot.to_csv(pivotMasterFile, float_format='%.2f', header=True)

    pivot.drop(['YrAvg','12moAvg','Yearly Average', 'Last 12 months Average'], axis=1, inplace=True) # only keep the data
    lineplotDF(pivot, title = 'Monthly Expenses by Master Categories', pF=pF)
    # ---------------------------------------------------
    # Create Pivot Table for  BenPlan
    # Only keep the 'OngoingExpenseCat' master category
    # Need to make deep copy
    agg3 = agg.loc[agg['BenPlan'] == 1].copy(deep=True)
    # Turn expenses to a positive humber
    agg3['Amount'] *= -1

    pivot = pd.pivot_table(agg3, index=['Category'], values =['Amount'], columns=['Month'])
    # Fill in NA with $0
    pivot.fillna(0.0, inplace=True)
    # Get rid of the multi-level index: pivot.columns.get_level_values(0) = ['Amount', 'Amount', ...]
    pivot.columns = pivot.columns.get_level_values(1)
    pivot['YrAvg'] = 12.0 * pivot.mean(axis=1)
    # last 12 months average
    pivot['12moAvg'] = pivot.iloc[:,-13:-1].sum(axis=1)  # Skip YrAvg column
    pivot.sort_values('12moAvg', axis=0, ascending=False,inplace=True, na_position='last')

    # Create an array for monthly  expenses
    monthly=pivot.sum(axis=0)
    monthlyDF = pd.DataFrame(monthly)
    monthlyDF.drop(['YrAvg','12moAvg'], inplace=True)
    monthlyDF.columns=['Monthly BenPlan Expenses']
    monthly_mean = monthlyDF['Monthly BenPlan Expenses'].mean()
    monthly_std = monthlyDF['Monthly BenPlan Expenses'].std()
    print_out(outf, 'Benplan Expenses Average (annualized): ${:,.2f} +/ ${:,.2f} (stddev)'.format(12*monthly_mean, 12*monthly_std))
    df2= monthlyDF.iloc[-12:,]  # get the most recent 12 months
    monthly_mean_12 = df2['Monthly BenPlan Expenses'].mean()
    monthly_std_12 = df2['Monthly BenPlan Expenses'].std()
    print_out(outf, 'Last 12 Months Benplan Expenses Average (annualized): ${:,.2f} +/ ${:,.2f} (stddev)'.format(12*monthly_mean_12, 12*monthly_std_12))
    plt.figure(figsize=(8,6))
    ax=plt.gca()
    x_tick = range(len(monthlyDF.index))
    plt.plot(x_tick, monthlyDF['Monthly BenPlan Expenses'],color='steelblue')
    # Plot lines for mean & rails for stddev
    # Build array of containing the mean value for all entries - as a Reference
    mean_array=np.array([monthly_mean for i in range(len(monthlyDF.index))])
    plt.plot(x_tick, mean_array, color='red')
    std_array=np.array([(monthly_mean + monthly_std) for i in range(len(monthlyDF.index))])
    plt.plot(x_tick, std_array, color='red',ls='dashed',alpha=0.5)
    std_array=np.array([(monthly_mean - monthly_std) for i in range(len(monthlyDF.index))])
    plt.plot(x_tick, std_array, color='red',ls='dashed',alpha=0.5)
    # Decorate plot w/ labels, title etc
    plt.xticks(x_tick, list(monthlyDF.index), rotation=270)
    plt.title('Monthly BenPlan Expenses')
    ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor('#E6E6E6')
    # Shrink current axis by 15% to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    plt.grid(axis='y', which='both', color='darkgreen', linestyle='-', linewidth=1, alpha=0.5)
    plt.savefig(pF, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    # Add a final row with monthly totals
    # Need to do this here so as not to screw up summaries & plot
    cat_list = list(pivot.index)
    pivot = pivot.append(pivot.sum(axis=0), ignore_index=True)  # Add Monthly totals
    cat_list += ['TOTAL']
    pivot['Category'] = pd.Series(cat_list)
    pivot.set_index('Category', drop=True, inplace=True)
    # pivot.to_csv(benPlanFile, float_format='%.2f', header=True)
    pivot.to_excel(xl_writer, sheet_name="BenPlan", float_format='%.2f', header=True)

    # ---------------------------------------------------
    # Create Pivot Table for Categories
    # Only keep the 'OngoingExpenseCat' master category
    # Need to make a deep copy
    agg2 = agg.loc[agg['MasterCat'] == 'OngoingExpenseCat'].copy(deep=True)
    # Turn expenses to a positive humber
    agg2['Amount'] *= -1.0

    pivot = pd.pivot_table(agg2, index=['Category'], values =['Amount'], columns=['Month'])
    # Fill in NA with $0
    pivot.fillna(0.0, inplace=True)
    # Get rid of the multi-level index: pivot.columns.get_level_values(0) = ['Amount', 'Amount', ...]
    pivot.columns = pivot.columns.get_level_values(1)
    nb_month = len(pivot.columns) # Number of months for which we have data
    print_out(outf, '#Months:' + str(nb_month))
    # ut.printDF(pivot, msg='pivot', verbose=True)
    # Compute the overall average Yearly expense
    pivot['YrAvg'] = 12.0 * pivot.mean(axis=1)
    # last 12 months average
    pivot['12moAvg'] = pivot.iloc[:,-13:-1].sum(axis=1)  # Skip YrAvg column
    pivot.sort_values('YrAvg', axis=0, ascending=False,inplace=True, na_position='last')
    # ut.printDF(pivot, msg='pivot', verbose=True)
    # pivot.to_csv(pivotFile, float_format='%.2f', header=True)
    pivot.to_excel(xl_writer, sheet_name="Monthly", float_format='%.2f', header=True)

    # Create an array for monthly  expenses
    monthly=pivot.sum(axis=0)
    monthlyDF = pd.DataFrame(monthly)
    monthlyDF.drop(['YrAvg','12moAvg'], inplace=True)
    monthlyDF.columns=['Monthly Expenses']
    plt.figure(figsize=(8,6))
    ax=plt.gca()
    x_tick = range(len(monthlyDF.index))
    plt.plot(x_tick, monthlyDF['Monthly Expenses'],color='steelblue')
    # Plot lines for mean & rails for stddev
    monthly_mean = monthlyDF['Monthly Expenses'].mean()
    monthly_std = monthlyDF['Monthly Expenses'].std()
    # Build array of containing the mean value for all entries - as a Reference
    mean_array=np.array([monthly_mean for i in range(len(monthlyDF.index))])
    plt.plot(x_tick, mean_array, color='red')
    std_array=np.array([(monthly_mean + monthly_std) for i in range(len(monthlyDF.index))])
    plt.plot(x_tick, std_array, color='red',ls='dashed',alpha=0.5)
    std_array=np.array([(monthly_mean - monthly_std) for i in range(len(monthlyDF.index))])
    plt.plot(x_tick, std_array, color='red',ls='dashed',alpha=0.5)
    # Decorate plot w/ labels, title etc
    plt.xticks(x_tick, list(monthlyDF.index), rotation=270)
    plt.title('Monthly Expenses')
    ax.yaxis.set_major_formatter(FuncFormatter(tick_format))
    ax.set_facecolor('#E6E6E6')
    # Shrink current axis by 15% to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    plt.grid(axis='y', which='both', color='darkgreen', linestyle='-', linewidth=1, alpha=0.5)
    plt.savefig(pF, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    # ---------------------------------------------------

    # Create a new summary array with top N category and a N+1 "Other" summing up the rest
    # need deeop copy, so that the drop works
    pivot['Yearly Average'] = pivot['YrAvg'].apply(to_dollar_str) # make it pretty
    pivot['Last 12 months Average'] =  pivot['12moAvg'].apply(to_dollar_str)
    print_out(outf, 'Averages for Categories\n' + repr(pivot[['Yearly Average','Last 12 months Average']]))
    avg_series = pivot['YrAvg'].copy(deep=True)
    pivot.drop(['YrAvg','12moAvg','Yearly Average', 'Last 12 months Average'], axis=1, inplace=True) # only keep the data
    # Figure out the categories which have negative value
    neg = avg_series[avg_series < 0].index.values
    if len(neg) > 0:
        print_out(outf, "Heads up: {} categories are still negative".format(repr(neg)))

    # Make a pretty version and save it
    # avg_series.to_csv(avgFile, header=True)
    avg_series.to_excel(xl_writer, sheet_name="Averages", float_format='%.2f', header=True)

    # Keep categories that contribute to 80% of expenses
    total_spend = avg_series.sum()
    print_out(outf, 'Average Annualized Total Spend: ' + to_dollar_str(total_spend))
    # Find the number of categories that cover 80% of expenses: other_idx
    for other_idx in range(len(avg_series)):
        other_spend = avg_series.iloc[other_idx:].sum()  # Accumulate the tail
        if other_spend / total_spend <= other_pct:
            # print_out(outf, 'other_idx:', other_idx, 'other_spend =', toDollarStr(other_spend))
            break
    # Relabel the last entry and assign it the cumulative tail spend
    avg_series.index.values[other_idx] = 'Other'
    avg_series.iloc[other_idx] = other_spend
    avg_series = avg_series.iloc[0:other_idx+1]
    avg_pct = avg_series / total_spend
    # Make it pretty by formatting the values
    # print(avg_series.apply(lambda x: toDollarStr(x)))
    print_out(outf, 'TopN Categories Average (annualized)\n' + repr(avg_series.apply(to_dollar_str)))

    # Add-up the smaller categories into 'Other'
    pivot.iloc[other_idx]=pivot.iloc[other_idx:].sum()
    # truncate to only keep Top N - including Other
    pivot = pivot.iloc[0:other_idx+1]

    plt.style.use('seaborn-deep')
    # Plot pie chart for top N average overall
    plt.figure(figsize=(8, 6))
    ax=plt.gca()
    plt.title('Monthly Expenses for TopN Categories - %')
    plt.pie(avg_pct.values, labels=avg_pct.index.values, autopct='%1.1f%%')
    ax.set_facecolor('#E6E6E6')
    plt.savefig(pF, format='pdf', dpi=300)
    # plt.show()

    lineplotDF(pivot, title = 'Monthly Expenses for TopN Categories', pF=pF)

    # Plot the trailing 12-month  yearly expenses
    # Create a DataFrame with the same categories
    run_avg=pd.DataFrame(pivot.index)
    run_avg.set_index('Category',drop=True,inplace=True) # same index as pivot
    col=list(pivot.columns)
    newcol = col[11:] # start at the 12th month - need 12 month of data
    for m, idx in zip(newcol, range(12, 12+len(newcol))):
        # sum by row the 12 columns, where the last one is month m
        run_avg[m] = pivot[col[idx-12:idx]].sum(axis=1)
    # ut.printDF(run_avg, msg='Trailing 12-month Yearly Expenses', verbose=True)
    lineplotDF(run_avg, title = 'Trailing 12-month Yearly Expenses for TopN Categories', pF=pF)


    # Plot the trailing quarterly expenses - scaled to yearly
    # Create a DataFrame with the same categories
    qtrly=pd.DataFrame(pivot.index)
    qtrly.set_index('Category',drop=True,inplace=True) # same index as pivot
    col=list(pivot.columns)
    # quarters = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
    quarter = {3: 'Q1', 6: 'Q2', 9: 'Q3', 12: 'Q4'}
    q_list =[]
    m_cnt = 0
    for m in col:
        mNb = int(m.split('-')[1])  # get month as a number
        q_list.append(m)
        m_cnt+=1
        # end of quarter or last month in series
        if mNb in quarter.keys() or m == col[len(col)-1]:  # end of quarter - add it up
            yr = m.split('-')[0] # get the year string
            # Handle the special case of last month of series to find the quarter
            while mNb not in quarter.keys():
                mNb += 1
            qtr = yr + '-' + quarter[mNb] # Build column name: yyyy-Qx
            # Add values for months available in that quarter - and scale to Yearly
            qtrly[qtr] = pivot[q_list].sum(axis=1) * 12/m_cnt
            # Reset the counters
            q_list = []
            m_cnt = 0
        # sum by row the 12 columns, where the last one is month m
    # ut.printDF(qtrly, msg='Quarterly  Expenses (scaled to yearly)', verbose=True)
    lineplotDF(qtrly, title = 'Quarterly Expenses for TopN Categories (annualized))', pF=pF)

    # Wrap up
    xl_writer.save()
    pF.close()
    outf.close()

    end_time = dt.datetime.now()
    logger.debug('\nEnd: {}\n'.format(str(end_time)))
    logger.debug('Run Time: {}\n'.format(str(end_time-start_time)))
    return


if __name__ == "__main__":

    # execute only if run as a script
    main(sys.argv)
    exit(0)
