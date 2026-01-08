#!/usr/bin/env python

"""
Compare expenses by Category during 2 periods
Option:
-s YYYY-MM: start month
-e YYYY-MM: end month
-d N: length of reference period - default: same length as target period
-r YYYY-MM: last month of the reference period - default: the month before the start month
Example:
    compare -s 2020-04 -e 2020-06  => compares spending between Apr - June 2020 and Jan - March 2020  (3 months each)
    compare -s 2020-04 -e 2020-06 -d 12  => compares spending between Apr - June 2020 and Apr 2019 - March 2020  (12
    months)
    compare -s 2020-04 -e 2020-06 -d 12 -r 2020-06 => compares spending between Apr - June 2020 and Apr - June 2019  (12
    months ending on June 2020)
"""

import re
import sys
import os
import getopt
import pandas as pd
import numpy as np

start_month = None
end_month = None
ref_start = None
ref_end = None
ref_duration = None
PROG_NAME = ''
SYNTAX: f"Syntax: {PROG_NAME} -s start_month -e end_month [-d reference_duration] [-r reference_end_month]"
MONTH_FORMAT = "\d\d\d\d-\d\d"  # format for month
DATA_DIR = "./Data"
DIR_NAME_PATTERN = '\d{4}-\d{2}-\d{2}'


def yyyy_mm_2_yr_mo(yyyy_mm: str) -> (int, int):
    """ Converts string YYYY-MM to int 2-uple: (YYYY, MM)"""
    if not re.match(MONTH_FORMAT, yyyy_mm):
        sys.exit("Months must be entered as: YYYY-MM")
    y, m = yyyy_mm.split('-')
    return int(y), int(m)


def yr_mo_2_yyyy_mm(yr: int, mo: int) -> str:
    if mo >= 10:
        return f"{yr:d}-{mo:d}"
    else:
        return f"{yr:d}-0{mo:d}"

def generate_yyyy_mm_labels(start_month: str, end_month: str) -> [str]:
    """ Generates a list of labels in the form YYYY-MM for all months between start_mo and end_mo """
    start_yr, start_mo = yyyy_mm_2_yr_mo(start_month)
    end_yr, end_mo = yyyy_mm_2_yr_mo(end_month)
    label_lst = [start_month]
    label_yr = start_yr
    label_mo = start_mo
    while True:
        label_mo += 1
        if label_mo == 13:
            label_mo = 1
            label_yr += 1
        label_month = yr_mo_2_yyyy_mm(label_yr, label_mo)
        label_lst.append(label_month)
        if label_month == end_month:
            break
    return label_lst


def compute_and_check_parameters(argv: [str]) -> (str, str, int, str, str, int):
    # set these parameters as None, so that we can validate that they have been set
    start_month = None
    end_month = None
    ref_duration = None
    ref_end = None

    try:
        #   opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
        opts, args = getopt.getopt(argv[1:], "hpqs:e:d:r:")
    except getopt.GetoptError:
        print(SYNTAX)
        sys.exit("Exception parsing options")
    for opt, arg in opts:
        if opt == '-h':
            print(SYNTAX)
            sys.exit()
        elif opt in "-p":  # show plots on display
            plot_flag = True
        elif opt in "-q":  # quick run - one set of values for hyperparameters
            quick_flag = True
        elif opt in "-s":  # quick run - one set of values for hyperparameters
            start_month = arg
        elif opt in "-e":  # quick run - one set of values for hyperparameters
            end_month = arg
        elif opt in "-d":  # quick run - one set of values for hyperparameters
            ref_duration = int(arg)
        elif opt in "-r":  # quick run - one set of values for hyperparameters
            ref_end = arg
        else:
            print(f"Error: Unrecognized option: {opt}")
            print(SYNTAX)
            sys.exit("Unknown option")

    assert start_month and end_month, f"Both start and end month must be specified via -s and -e options"
    # Make sure we have all the parameters we need
    if not start_month or not end_month:
        sys.exit("Both start and end month need to be specified")
    start_yr, start_mo = yyyy_mm_2_yr_mo(start_month)
    end_yr, end_mo = yyyy_mm_2_yr_mo(end_month)
    target_duration = 12 * (end_yr - start_yr) + end_mo - start_mo + 1
    if start_mo > end_mo:
        target_duration -= 12
    if not ref_duration:
        ref_duration = target_duration

    assert target_duration > 0, f"Target duration ({target_duration}) must be > 0"
    assert ref_duration > 0, f"Reference_duration ({ref_duration}) must be > 0"

    if not ref_end:
        if start_mo == 1:
            ref_end_mo = 12
            ref_end_yr = start_yr - 1
        else:
            ref_end_mo = start_mo - 1
            ref_end_yr = start_yr
    else:
        ref_end_yr, ref_end_mo = yyyy_mm_2_yr_mo(ref_end)

    ref_start_mo = ref_end_mo - ref_duration + 1
    ref_start_yr = ref_end_yr
    if ref_start_mo <= 0:
        ref_start_mo += 12
        ref_start_yr -= 1

    ref_start = yr_mo_2_yyyy_mm(ref_start_yr, ref_start_mo)
    ref_end = yr_mo_2_yyyy_mm(ref_end_yr, ref_end_mo)

    return start_month, end_month, target_duration, ref_start, ref_end, ref_duration


# ---- MAIN
def main(argv):
    global PROG_NAME
    global plot_flag, quick_flag

    # get program name by removing './' at start and '.py' at end
    prog_name = re.sub('^\.\/', '', argv[0])
    prog_name = re.sub('\\.py$', '', prog_name)

    start_month, end_month, target_duration, ref_start, ref_end, ref_duration = compute_and_check_parameters(argv)
    print(f"start_month : {start_month} - end_month: {end_month} - target_duration: {target_duration}")
    print(f"ref_start: {ref_start} - ref_end: {ref_end} - ref_duration: {ref_duration}")

    # Read the data from the most recent BenPlan
    # Get the list of all directories in DATA_DIR
    d_files = [d.name for d in os.scandir(DATA_DIR) if d.is_dir() is True]
    # fullmatch returns None if there is No match
    dir_files = [d for d in d_files if re.fullmatch(DIR_NAME_PATTERN, d)]
    # Get the most recent directory
    current_dir = max(dir_files)
    in_file_name = DATA_DIR + '/' + current_dir + '/Benplan-' + current_dir + '.xlsx'
    print(f"Reading data from: {in_file_name}")
    df_benplan = pd.read_excel(in_file_name, sheet_name="BenPlan Categories", index_col=0, header=0, engine='openpyxl')
    print(df_benplan.head())

    # Copy relevant data to 2 DF: target and reference
    target_lst = generate_yyyy_mm_labels(start_month, end_month)
    ref_lst = generate_yyyy_mm_labels(ref_start, ref_end)
    print(f"Target Labels: {target_lst}")
    print(f"Reference Labels: {ref_lst}")
    target_df = df_benplan[target_lst].copy(deep=True)
    ref_df = df_benplan[ref_lst].copy(deep=True)
    compare_df = pd.DataFrame(index=df_benplan.index)
    compare_df['Target'] = np.mean(target_df, axis=1)
    compare_df['Reference'] = np.mean(ref_df, axis=1)
    compare_df['Delta'] = compare_df['Target'] - compare_df['Reference']
    compare_df['Delta-%'] = compare_df.apply(lambda x: 100*x['Delta']/x['Reference'], axis=1)
    compare_df.sort_values(by='Delta-%', axis=0, ascending=False, inplace=True)
    # make it pretty for printing
    display_df = compare_df.applymap(lambda x: f"{x:,.0f}")
    print(f"Comparison Results:\n{display_df}")

    return


if __name__ == "__main__":
    main(sys.argv)
    exit(0)
