#!/usr/local/bin/python3

"""
Implements algorithm to adapt discretionary spending in order to always have money at the end
# ToDo: reconcile START_FUNDS with "initial" Assets in Goals DF ... should be only 1
"""

import matplotlib.pyplot as plt
import sys
import getopt
import datetime as dt
from pprint import pformat
from pathlib import Path
import tomli
import numpy as np
import pandas as pd
import math
import mc_new_utilities as UT
import plot_color_matrix as plot_c_m

plt.style.use('seaborn-v0_8-deep')
ProgName = ''  # Place holder for program name
goals_col_2_keep = ['Amount', 'Start_age', 'End_age', 'Inflation', 'Discretionary']
DEBUG = False


def get_params(cmd_args: [str]) -> dict:
    usage_str = f"Usage: {ProgName} -t toml_file\n"
    # Check if any of the parameters are overridden by command-line flags
    cmd_line_param_dict = dict()
    try:
        opts, args = getopt.getopt(cmd_args[1:], "ht:")
    except getopt.GetoptError:
        print(f'Error: Unrecognized option in: {" ".join(cmd_args)}')
        print(f"{usage_str}\n{__doc__}")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(f"{usage_str}\n{__doc__}")
            sys.exit(2)
        elif opt in "-t":  # different config file
            cmd_line_param_dict['TOML_FILE'] = arg
        else:
            print(f'Error: Unrecognized option: {opt}')
            print(f"{usage_str}\n{__doc__}")
            sys.exit(2)

    # Read parameters from JSON file
    # print_out(outf, f"Using JSON file: {config_file}")
    config_file = './src/' + ProgName + ".toml"
    config_file = cmd_line_param_dict.get("TOML_FILE", config_file)  # use config_file from command line, if specified
    print(f"Using TOML file: {config_file}")
    with open(config_file, mode='rb') as fp:
        param_dict = tomli.load(fp)
    # Create flag to indicate whether we iterate on initial funds

    # Override, or add, the params from config file with values from command line
    for ky in cmd_line_param_dict.keys():
        param_dict['config'][ky] = cmd_line_param_dict[ky]

    return param_dict

def cumulative_interest(rate: float, nb_period: int) -> float:
    return (1 - pow(rate, nb_period + 1)) / (1 - rate)

def pct_to_save(years_saving: int, yrs_retirement: int, ret_work_ratio: float, work_int_rate: float, ret_int_rate:
    float) -> float:

    monthly_work_rate = work_int_rate / 12
    monthly_ret_rate = ret_int_rate / 12
    months_retirement = yrs_retirement * 12
    months_saving = years_saving * 12

    ### Compute money that needs to be accumulated by start of retirement
    # Compute monthly expense during retirement
    avail_income_ret_month = ret_work_ratio / 12
    x = 1.0 / (1.0 + monthly_ret_rate)
    pot_of_gold = avail_income_ret_month * cumulative_interest(x, months_retirement)
    print(f"pot of gold = {pot_of_gold:,.2f}")

    x = 1.0 + monthly_work_rate
    monthly_savings = pot_of_gold / cumulative_interest(x, months_saving)
    # message(monthly_savings)
    pct_income_to_save = 1200 * monthly_savings  # Percent of yearly income to save
    return pct_income_to_save


def fff(yrs:int) -> float:
    """  fff fixes all the paremeters to the inputs selected - except number of years"""



def main(argv: [str]) -> None:
    """
    Read the paramaters, financial goals (including inflation projection) - the financial model and get the Stock
    Statistics
    Run the simulation
    Print the results
    @param argv: command line arguments
    @return: None
    """
    # Get configuration parameters
    param_dict = get_params(argv)['default']
    print(f"Parameters:\n{pformat(param_dict)}")
    xl_file = ProgName + "_out.xlsx"
    plt_file = ProgName + "_out.pdf"
    years_retirement = param_dict['END_PLAN_AGE'] - param_dict['RETIREMENT_AGE']
    retirement_work_ratio = param_dict['RETIREMENT_SPENDING_RATE']  # Ratio of expenditures when retired vs when working
    work_interest_rate =  param_dict['ROR_WORKING']  # Rate of return during working period
    ret_interest_rate = param_dict['ROR_RETIREMENT']  # Rate of return during retirement period

    #
    # Main
    #
    print(f"Years of retirement:  {years_retirement}")
    print(f"Ratio of retirement income to income while working: {retirement_work_ratio}")
    print(f"Returns earned during working period: {100.0 * work_interest_rate}%")
    print(f"Returns earned during retirement period: {100.0 * ret_interest_rate}%")

    out_df = pd.DataFrame(columns=["Years Saving", "Pct to Save", "Percentage of Income to Save"])
    for years in range(1, 41):
        pct_2_save = pct_to_save(years, years_retirement, retirement_work_ratio, work_interest_rate, ret_interest_rate)
        out_df.loc[years-1] = [years, pct_2_save, f"{pct_2_save:,.2f}%"]
    # Drop entries where the savings % is greater than 100% - i.e. impossible
    out_df.drop(out_df[out_df["Pct to Save"] > 100].index, inplace=True, axis=0)
    # out_df.sort_values("Pct to Save", inplace=True, ignore_index=True)
    out_df["Starting Age"] = param_dict['RETIREMENT_AGE'] - out_df['Years Saving']
    out_df.sort_values("Starting Age", inplace=True, ignore_index=True)
    print(f"Results:\n{out_df}")
    with pd.ExcelWriter(xl_file, mode='w', engine="openpyxl") as xl_wr:
        out_df.to_excel(xl_wr, sheet_name="Financing Retirement", header=True, index=False)

    # Plot
    plt.figure(figsize=[11, 8.5], num=f"Percent of Income to Save by Years of Saving", clear=True)
    plt.title(f"Percent of Income to Save by Years of Saving\nRetirement Age = {param_dict['RETIREMENT_AGE']}")
    plt.plot(list(out_df["Starting Age"]), list(out_df["Pct to Save"]))
    min_x = 5 * math.floor(min(out_df["Starting Age"]) / 5)
    max_x = 5 * math.ceil(max(out_df["Starting Age"]) / 5)
    plt.xlim(min_x, max_x)  # keep the scale constant for all iterations 0%-100%
    plt.ylim(0, 100)  # keep the scale constant for all iterations 0%-100%
    df_5 = out_df[out_df['Starting Age'] % 5 == 0].copy(deep=True)
    df_5.set_index('Starting Age', inplace=True, drop=True)
    plt.plot(list(df_5.index), list(df_5["Pct to Save"]), 'o', color='red')
    for yr in df_5.index:  # Adding the values on the plot
        # Using +1 and -1 to offset the labels from the curve
        plt.text(yr+1, df_5.at[yr, "Pct to Save"] -1 , df_5.at[yr, "Percentage of Income to Save"])
    plt.xlabel("Starting Age")
    plt.ylabel("Percent to Save per Year (%)")
    plt.grid()
    plt.savefig(plt_file, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()  # FYI - Use backend=WebAgg in matplotlibrc
    return


if __name__ == "__main__":
    # execute only if run as a script
    ProgName = Path(sys.argv[0]).stem
    main(sys.argv)
    sys.exit("All Done")
