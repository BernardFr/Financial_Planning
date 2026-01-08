"""
Compare the current financial status with the Envision plan

Investment Portfolio needs are:  - Payments to PCL (interest and Principal) - Partech Investment outflow + Partech
Principal income (+ Misc)
PCL_Cashflow = Payments to PCL - Distributions to BF - Interest accrued on PCL balance
Distributions to BF = retirement spending + Home Improvement + Kayla + 2 mortgages + Travel - CalPers - consulting -
Harlan Net - PPP Net .... + Income Taxes (if we want to keep track separately)

"""

import typing

import pandas as pd

from utils import make_dollar_pretty, my_age, print_out, this_function
HOME_DIR = '/Users/bfraenkel/Documents/Code/BenPlan/'
DATA_DIR = HOME_DIR + '/Data/'
ENVISION_FILE = DATA_DIR + 'Envision/Envision_Projections.xlsx'
HOUSEHOLD_INCOME_TAX = -7500.0  # note this is a negative number

# Aggregate some BlenPlan rows into a single row to match Envision
DEL_ROWS = ['Alimony', 'Dad', 'Ignore', 'Julia', 'Transfer', 'Family_Loan', 'Work_Exp']
# BP_NET_CASHFLOW & ENV_CASHFLOW should be the same
SPENDING_DISCRETIONARY = ['Consulting_Income', 'Kayla', 'Remodel', 'Rtmt_Spending', 'Travel']
BP_NET_CASHFLOW = SPENDING_DISCRETIONARY + ['CalPERS', 'Emryvil_Mortgage', 'Loft_Rent', 'Mortgage',
                                               'PPP_Rent', 'Invest_Income']

ENV_NET_CASHFLOW = SPENDING_DISCRETIONARY + ['CalPERS', 'Emryvil_Mortgage', 'Loft_Rent', 'Mortgage',
                                                 'Social Security - Bernard']

PCL_FLOW =  ['Pay_off_PCL', 'XTRA']  # Net flows in/out PCL
# Group all the VC investment rows into a single row -> VC_Invest
VC_INVEST_ROWS = ['Other_VC', 'Partech_I', 'Partech_II', 'Partech III', 'Partech I & II', 'Partech IV']
ENV_VC_PRINCIPAL = ['Partech I Principal Return', 'Partech II Principal Return', 'Partech III Principal Return']
# Net flows for VC drawdowns and distributions
VC_FLOW = ['VC_Principal', 'VC_Invest']
#  Compute Investment Portfolio Need from BenPlan / actual data
# Invest_Portfolio_Need
# Aggregate some Envision rows into a single row to match BenPlan
# These are all the $$ flows in and out of the investment portfolio
BP_Invest_Portfolio_Need = VC_FLOW + BP_NET_CASHFLOW + \
                            ['Pay_off_PCL', 'Vanguard', 'Cornelia', 'PPP_Sale']
# These are one time events
One_Time_Transactions = ['Cornelia', 'PPP_Sale']
# These moving to different accounts but not affecting the global asset value
# Global_Asset_Delta = BP_Invest_Portfolio_Need - Net_Portfolio_Need_Adjust ... Cumulative delta across all asset
# accounts: Vanguard, PCL, Investment accounts
Net_Portfolio_Need_Adjust = ['Pay_off_PCL', 'Vanguard']  

SUMMARY_ROWS = ['Spending_Discretionary', 'Net_Cashflow', 'XTRA', 'Vanguard', 'Income_Tax_Investing', 'PCL_Flow',
                'VC_Flow', 'Invest_Portfolio_Need', 'Regular_Invest_Portfolio_Need',
                'Global_Asset_Delta', 'Regular_Global_Asset_Delta']

bp_suffix = '_actual'  # Suffix for data coming from BenPlan
env_suffix = '_envsn'  # Suffix for data coming from Envision
delta_suffix = '_delta'  # Suffix for delta data


Envision_Notes = {
"CalPERS": "CalPERS retirement income",
"Consulting_Income": "Consulting income",
"Cornelia": "Cornelia Trust one time income 2024",
"Emryvil_Mortgage": "Loft Mortgage",
"Invest_Income": "Investment income from other sources e.g. Lending Club",
"Kayla": "Kayla education",
"Loft_Rent": "Loft rent net of expenses",
"Mortgage": "Mortgage - House",
"PCL_Interest": "Interest accrued on PCL",
"PPP_Rent": "PPP Rent net of expenses",
"PPP_Sale": "PPP Sale one time income 2024",
"Pay_off_PCL": "Payments to/from PCL from/to WF accounts",
"Remodel": "Remodel - House",
"Rtmt_Spending": "Retirement/Household spending",
"Social Security - Bernard": "Social Security - Bernard",
"Travel": "Travel",
"VC_Invest": "VC drawdowns",
"VC_Principal": "VC distributions",
"Spending_Discretionary": "Aggregated discretionary lifestyple spending - ideally equal to Envision value",
"Net_Cashflow": "Aggregated cashflow in the personal accounts, discretionary and mandatory (mortgage, rentals, etc)",
"XTRA": "Net flow between PCL and personal accounts",
"Vanguard": "Contributions to Vanguard 401K",
"Income_Tax_Investing": "Federal income tax on WF investment income",
"PCL_Flow": "Net flow in/out of PCL, including interest accrued",
"VC_Flow": "Net VC drawdowns and distributions in investment accounts",
"Invest_Portfolio_Need": "Net amount of money that comes out of the investment accounts.",
"Regular_Invest_Portfolio_Need": "Net amount of money that comes out of the investment accounts - "
                         "excluding 1-time events",
"Global_Asset_Delta": "Cumulative delta across all asset accounts: Vanguard, PCL, Investment accounts",
"Regular_Global_Asset_Delta": "Cumulative delta across all asset accounts: Vanguard, PCL, Investment "
                      "accounts - excluding 1-time events"
}



def sub_total(df1: pd.DataFrame, label_lst1: [str], df2: pd.DataFrame, label_lst2: [str], label: str, axs: int) -> None:
    """
    Sum the rows/cols in label_lst1 and label_lst2 and put the result in label for df1 and df2
    axs = 0 for summing rows, 1 for summing columns
    """
    # Make sure the label is in the index
    lst1 = [x for x in label_lst1 if x in list(df1.index)]
    lst2 = [x for x in label_lst2 if x in list(df2.index)]

    if len(lst1) != len(label_lst1):
        missing = [x for x in label_lst1 if not x in list(df1.index)]
        print(f"Warning: {this_function()} for {label}: label_lst1 labels are not in df1 {missing}")
    if len(lst2) != len(label_lst2):
        missing = [x for x in label_lst2 if not x in list(df2.index)]
        print(f"Warning: {this_function()} for {label}: label_lst2 labels are not in df2 {missing}")
    if len(lst1) == 0:
        df1.loc[label] = 0.0
    else:
        df1.loc[label] = df1.loc[lst1].sum(axis=axs)
    if len(lst2) == 0:
        df2.loc[label] = 0.0
    else:
        df2.loc[label] = df2.loc[lst2].sum(axis=axs)
    return


def build_delta_df(bp_df: pd.DataFrame, ev_df: pd.DataFrame, outf: typing.TextIO) -> pd.DataFrame:
    """ Merge BenPlan and Envision dataframes and compute the delta for the available ages """
    # Remove the TOTAL row from both dataframes - if it exists in either
    if 'TOTAL' in list(bp_df.index):
        bp_df = bp_df.drop('TOTAL')
    if 'TOTAL' in list(ev_df.index):
        ev_df = ev_df.drop('TOTAL')
    # Add suffix to column names of both arrays to distinguish source
    bp_ages = list(bp_df.columns)
    ev_ages = list(ev_df.columns)
    new_names = [str(x) + bp_suffix for x in bp_df.columns]
    bp_df.columns = new_names
    new_names = [str(x) + env_suffix for x in ev_df.columns]
    ev_df.columns = new_names
    delta_df = bp_df.merge(ev_df, how='outer', left_index=True, right_index=True, copy=True)
    delta_df.fillna(0.0, inplace=True)  # replace the NaN

    # Re-order delta_df so that the summary roq are at the end
    index_old = list(delta_df.index)
    index_new = [x for x in index_old if not x in SUMMARY_ROWS]
    index_new += SUMMARY_ROWS
    delta_df = delta_df.reindex(index_new)

    # Find ages for which we have data in both BenPlan and Envision
    common_ages = [x for x in bp_ages if x in ev_ages]
    assert len(common_ages) > 0, f"{this_function()}: No common ages found between BenPlan and Envision"
    # sort to start with the oldest age
    common_ages.sort(reverse=True)
    # Compute the delta for the common ages
    summary_col = []
    for age in common_ages:
        lft = str(age) + bp_suffix
        rgt = str(age) + env_suffix
        col = str(age) + delta_suffix
        delta_df[col] = delta_df[lft] - delta_df[rgt]
        summary_col += [col, lft, rgt]

    # Re-order columns so that ages that have both BenPlan and Envision data are first
    new_cols = summary_col + [x for x in list(delta_df.columns) if not x in summary_col]
    delta_df = delta_df[new_cols]

    # Write summary to console
    tmp_df = delta_df[summary_col].copy(deep=True)
    # Drop rows that are all zeros
    tmp_df = tmp_df.loc[(tmp_df != 0).any(axis=1)]
    summary_df = make_dollar_pretty(tmp_df)
    print_out(outf, 'BenPlan vs Envision\n' + summary_df)

    return delta_df


def net_portfolio_adjust(df_in: pd.DataFrame, row_label: str, row_lst: [str]) -> pd.Series:
    """  Adjust the Net Portfolio Need (row_label) by the values in row_lst (hack) """
    # Make sure at least one item in row_lst is in the index of df_in
    lst = [x for x in row_lst if x in list(df_in.index)]
    if len(lst) == 0:
        return df_in.loc[row_label]
    else:
        # print(f"Net Portfolio Need Adjust: {row_label} - {lst}")
        # print(f"Net Portfolio Need Adjust df_in.loc[row_label]: {df_in.loc[row_label]}")
        # for x in lst:
        #     print(f"Net Portfolio Need Adjust df_in.loc[{x}]: {df_in.loc[x]}")
        # print(f"Net Portfolio Need Adjust df_in.loc[lst]: {df_in.loc[lst].sum(axis=0)}")
        return df_in.loc[row_label] - df_in.loc[lst].sum(axis=0)

def adjust_income_tax(df_in: pd.DataFrame, default_tax_adjustment: float) -> pd.DataFrame:
    # Split Tax Income Tax between household and investment related
    for col in df_in.columns:
        # Tax adjustment is the min of the default_tax_adjustment and the actual Income_Tax - use max() because they
        # are negative numbers
        tax_adjusment = max(df_in.at['Income_Tax', col], default_tax_adjustment)
        df_in.at['Income_Tax', col] -= tax_adjusment
        df_in.at['Rtmt_Spending', col] += tax_adjusment
    # rename the Income_Tax row to Income_Tax_Investing
    df_in.rename(index={'Income_Tax': 'Income_Tax_Investing'}, inplace=True)
    return df_in


def add_envision_notes(df: pd.DataFrame, notes: dict) -> pd.DataFrame:
    """ Add notes to the dataframe """
    for key, value in notes.items():
        if key in list(df.index):
            df.at[key, 'Notes'] = value
    # Replace NaN in df with ''
    # df['Notes'].fillna('', inplace=True)
    # df.fillna(value={'Notes', ''}, inplace=True)
    df['Notes'] = df['Notes'].fillna('')
    return df

def compare_w_envision(pivot: pd.DataFrame, age_set: list, outf: typing.TextIO) -> pd.DataFrame:
    xl = pd.ExcelFile(ENVISION_FILE, engine='openpyxl')
    # Sheet names are YYYY-MM-DD so that max() provides the latest
    latest_sheet = max(xl.sheet_names)
    envision_df = pd.read_excel(ENVISION_FILE, sheet_name=latest_sheet, header=1, index_col=0, engine='openpyxl')
    envision_df.fillna(0.0, inplace=True)

    # ---- Compute aggregate values to compare BenPlan with Envision ----
    # For comparison at current age, use the Last12Mo - rather than the
    # interpolated value
    current_age = my_age()
    print_out(outf, f'\n{this_function()} - Current Age: {current_age}\n')
    bplan_df = pivot[age_set].copy(deep=True)
    bplan_df[current_age] = pivot['Last12Mo']  # add column for current age
    # Drop rows that are not interesting and low $$ amount
    bplan_df = bplan_df.drop(DEL_ROWS, axis=0, errors='ignore')  # ignore if row not in df
    bplan_df.loc['XTRA'] *= -1.0
    bplan_df = adjust_income_tax(bplan_df, HOUSEHOLD_INCOME_TAX)



    # Aggregate all the VC investment rows into a single row and delete the originals
    row_to_sum = [x for x in VC_INVEST_ROWS if x in list(envision_df.index)]
    envision_df.loc['VC_Invest'] = envision_df.loc[row_to_sum].sum(axis=0)
    envision_df = envision_df.drop(row_to_sum, axis=0)
    # Aggregate all the VC Principal rows into a single row and delete the originals for Envision
    # find the values in ENV_VC_PRINCIPAL that are in envision_df
    env_vc_principal_lst = [x for x in ENV_VC_PRINCIPAL if x in list(envision_df.index)]
    envision_df.loc['VC_Principal'] = envision_df.loc[env_vc_principal_lst].sum(axis=0)
    envision_df = envision_df.drop(env_vc_principal_lst, axis=0)
    envision_df.fillna(0.0, inplace=True)
    sub_total(bplan_df, VC_FLOW, envision_df, VC_FLOW, 'VC_Flow', 0)

    # print(f"SPENDING_DISCRETIONARY: {SPENDING_DISCRETIONARY}")
    sub_total(bplan_df, SPENDING_DISCRETIONARY, envision_df, SPENDING_DISCRETIONARY, 'Spending_Discretionary', 0)
    sub_total(bplan_df, BP_NET_CASHFLOW, envision_df, ENV_NET_CASHFLOW, 'Net_Cashflow', 0)
    bplan_df.loc['Invest_Portfolio_Need'] = bplan_df.loc[BP_Invest_Portfolio_Need].sum(axis=0)
    # print(f"Invest_Portfolio_Need\n{bplan_df.loc['Invest_Portfolio_Need']}")
    # Compute the Regular Investment Portfolio Needs ... i.e. subtract the one time transactions
    for df in [bplan_df, envision_df]:
        df.loc['Regular_Invest_Portfolio_Need'] = df.loc['Invest_Portfolio_Need']
        for transaction in One_Time_Transactions:
            if transaction in list(df.index):
                df.loc['Regular_Invest_Portfolio_Need'] -= df.loc[transaction]
    # print(f"Regular Invest_Portfolio_Need\n{bplan_df.loc['Regular_Invest_Portfolio_Need']}")
    bplan_df.loc['Global_Asset_Delta'] = net_portfolio_adjust(bplan_df, 'Invest_Portfolio_Need', Net_Portfolio_Need_Adjust)
    envision_df.loc['Global_Asset_Delta'] = net_portfolio_adjust(envision_df,'Invest_Portfolio_Need', Net_Portfolio_Need_Adjust )
    # Compute the Regular Global Asset Delta ... i.e. subtract the one time transactions
    for df in [bplan_df, envision_df]:
        df.loc['Regular_Global_Asset_Delta'] = df.loc['Global_Asset_Delta']
        for transaction in One_Time_Transactions:
            if transaction in list(df.index):
                df.loc['Regular_Global_Asset_Delta'] -= df.loc[transaction]
    # PCL_FLOW is positive when we are paying off PCL - i,e, decreasing the loan ... XTRA is negative
    # Pay_off_PCL is computed as outflows (negative value) from Portfolio - so we need to reverse the sign
    bplan_df.loc['PCL_Flow'] = - bplan_df.loc['Pay_off_PCL'] + bplan_df.loc['XTRA']
    envision_df.loc['PCL_Flow'] = 0.0

    delta_df = build_delta_df(bplan_df, envision_df, outf)
    # Add notes to the delta_df
    delta_df = add_envision_notes(delta_df, Envision_Notes)
    return delta_df
