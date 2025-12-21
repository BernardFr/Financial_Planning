#!/Library/Frameworks/Python.framework/Versions/Current/bin/python3

"""
Get Statistics from Morningstar
"""

import datetime as dt
import os
import re
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

# The URL of the ETF profile page
URL_STATS = "https://admainnew.morningstar.com/webhelp/dialog_boxes/cs_db_editassumptions.htm"
ULR_CORR = "https://admainnew.morningstar.com/webhelp/Practice/Plans/Correlation_Matrix_of_the_14_Asset_Classes.htm"

# FYI: df_stat_names and df_corr_names are the names of the assets in the 2 dataframes - they NEED to be in the same
#  order
df_stat_names = [
        "US Large Cap Growth",
        "US Large Cap Value",
        "US Mid Cap Growth",
        "US Mid Cap Value",
        "US Small Cap Growth",
        "US Small Cap Value",
        "Non-US Dev Stk",
        "Non-US Emrg Stk",
        "US Inv Grade Bonds",
        "US High Yield Bonds",
        "Non-US Dev Bonds",
        "Cash",
        "Commodities",
        "Real Estate",
]

df_corr_names = [
        "U.S. Lg Cap Growth",
        "U.S. Lg Cap Val",
        "U.S. Mid Cap Growth",
        "U.S. Mid Cap Val",
        "U.S. Sm Cap Growth",
        "U.S. Sm Cap Val",
        "Foreign Industrialzed Mkts Stocks",
        "Emerging Mkts Stks",
        "U.S. Investment Grade Bonds",
        "U.S. High Yield Bonds",
        "Non-U.S. Bonds",
        "Cash",
        "Commodities",
        "Real Estate",
]


def remap_names(names: [str], old_names: [str], new_names: [str]) -> [str]:
    """
    Remap the names in names according to the mapping of old_names new_names
    """
    # New desired values
    remap_dict = dict(zip(old_names, new_names))
    result_names = [remap_dict.get(x, "Error") for x in names]  # remap
    if "Error" in result_names:
        print(f"Error(s) in remapping names: {names} -> {result_names}")
        exit(-1)
    return result_names


def get_morningstar_stats(url_stats: str) -> pd.DataFrame:
    try:
        response = requests.get(url_stats)
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
        print(f"Failed to retrieve the page for URL_STATS Error: {e}\nURL: {url_stats}")
        return None
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the paragraph containing the Morningstar Basic Statistics which is a class of "Tip-Note-Heading"
        element_all = soup.find_all("p", {"class": "Tip-Note-Heading"})
        print(f"element_all: {element_all}")
        # find the one that has text 'ETF Database Themes'
        for elt in element_all:
            if elt.text.strip() == "Morningstar Basic":
                # print(f"\nelement: {elt.text}")
                break
        # Find the first table after elt containing the Morningstar Basic Statistics which is a class of "MsoTableGrid"
        if elt is not None:
            tbl = elt.find_next("table", {"class": "MsoTableGrid"})
        else:
            exit(-1)
        # print(f"tbl: {tbl}")

        tbl_row = tbl.find_all('tr')
        for idx, row in enumerate(tbl_row):
            row_list = []
            for td in row.find_all('td'):
                row_list.append(td.text.strip())
            print(f"\nrow_list: {row_list}")
            if idx == 0:
                col_names = row_list
                result_df = pd.DataFrame(columns=col_names)
            else:
                result_df.loc[idx - 1] = row_list
    else:
        print(f"Failed to retrieve the page for url {url_stats} Status code: {response.status_code}")
        return None

    # Make Asset Class the index
    result_df.set_index('Asset Class', inplace=True)
    result_df = result_df.astype(float)
    if 'Inflation' in result_df.index:  # get rid of it
        print('Removing Inflation from Stats')
        result_df.drop(labels='Inflation', axis=0, inplace=True)
    return result_df


def get_morningstar_corr(url_corr: str) -> pd.DataFrame:
    try:
        response = requests.get(url_corr)
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
        print(f"Failed to retrieve the page for URL_CORR Error: {e}\nURL: {url_corr}")
        return None
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the paragraph containing the Morningstar Basic Statistics which is a class of "Tip-Note-Heading"
        element_all = soup.find_all("h1")
        print(f"element_all: {element_all}")
        # find the one that has text 'ETF Database Themes'
        for elt in element_all:
            if elt.text.strip() == "Correlation Matrix for the 14 Asset Classes":
                print(f"\nelement: {elt.text}")
                break
        # Find the first table after elt containing the Morningstar Basic Statistics which is a class of "MsoTableGrid"
        if elt is not None:
            tbl = elt.find_next("table")
        else:
            exit(-1)
        # print(f"tbl: {tbl}")

        tbl_row = tbl.find_all('tr')
        for idx, row in enumerate(tbl_row):
            row_list = []
            for td in row.find_all('td'):
                row_list.append(td.text.strip())
            print(f"\nrow_list: {row_list}")
            if idx == 0:
                row_list[0] = "Asset Class"
                col_names = row_list
                result_df = pd.DataFrame(columns=col_names)
            else:
                result_df.loc[idx - 1] = row_list
    else:
        print(f"Failed to retrieve the page for url {url_corr} Status code: {response.status_code}")
        return None

    # Make Asset Class the index
    result_df.set_index('Asset Class', inplace=True)
    result_df = result_df.astype(float)
    return result_df


def get_asset_stats(url_stats: str, url_corr: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Get the asset statistics and correlation matrix from Morningstar
    @param url_stats:
    @param url_corr:
    @return: 2 DF for asset statistics and correlation matrix
    """
    # Get statistics of the assets
    df_stat = get_morningstar_stats(url_stats)
    # Get the correlation matrix
    df_corr = get_morningstar_corr(url_corr)
    # Remap the name of asset classes to match the names in df_corr
    asset_class_names = remap_names(list(df_stat.index), df_stat_names, df_corr_names)
    # Force the index of df_stat to be asset_class_names
    df_stat['IDX'] = asset_class_names
    df_stat.set_index('IDX', drop=True, inplace=True)
    # Reorder  df_stat index to match df_corr index
    df_stat = df_stat.reindex(index=df_corr.index)

    return df_stat, df_corr


def main(argv):
    prog_name = re.sub("\.py$", "", os.path.relpath(argv[0]))
    plt_file = prog_name + "_out.pdf"  # replace the ".xlsx" extension
    config_file = prog_name + '.json'
    today, _ = str(dt.datetime.now()).split(' ')
    xl_wr = pd.ExcelWriter(prog_name + '_' + today + '.xlsx')
    df_stat, df_corr = get_asset_stats(URL_STATS, ULR_CORR)
    print(f'\nAsset Class Statistics\n{df_stat}')
    print(f'\nAsset Class Correlations\n{df_corr}')
    df_stat.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)
    df_corr.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)
    xl_wr.close()

    return


if __name__ == '__main__':
    main(sys.argv)
    exit(0)
