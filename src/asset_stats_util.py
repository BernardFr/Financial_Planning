import sys
import requests
from lxml import etree
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky
from sklearn.datasets import make_spd_matrix
from bs4 import BeautifulSoup

# Random state for the random number generator used to generate the random variables to ensure we have the same
# numbers across runs
RvsRandomState = 42


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
        # print(f"element_all: {element_all}")
        # find the one that has text 'ETF Database Themes'
        for elt in element_all:
            if elt.text.strip() == "Correlation Matrix for the 14 Asset Classes":
                # print(f"\nelement: {elt.text}")
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
            # print(f"\nrow_list: {row_list}")
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


def get_asset_stats_old(url_stats, url_corr, xl_wr):
    # Get statistics of the assets
    page = requests.get(url_stats)
    # Store the contents of the website under doc
    doc = etree.fromstring(page.content)
    # Parse data that are stored between <tr>..</tr> of HTML
    tr_elements = doc.xpath('//tr')
    # Check the length of the rows
    row_len = [len(T) for T in tr_elements]
    if len(set(row_len)) != 1:  # all values should be the same
        print("We have a problem - the row lengths are not equal")
        print(row_len)

    row_list = []  # List of all the rows in the table
    for row in tr_elements:
        ll = [t.text_content().replace('\r\n', '').replace('\t\t', '').replace('  ', ' ') for t in row]
        row_list.append(ll)  # We do not need the first element of each row

    col_name = row_list[0]  # First row of the table
    df_stat = pd.DataFrame(data=row_list[1:], columns=col_name)
    # the page contains 3 tables with duplicate entries
    # We only need the first table
    # df_stat.drop_duplicates(inplace=True)  # Eliminate duplicates
    # The other tables start at a row with "Asset Class' in 1st column
    stop_idx = list(df_stat[df_stat['Asset Class'] == 'Asset Class'].index)[0]  # Pick the 1st value
    df_stat.drop(index=range(stop_idx, len(df_stat.index)), inplace=True)
    stat_asset_list = list(df_stat['Asset Class'])
    # Make Asset Class the index
    df_stat.set_index('Asset Class', inplace=True)
    df_stat = df_stat.astype(float)
    if 'Inflation' in df_stat.index:  # get rid of it
        print('Removing Inflation from Stats')
        df_stat.drop(labels='Inflation', axis=0, inplace=True)
    df_stat.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)

    # Get the correlation matrix
    # Create a handle, page, to handle the contents of the website
    page = requests.get(url_corr)
    # Store the contents of the website under doc
    doc = etree.fromstring(page.content)
    # Parse data that are stored between <tr>..</tr> of HTML
    tr_elements = doc.xpath('//tr')
    # Check the length of the rows
    row_len = [len(T) for T in tr_elements]
    if len(set(row_len)) != 1:  # all values should be the same
        print("We have a problem - the row lengths are not equal")
        print(row_len)

    row_list = []  # List of all the rows in the table
    for row in tr_elements:
        ll = [t.text_content().replace('\r\n', '').replace('U.S.', "US") for t in row]
        ll = [t.replace('Lg', 'Large').replace('Sm', 'Small').replace('  ', ' ') for t in ll]
        row_list.append(ll[1:])  # We do not need the first element of each row

    asset_class = row_list[0]  # First row of the table - list of assets -> index and column name
    # There is an error where 'US Mid Cap Growth' is repeated twice - the second one should be 'US Mid Cap Value
    idx = asset_class.index('US Mid Cap Growth')
    if 'US Mid Cap Growth' in asset_class[idx + 1:]:  # error is still there
        idx2 = asset_class[idx + 1:].index('US Mid Cap Growth')
        asset_class[idx + 1 + idx2] = 'US Mid Cap Value'
    # Map asset class names that are a lot different to their corresponding values in stats

    # Remap some of the names in asset_class to those in stat_asset_list - so that we have common names
    # Fill the dict with values for all elements to avoid key errors
    map_dict = dict(zip(asset_class, asset_class))
    # New desired values
    remap_dict = {'Foreign Industrialzed Mkts Stocks': 'Non-US Dev Stk',
                  'Emerging Mkts Stks': 'Non-US Emrg Stk',
                  'USInvestment Grade Bonds': 'US Inv Grade Bonds',
                  'Non-US Bonds': 'Non-US Dev Bonds',
                  'US Small Cap Val': 'US Small Cap Value'}
    map_dict.update(remap_dict)  # update the selected keys
    asset_class = [map_dict[x] for x in asset_class]  # remap

    df_corr = pd.DataFrame(data=row_list[1:], index=asset_class, columns=asset_class, dtype=float)
    df_corr.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)

    # Compare the 2 lists of assets
    stat_set = set(stat_asset_list)
    corr_set = set(asset_class)
    print('Intersection of the 2 lists of assets: ', stat_set & corr_set)
    print('Intersection of the 2 lists of assets length: ', len(stat_set & corr_set))
    print('Delta of the 2 lists of assets: ', stat_set - corr_set)

    return df_stat, df_corr


def eigen_values_positive(df_corr: pd.DataFrame) -> bool:
    """ Returns True if the eigen values of a correlation matrix are positive"""
    # ToDo: add teststo ensure that df_corr is a correlation matrix
    try:
        eigen_values = np.linalg.eigvals(df_corr.astype(float))
    except np.linalg.LinAlgError:
       return False  # not a viable matrix
    positive_eigen = [eigen >= 0 for eigen in eigen_values ]
    return True if all(positive_eigen) else False



def matrix_equal(df_1: pd.DataFrame, df_2: pd.DataFrame, error_margin: float) -> (bool, float):
    """
    Compares 2 DF with can be 1- or 2-dimensional and returns a tuple (flag, error)
    The error is normalized by the number of elements in the DF
    @param df_1: DF 1
    @param df_2: DF 2
    @param error_margin: normalized error margin to determine if the 2 DF are equal
    @return: (flag, error). Flag is True if the 2 DF are equal (i.e. normalized delta < error_margin). error is the
    computated normalized difference
    """
    fct_name = sys._getframe().f_code.co_name
    assert df_1.shape == df_2.shape, f"{fct_name}: df_1 and df_2  have different shapes\n{df_1.shape}\n{df_2.shape}"
    assert list(df_1.index) == list(df_2.index), f"{fct_name}: df_1 and df_2 don't have same index\n" \
                                                 f"{df_1.index}\n{df_2.index}"
    delta = df_1 - df_2
    # denom is the total number of samples in each DF
    denom = len(df_1.index) if df_1.ndim == 1 else len(df_1.index) * len(df_1.columns)
    err = np.linalg.norm(delta) / denom
    flag = True if abs(err) <= error_margin else False
    return flag, err



def make_random_correlation_matrix(labels: [str], seed=None) -> pd.DataFrame:
    """
    Generates a cross correlation matrix with labels as labels for index and columns
    the size of the matrix is NxN where N = len(labels)
    They are generated by the random number generator rng - provided as parameter
    @poram labels: labels
    @param seed: seed for random number generator
    @return cross_correlation matrix
    """
    # corr_matrix = pd.DataFrame(index=labels, columns=labels)
    dimension = len(labels)
    corr_matrix = pd.DataFrame(make_spd_matrix(dimension, random_state=seed), index=labels, columns=labels)
    # Make it a correlation matrix
    diag_inv = [1/np.sqrt(corr_matrix.iloc[i, i]) for i in range(dimension)]
    # for i in range(dimension):
    #     for j in range(dimension):
    #         corr_matrix.iloc[i, j] *= diag_inv[i]
    # for i in range(dimension):
    #     for j in range(dimension):
    #         corr_matrix.iloc[i, j] *= diag_inv[j]

    for idx, ro in enumerate(corr_matrix.index):  # normalize the rows
        corr_matrix.loc[ro] *= diag_inv[idx]
    for idx, cl in enumerate(corr_matrix.columns):  # normalize the rows
        corr_matrix[cl] *= diag_inv[idx]
    assert eigen_values_positive(corr_matrix), f"Failed to create a correlation matrix"
    return corr_matrix


def correlated_rvs(df_stat: pd.DataFrame, df_corr: pd.DataFrame, nb_smpl: int) -> pd.DataFrame:
    """ Generate nb_smpl of Random Variable which are cross-correlated
    df_stat contains 2 columns: "Expected Return" i.e. mean and "Standard Deviation"
    df_corr is the cross-correlation matrix of the variables
    Must have df_stat.index == df_corr.index == df_corr.columns
    nb_smpl is the number of samples that are generated
    Returns an list of  'nb_asset" lists - each list is nb_smpl long
    nb_asset = len(df_stat.index)
    """

    # Create  N series of cross-correlated samples
    nb_asset = df_corr.shape[0]  # square matrix
    x = []  # list of nb_asset data series
    return_list = list(df_stat['Expected Return'])
    stddev_list = list(df_stat['Standard Deviation'])
    # asset_class = list(df_corr.columns)  # List of assets
    for exp_ret, std_dev in zip(return_list, stddev_list):
        # ToFix the random_state option is supposed to make the random number generator repeatable - however it
        #  seems to break mean/loc and scale/stddev
        # x1 = norm.rvs(size=nb_smpl, loc=0.0, scale=1.0, random_state=RvsRandomState)  # create a series w/ the desired
        x1 = norm.rvs(size=nb_smpl, loc=0.0, scale=1.0)  # create a series w/ the desired stats
        x.append(x1)  # add data series list to list of lists

    # Compute eigen values to make sure none is negative - otherwise, cholesky will fail
    try:
        eigen_values = np.linalg.eigvals(df_corr.astype(float))
    except np.linalg.LinAlgError as e:
        sys.exit(f"correlated_rvs could not converge on eigen value decomposition")
    negative_eigen = [eigen < 0 for eigen in eigen_values ]
    if any(negative_eigen):
        sys.exit(f"correlated_rvs cannot handle matrices with negative eigen values:\n{eigen_values}")

    c_matrix = cholesky(df_corr, lower=True)
    data_df = pd.DataFrame(np.dot(c_matrix, x))
    # Assign the desired mean and std_dev to each series
    def mk_lin_interp(mean_val, std_val):
        def f(x):
            return mean_val + std_val * x
        return f

    for rw, mn, stdv in zip(data_df.index, return_list,stddev_list):
        data_df.loc[rw] = data_df.loc[rw].map(mk_lin_interp(mn, stdv))

    return data_df


def make_ben_model(alloc_df):
    # Create the list of allocation choices  %stock/%bond
    allocation_choices = []
    for stock in range(20, 81, 5):
        allocation_choices.append(str(stock) + '/' + str(100-stock))
    print('Allocation Choices: ', allocation_choices)

    # Create DF for Ben Models
    ben_df = alloc_df[allocation_choices + ['Morningstar Category']].copy(deep=True)
    ben_df.dropna(axis=0, inplace=True)  # drop rows that are either blank or are used as sub-totals
    ben_df.set_index('Morningstar Category', drop=True, inplace=True)
    print('Ben Model:\n', ben_df)

    # Figure out the mapping using BF Categories
    map_df = alloc_df[['Morningstar Category', 'Mapping']].copy(deep=True)
    map_df.reset_index(drop=True, inplace=True)
    map_df.dropna(axis=0, inplace=True)  # drop rows that are blank
    # Create a matrix of mapping weights - Initialize to 0.
    morning_cat = list(alloc_df['BF Morningstar'].dropna())
    map_weights = pd.DataFrame(0.0, index=morning_cat, columns=map_df['Morningstar Category'], dtype=float)
    print('Mapping:\n', map_df)
    print('\n----\n')

    for col, mapg in zip(map_df['Morningstar Category'], map_df['Mapping']):
        # Determine if there is one or more entries
        map_lst = mapg.split('+')
        map_lst = [x.strip(' ') for x in map_lst]  # get rid of ' '
        wgt = 1.0 / len(map_lst)
        for idx in map_lst:
            print(mapg, idx, col, wgt)
            map_weights.loc[idx, col] = wgt
    print('Mapping Weight Matrix:\n', map_weights)
    print('\nSum of weights by column= \n', map_weights.sum(axis=0))
    print('\nSum of weights by row= \n', map_weights.sum(axis=1))
    print('\nSum of total weights = ', map_weights.sum().sum())

    bf_alloc = map_weights.dot(ben_df)
    return bf_alloc


def validate_cross_correlation(data_df: pd.DataFrame, stats_matrix: pd.DataFrame, corr_matrix: pd.DataFrame,
                               error_margin: float) -> bool:
    """
    Validate that a data series has the statistics and cross-correlation passed as parameters
    @param data_df: NxM DF - N series of M samples
    @param stats_matrix: Nx2 DF with expected mean and std_dev as columns
    @param corr_matrix: NxN cross-correlation matrix for series
    @param error_margin: margin of error when testing equality
    @return: True if validation passes
    """
    fct_name = sys._getframe().f_code.co_name
    assert list(data_df.index) == list(stats_matrix.index), f"{fct_name}: data_df and stats_matrix don't have same " \
                                                    f"index\n{data_df.index}\n{stats_matrix.index}"
    assert list(data_df.index) == list(corr_matrix.index), f"{fct_name}: data_df and corr_matrix don't have same index\n" \
                                                   f"{data_df.index}\n{corr_matrix.index}"
    assert list(corr_matrix.index) == list(corr_matrix.columns), f"{fct_name}: corr_matrix index and columns are " \
                                                             f"different " \
                                                      f"index\n" \
                                                     f"{corr_matrix.index}\n{corr_matrix.columns}"
    mean_ser = data_df.mean(axis=1)

    # FYI: the _validate variables are (flag, error) tuples
    mean_validate = matrix_equal(mean_ser, stats_matrix.iloc[:,0], error_margin)  # mean is the first column

    std_ser = data_df.std(axis=1)
    stddev_validate = matrix_equal(std_ser, stats_matrix.iloc[:,1], error_margin)  # std_dev is second column

    # IMPORTANT: astype(float) is needed otherwise, corr() returns empty DF
    data_corr = data_df.T.astype(float).corr()  # Corr works on columns
    corr_validate = matrix_equal(data_corr, corr_matrix, error_margin)

    return mean_validate,  stddev_validate, corr_validate
