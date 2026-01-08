#!/usr/bin/env python

"""
Get major asset types cross-correlation from Morningstar Webpage
Ref: https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
Create N pseudo-random series with the same cross correlation
Correlated series Ref: https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html
"""
import requests
import lxml.html as lh
import pandas as pd
import datetime as dt
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm


url_corr = 'https://admainnew.morningstar.com/webhelp/Practice/Plans/Correlation_Matrix_of_the_14_Asset_Classes.htm'
# url_stats = 'https://admainnew.morningstar.com/webhelp/Advisor_Workstation_Office.htm#Practice/Plans/'
url_stats = 'https://admainnew.morningstar.com/webhelp/dialog_boxes/cs_db_editassumptions.htm'
model_file = 'models.xlsx'
xl_out_name = 'morningstar_asset_correlations'
today, _ = str(dt.datetime.now()).split(' ')
xl_wr = pd.ExcelWriter(xl_out_name + '_' + today + '.xlsx')
# Number of pseudo-random samples to generate - need 1M for correlation to match exactly with original
nb_smpl = 1000000

# Get statistics of the assets
page = requests.get(url_stats)
#Store the contents of the website under doc
doc = lh.fromstring(page.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')
# Check the length of the rows
row_len = [len(T) for T in tr_elements]
if len(set(row_len)) != 1:  # all values should be the same
    print("We have a problem - the row lengths are not equal")
    print(row_len)

row_list = []  # List of all the rows in the table
for row in tr_elements:
    ll = [t.text_content().replace('\r\n', '').replace('\t\t','').replace('  ', ' ') for t in row]
    row_list.append(ll)  # We do not need the first element of each row

col_name = row_list[0]  # First row of the table
df_stat = pd.DataFrame(data=row_list[1:],columns=col_name)
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
df_stat.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)

print(df_stat)

# Get the correlation matrix
#Create a handle, page, to handle the contents of the website
page = requests.get(url_corr)
#Store the contents of the website under doc
doc = lh.fromstring(page.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')
# Check the length of the rows
row_len = [len(T) for T in tr_elements]
if len(set(row_len)) != 1:  # all values should be the same
    print("We have a problem - the row lengths are not equal")
    print(row_len)

row_list = []  # List of all the rows in the table
for row in tr_elements:
    ll = [t.text_content().replace('\r\n', '').replace('U.S.', "US") for t in row]
    ll = [t.replace('Lg', 'Large').replace('Sm','Small').replace('  ', ' ') for t in ll]
    row_list.append(ll[1:])  # We do not need the first element of each row

asset_class = row_list[0]  # First row of the table - list of assets -> index and column name
# There is an error where 'US Mid Cap Growth' is repeated twice - the second one should be 'US Mid Cap Value
idx = asset_class.index('US Mid Cap Growth')
if 'US Mid Cap Growth' in asset_class[idx+1:]:   # error is still there
    idx2 = asset_class[idx+1:].index('US Mid Cap Growth')
    asset_class[idx+1+idx2] = 'US Mid Cap Value'
# Map asset class names that are a lot different to their corresponding values in stats

# Remap some of the names in asset_class to those in stat_asset_list - so that we have common names
# Fill the dict with values for all elements to avoid key errors
map_dict = dict(zip(asset_class, asset_class))
# New desired values
remap_dict = {'Foreign Industrialzed Mkts Stocks': 'Non-US Dev Stk',
            'Emerging Mkts Stks': 'Non-US Emrg Stk',
            'USInvestment Grade Bonds': 'US Inv Grade Bonds',
            'Non-US Bonds': 'Non-US Dev Bonds',
            'US Small Cap Val':'US Small Cap Value'}
map_dict.update(remap_dict)  # update the selected keys
asset_class = [map_dict[x] for x in asset_class]  # remap

df_corr = pd.DataFrame(data=row_list[1:],index=asset_class,columns=asset_class,dtype=float)
print(df_corr)
df_corr.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)

# Compare the 2 lists of aasets
stat_set = set(stat_asset_list)
corr_set = set(asset_class)
print('Intersection of the 2 lists of assets: ', stat_set & corr_set)
print('Intersection of the 2 lists of assets length: ', len(stat_set & corr_set))
print('Delta of the 2 lists of assets: ', stat_set - corr_set)


# Create  N series of cross-correlated samples
nb_asset = df_corr.shape[0]  # square matrix
print('nb_asset: {}'.format(nb_asset))
x = []  # list of nb_asset data series
return_list = list(df_stat['Expected Return'])
stddev_list = list(df_stat['Standard Deviation'])
for cls, exp_ret, std_dev in zip(corr_set, return_list, stddev_list):
    x1 = norm.rvs(size=nb_smpl, loc=exp_ret, scale=std_dev)  # create a series w/ the desired stats
    x.append(x1)  # add data series list to list of lists

c_matrix = cholesky(df_corr, lower=True)
# Adjust for stddev of series different from 1.0
for i, mult in zip(range(nb_asset), stddev_list):
    c_matrix[:, i] *= 1 / mult
# Apply the cross-correlation
y = np.dot(c_matrix, x)
corr_out = np.corrcoef(y)
df2 = pd.DataFrame(corr_out,index=asset_class,columns=asset_class,dtype=float)
df2.to_excel(xl_wr, sheet_name='validation', float_format='%0.2f', header=True, index=True)
delta = df_corr - df2
delta.to_excel(xl_wr, sheet_name='delta', float_format='%0.2f', header=True, index=True)

# Perform portfolio mappying from Ben's Portfolio
alloc_df = pd.read_excel(model_file, sheet_name='Models', header=0, engine='openpyxl')
alloc_df.to_excel(xl_wr, sheet_name='Mappings', float_format='%0.2f', header=True, index=True)
# Create the list of allocation choices  %stock/%bond
allocation_choices = []
for stock in range(20, 81, 5):
    allocation_choices.append(str(stock) + '/' + str(100-stock))
print(allocation_choices)

# Create DF for Ben Models
ben_df = alloc_df[allocation_choices + ['Morningstar Category']].copy(deep=True)
ben_df.dropna(axis=0, inplace=True)  # drop rows that are either blank or are used as sub-totals
ben_df.set_index('Morningstar Category', drop=True, inplace=True)
print(ben_df)

# Figure out the mapping using BF Categories
map_df = alloc_df[['Morningstar Category', 'Mapping']].copy(deep=True)
map_df.reset_index(drop=True, inplace=True)
map_df.dropna(axis=0, inplace=True)  # drop rows that are blank
# Create a matrix of mapping weights - Initialize to 0.
morning_cat = list(alloc_df['BF Morningstar'].dropna())
map_weights = pd.DataFrame(0.0, index=morning_cat, columns=map_df['Morningstar Category'], dtype=float)
print(map_df)
print('\n----\n')

for col, mapg in zip(map_df['Morningstar Category'], map_df['Mapping']):
    # Determine if there is one or more entries
    map_lst = mapg.split('+')
    map_lst = [x.strip(' ') for x in map_lst]  # get rid of ' '
    wgt = 1.0 / len(map_lst)
    for idx in map_lst:
        print(mapg, idx, col, wgt)
        map_weights.loc[idx, col] = wgt
print(map_weights)
print('Sum of weights by column= ', map_weights.sum(axis=0))
print('Sum of weights by row= ', map_weights.sum(axis=1))
print('Sum of total weights = ', map_weights.sum().sum())

bf_alloc = map_weights.dot(ben_df)
bf_alloc.to_excel(xl_wr, sheet_name='Model', float_format='%0.2f', header=True, index=True)
print('Portfolio allocation')
print(bf_alloc)
print('Sum of weights by column= ', bf_alloc.sum(axis=0))

# Verify Stock/Bond ratios
# Remove empty cells and get rid of index
bf_alloc['Stock/Bond'] = list(alloc_df['Stock/Bond'].dropna())
stock_bond = bf_alloc.groupby('Stock/Bond',  axis=0).sum()
stock_bond.to_excel(xl_wr, sheet_name='Stock-Bond Ratios', float_format='%0.2f', header=True, index=True)
print('Stock-Bond Ratio')
print(stock_bond)

# Wrap up
xl_wr.save()
exit(0)