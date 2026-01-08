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
xl_out_name = 'morningstar_asset_correlations'
today, _ = str(dt.datetime.now()).split(' ')
xl_wr = pd.ExcelWriter(xl_out_name + '_' + today + '.xlsx')
# Number of pseudo-random samples to generate - need 1M for correlation to match exactly with original
nb_smpl = 1000000

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

df = pd.DataFrame(data=row_list[1:],index=asset_class,columns=asset_class,dtype=float)
print(df)
df.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)

# Create  N series of cross-correlated samples
nb_asset = df.shape[0]  # square matrix
print('nb_asset: {}'.format(nb_asset))
x = norm.rvs(size=(nb_asset, nb_smpl))

c_matrix = cholesky(df, lower=True)
# Apply the cross-correlation
y = np.dot(c_matrix, x)
corr_out = np.corrcoef(y)
df2 = pd.DataFrame(corr_out,index=asset_class,columns=asset_class,dtype=float)
df2.to_excel(xl_wr, sheet_name='validation', float_format='%0.2f', header=True, index=True)
delta = df - df2
delta.to_excel(xl_wr, sheet_name='delta', float_format='%0.2f', header=True, index=True)

# Handle samples that are Not N(0.0, 1.0)
# Test with random variables that are Not N(0.0, 1.0
loc1, scale1 = (2.0, 2.0)
loc2, scale2 = (7., 3.0)
loc3, scale3 = (100., 0.1)
# loc1, scale1 = (5.0, 3.0)
# loc2, scale2 = (0.5, 0.3)
# loc3, scale3 = (0.5, 4.0)
x1 = norm.rvs(size=nb_smpl, loc=loc1, scale=scale1)
x2 = norm.rvs(size=nb_smpl, loc=loc2, scale=scale2)
x3 = norm.rvs(size=nb_smpl, loc=loc3, scale=scale3)
x = [x1, x2, x3]
dfx = df.iloc[0:3, 0:3].copy(deep=True)
dfx.to_excel(xl_wr, sheet_name='Test', float_format='%0.2f', header=True, index=True)

c_matrix = cholesky(dfx, lower=True)
# Adjust for stddev of series different from 1.0
for i, mult in zip(range(3), [scale1, scale2, scale3]):
    c_matrix[:,i] *= 1/mult
# Apply the cross-correlation
y = np.dot(c_matrix, x)
corr_out = np.corrcoef(y)
dfy = pd.DataFrame(corr_out,index=asset_class[0:3],columns=asset_class[0:3], dtype=float)
dfy.to_excel(xl_wr, sheet_name='validation scaled', float_format='%0.2f', header=True, index=True)
delta = dfx - dfy
delta.to_excel(xl_wr, sheet_name='delta scaled', float_format='%0.2f', header=True, index=True)
print(str(delta))

# Wrap up
xl_wr.save()
exit(0)