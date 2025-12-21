#!/usr/local/bin/python3

"""
Get major asset types cross-correlation from Morningstar Webpage
Ref: https://towardsdatascience.com/web-scraping-html-tables-with-python-c9baba21059
Create N pseudo-random series with the same cross correlation
Correlated series Ref: https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html
"""

import sys
import os
import re
import datetime as dt
import tomli

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm as matrix_norm
from asset_stats_util import get_asset_stats, correlated_rvs, make_ben_model
import getopt

# Use helper functions from Color_Map
sys.path.insert(0, '../Color_Map/')  # Add path to the directory containing plot_color_matrix
from plot_color_matrix import plot_color_matrix as pcm

must_have_param = ['url_stats', 'url_corr', 'nb_smpl', 'quick_flag', 'model_file', 'xl_out_name']
XLabelLen = 5  # Length of the X axis labels


def config_param(config_file, argv):
    with open(config_file, mode='rb') as f:
        toml_dict = tomli.load(f)
    param_dict = toml_dict['default']
    # Make sure that all parameters have been configured - i.e. have a default value
    if sorted(must_have_param) != sorted(param_dict.keys()):
        print("Error: missing configuration parameters")
        for p in must_have_param:
            if p not in param_dict.keys():
                print("Must have parameter missing: ", p)
        for p in param_dict.keys():
            if p not in must_have_param:
                print("Unexpected parameter: ", p)
        sys.exit(-1)

    # Check if any of the parameters are overridden by command-line flags
    try:
        opts, args = getopt.getopt(argv[1:], "hq")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print(__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt in "-q":  # Quick run - don't iterate
            param_dict['quick_flag'] = True
        else:
            print(__doc__)
            assert False, "unhandled option"
    return param_dict


def make_xlabel(in_str: str, label_len) -> str:
    """
    Shorten the X axis labels to XLabelLen characters
    """
    # remove white space
    in_str = in_str.replace(' ', '')
    return in_str[0:label_len]


def main(argv):
    prog_name = re.sub("\\.py$", "", os.path.relpath(sys.argv[0]))
    plt_file = prog_name + "_out.pdf"  # replace the ".xlsx" extension
    config_file = prog_name + '.toml'
    today, _ = str(dt.datetime.now()).split(' ')

    param_dict = config_param(config_file, argv)
    nb_smpl = param_dict['nb_smpl']
    model_file = param_dict['model_file']
    xl_out_name = param_dict['xl_out_name']
    xl_wr = pd.ExcelWriter(xl_out_name + '_' + today + '.xlsx')

    df_stat, df_corr = get_asset_stats(param_dict['url_stats'], param_dict['url_corr'])
    # FIXME: the morningstar webpage has the wrong column names for the correlation matrix
    df_corr.columns = df_corr.index
    print(f'\nAsset Class Statistics\n{df_stat}')
    print(f'\nAsset Class Correlations\n{df_corr}')
    df_stat.to_excel(xl_wr, sheet_name='Stats', float_format='%0.2f', header=True, index=True)
    df_corr.to_excel(xl_wr, sheet_name='Correlation', float_format='%0.2f', header=True, index=True)
    asset_class = list(df_corr.index)  # List of assets

    y = correlated_rvs(df_stat, df_corr, nb_smpl)
    # Set index to asset_class
    y['AssetClass'] = asset_class
    y.set_index('AssetClass', drop=True, inplace=True)

    print('\nMean')
    print(y.mean(axis=1))
    print('\nMean Delta')
    print(df_stat['Expected Return'] - y.mean(axis=1))
    print('\nStddev')
    print(y.std(axis=1))
    print('\nStddev Delta')
    print(df_stat['Standard Deviation'] - y.std(axis=1))

    # Validate by computing cross-correlation on generated samples
    corr_out = np.corrcoef(y)

    asset_corr_df = pd.DataFrame(corr_out, index=asset_class,columns=asset_class,dtype=float)
    asset_corr_df.to_excel(xl_wr, sheet_name='validation', float_format='%0.2f', header=True, index=True)
    delta = df_corr - asset_corr_df   # this matrix should be all 0
    print("\nNorm of Delta Matrix (should be 0.0):", matrix_norm(delta, ord='fro'),'\n')
    delta.to_excel(xl_wr, sheet_name='delta', float_format='%0.2f', header=True, index=True)

    # Perform portfolio mapping from Ben's Portfolio
    alloc_df = pd.read_excel(model_file, sheet_name='Models', header=0, engine='openpyxl')
    alloc_df.to_excel(xl_wr, sheet_name='Mappings', float_format='%0.2f', header=True, index=True)
    bf_alloc = make_ben_model(alloc_df)
    print('\n\nPortfolio allocation')
    print(bf_alloc)
    print('Sum of weights by column= \n', bf_alloc.sum(axis=0))

    # Verify Stock/Bond ratios
    # Remove empty cells and get rid of index
    bf_alloc['Stock/Bond'] = list(alloc_df['Stock/Bond'].dropna())
    bf_alloc.to_excel(xl_wr, sheet_name='Model', float_format='%0.2f', header=True, index=True)
    stock_bond = bf_alloc.groupby('Stock/Bond',  axis=0).sum()
    stock_bond.to_excel(xl_wr, sheet_name='Stock-Bond Ratios', float_format='%0.2f', header=True, index=True)
    print('\nStock-Bond Ratio:\n', stock_bond)
    bf_alloc.drop(['Stock/Bond'], axis=1, inplace=True)  # no longer needed
    # Save the data file
    xl_wr.close()

    # Plot Asset Correlation Matrix
    print('\nPlotting Asset Correlation Matrix:\n')
    legend_dict = dict()
    legend_dict['title'] = "Asset Class Correlation Matrix"
    legend_dict['x_label'] = "Asset Class"
    legend_dict['y_label'] = "Asset Class"
    legend_dict['xticklabels'] = [make_xlabel(ss, XLabelLen) for ss in asset_corr_df.columns]  # Asset Classes, truncated
    legend_dict['yticklabels'] = [ss.replace(' ', '') for ss in asset_corr_df.columns]  # Asset Classes, 'truncated
    pcm(asset_corr_df, legend_dict=legend_dict, vmin_val=-1, vmax_val=1, plot_file=plt_file)  #
    # Alternative plot
    # plt.imshow(asset_corr_df)
    # plt.colorbar()
    # plt.show()

    return


if __name__ == '__main__':

    main(sys.argv)
    exit(0)
