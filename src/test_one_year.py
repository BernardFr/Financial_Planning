#!/usr/local/bin/python3
"""
Test the one_iter function.
Note we keep the same ror_ser, cashflow_val, and mgt_fee for each year.
Note also that a net negative ROR is not realistic - a negative return will be a small positive number
"""
import pandas as pd
from logger import logger
from typing import List, Dict, Any
import logging

# Set logger level to DEBUG for this test file
logger.setLevel(logging.DEBUG)
# Also set handlers to DEBUG level
for handler in logger.handlers:
    handler.setLevel(logging.DEBUG)

Assets = ['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4']
init_portfolio_ser = pd.Series([100000, 200000, 300000, 400000], index= Assets)
ror_ser = pd.Series([1.2, -1.1, 1.5, 0.4], index=Assets)
cashflow_val = 10000
mgt_fee = 0.01
run_cnt = 10
busted_value = 0.01 * init_portfolio_ser.sum() # 1% of the initial portfolio value

def rebalance_portfolio(portfolio_ser, target_asset_allocation):
    portfolio_value = portfolio_ser.sum()
    new_portfolio_ser = portfolio_value * target_asset_allocation
    return new_portfolio_ser

def print_series(msg: str, ser: pd.Series):
    # print the series with the index and the values - with the values formatted to 2 decimal places and a comma and dollar sign
    print(f"{msg}:")
    for index, value in ser.items():
        print(f"{index}: ${value:,.2f}")

def printable_series(msg: str, ser: pd.Series) -> str:
    # print the series with the index and the values - with the values formatted to 2 decimal places and a comma and dollar sign
    out_str = f"{msg}:\n"
    for index, value in ser.items():
        out_str += f"{index}: ${value:,.2f}\n"
    return out_str

def run_one_year(portfolio_ser: pd.Series, ror_ser: pd.Series, cashflow_val: float, mgt_fee: float, rebalance_flag: bool = True):
    portfolio_value = portfolio_ser.sum()
    initial_portfolio_value = portfolio_value
    target_asset_allocation = (portfolio_ser / portfolio_value)
    logger.debug(printable_series("Portfolio Series", portfolio_ser))
    logger.debug(f"Target Asset Allocation:\n{target_asset_allocation}")
    new_portfolio_ser = portfolio_ser.mul(ror_ser, axis=0)  # add the returns to the portfolio
    logger.debug(printable_series("New Portfolio Series after ror", new_portfolio_ser))
    value_after_ror = new_portfolio_ser.sum()
    # rebalance the portfolio - this way the distribution is done in the same proportion as the market value of each asset
    # and won't cause any holding to go negative
    # Zero out any negative values in the portfolio
    new_portfolio_ser = new_portfolio_ser.clip(lower=0) 
    logger.debug(printable_series("New Portfolio Series after clipping negative values", new_portfolio_ser))
    value_after_clipping = new_portfolio_ser.sum()
    if rebalance_flag:
        new_portfolio_ser = rebalance_portfolio(new_portfolio_ser, target_asset_allocation)
    logger.debug(printable_series("New Portfolio Series after (optional) rebalancing", new_portfolio_ser))
    value_after_rebalancing = new_portfolio_ser.sum()
    management_fee_value = portfolio_value * mgt_fee
    logger.debug(f"Management Fee Value ${management_fee_value:,.2f}")
    distribution_value = cashflow_val + management_fee_value  # money going out
    logger.debug(f"Distribution Value ${distribution_value:,.2f}")
    # Make a vector where the distribution is distributed to the portfolio proportionally to the market value of each asset
    distribution_vector = portfolio_ser  * distribution_value / portfolio_value 
    logger.debug(printable_series("Distribution Vector", distribution_vector))
    new_portfolio_ser -= distribution_vector
    logger.debug(printable_series("New Portfolio Series after distribution", new_portfolio_ser))
    value_after_distribution = new_portfolio_ser.sum()
    logger.debug(f"Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
    logger.debug(f"Value after ROR: ${value_after_ror:,.2f}")
    logger.debug(f"Value after clipping negative values: ${value_after_clipping:,.2f}")
    logger.debug(f"Value after rebalancing: ${value_after_rebalancing:,.2f}")
    logger.debug(f"Value after distribution: ${value_after_distribution:,.2f}")
    return new_portfolio_ser

if __name__ == "__main__":
    """
    This is the main function that runs the simulation.
    """
    portfolio_ser = init_portfolio_ser.copy(deep=True)
    logger.debug(f"ROR:\n{ror_ser}")
    logger.debug(f"Cashflow ${cashflow_val:,.2f}")
    logger.debug(f"Management Fee: {100* mgt_fee:.2f}%")
    logger.debug(printable_series("Initial Portfolio Series", init_portfolio_ser))

    for i in range(1, run_cnt + 1):
        portfolio_ser = run_one_year(portfolio_ser, ror_ser, cashflow_val, mgt_fee)
        logger.debug(printable_series(f"Portfolio Series after {i} years", portfolio_ser))
        portfolio_value = portfolio_ser.sum()
        logger.debug(f"Year: {i} - Portfolio Value: ${portfolio_value:,.2f}\n")
        if portfolio_value < busted_value:
            logger.debug(f"Portfolio busted at year {i}")
            break
