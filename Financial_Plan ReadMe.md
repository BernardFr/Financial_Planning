Financial Plan
===

# Synopsis

* Load Config
* Load goals and compute cashflow by year
* Load Morningstar stats
* Load holdings - map to Morningstar asset classes -> generate portfolio with assets, %, Morninstar stats
  * Option: load other stat sources ... and corresponding holding <-> asset class matcher
* From the Morningstar stats, generate either the sequence of RoR -- see `cross_correlated_rvs_flag` for options
* Run simulation with cashflows and portfolio
* Show results

## Montecarlo_Simulation

### Options / Flags

* `rebalance_flag` if True, assets are rebalanced to initial allocation each year
* `cross_correlated_rvs_flag` Determines whether we use Morningstar stats as is (false) or if we create cross-correlated RoR
* "make it work": i.e. adjust spending in down years

### Inputs

* N = `run_cnt` number of iterations 
* `initial_holdings`:  DF of asset classes x Market Value
* `asset_alloc`: is the allocation of assets.  `target_asset_alloc`  is equal to `initial_holdings` . When one of the assets for an asset class goes negative/zero - then the portfolio is rebalanced to match `target_asset_alloc` 
* `cashflow_ser`: series of cashflows indexed by age
* `ror_df`: DF indexed by asset classes with `run_cnt * nb_ages` columns of rate of returns. 

### Each Step

* Computes the new values of each holding based on the returns of the corresponding asset class, read from `ror_df`
* Subtracts the `cashflow` for that age, and the management fee, proportianately to each asset class
* Detects if the portfolio busted
* Rebalances the holdings to `target_asset_alloc`

### To Fix

* `Morningstar - cross_correlated_rvs`: should use a generator rather than computing all the RoR at once

# Programs

| Program | What It Does |
| --- | --- |
| `financing_retirement` | Creates a plot of how much you need to save based on when you start saving |
| `get_morningstar_stats` | Retrieves the latest stats (mean, stddev, correlation) for main asset classes published by Morningstar. Output is `"./Data/Morningstar_asset_stats"` |
| ----- | ----- |
| `make_it_work_simple` | |
| `montecarlo_simulate` | |
| `montecarlo_scenario` | |
| `montecarlo_scenario_rtmt_trvl` | |
| `montecarlo_scenario_stock_val` | |
| `compute_cashflow_discount` | |
| `init_asset_scenario` | |
| `montecarlo_multi` | |
| `miw_plt` | |
| `MonteCarlo_utilities` | |
| `asset_stats_util` | |
| `WF_capital_market_data` | |

# Notes on Rate of Returns

Per Gemini on 1/12/2026

The S&P 500's biggest yearly gain was over **+40%** (with some sources citing over 50% in periods like 1928 or 1997), while its worst annual loss was around **-38% to -47%**, with significant drops in 2008 and 1931, showcasing extreme volatility with massive booms and devastating crashes in market history. [[1](https://www.macrotrends.net/2526/sp-500-historical-annual-returns#:~:text=Interactive chart showing the annual percentage change,the last trading day of each year.), [2](https://www.pyrfordfp.com/post/the-s-p-500-lost-decade-how-to-protect-your-retirement#:~:text=If we look over the whole period,a greater maximum drawdown (-50.97% vs -34.86%).)]  

**Biggest Returns (Gains)** 

- **+53.1%** (around 1950s era). 
- **+49.5%** (1947-1957 period). 
- **+40% to +50%** (multiple years). 
- **+37.88%** in 1928. 
- **+31.01%** in 1997. [[1](https://www.macrotrends.net/2526/sp-500-historical-annual-returns#:~:text=Interactive chart showing the annual percentage change,the last trading day of each year.), [3](https://www.investopedia.com/ask/answers/042415/what-average-annual-return-sp-500.asp#:~:text=Annual Real Returns for the S&P 500,−6.1% 28.2% 16.5% 19.9% −23.0% 22.3% 22.7%), [4](https://www.sensiblefinancial.com/how-has-the-sp-500-performed-over-the-last-98-years/)]  

**Smallest Returns (Losses)** 

- **-47.07%** in 1931 (Great Depression). 
- **-38.49%** in 2008 (Great Financial Crisis). 
- **-36.61%** in 2008. 
- **-30%** in 1987 (Black Monday). [[1](https://www.macrotrends.net/2526/sp-500-historical-annual-returns#:~:text=Interactive chart showing the annual percentage change,the last trading day of each year.), [4](https://www.sensiblefinancial.com/how-has-the-sp-500-performed-over-the-last-98-years/), [5](https://www.sofi.com/learn/content/average-stock-market-return/#:~:text=Over the past 25 years%2C from 1998,is precisely 10% in any given year.), [6](https://carry.com/learn/average-stock-market-returns)]  

*AI responses may include mistakes.*

[1] [https://www.macrotrends.net/2526/sp-500-historical-annual-returns](https://www.macrotrends.net/2526/sp-500-historical-annual-returns#:~:text=Interactive chart showing the annual percentage change,the last trading day of each year.)

[2] [https://www.pyrfordfp.com/post/the-s-p-500-lost-decade-how-to-protect-your-retirement](https://www.pyrfordfp.com/post/the-s-p-500-lost-decade-how-to-protect-your-retirement#:~:text=If we look over the whole period,a greater maximum drawdown (-50.97% vs -34.86%).)

[3] [https://www.investopedia.com/ask/answers/042415/what-average-annual-return-sp-500.asp](https://www.investopedia.com/ask/answers/042415/what-average-annual-return-sp-500.asp#:~:text=Annual Real Returns for the S&P 500,−6.1% 28.2% 16.5% 19.9% −23.0% 22.3% 22.7%)

[4] https://www.sensiblefinancial.com/how-has-the-sp-500-performed-over-the-last-98-years/

[5] [https://www.sofi.com/learn/content/average-stock-market-return/](https://www.sofi.com/learn/content/average-stock-market-return/#:~:text=Over the past 25 years%2C from 1998,is precisely 10% in any given year.)

[6] https://carry.com/learn/average-stock-market-returns



# References

* [Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) - see "Applications/Monte Carlo" simulation section	
