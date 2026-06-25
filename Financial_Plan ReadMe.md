Financial Plan
===

# Synopsis

1. Get the latest data files - if necessary
   1. Holdings from WF -- Manually update the holdings file in `/Users/bfraenkel/Documents/Code/BenPlan/Holdings` - format: `WFA_Portfolio_Positions_mmddyy_nnnn.xlsx`
   2. Capital Markets Assumptions / Asset statistics: Morningstart, JPM, Vanguard, etc.
      1. Manually download the "raw" files
      2. Use the provider-specific to extract asset_statistics in 1 Excel file with Return & Stddev in 1 tab `Stats` and, if available, asset cross-correlation matrix, tab: `Correlation`
         1. e.g. Use `morningstar_stats_class.py` to generate `Morningstar_asset_stats_2025_12_21.xlsx`
      
   3. if either of the above is updated: 
      1. Map holdings to positions by asset class. `WFA_to_positions.py`: Find the most recent holdings file, and aggregate positions by ETF ticker - save in `Positions_YYYY_mm_dd.xlsx`: 1 row per ticker with number of shares and market value (computed at the time when the holdings file was downloaded)
      2. Run `map_etf_asset_class.py`
         1. Update ETF- to-asset-class mapping -> `etf_asset_class_map_YYYY_mm_dd.xlsx`
         2. Map portfolio to assset classes (from tickers) -> `portfolio_by_Asset_class_YYYY_mm_dd.xlsx`
      
   4. Load expected cashflow from latest Envision/Money file - don't recompute if the cashflow data  has not changed.  `extract_outflows_from_lifeplan.py` `/Users/bfraenkel/Documents/Home/Finances/WFAdvisors/Envision_and_Performance` - format: `fraenkel.davis.life.plan.m{m}.d{d}.yy.pdf`  (month and day can be 1 or 2 digits). Output: `./Data/WFA_life_plan_YYYY_MM_DD.xlsx`
   
2. Run Monte Carlo Simulation
   1. Load config
      1. Inputs are in `./Data: 
         1. Positions (by asset class)
         2. Stats for selected Capital Markets Models
         3. Life plan cashflow

   2. Generate N portfolios of asset class, $_amount for each of the Asset Capital MarketsModels - (N=nb_cpu)
   3. For each of the Asset Class Statistics Models, generate sequences of RoR - Use `cross_correlated_rvs_flag` , if asset cross correlations are available (unless command line disables it)
   4. Run simulation with cashflows and portfolio for each model
   5. Show results for each model


### Options / Flags

* `rebalance_flag` if True [default **??**], assets are rebalanced to initial allocation each year
* `cross_correlated_rvs_flag` Determines whether we use Morningstar stats as is (false) or if we create cross-correlated RoR ===> **modify** so that by default we use cross correlation for the models that have it with option to disable it
* Asset_class_model: only run the specified Asset Class Statistics Model
* "make it work": i.e. adjust spending in down years

## Capital Market Models

### `capital_markets_stats_class.py`

Read the stats (and correlation matrix) for the capital markets stats from input file 
e.g. Morningstar / JP Morgan Capital Market 

Synopsis:
- Determine which capital market model to use (Morningstar, JP Morgan, etc.) based on the configuration file
- Read the stats and correlation matrix from the input file
- Validate that the correlation matrix is positive definite (i.e. all eigen values are positive)
- Validate that the stats and correlation matrix are consistent with each other (i.e. the correlation matrix is consistent with the stats) and consistent with the assets in the portfolio
- Generate correlated random variables based on the stats and correlation matrix

#### Filename Conventions

Files are in sub-directory of the master directory. Each capital market model has its directory.
The directory contains one Excel file for Stats and Correlation matrix. The file is in Excel format with 2 sheets: Stats and Correlation. 
The directory also contains the asset allocation of the portfolio in a separate file - and a file to map ETF to Asset Class 

Filename convention - assume Capital Market Model is CMM (Morningstar, JPM, etc.)
Master directory: `./Data`
Sub-directory: `./Data/CMM`
Stats and Correlation matrix file: `./Data/CMM/CMM_Stats_YYYY_MM_DD.xlsx`
Portfolio allocation file: `./Data/CMM/CMM_Portfolio_Allocations_YYYY_MM_DD.xlsx`
ETF to Asset Class mapping file: `./Data/CMM/CMM_ETF_Asset_Class_Mapping_YYYY_MM_DD.xlsx`

Note that `YYYY_MM_DD` are the date of the file creation and will be different for each file. The program will use the latest file based on the date in the filename.

## ETF Mapping



## Montecarlo_Simulation

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
*  `MorningstarStats - generate_correlated_rvs`   Figure out how to clip the rvs_df - like `ArrayRandGen` class

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



# Random Number Generator

* `spipy.stats.norm.rvs`  is better than `np.random.default_rng.normal` - see test in `default_rng_vs_rvs.py`

# Managing Seeds for RNGs for Multi-CPU

Assuming we need X random number generators (# of assets) and running N (4) CPUs

**NOTE**: **we use the seed sequence to generate rngs, which in turn are passed to `spipy.stats.norm.rvs` as `random_state` arguments**

```python
import numpy as np

master = np.random.SeedSequence(seed)

cpu_seqs = master.spawn(N)  *# one per CPU*

rngs_by_cpu = []  # random number generators

for cpu_seq in cpu_seqs:

​    child_seqs = cpu_seq.spawn(X)          # one per RNG on that CPU*

​    rngs = [np.random.default_rng(s) for s in child_seqs]

​    rngs_by_cpu.append(rngs) # these are fed to the scipy.stats.norm.rvs as randow_state
```

# References

* [Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) - see "Applications/Monte Carlo" simulation section		
