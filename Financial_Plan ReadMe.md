Financial Plan
===

# Synopsis

* Load Config
* Load goals and compute cashflow by year
* Load Morningstar stats
* Load holdings - map to Morningstar asset classes -> generate portfolio with assets, %, Morninstar stats
  * Option: load other stat sources ... and corresponding holding <-> asset class matcher
* Run simulation with cashflows and portfolio
* Show results

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

### Options

* "make it work": i.e. adjust spending in down years
* Rebalancing

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

