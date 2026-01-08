Financial Plan
===

# Synopsis

* Load Config
* Load goals and compute cashflow by year
* Load Morningstar stats
* Load holdings - map to Morningstar asset classes -> generate portfolio with assets, %, Morninstar stats
  * Option: load other stat sources ... and corresponding holding <-> asset class matcher
* Run simulation with cashflows and portfolio
  * Option: "make it work": i.e. adjust spending in down years
* Show results

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

