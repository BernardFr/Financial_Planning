# Financial_Planning

## ToDo

* Fix seed - so that we get same RoR DF when we re_initialize funds
* **Fix cross_correlated_rvs**
  * Seed initialization does not work (i.e. does not produce same ror each run)
  * Std in final result stats is very high
  * `Morningstar - cross_correlated_rvs`: should use a generator rather than computing all the RoR at once

* Fix `nb_cpu > 1`: need to handle the seed for each instance of `montecarlo_simulation`
* in `portfolio_class.py` : get today's market value of assets rather than the one at time of download
* Find another source for historical asset statistics - Morningstar seems to be dated 2023
  * See JP [Morgan	](https://am.jpmorgan.com/us/en/asset-management/liq/insights/portfolio-insights/ltcma/interactive-assumptions-matrices/)	
  * See [Guggenheim Investments](https://www.guggenheiminvestments.com/advisor-resources/interactive-tools/asset-class-correlation-map) (2024)
  * Using different capital models (WF, JPMC) - will require remapping the holdings to the selected model
    * Note most models don't have the cross-correlation matrix


# Release Notes

### 2/20/2026

* Working version of `run_montecarlo_simulation` - including with `cross_correlated_rvs` 
  * `correlated_rvs_flag` true works ... but likely bugs in `mk_ror_df`
* WIP - montecarlo_simiulation_class is probably good - run_montecarlo_simulation Not good
* WIP - structure is correct - calculations are wrong

### 2/11/2026

* Created `test_rng.py` to validate the proper way to run the random number generator - including w/ multiple CPUs
  * Removed the tests in `arrayrandgen_class.py`

### 2/10/2026

* Added various tests to validate RoR generation

### 1/21/2026

* added option to use `correlated_rvs` to `montecarlo_simulation_class`

### 1/20/2026

* working version of `montecarlo_simulation_class`

### 1/19/2026

* working version of `montecarlo_simulation_class`

### 1/12/2026

* ``test_ror.py`: test creating the ror array
  * **NOTE**: added a floor and a ceiling (currently +/- 40%) for a given year to align with reality -- See ReadMe
* test_one_year.py`: test the one year loop

### 1/11/2026

* Merged `Morningstar_stats_class`, `cashflow_class` and `holdings_class` into `montecarlo_multi`

### 1/10/2026

* Basic version `holdings_class.py`: reads holdings, maps them to asset class - with market value
* Basic `cashflow_class`

### 1/9/2026

* Optimized `Morningstar_stats_class.py`

### 1/7/2026

* Basic version of `Morningstar_stats_class.py`:  fetches the data correctly from Morningstar
