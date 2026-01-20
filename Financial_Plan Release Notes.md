# Financial_Planning

## ToDo

* **Consolidate .toml files into `montecarlo_multi.toml`** - e.g. cashflow, Morningstar_stats, ... ... i.e. update `montecarlo_multi.toml` with the correct values from the individual .toml files
* in `holdings_class.py` : get today's market value of assets rather than the one at time of download
* Find another source for historical asset statistics - Morningstar seems to be dated 2023
  * See JP [Morgan	](https://am.jpmorgan.com/us/en/asset-management/liq/insights/portfolio-insights/ltcma/interactive-assumptions-matrices/)	
  * See [Guggenheim Investments](https://www.guggenheiminvestments.com/advisor-resources/interactive-tools/asset-class-correlation-map) (2024)

# Release Notes

### 1/29/2026

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
