Fin_Think Release Notes
=============


# Goals
* Capabilities
 * Define spending goals -> compute yearly cashflow
 * Get latest rate statistics / projections
 * Define portfolio models -> match stategy to Morningstar indexes

 
1. ✓ Get %confidence given portfolio strategy and initial funds 
2. ✓ Get %confidence for a range of portfolio strategies and range of initial funds
	3. ✓ Use "OrangeBlue" colormap to plot confidence level for Starting Fungs x Strategy matrix:  https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html 
3. Use multiple CPUs for heavy computations
4. Given a portfolio strategy, adapt discretionary spending to "make it work" - i.e. reach target confidence level. Make it work both ways (i.e. spend less or more depending on how poorly/well the portfolio is doing)
5. Update Morningstar files automatically

# ToDo
* Measure execution time
* For each starting fund - find Strategy to leads to max confidence level
* downsample the maps to a total of 20 so that colors are blocks of 5%
* plot colormap from an existing Excel file

# 4/11 & 12/2023 
* Added asset cross correlation plot
* Misc refactoring of `TOML` file


# 12/21/2022 - Added `financing_retirement`
* Added `financing_retirement`: plots how much to save based on the year at which one starts saving
* Added `mc_new_mp_test`in the `Tests` directory to test various methods for multi-processing

# 12/1/2022
* Separated `plot_color_matrix()` into its own file in `Color_Map` directory
 * Added code to read a DF from file and plot it
* Compute strategy that leads to highest confidence level for given starting funds

# 11/28/2022
* Added matrix showing confidence level for range of starting funds and various portfolio strategies
* Added `Color_Map_Play` directory with `colormap_play.py` to demo colormap manipulations. 
* References:
 * "OrangeBlue" colormap: https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
 * Downloaded code example: `colormap-manipulation.py`
 * Tick Placement Reference: https://www.tutorialspoint.com/matplotlib/matplotlib_setting_ticks_and_tick_labels.htm

# 11/24/2022
* Added ability to iterate over Strategies


# 11/23/2022
* Fixed the segmentation fault for `matplotlib.show()`. Had to
 * Create new environment `fin_think`
 * edit **`matplotlibrc`** in `~/.matplotlib`. Get template in `/Users/bfraenkel/opt/anaconda3/envs/fin_think/lib/python3.8/site-packages` - and set `backend=WebAgg`
 	 * `backend=Template` does not crash, but does not show the plot either
 * Saved a copy in the current directory FYI `matplotlibrc_works_2022-11-23`
 * Reference: https://matplotlib.org/stable/users/explain/backends.html#selecting-a-backend
* Added ability to iterate over Starting Funds

 
#11/17/2022
* Progress release - assumptions and results are saved in Excel file `xxx_out.xlsx`

#11/15/2022
* Cleaned up 11/14/2022 release


#11/14/2022
* Progress release for `mc_new.py`: added run of Monte Carlo simulations for 1 asset - uses S&P 500 statistics for last 10-years
	* Reference: ` https://www.morningstar.com/indexes/spi/spx/risk`

#11/13/2022
* Started cleaning up by creating a new program `mc_new.py` and associated utilities in `mc_new_utilities.py`
	* Config file is now TOML `mc_new.toml`
	* This release works for use case of `blended_rate` only

# 2/16/2022
* Fixed `montecarlo_simulate.py` ... bunch of bugs
* Added option `-s start_funds`
* **Fatal bug: ** `matplotlib.pyplot.show()` causes a segmentation fault crash
	* Re-installed all packages, including `matplotlib` to no avail

# 8/27/2021
* Introduced 2 modes of optimization:
	* `-o s`: adjusts the starting funds up or down to meet the target success rate
	* `-o d`: adjusts the discretionary spend up or down to meet the target success rate


# 8/27/2021
* Changed the way goals are handled
	* Added ability to use "Default" for inflation rate
	* Initial Assets `start_funds`, Target end funds `target_end_funds`, and Default inflation rate `default_inflation_rate ` are now stored in the JSON file
	* Inflation_pct is now a % value: e.g. 2.5% in Excel -> value of 0.025
* Added `target_success_rate` in JSON to make results more readable

# 8/26/2021 - Not fully tested
* Changed how Goals data is read and how cashlow is computed. Took it out of JSON file, and use Excel file instead (more readable)
* Goals are now in `./Envision/Envision_Goals.xlsx`
	* Columns are: `Cashflow (index), Amount, Start_age, End_age, Inflation_pct, Discretionary`
	* Required rows are: `Initial Assets` (start_funds) and `End Assets` (target_funds)
* Wrote, or rewrote the following files and moved them to `MonteCarlo_utilities`
	* `process_goals_file`: reads the goals file, cleans it up and generates a DF
	* `make_cashflow`: computes the cashflow for each line item and each age year, based on Goals (including inflation for each item)
	* `lineitem_cashflow`: used by `make_cashflow` - processes one row of the Goals DF

# 8/25/2021
* Updated packages
* Minor fixes in gathering the output DF of `run_mc_multi`

# 1/27/2021
* Minor fixes: e.g adding `engine='openpyxl'` to `read_excel`

# 7/21/20
* Added `Tests/test_correlated_rvs` to fix and validate the generation of cross-correlatated pseudo-random data series
* Updated `Morningstar_Assets_Stats` with new (correct) version of `correlated_rvs`

# 6/10/20
* Renamed `init_hover` to `tooltip_plot_on_hover`
* Enabled `tooltip_plot_on_hover` in last plot of `init_asset_scenario`


# 6/8/20
* Added Play directory for experimentation programs
* `init_hover` demos how to label data points on a plot with mouse hover

# v1.1 - 4/17/2020
* **Final (?)** version of the `make_it_work` both in terms of code and results. Includes:
 * `make_it_work.py`
 * `montecarlo_class.py`
 * `MonteCarlo_utilities.py`
* **ToDo:** documentation
