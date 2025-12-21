Fin_Think New Documentation
===

This is intended to be the end-all be-all of documentation for Fin_Think. All other docs, except `Fin_Think Release Notes` are obsolete (once this document is complete)


# Process
* If necessary update, and read, the Morningstar statistics
* Read Envision goals, and create yearly cashflow stream
* Read portfolio model
* Run Monte Carlo simulation with options:
 * Single asset vs multiple assets
 * Iterate with starting funds or discretionary spending to meet goals
 * Single vs. multi-processor
* Print & plot results

# Programs
| Program | What It Does |
| --- | --- |
| `make_it_work` | |
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

etc

# Parameter Dict
## Example
`Parameters: 
{'config': {'AGE_BIN': 5,
            'BF_BDAY': '04/09/1958',
            'DEATH_AGE': 95,
            'DEBUG': False,
            'DEFAULT_INFLATION_RATE': 0.03,
            'DELTA_NORM': 1000000.0,
            'END_AGE': 101,
            'GOALS_FILE': './Envision/Envision_Goals_Bernard.xlsx',
            'MANAGEMENT_FEE': 0.007,
            'MAX_ITER': 100,
            'MC_TARGET': 0.8,
            'MC_THRESHOLD': 0.02,
            'MIN_DELTA': 10000.0,
            'MORNINGSTAR_DIR': 'Morningstar_Asset_Stats',
            'MORNINGSTAR_ROOT': 'Morningstar_asset_stats_',
            'NB_ITER': 100000,
            'PLOT_FLAG': True,
            'ROR_INCREMENT': 0.001,
            'RUN_ALL_STRATEGIES': False,
            'RUN_MAKE_IT_WORK': False,
            'RUN_START_FUNDS': True,
            'SEED': 42,
            'SP500_MEAN': 0.1061,
            'SP500_STDDEV': 0.1453,
            'START_FUNDS': 2500000.0,
            'STRATEGY': '80/20',
            'iterate_start_funds': True},
 'make_it_work': {'asset_delta_neg': 1.5,
                  'asset_delta_pos': 2.0,
                  'disc_adjust': 0.05,
                  'min_disc_pct': 0.5},
 'start_funds': {'max_start_funds': 3500000.0,
                 'min_start_funds': 1500000.0,
                 'step_start_funds': 100000.0}}`
