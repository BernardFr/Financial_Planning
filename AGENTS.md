# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python-based Monte Carlo retirement simulation tool ("Financial_Plan" / "Fin_Think"). It uses Morningstar asset class statistics to project retirement outcomes. There is no web framework, database, or Docker dependency. All data is stored in Excel files.

### Running scripts

All Python scripts live in `src/`. Run them from the **workspace root** (`/workspace`) with `PYTHONPATH` and headless matplotlib:

```bash
PYTHONPATH=/workspace/src MPLBACKEND=Agg python3 src/<script_name>.py
```

Each script has a co-located `.toml` config file (e.g., `src/test_rng.toml` for `src/test_rng.py`). The `ConfigurationManager` class automatically reads the TOML from the script's own directory.

### Scripts that run without external data

| Script | Description |
|---|---|
| `src/test_rng.py` | Tests random number generation with multi-CPU support |
| `src/test_one_year.py` | Tests single-year Monte Carlo simulation step |
| `src/financing_retirement.py` | Computes savings needed by starting age (produces PDF + Excel output) |
| `src/get_morningstar_stats.py` | Scrapes Morningstar asset statistics (requires internet) |

### Scripts that require developer-local portfolio data

The following scripts fail because their TOML configs reference hardcoded macOS paths for portfolio holdings data (`/Users/bfraenkel/Documents/Code/BenPlan/Holdings/`):

- `src/run_montecarlo_simulation.py` (main simulation)
- `src/test_mk_ror_df.py`
- `src/test_run_one_year.py`

These scripts require brokerage position files (`.xls` format) and an ETF-to-asset-class mapping file that are not included in the repository.

### Key gotchas

- **`sys.exit("Done!")` produces exit code 1**: Many scripts exit with `sys.exit("---\nDone!")`. Passing a string to `sys.exit` causes a non-zero exit code. This is expected behavior, not an error.
- **`plt.show()` warning**: When using `MPLBACKEND=Agg`, scripts that call `plt.show()` emit a `UserWarning: FigureCanvasAgg is non-interactive` warning. This is harmless; PDF output still works.
- **`test_ror.py` is slow**: This script runs 10,000 iterations and can take over 60 seconds.
- **Data paths**: Morningstar model data is at `/workspace/Morningstar_Asset_Stats/Models.xlsx`, but some TOML configs reference it as `./Data/Morningstar_Asset_Stats/Models.xlsx`. The actual `./Data/` directory only contains `Envision_Goals.xlsx` and Morningstar stats exports.
