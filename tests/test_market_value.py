#!/usr/local/bin/python3
"""
Tests for MarketValueProcessor in src/market_value.py.

    - _save_results: creates and updates the market value history.
    - _log_summary: runs and does not change its input.
    - _plot_market_value: places points on the x-axis in proportion to the time between dates.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from market_value import DATE_COL, MARKET_VALUE_COL, MARKET_VALUE_SHEET, MarketValueProcessor  # noqa: E402

TEST_XL_FILE = Path(__file__).resolve().parent / 'test_market_value.xlsx'
PLOT_DATES = ['2026-01-01', '2026-01-15', '2026-03-01', '2026-09-20', '2026-10-31']
PLOT_VALUES = [100_000.0, 105_000.0, 98_000.0, 120_000.0, 118_000.0]
# Days between consecutive PLOT_DATES: 14, 45, 184, 30
EXPECTED_GAPS_DAYS = np.diff(pd.to_datetime(PLOT_DATES)).astype('timedelta64[D]').astype(float)


@pytest.fixture
def processor():
    """Make a processor without config files. Point the output to the tests/ directory."""
    TEST_XL_FILE.unlink(missing_ok=True)
    proc = MarketValueProcessor.__new__(MarketValueProcessor)
    proc.config_manager = SimpleNamespace(output_paths={'xl_out_filename': str(TEST_XL_FILE)})
    yield proc
    TEST_XL_FILE.unlink(missing_ok=True)
    plt.close('all')


def make_cumul_df(total: float) -> pd.DataFrame:
    return pd.DataFrame({'Symbol': ['AAA', 'Cash'], 'Sub_Total': [total - 100.0, 100.0]})


def test_save_results_creates_and_updates_history(processor):
    first_df = processor._save_results(make_cumul_df(1000.0))
    assert len(first_df) == 1
    assert first_df[MARKET_VALUE_COL].iloc[0] == 1000.0

    # A second run on the same day updates today's row
    second_df = processor._save_results(make_cumul_df(2000.0))
    assert len(second_df) == 1
    assert second_df[MARKET_VALUE_COL].iloc[0] == 2000.0

    saved_df = pd.read_excel(TEST_XL_FILE, sheet_name=MARKET_VALUE_SHEET)
    assert saved_df[MARKET_VALUE_COL].tolist() == [2000.0]


def test_save_results_appends_to_existing_history(processor):
    history_df = pd.DataFrame({DATE_COL: ['2000-01-01'], MARKET_VALUE_COL: [500.0]})
    with pd.ExcelWriter(TEST_XL_FILE, mode='w', engine='openpyxl') as writer:
        history_df.to_excel(writer, sheet_name=MARKET_VALUE_SHEET, index=False)

    market_value_df = processor._save_results(make_cumul_df(1000.0))
    assert market_value_df[MARKET_VALUE_COL].tolist() == [500.0, 1000.0]


def test_log_summary_runs(processor):
    market_value_df = pd.DataFrame({DATE_COL: ['2000-01-01', '2000-01-02'], MARKET_VALUE_COL: [500.0, 1500.0]})
    processor._log_summary(market_value_df, 1000.0)
    # _log_summary must not change its input
    assert market_value_df[MARKET_VALUE_COL].tolist() == [500.0, 1500.0]


@pytest.mark.parametrize('date_values', [PLOT_DATES, pd.to_datetime(PLOT_DATES)], ids=['str_dates', 'datetime_dates'])
def test_plot_x_spacing_is_proportional_to_time(processor, date_values):
    market_value_df = pd.DataFrame({DATE_COL: date_values, MARKET_VALUE_COL: PLOT_VALUES})
    with patch.object(plt, 'show'):
        processor._plot_market_value(market_value_df)
    ax = plt.gcf().axes[0]

    # If the dates were categories, x would be 0, 1, 2, ... (gaps of 1). Matplotlib date units are days.
    line_x = np.asarray(ax.convert_xunits(ax.lines[0].get_xdata()), dtype=float)
    scatter_x = np.asarray(ax.collections[0].get_offsets()[:, 0], dtype=float)
    np.testing.assert_allclose(np.diff(line_x), EXPECTED_GAPS_DAYS)
    np.testing.assert_allclose(np.diff(scatter_x), EXPECTED_GAPS_DAYS)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
