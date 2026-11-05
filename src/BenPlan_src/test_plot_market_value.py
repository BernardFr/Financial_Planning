#!/usr/bin/env python3
"""
Validate that MarketValueProcessor._plot_market_value places points on the x-axis
proportionally to the elapsed time between dates (i.e. unevenly spaced dates are
NOT drawn at evenly spaced positions).

During the tests plt.show() is patched, so no window opens. The x positions of both the scatter markers and the
connecting line are read back from the figure and compared to the actual day gaps.

Two input flavors are tested:
    - Date column as datetime64 values
    - Date column as 'YYYY-MM-DD' strings (what _save_results writes/reads today)

When run as a script, the tests run first, then the test series is displayed with
_plot_market_value for visual inspection.
"""

import logging
import sys
import types
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Stub the logger modules (old `logger` and new `custom_logger` names) so the test does
# not depend on the logger module layout or create log files.
_TEST_LOGGER = logging.getLogger('test_plot_market_value')
for _module_name in ('logger', 'custom_logger'):
    _logger_stub = types.ModuleType(_module_name)
    _logger_stub.logger = _logger_stub.custom_logger = _TEST_LOGGER
    sys.modules[_module_name] = _logger_stub

from market_value import MarketValueProcessor, MARKET_VALUE_COL  # noqa: E402

TEST_DATES = ['2026-01-01', '2026-01-15', '2026-03-01', '2026-09-20', '2026-10-31']
TEST_VALUES = [100_000.0, 105_000.0, 98_000.0, 120_000.0, 118_000.0]
# Days between consecutive TEST_DATES: 14, 45, 184, 30
EXPECTED_GAPS_DAYS = np.diff(pd.to_datetime(TEST_DATES)).astype('timedelta64[D]').astype(float)


class TestPlotMarketValueDateSpacing(unittest.TestCase):
    """x-axis spacing of _plot_market_value must reflect real time gaps."""

    def setUp(self):
        # Bypass __init__: it loads configuration files not needed for plotting
        self.processor = MarketValueProcessor.__new__(MarketValueProcessor)

    def tearDown(self):
        plt.close('all')

    def _render(self, date_values) -> plt.Axes:
        """Call _plot_market_value on TEST_VALUES and return the resulting Axes."""
        df = pd.DataFrame({'Date': date_values, MARKET_VALUE_COL: TEST_VALUES})
        with patch.object(plt, 'show'):
            self.processor._plot_market_value(df)
        return plt.gcf().axes[0]

    @staticmethod
    def _line_x(ax: plt.Axes) -> np.ndarray:
        """x positions of the connecting line, in axis (float) units."""
        return np.asarray(ax.convert_xunits(ax.lines[0].get_xdata()), dtype=float)

    @staticmethod
    def _scatter_x(ax: plt.Axes) -> np.ndarray:
        """x positions of the scatter markers, in axis (float) units."""
        return np.asarray(ax.collections[0].get_offsets()[:, 0], dtype=float)

    def _assert_spacing_proportional_to_time(self, ax: plt.Axes) -> None:
        # If dates were treated as categories, x would be 0,1,2,3,4 (gaps of 1)
        for label, x in (('line', self._line_x(ax)), ('scatter', self._scatter_x(ax))):
            with self.subTest(series=label):
                # Matplotlib date units are days, so gaps must match day counts exactly
                np.testing.assert_allclose(np.diff(x), EXPECTED_GAPS_DAYS,
                                           err_msg=f'{label} x-spacing does not match day gaps')

    def test_datetime_dates_are_time_scaled(self):
        ax = self._render(pd.to_datetime(TEST_DATES))
        self._assert_spacing_proportional_to_time(ax)

    def test_string_dates_are_time_scaled(self):
        ax = self._render(TEST_DATES)
        self._assert_spacing_proportional_to_time(ax)


def display_test_series() -> None:
    """Display the test series with the production plotting code (string dates, as in real data)."""
    processor = MarketValueProcessor.__new__(MarketValueProcessor)
    processor._plot_market_value(pd.DataFrame({'Date': TEST_DATES, MARKET_VALUE_COL: TEST_VALUES}))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
    display_test_series()
