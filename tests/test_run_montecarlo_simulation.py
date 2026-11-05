#!/usr/local/bin/python3
"""
Tests for RunMontecarloSimulation: every iteration starts from the same random-number state (common random numbers),
for single- and multi-CPU runs - so repeating an iteration with the same assets_multiplier gives identical results
Uses the real input files in ./Data and the settings in src/run_montecarlo_simulation.toml
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TESTS_DIR))

from run_montecarlo_simulation import RunMontecarloSimulation  # noqa: E402
from test_montecarlo_data_loader import FileConfigManager  # noqa: E402

RUN_CNT = 30
ASSETS_MULTIPLIER = 1.1


def make_runner(nb_cpu: int, cross_correlated_rvs_flag: bool) -> RunMontecarloSimulation:
    config_manager = FileConfigManager(cross_correlated_rvs_flag)
    config_manager.full_config["default"].update({"nb_cpu": nb_cpu, "run_cnt": RUN_CNT})
    config_manager.config = config_manager.get_class_config("ConfigurationManager")  # type: ignore[attr-defined]
    return RunMontecarloSimulation(config_manager)  # type: ignore[arg-type]


def run_once(runner: RunMontecarloSimulation) -> tuple[pd.Series, int]:
    runner.reinitialize_data(ASSETS_MULTIPLIER)
    runner.run()
    return runner.final_result_series.copy(), runner.busted_cnt


@pytest.mark.parametrize("nb_cpu", [1, 2])
@pytest.mark.parametrize("cross_correlated_rvs_flag", [False, True])
def test_repeated_iteration_gives_identical_results(nb_cpu: int, cross_correlated_rvs_flag: bool) -> None:
    runner = make_runner(nb_cpu, cross_correlated_rvs_flag)
    try:
        first_series, first_busted_cnt = run_once(runner)
        second_series, second_busted_cnt = run_once(runner)
    finally:
        runner.close()

    assert first_busted_cnt == second_busted_cnt
    pd.testing.assert_series_equal(first_series, second_series)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
