#!/usr/local/bin/python3
"""
Tests for SimulationWorker: per-worker setup runs once, every iteration starts from the same state,
the parent's assets_multiplier is applied, and worker errors reach the parent
"""
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from simulation_worker_class import SimulationWorker  # noqa: E402

NB_ITERATIONS = 3


class FakeSimulation:
    """Picklable stand-in for MontecarloSimulation that records what the worker did to it"""
    def __init__(self, cross_correlated_rvs_flag: bool = True, fail: bool = False) -> None:
        self.cross_correlated_rvs_flag = cross_correlated_rvs_flag
        self.correlated_ror: list[int] | None = None
        self.generate_cnt = 0
        self.run_cnt_on_this_copy = 0  # > 0 at run() time would mean state leaked from a previous iteration
        self.assets_multiplier = 1.0
        self.fail = fail

    def generate_correlated_ror(self) -> None:
        self.generate_cnt += 1
        self.correlated_ror = [1, 2, 3]

    def initialize_data(self) -> None:
        pass

    def run(self) -> tuple[int, int, float]:
        if self.fail:
            raise ValueError("simulation failed")
        self.run_cnt_on_this_copy += 1
        return self.generate_cnt, self.run_cnt_on_this_copy, self.assets_multiplier


def run_iterations(worker: SimulationWorker, multipliers: list[float]) -> list[tuple[int, int, float]]:
    results = []
    for multiplier in multipliers:
        worker.submit(multiplier)
        results.append(worker.result())
    return results


def test_correlated_ror_generated_once_across_iterations() -> None:
    worker = SimulationWorker(FakeSimulation())
    try:
        results = run_iterations(worker, [1.0] * NB_ITERATIONS)
    finally:
        worker.close()
    assert [generate_cnt for generate_cnt, _, _ in results] == [1] * NB_ITERATIONS


def test_each_iteration_starts_from_a_fresh_copy() -> None:
    worker = SimulationWorker(FakeSimulation())
    try:
        results = run_iterations(worker, [1.0] * NB_ITERATIONS)
    finally:
        worker.close()
    assert [run_cnt for _, run_cnt, _ in results] == [1] * NB_ITERATIONS


def test_assets_multiplier_is_applied() -> None:
    multipliers = [1.0, 1.1, 0.95]
    worker = SimulationWorker(FakeSimulation())
    try:
        results = run_iterations(worker, multipliers)
    finally:
        worker.close()
    assert [multiplier for _, _, multiplier in results] == multipliers


def test_no_generation_in_independent_mode() -> None:
    worker = SimulationWorker(FakeSimulation(cross_correlated_rvs_flag=False))
    try:
        results = run_iterations(worker, [1.0])
    finally:
        worker.close()
    assert results[0][0] == 0


def test_worker_error_is_raised_in_parent_and_worker_survives() -> None:
    worker = SimulationWorker(FakeSimulation(fail=True))
    try:
        for _ in range(2):  # second call proves the worker is still responsive
            worker.submit(1.0)
            with pytest.raises(ValueError, match="simulation failed"):
                worker.result()
    finally:
        worker.close()
    assert not worker.process.is_alive()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
