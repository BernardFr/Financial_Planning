#!/usr/local/bin/python3
"""
SimulationWorker: runs one MontecarloSimulation in a dedicated, long-lived process

The process keeps the simulation alive across iterations, so expensive per-CPU setup - generating the
cross-correlated RoR table - happens once, inside the worker. Each iteration runs on a fresh copy of the
prepared simulation, so every iteration starts from the same random-number state (common random numbers):
changing assets_multiplier is the only difference between iterations.

Protocol over the pipe: parent sends an assets_multiplier (float) to run one iteration, or None to stop.
Worker replies with the result of MontecarloSimulation.run() - or the exception it raised.
"""

import copy
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any

from custom_logger import setup_logger

STOP_COMMAND = None


def _worker_loop(simulation: Any, conn: Connection, prog_name: str) -> None:
    """Entry point of the worker process"""
    setup_logger(prog_name)  # a spawned process starts with an unconfigured logger
    if simulation.cross_correlated_rvs_flag and simulation.correlated_ror is None:
        simulation.generate_correlated_ror()  # once per worker
    while (assets_multiplier := conn.recv()) is not STOP_COMMAND:
        try:
            iteration_simulation = copy.deepcopy(simulation)
            iteration_simulation.assets_multiplier = assets_multiplier
            iteration_simulation.initialize_data()
            conn.send(iteration_simulation.run())
        except Exception as exc:  # report to the parent instead of leaving it blocked on recv()
            conn.send(exc)
    conn.close()


class SimulationWorker:
    """Parent-side handle on a worker process that owns one MontecarloSimulation"""

    def __init__(self, simulation: Any, prog_name: str) -> None:
        self.conn, child_conn = Pipe()
        self.process = Process(target=_worker_loop, args=(simulation, child_conn, prog_name), daemon=True)
        self.process.start()
        child_conn.close()  # the child owns its end

    def submit(self, assets_multiplier: float) -> None:
        """Start one iteration - returns immediately so all workers run in parallel"""
        self.conn.send(assets_multiplier)

    def result(self) -> Any:
        """Wait for the iteration started by submit() and return its result"""
        result = self.conn.recv()
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        """Stop the worker process"""
        if self.process.is_alive():
            self.conn.send(STOP_COMMAND)
            self.process.join()
        self.conn.close()
