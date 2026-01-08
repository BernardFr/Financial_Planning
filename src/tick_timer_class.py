
import time 
from logger import logger
VERBOSE = False

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class TickTimer:
    """ Timer that provides interval time as well as total time
    Reference: https://realpython.com/python-timer/#a-python-timer-class
    """

    def __init__(self, name=None) -> None:
        self._start_time = None
        self._tick_time = None
        self._tick_dict = {}
        self._tick_count = 0
        self._name = name if name is not None else "Tick"
        return

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")
        if self._tick_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
        self._tick_time = time.perf_counter()
        self._tick_count = 0
        return

    def tick(self, msg: str = ''  ) -> None:
        """ Report the elapsed time since the last tick """
        if self._tick_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        now_time = time.perf_counter()
        assert self._tick_time is not None
        assert self._start_time is not None
        elapsed_time = now_time - self._tick_time
        total_time = now_time - self._start_time
        self._tick_time = now_time
        if VERBOSE:
            if msg is not None:
                logger.info(f"{self._name}: {msg}: Elapsed time: {elapsed_time:,.2f} secs - \
                Total time: {total_time:,.2f} secs")
            else:
                logger.info(f"{self._name}: Elapsed time since last Tick: {elapsed_time:,.2f} secs\
                 - Total time: {total_time:,.2f} secs")
        self._tick_dict[self._tick_count] = [msg, elapsed_time]
        self._tick_count += 1
        return

    def stop(self, quiet_flag: bool = False) -> None:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        sorted_intervals = sorted(self._tick_dict.items())
        elapsed_min = int(elapsed_time / 60.0)
        elapsed_hr = int(elapsed_min / 60.0)
        elapsed_min -= 60 * elapsed_hr
        # print(f"{self._name}:
        # Total Elapsed time: {elapsed_hr:02,d}:{elapsed_min:02,d}:{elapsed_sec:02,d}")
        if not quiet_flag:
            logger.info(f"{self._name}: Timer Intervals:")
            for _, itm in sorted_intervals:  # don't need the key
                logger.info(f"{itm[0]}: {itm[1]:,.2f} seconds")  # -> msg: tick time
        self._start_time = None

        return
