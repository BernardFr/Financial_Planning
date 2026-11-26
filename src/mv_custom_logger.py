"""Project logger: one named logger writing to the console and to ``<prog_name>.log``.

Every module logs through ``custom_logger``::

    from custom_logger import custom_logger as logger

The main program configures it once, passing its own root name::

    setup_logger(Path(__file__).stem)   # e.g. "holdings_cross_corr"

Its starting level is ``log_level`` from the [meta] table of
``<prog_name>.toml``; when a different config is loaded (``--config``) or
``--verbose`` is passed, the loader calls :func:`set_log_level` to apply that
config's level instead.
"""

import logging
import tomllib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOGGER_NAME = "custom_logger"
FILE_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
CONSOLE_LOG_FORMAT = "%(levelname)s - %(funcName)s - %(message)s"

custom_logger = logging.getLogger(LOGGER_NAME)


def _default_log_level(toml_path: Path) -> str:
    """``log_level`` from the config's [meta] table."""
    meta = tomllib.loads(toml_path.read_text())["meta"]
    return str(meta["log_level"]).upper()


def set_log_level(level: str) -> None:
    """Set the logger and all its handlers to ``level``."""
    custom_logger.setLevel(level)
    for handler in custom_logger.handlers:
        handler.setLevel(level)


def setup_logger(prog_name: str) -> logging.Logger:
    """Attach the file and console handlers for ``prog_name`` (idempotent)."""
    toml_default = f"{prog_name}.toml"
    log_file = f"{prog_name}.log"

    if not custom_logger.handlers:
        file_handler = logging.FileHandler(PROJECT_DIR / log_file)
        file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
        custom_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(CONSOLE_LOG_FORMAT))
        custom_logger.addHandler(console_handler)

        # Keep messages out of the root logger so they are not printed twice.
        custom_logger.propagate = False
        set_log_level(_default_log_level(PROJECT_DIR / toml_default))

    return custom_logger
