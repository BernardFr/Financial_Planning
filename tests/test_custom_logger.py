#!/usr/local/bin/python3
"""
Tests for custom_logger: the log level comes from [meta] log_level in <prog_name>.toml.
"""
import logging
import sys
import tomllib
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parent.parent / "src" / "Using_Cross_Cor"
sys.path.insert(0, str(CODE_DIR))

import custom_logger  # noqa: E402

PROGRAMS = ["run_montecarlo_simulation", "holdings_cross_corr"]


@pytest.fixture
def clean_logger(tmp_path: Path, monkeypatch):
    """Point custom_logger at tmp_path and give each test a logger without handlers."""
    logger = custom_logger.custom_logger
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers.clear()
    monkeypatch.setattr(custom_logger, "PROJECT_DIR", tmp_path)
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers[:], logger.level, logger.propagate = saved


@pytest.mark.parametrize("level", ["DEBUG", "WARNING"])
def test_setup_logger_uses_toml_level(clean_logger, tmp_path: Path, level: str) -> None:
    (tmp_path / "prog.toml").write_text(f'[meta]\nlog_level = "{level.lower()}"\n')

    custom_logger.setup_logger("prog")

    assert clean_logger.level == getattr(logging, level)
    assert all(h.level == getattr(logging, level) for h in clean_logger.handlers)
    assert (tmp_path / "prog.log").exists()


@pytest.mark.parametrize("prog_name", PROGRAMS)
def test_program_toml_has_valid_log_level(prog_name: str) -> None:
    meta = tomllib.loads((CODE_DIR / f"{prog_name}.toml").read_text())["meta"]

    assert isinstance(logging.getLevelName(str(meta["log_level"]).upper()), int)
