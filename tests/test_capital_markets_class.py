#!/usr/local/bin/python3
"""
Tests for CapitalMarkets: seeded correlated RoR generation and matching stats to portfolio asset classes
"""
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from capital_markets_class import CapitalMarkets  # noqa: E402

ASSET_CLASSES = ["Stocks", "Bonds", "Cash"]
STATS_DF = pd.DataFrame(
    {"Expected Return": [7.0, 4.0, 2.0], "Standard Deviation": [15.0, 6.0, 0.5]},
    index=ASSET_CLASSES,
)
CORR_DF = pd.DataFrame(
    [[1.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 1.0]],
    index=ASSET_CLASSES, columns=ASSET_CLASSES,
)
NB_SMPL = 2_000
SEED = 42


class FakeConfigManager:
    """Minimal stand-in for ConfigurationManager - CapitalMarkets only needs get_class_config"""
    def get_class_config(self, class_name: str) -> dict[str, Any]:
        return {}


def make_capital_markets() -> CapitalMarkets:
    capital_markets = CapitalMarkets(FakeConfigManager())
    capital_markets.set_stat_df_and_corr_df(STATS_DF.copy(), CORR_DF.copy())
    capital_markets.set_nb_smpl(NB_SMPL)
    return capital_markets


def make_rng_list(seed: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(ASSET_CLASSES))]


def test_seeded_rngs_are_reproducible() -> None:
    ror_1 = make_capital_markets().generate_correlated_ror(make_rng_list(SEED))
    ror_2 = make_capital_markets().generate_correlated_ror(make_rng_list(SEED))
    pd.testing.assert_frame_equal(ror_1, ror_2)


def test_different_rngs_give_different_samples() -> None:
    ror_1 = make_capital_markets().generate_correlated_ror(make_rng_list(SEED))
    ror_2 = make_capital_markets().generate_correlated_ror(make_rng_list(SEED + 1))
    assert not ror_1.equals(ror_2)


def test_generated_ror_matches_target_stats() -> None:
    ror_df = make_capital_markets().generate_correlated_ror(make_rng_list(SEED)).astype(float)
    assert list(ror_df.index) == ASSET_CLASSES
    assert ror_df.shape == (len(ASSET_CLASSES), NB_SMPL)
    pct_df = (ror_df - 1.0) * 100.0  # multipliers -> percent
    np.testing.assert_allclose(pct_df.mean(axis=1), STATS_DF["Expected Return"], atol=1.0)
    np.testing.assert_allclose(pct_df.std(axis=1), STATS_DF["Standard Deviation"], rtol=0.1)
    assert pct_df.T.corr().loc["Stocks", "Bonds"] == pytest.approx(0.3, abs=0.1)


def test_unseeded_generation_still_works() -> None:
    ror_df = make_capital_markets().generate_correlated_ror()
    assert ror_df.shape == (len(ASSET_CLASSES), NB_SMPL)


def test_wrong_number_of_rngs_exits() -> None:
    with pytest.raises(SystemExit):
        make_capital_markets().generate_correlated_ror(make_rng_list(SEED)[:-1])


def test_match_stats_vs_assets_drops_extras_and_uses_portfolio_order() -> None:
    portfolio_ser = pd.Series([100.0, 300.0], index=["Bonds", "Stocks"])
    stats_df, corr_df = make_capital_markets().match_stats_vs_assets(portfolio_ser)
    assert list(stats_df.index) == ["Bonds", "Stocks"]
    assert list(corr_df.index) == ["Bonds", "Stocks"]
    assert list(corr_df.columns) == ["Bonds", "Stocks"]
    assert stats_df.loc["Stocks", "Expected Return"] == 7.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
