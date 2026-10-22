"""E3 conformance for reghdfe, the user-written package.

``reghdfe`` is not on SSC from this network, so it was installed from the
author's GitHub sources with ``scripts/install_reghdfe.py``.  These fixtures let
the comparison replay without Stata.

The interesting finding encoded here: under clustering, reghdfe's ``e(df_r)`` is
``G - 1``, not ``N - rank``, because a fixed effect nested in the cluster
variable is marked redundant.  westata had this wrong until these fixtures
exposed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.conformance.oracles.stata import StataCapture

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"


def panel_frame(n_entities: int = 30, n_periods: int = 6, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    period = rng.normal(size=n_periods)[time - 1]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + period + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def _load(name: str) -> StataCapture:
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name}.json missing; run scripts/capture_reghdfe_fixtures.py")
    return StataCapture.load(path)


def _rel(actual, expected) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1e-12)))


def _slope_positions(result, capture):
    """Compare slopes only: reghdfe reports a _cons an absorbed model lacks."""

    westata_positions = []
    stata_positions = []
    for index, name in enumerate(capture.names):
        key = "Intercept" if name == "_cons" else name
        if key in result.param_names:
            westata_positions.append(list(result.param_names).index(key))
            stata_positions.append(index)
    return np.asarray(westata_positions), np.asarray(stata_positions)


def _assert_matches(name: str, formula: str, cov_type: str, cov_kwds=None):
    capture = _load(name)
    frame = panel_frame()
    result = ws.FEOLS.from_formula(formula, frame).fit(
        cov_type=cov_type, cov_kwds=cov_kwds
    )
    westata_positions, stata_positions = _slope_positions(result, capture)

    # Measured across all five cases: 4.4e-16 on parameters, 2.0e-16 on SEs.
    assert _rel(
        result.params.to_numpy()[westata_positions],
        np.asarray(capture.b, dtype=float)[stata_positions],
    ) < 1e-12
    assert _rel(
        result.bse.to_numpy()[westata_positions],
        np.sqrt(np.diag(capture.V))[stata_positions],
    ) < 1e-12
    assert result.e["N"] == capture.scalars["N"]
    assert result.e["r2"] == pytest.approx(capture.scalars["r2"], rel=1e-12)
    assert result.e["df_r"] == capture.scalars["df_r"]
    return result


def test_reghdfe_one_way_fe():
    _assert_matches("reghdfe_firm", "y ~ x1 + x2 | firm", "nonrobust")


def test_reghdfe_two_way_fe():
    _assert_matches("reghdfe_firm_year", "y ~ x1 + x2 | firm + year", "nonrobust")


def test_reghdfe_robust():
    _assert_matches("reghdfe_firm_robust", "y ~ x1 + x2 | firm", "HC1")


def test_reghdfe_clustered_uses_g_minus_1_for_df_r():
    result = _assert_matches(
        "reghdfe_firm_cluster",
        "y ~ x1 + x2 | firm",
        "cluster",
        {"cluster": panel_frame()["firm"]},
    )
    # The firm effect is nested in the cluster variable, so reghdfe treats it as
    # redundant and e(df_r) becomes G - 1 rather than N - rank.  westata was
    # reporting 177 here before this test existed.
    assert result.e["df_r"] == 29


def test_reghdfe_two_way_clustered():
    result = _assert_matches(
        "reghdfe_firm_year_cluster",
        "y ~ x1 + x2 | firm + year",
        "cluster",
        {"cluster": panel_frame()["firm"]},
    )
    assert result.e["df_r"] == 29


def test_fixtures_record_the_command():
    for name in [
        "reghdfe_firm",
        "reghdfe_firm_year",
        "reghdfe_firm_robust",
        "reghdfe_firm_cluster",
        "reghdfe_firm_year_cluster",
    ]:
        capture = _load(name)
        assert capture.macros.get("cmd"), f"{name} lacks e(cmd)"
        assert capture.names
