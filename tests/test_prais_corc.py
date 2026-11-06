"""``prais ..., corc``: the Cochrane-Orcutt variant, added in round 3 of the v2 report.

The v2 report listed ``prais, corc`` as a missing option (P1, C08-05). It is the same
iteration as the Prais-Winsten default with one difference: the transformed first observation
is DISCARDED rather than scaled by ``sqrt(1 - rho^2)`` (``prais.ado:6``: ``syntax ... noDW
CORC``, and ``prais.ado:255`` records ``e(tranmeth)`` as either ``corc`` or ``prais``).

That is enough to change the answer materially, so the two are separate estimators:

    prais y k l          b = .15651652   .93948861   -7.5439159    rho = .12543124
    prais y k l, corc    b = .15687809   .92974899   -6.0923288    rho = .11463695

Both from a live StataNow run over cobb_douglas.dta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")


def _params(result) -> dict[str, float]:
    names = [str(name) for name in result._names]
    return {name: float(value) for name, value in zip(names, np.asarray(result.params))}


def test_corc_matches_stata(frame) -> None:
    result = ws.prais(frame, "y ~ k + l", corc=True)
    values = _params(result)
    assert values["k"] == pytest.approx(0.15687809, abs=1e-7)
    assert values["l"] == pytest.approx(0.92974899, abs=1e-7)
    # The intercept is an order of magnitude larger than the slopes, so an absolute
    # tolerance tight enough for them is too tight here; both fits converge to ~1e-6.
    assert values["Intercept"] == pytest.approx(-6.0923288, abs=1e-5)
    assert float(result.rho) == pytest.approx(0.11463695, abs=1e-6)


def test_prais_winsten_remains_unchanged(frame) -> None:
    """Guard: adding corc must not move the default estimator."""

    result = ws.prais(frame, "y ~ k + l")
    values = _params(result)
    assert values["k"] == pytest.approx(0.15651652, abs=1e-7)
    assert values["l"] == pytest.approx(0.93948861, abs=1e-7)
    assert values["Intercept"] == pytest.approx(-7.5439159, abs=1e-5)
    # rho converges to the package default tolerance (1e-8 on rho); the printed
    # Stata value is rounded to 8 decimals, so compare at that scale.
    assert float(result.rho) == pytest.approx(0.12543124, abs=1e-6)


def test_the_two_variants_actually_differ(frame) -> None:
    """If the switch were cosmetic this test would pass trivially; it must not."""

    winsten = ws.prais(frame, "y ~ k + l")
    corc = ws.prais(frame, "y ~ k + l", corc=True)
    assert abs(float(winsten.rho) - float(corc.rho)) > 0.005
    assert abs(_params(winsten)["Intercept"] - _params(corc)["Intercept"]) > 0.5
