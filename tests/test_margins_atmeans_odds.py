"""Pin ``margins(..., atmeans=True)`` and ``odds_ratios`` against Stata.

References from live runs on grilic.dta after ``quietly logit rns s expr tenure``::

    margins, dydx(s expr)
        s     -.0135532   .0075125
        expr  -.0002224   .0079914

    margins, dydx(s expr) atmeans
        At: s = 13.40501, expr = 1.735429, tenure = 1.831135 (mean)
        s     -.0135868   .0075583
        expr   -.000223   .0080113

    logit rns s expr tenure, or
        s      .9330414   .036064
        expr   .9988632  .0408187
        tenure .9457414  .0496525
        _cons  1.027265  .5595364

``atmeans`` is not a cosmetic variant of the default: it reports a CONDITIONAL effect at the
covariate means, where the default averages the effect over the sample. The two differ in the
fourth decimal on both coefficients, which is exactly the size that survives review unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.Logit.from_formula("rns ~ s + expr + tenure", frame).fit()


def test_average_marginal_effects_match_stata(fitted) -> None:
    rows = ws.margins(fitted, dydx=["s", "expr"])
    observed = {row.label: (row.estimate, row.std_err) for row in rows}
    assert observed["s"][0] == pytest.approx(-0.0135532, abs=5e-8)
    assert observed["s"][1] == pytest.approx(0.0075125, abs=5e-8)
    assert observed["expr"][0] == pytest.approx(-0.0002224, abs=5e-8)


def test_atmeans_marginal_effects_match_stata(fitted) -> None:
    rows = ws.margins(fitted, dydx=["s", "expr"], atmeans=True)
    observed = {row.label: (row.estimate, row.std_err) for row in rows}
    assert observed["s"][0] == pytest.approx(-0.0135868, abs=5e-8)
    assert observed["s"][1] == pytest.approx(0.0075583, abs=5e-8)
    assert observed["expr"][0] == pytest.approx(-0.000223, abs=5e-8)


def test_atmeans_is_not_the_default(fitted) -> None:
    """A conditional effect at the means is a different number from average effects."""

    # A single-term request returns one MarginsResult rather than a list.
    average = ws.margins(fitted, dydx=["s"]).estimate
    conditional = ws.margins(fitted, dydx=["s"], atmeans=True).estimate
    assert abs(average - conditional) > 1e-5


def test_at_and_atmeans_cannot_be_combined(fitted) -> None:
    with pytest.raises(Exception):
        ws.margins(fitted, dydx=["s"], at={"s": 12.0}, atmeans=True)


def test_odds_ratios_match_stata(fitted) -> None:
    ratios = fitted.odds_ratios
    assert float(ratios["s"]) == pytest.approx(0.9330414, abs=5e-7)
    assert float(ratios["expr"]) == pytest.approx(0.9988632, abs=5e-7)
    assert float(ratios["tenure"]) == pytest.approx(0.9457414, abs=5e-7)
    assert float(ratios["Intercept"]) == pytest.approx(1.027265, abs=5e-6)


def test_odds_ratios_are_exp_of_the_coefficients(fitted) -> None:
    ratios = fitted.odds_ratios
    for name in ratios.index:
        assert float(ratios[name]) == pytest.approx(
            float(np.exp(float(fitted.params[name]))), rel=1e-12
        )


def test_odds_ratios_refuse_non_log_odds_models() -> None:
    """Returning exp(beta) for a Poisson mean would be a number with no meaning."""

    frame = pd.DataFrame(
        {
            "y": np.random.default_rng(3).normal(size=120),
            "x": np.random.default_rng(4).normal(size=120),
        }
    )
    ols = ws.OLS.from_formula("y ~ x", frame).fit()
    with pytest.raises(Exception):
        _ = ols.odds_ratios
