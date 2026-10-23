"""E3 conformance for etregress: comparing two-step with two-step.

The registry had etregress at E1 with "no Stata comparison yet". When I finally made
one, the first attempt compared westata's control-function estimate against Stata's
DEFAULT, which is maximum likelihood -- two different estimators, so the 5.3e-2 gap on
the treatment effect measured the difference between methods, not a defect.

Stata exposes the two-step version as ``etregress, twostep``, and against that the
agreement is exact:

    parameter   westata            Stata twostep        gap
    x1          2.03392809         2.033928092563467    8e-10
    treated     1.49840416         1.4984041587868442   1e-9
    _cons       0.93480350         0.9348034986765116   1e-9
    z           0.58318977         0.5831897686471033   1e-9
    lambda      0.38980699         0.389806987829436   1e-9

The ML variant is deliberately not implemented (``method="ml"`` raises
UnsupportedError), so there is nothing to compare it to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
REFERENCE = FIXTURES / "e_upgrade_etregress.txt"

if not REFERENCE.exists():
    pytest.skip("etregress fixture missing", allow_module_level=True)


def _require(name: str):
    attribute = getattr(ws, name, None)
    if attribute is None:
        pytest.skip(f"{name} is not exposed on westata yet")
    return attribute


def treatreg_frame(n: int = 800, seed: int = 107) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x1 = rng.normal(size=n)
    common = rng.normal(size=n)
    u = 0.6 * common + 0.8 * rng.normal(size=n)
    e = 0.6 * common + 0.8 * rng.normal(size=n)
    treated = (0.6 * z + u) > 0
    y = 1.0 + 2.0 * x1 + 1.5 * treated.astype(float) + e
    return pd.DataFrame({"y": y, "x1": x1, "z": z, "treated": treated.astype(float)})


STATA_TWOSTEP = {
    "y:x1": 2.033928092563467,
    "y:treated": 1.4984041587868442,
    "y:Intercept": 0.9348034986765116,
    "treat:z": 0.5831897686471033,
    "treat:Intercept": -0.022583960821548055,
    "lambda": 0.38980698782943624,
}


def test_etregress_twostep_matches_stata():
    """Two-step against two-step, which is the only like-for-like comparison."""

    Etregress = _require("Etregress")
    result = Etregress.from_formula(
        "y ~ x1 + treat(treated ~ z)", treatreg_frame()
    ).fit(method="control")

    for name, expected in STATA_TWOSTEP.items():
        assert float(result.params[name]) == pytest.approx(expected, rel=1e-6), name


def test_etregress_ml_is_refused_rather_than_approximated():
    """Stata's default is ML; westata implements the two-step version only.

    Refusing is what keeps the comparison above meaningful -- if `method="ml"` quietly
    returned a two-step estimate, a caller could believe they had ML.
    """

    Etregress = _require("Etregress")
    model = Etregress.from_formula("y ~ x1 + treat(treated ~ z)", treatreg_frame())
    with pytest.raises(Exception):
        model.fit(method="ml")


def test_etregress_reports_the_inverse_mills_coefficient():
    """``lambda`` is the coefficient on the generated regressor, and Stata prints it."""

    Etregress = _require("Etregress")
    result = Etregress.from_formula(
        "y ~ x1 + treat(treated ~ z)", treatreg_frame()
    ).fit(method="control")
    assert "lambda" in result.param_names
    assert np.isfinite(float(result.params["lambda"]))
