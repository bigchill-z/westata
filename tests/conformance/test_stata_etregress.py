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


def test_etregress_ml_matches_stata():
    """Stata's DEFAULT for `etregress` is ML; `twostep` is the option.

    Pinned after implementing `method='ml'`. The likelihood was verified separately at STAT'S estimates (2.0e-16
    relative), which is the check that catches a wrong model -- the etregress and heckman likelihoods are NOT the
    same and using heckman's here would converge to a different answer while looking plausible.

    Reference from a live Stata session on the sample below: likelihood -1124.693682764672 with rho 0.42409865
    and sigma 0.96618228.
    """

    import numpy as np
    import pandas as pd

    import westata as ws

    rng = np.random.default_rng(7)
    n = 600
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    u = rng.normal(size=n)
    v = 0.6 * u + np.sqrt(0.64) * rng.normal(size=n)
    treated = (0.5 + 0.7 * z1 - 0.4 * z2 + u > 0).astype(float)
    frame = pd.DataFrame(
        {"y": 1.0 + 0.9 * x1 + v, "x1": x1, "z1": z1, "z2": z2, "t": treated}
    )

    result = ws.Etregress.from_formula("y ~ x1 + treat(t ~ z1 + z2)", frame).fit(method="ml")

    assert float(result.e["ll"]) == pytest.approx(-1124.693682764672, rel=1e-12)
    assert float(result.e["rho"]) == pytest.approx(0.4240986539296641, rel=1e-6)
    assert float(result.e["sigma"]) == pytest.approx(0.9661822784535223, rel=1e-6)

    # Coefficients, in our block order rather than Stata's.
    values = dict(zip(result.param_names, np.asarray(result.params, dtype=float)))
    assert values["y:x1"] == pytest.approx(0.92442027, rel=1e-5)
    assert values["y:t"] == pytest.approx(0.14566395, rel=1e-4)
    assert values["y:Intercept"] == pytest.approx(0.90985491, rel=1e-5)
    assert values["treat:z1"] == pytest.approx(0.69259854, rel=1e-5)
    assert values["treat:z2"] == pytest.approx(-0.34114759, rel=1e-5)
    assert values["treat:Intercept"] == pytest.approx(0.47796736, rel=1e-5)


def test_etregress_reports_the_inverse_mills_coefficient():
    """``lambda`` is the coefficient on the generated regressor, and Stata prints it."""

    Etregress = _require("Etregress")
    result = Etregress.from_formula(
        "y ~ x1 + treat(treated ~ z)", treatreg_frame()
    ).fit(method="control")
    assert "lambda" in result.param_names
    assert np.isfinite(float(result.params["lambda"]))
