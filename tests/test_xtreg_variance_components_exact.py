"""The three xtreg estimators now agree with Stata on sigma_u / sigma_e / rho.

Round 46 fixed `xtreg, re`'s sigma_e: it is the WITHIN regression's rmse, read from
`xtreg_re.ado:128/338`, not the random-effects fit's own residual sum of squares divided by N. That
correction relays through Swamy-Arora into sigma_u and then rho.

Measured against a live Stata on munnell.dta, `lny ~ lnk1 + lnk2 + lnlabor`:

    estimator      quantity    westata              Stata
    xtreg, fe      sigma_u     0.10711260397        0.1071126040
                   sigma_e     0.03882160876        0.0388216088
                   rho         0.8838913003         0.8838913003
    xtreg, re      sigma_u     0.08186286861        0.0818628686
                   sigma_e     0.03882160876        0.0388216088
                   rho         0.8163988065         0.8163988065

`xtreg, be` stores none of the three -- Stata prints "." for all of them, and so does this package,
because the between estimator runs OLS on entity means and has no decomposed error structure to
report.

The two estimators share sigma_e HERE, which is not a coincidence: for `re` it is the within rmse by
construction (see above), and for `fe` it is that same fit's rmse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"
FORMULA = "lny ~ lnk1 + lnk2 + lnlabor"

#: Captured from `xtreg ..., fe` and `xtreg ..., re` at %20.10f.
FE = (0.1071126040, 0.0388216088, 0.8838913003)
RE = (0.0818628686, 0.0388216088, 0.8163988065)


@pytest.fixture(scope="module")
def frame():
    try:
        loaded = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    loaded = loaded.copy()
    loaded["sid"] = pd.Categorical(loaded["st_abb"]).codes
    loaded["year"] = loaded["year"].astype(int)
    return loaded


@pytest.mark.parametrize(
    "klass, expected",
    [(ws.PanelOLS, FE), (ws.RandomEffects, RE)],
    ids=["fe", "re"],
)
def test_the_variance_components_are_exact(frame, klass, expected) -> None:
    fitted = klass.from_formula(FORMULA, frame, entity="sid", time="year").fit()
    sigma_u, sigma_e, rho = expected
    assert float(fitted.e["sigma_u"]) == pytest.approx(sigma_u, abs=1e-10)
    assert float(fitted.e["sigma_e"]) == pytest.approx(sigma_e, abs=1e-10)
    assert float(fitted.e["rho"]) == pytest.approx(rho, abs=1e-10)


def test_fe_and_re_share_sigma_e(frame) -> None:
    """Not an accident: for `re` it IS the within rmse, and for `fe` it is that fit's rmse.

    A regression here would mean one of the two is being computed from the wrong residual.
    """

    fe = ws.PanelOLS.from_formula(FORMULA, frame, entity="sid", time="year").fit()
    re = ws.RandomEffects.from_formula(FORMULA, frame, entity="sid", time="year").fit()
    assert float(fe.e["sigma_e"]) == pytest.approx(float(re.e["sigma_e"]), abs=1e-12)


def test_the_between_estimator_stores_none_of_them(frame) -> None:
    """`xtreg, be` prints "." for all three; the package must not invent values."""

    fitted = ws.BetweenOLS.from_formula(FORMULA, frame, entity="sid", time="year").fit()
    for name in ("sigma_u", "sigma_e", "rho"):
        assert name not in fitted.e or not np.isfinite(float(fitted.e[name]))
