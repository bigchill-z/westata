"""`rho`'s confidence interval: the probit link from `xtreg_ml.ado:329-332`.

`xtreg ..., mle` on lin_1992.dta prints

         rho |   .8028448   .0486812                      .6942946    .8840722

westata printed the estimate and the standard error but NO interval, and the code comment explained
why: the bounds sit 2.23 se below the estimate and only 1.67 se above, so a symmetric interval is out,
and log, logit, atanh and Fieller had all been tried without success.

THE MISSING PIECE IS THE SCALE. `xtreg_ml.ado:329-332`:

    est hidden local diparm3 sigma_u sigma_e, label(rho)        /*
        */ func(@1^2/(@1^2+@2^2))                                /*
        */ der( 2*@1*(@2/(@1^2+@2^2))^2 -2*@2*(@1/(@1^2+@2^2))^2 ) /*
        */ ci(probit)

Two things follow, and both are needed:

1. rho is a delta-method function of BOTH variance components, evaluated at (sigma_u, sigma_e) with
   their 2x2 covariance. Doing it from se(rho) alone cannot work -- it loses the covariance term.
2. `ci(probit)` builds the z-interval on `Phi^-1(rho)` and maps back through `Phi`. That produces the
   asymmetric spread directly.

Reproduced: both bounds match Stata to the last printed digit.

The SE is still .0486811 against Stata's .0486812 -- sigma_u differs in its ninth significant digit,
a convergence residual tracked separately and deliberately NOT papered over here.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

LIN = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
FORMULA = "ltvfo ~ ltlan + ltwlab + ltpow + ltfer + hrs + mci + ngca"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(LIN).exists(), reason="lin_1992.dta is not available"
)


@pytest.fixture(scope="module")
def fitted():
    frame = pd.read_stata(LIN).copy()
    return ws.RandomEffectsMLE.from_formula(
        FORMULA, frame, entity="province", time="t"
    ).fit()


def _rho_row(fitted) -> str:
    return next(
        line
        for line in str(fitted.summary()).splitlines()
        if line.startswith("         rho")
    )


def test_the_row_carries_both_interval_bounds(fitted) -> None:
    """The two numbers Stata prints on this row."""

    row = _rho_row(fitted)
    assert ".6942946" in row, row
    assert ".8840722" in row, row


def test_the_estimate_is_unchanged(fitted) -> None:
    assert ".8028448" in _rho_row(fitted)


def test_the_interval_is_asymmetric(fitted) -> None:
    """A symmetric interval would be [.7074314, .8982582]; Stata's is not that.

    This is why the earlier version printed nothing: it assumed a symmetric `rho -/+ z*se` or a
    single-variable transform, and the probit scale was never tried.
    """

    row = _rho_row(fitted)
    tokens = row.split()
    estimate = float(tokens[2])
    low = float(tokens[4])
    high = float(tokens[5])
    assert abs(estimate - low) > abs(high - estimate) * 1.2, (low, estimate, high)


def test_the_bounds_are_the_probit_transform(fitted) -> None:
    """Recompute them from the components the same way `diparm ..., ci(probit)` does."""

    import numpy as np
    from scipy.special import ndtri
    from scipy.stats import norm

    stored = fitted.e or {}
    sigma_u = float(stored["sigma_u"])
    sigma_e = float(stored["sigma_e"])
    rho = sigma_u**2 / (sigma_u**2 + sigma_e**2)
    se = float(stored.get("rho_se") or 0.0)
    assert se > 0.0

    transformed = float(ndtri(rho))
    spread = 1.959963985 * se / float(norm.pdf(transformed))
    low = float(norm.cdf(transformed - spread))
    high = float(norm.cdf(transformed + spread))

    row = _rho_row(fitted)
    assert f"{low:.7f}".lstrip("0")[:8] in row.replace("0.", "."), (low, row)
    assert f"{high:.7f}".lstrip("0")[:8] in row.replace("0.", "."), (high, row)


def test_style_statsmodels_does_not_crash(fitted) -> None:
    """The link change must not break the other renderer."""

    assert isinstance(str(fitted.summary(style="statsmodels")), str)
