"""`prais, corc` on icecream.dta: BOTH Stata and this package fail to converge, and agree on where.

C08-05 scores `prais, corc` at 0.2593. Measured on icecream.dta:

    tsset time
    prais consumption price income, corc
        rho |   .9724118
        Durbin-Watson statistic (original)    = 0.423514
        Durbin-Watson statistic (transformed) = 1.568047
        convergence not achieved
        r(430);

STATA RAISES r(430) HERE TOO. This is not a case of the package failing where Stata succeeds -- the
data does not support the iterative Cochrane-Orcutt update, and both implementations report it.

THE PACKAGE GOES ONE STEP FURTHER and carries the reached rho on the exception as `failure.rho`, so a
caller can see that the estimate was travelling toward .9724 rather than diverging. That matters:
a rho of .97 that is still creeping and one that is jumping wildly look identical from a bare error
message, and only one of them is worth re-running with a looser tolerance.

The guards below pin that the default limit fails where Stata fails, carrying the reached rho, and
that a raised limit then converges -- so this is slowness rather than divergence.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConvergenceError

ICECREAM = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(ICECREAM)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("icecream.dta is not available")


def test_corc_raises_on_this_data_as_stata_does(frame) -> None:
    """Stata raises r(430) on the same input, so success here would be the surprise."""

    with pytest.raises(ConvergenceError):
        ws.prais(frame, "consumption ~ price + income", corc=True)


def test_the_reached_rho_travels_on_the_exception(frame) -> None:
    """Stata prints rho .9724118 before reporting the failure; the package reaches .9724151."""

    with pytest.raises(ConvergenceError) as caught:
        ws.prais(frame, "consumption ~ price + income", corc=True)

    reached = getattr(caught.value, "rho", None)
    assert reached is not None, "the reached rho must survive the exception"
    assert float(reached) == pytest.approx(0.9724, abs=5e-4)


def test_a_higher_iteration_limit_does_converge(frame) -> None:
    """It is a SLOW series, not a divergent one -- 1000 passes settle at rho 0.9724596.

    This is the opposite of what I first assumed. The default limit of 100 reports failure exactly
    where Stata does, and raising it converges, so the two systems stop at the same place for the
    same reason rather than one of them being wrong.
    """

    fitted = ws.prais(frame, "consumption ~ price + income", corc=True, maxiter=1000)
    assert float(fitted.rho) == pytest.approx(0.9724596, abs=1e-6)


def test_the_converged_coefficients_match_stata(frame) -> None:
    """Stata with `iterate(1000)`: price -1.075987, income -.0033641, _cons 1.173607, rho .9724596.

    This turns the non-convergence note into a complete account: at the limit where Stata stops the
    package stops too, and at the limit where it finishes the numbers agree.
    """

    fitted = ws.prais(frame, "consumption ~ price + income", corc=True, maxiter=1000)

    assert float(fitted.params["price"]) == pytest.approx(-1.075987, abs=1e-6)
    assert float(fitted.params["income"]) == pytest.approx(-0.0033641, abs=1e-7)
    assert float(fitted.params["Intercept"]) == pytest.approx(1.173607, abs=1e-6)
    assert float(fitted.rho) == pytest.approx(0.9724596, abs=1e-7)
