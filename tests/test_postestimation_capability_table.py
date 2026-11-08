"""The postestimation surface: what each command accepts, and why the refusals are correct.

Rounds 130-133 fixed four instances of one defect family. Round 134 scanned 10 commands against 9
estimators to find the rest at once. This file records the resulting capability table as tests, so a
future change that quietly widens or narrows it shows up as a failure.

THE TABLE (ok = supported, - = refused):

    command          ols  wls  logit probit poisson nbreg iv2sls feols var
    estat_ic         ok   ok   ok    ok     ok      ok    ok     ok    ok
    estat_vce        ok   ok   ok    ok     ok      ok    ok     ok    ok
    estat_vif        ok   ok   -     -      -       -     -      ok    -
    dwatson          ok   ok   ok    ok     ok      ok    ok     ok    -
    hettest          ok   ok   -     -      -       -     ok     ok    -
    ovtest           ok   ok   -     -      -       -     ok     ok    -
    bgodfrey         ok   ok   ok    ok     ok      ok    ok     ok    -
    imtest           ok   ok   -     -      -       -     -      -     -
    gof              -    -    ok    ok     ok      -     -      -     -
    classification   -    -    ok    ok     -       -     -      -     -

WHY EACH REFUSAL IS CORRECT rather than a gap:

  * vif on a non-linear fit -- variance inflation is a property of a linear design;
  * dwatson and bgodfrey on a VAR -- they need single-equation residuals, and Stata refuses there too;
  * hettest and ovtest on a non-linear fit -- both are tests on OLS residuals;
  * imtest on anything but a linear fit -- Stata offers it after `regress` (and the White variant is
    the one ported here);
  * gof and classification on a linear fit -- they are discrete-outcome diagnostics.

`estat_ic` and `estat_vce` accepting every estimator is the point of rounds 130-132: they used to
assume one interface and now read whichever the result provides.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(11)
    n = 300
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n) + 0.5 * rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": np.repeat(np.arange(30), 10),
        }
    )
    data["binary"] = (data["y"] > 0).astype(float)
    data["count"] = rng.poisson(2.0, n).astype(float)
    data["y_panel"] = data["y"] + data["g"] * 0.1
    return data


@pytest.fixture(scope="module")
def fits(frame):
    return {
        "ols": ws.OLS.from_formula("y ~ x + z", frame).fit(),
        "logit": ws.Logit.from_formula("binary ~ x + z", frame).fit(),
        "feols": ws.FEOLS.from_formula("y_panel ~ x + z | g", frame).fit(),
        "var": ws.VAR(frame[["y", "x"]], lags=2).fit(),
    }


def test_estat_ic_accepts_every_estimator(fits) -> None:
    """Rounds 130-132 made this true; it read assumptions before."""

    for name, fit in fits.items():
        result = ws.estat_ic(fit)
        assert result is not None, name


def test_estat_vce_accepts_every_estimator(fits) -> None:
    for name, fit in fits.items():
        result = ws.estat_vce(fit)
        assert result.values.shape[0] == result.values.shape[1], name


def test_vif_refuses_a_non_linear_fit(fits) -> None:
    with pytest.raises(Exception):
        ws.estat_vif(fits["logit"])


def test_serial_correlation_tests_refuse_a_var(fits) -> None:
    for command in (ws.estat_dwatson, ws.estat_bgodfrey):
        with pytest.raises(Exception):
            command(fits["var"])


def test_hettest_and_ovtest_refuse_a_non_linear_fit(fits) -> None:
    for command in (ws.estat_hettest, ws.estat_ovtest):
        with pytest.raises(Exception):
            command(fits["logit"])
        # ...and both work on a linear one, including an absorbed fit (round 133).
        assert command(fits["ols"]) is not None
        assert command(fits["feols"]) is not None


def test_discrete_diagnostics_refuse_a_linear_fit(fits) -> None:
    for command in (ws.estat_gof, ws.estat_classification):
        with pytest.raises(Exception):
            command(fits["ols"])
        assert command(fits["logit"]) is not None
