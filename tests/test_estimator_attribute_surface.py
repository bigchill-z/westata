"""The attribute surface of every estimator, pinned so entry-point guesses become lookups.

Rounds 130, 132, 145 and 153 each lost time to the same class of mistake: guessing where a quantity
lives, getting an AttributeError or TypeError, and briefly concluding a feature was absent. Every one of
those features worked. The four cases:

  * `cov_params` is a METHOD on OLS/Logit/Poisson/FEOLS and an ATTRIBUTE on VARResults -- calling it
    unconditionally raised "'numpy.ndarray' object is not callable" after a `var` (round 130);
  * `irf_table` and `fevd_table` live on the VAR fit, NOT on the statsmodels object that `irf()`
    returns (rounds 145, 149);
  * leverage is the dict KEY `get_influence()["hat_matrix_diag"]`, not an attribute (round 153).

The table, measured rather than assumed:

    attribute       ols   logit  poisson  feols  var
    params          yes   yes    yes      yes    yes
    bse             yes   yes    yes      yes    yes
    resid           yes   yes    yes      yes    -
    fittedvalues    yes   yes    yes      yes    -
    get_influence   call  call   call     call   -
    cov_params      call  call   call     call   attr
    param_names     yes   yes    yes      yes    -
    model_name      yes   yes    yes      yes    yes
    response_name   yes   yes    yes      yes    yes
    irf_table       -     -      -        -      call
    fevd_table      -     -      -        -      call

THE POINT OF THIS FILE IS NOT THE TABLE ITSELF but that a future reader can check it instead of probing
in a REPL for a round.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fits():
    rng = np.random.default_rng(17)
    n = 300
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": np.repeat(np.arange(30), 10),
        }
    )
    frame["binary"] = (frame["y"] > 0).astype(float)
    frame["y_panel"] = frame["y"] + frame["g"] * 0.1
    return {
        "ols": ws.OLS.from_formula("y ~ x + z", frame).fit(),
        "logit": ws.Logit.from_formula("binary ~ x + z", frame).fit(),
        "feols": ws.FEOLS.from_formula("y_panel ~ x + z | g", frame).fit(),
        "var": ws.VAR(frame[["y", "x"]], lags=2).fit(),
    }


def test_cov_params_is_a_method_on_the_single_equation_fits(fits) -> None:
    for name in ("ols", "logit", "feols"):
        assert callable(fits[name].cov_params), name


def test_cov_params_is_an_attribute_on_the_var_fit(fits) -> None:
    """The asymmetry that raised TypeError in round 130."""

    attribute = fits["var"].cov_params
    assert not callable(attribute)
    assert np.asarray(attribute).ndim == 2


def test_the_var_fit_exposes_model_and_response_names(fits) -> None:
    """`estat vce` builds its title from these; they were absent before round 132."""

    assert fits["var"].model_name == "var"
    # VAR renames unnamed columns to y1, y2, ... because statsmodels needs distinct equation labels.
    # The property reports whatever names the fit actually carries.
    assert fits["var"].response_name == " ".join(fits["var"].names)


def test_irf_and_fevd_tables_are_on_the_fit_not_on_irf(fits) -> None:
    fit = fits["var"]
    assert callable(fit.irf_table)
    assert callable(fit.fevd_table)
    assert not hasattr(fit.irf(periods=2), "irf_table")


def test_leverage_is_a_dict_key_on_the_influence_object(fits) -> None:
    for name in ("ols", "feols"):
        influence = fits[name].get_influence()
        assert "hat_matrix_diag" in influence
        assert not hasattr(influence, "hat_matrix_diag")


def test_the_residual_and_fitted_values_agree_in_length(fits) -> None:
    """A shape mismatch here is how a library-version change would first show up."""

    for name in ("ols", "logit", "feols"):
        fit = fits[name]
        assert len(np.asarray(fit.resid)) == len(np.asarray(fit.fittedvalues))


def test_the_var_fit_has_neither_residual_nor_fitted_values(fits) -> None:
    """Not an omission: a VAR models several equations at once, so there is no single residual vector.

    The serial-correlation tests refuse a VAR for this reason, which round 134 pinned separately.
    """

    fit = fits["var"]
    assert not hasattr(fit, "resid")
    assert not hasattr(fit, "fittedvalues")
