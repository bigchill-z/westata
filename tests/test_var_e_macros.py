"""``var``'s ``e()`` macros, checked against ``var.sthlp``'s stored-results block.

The block lists a set of macros that a VAR is expected to carry -- ``e(depvar)``,
``e(eqnames)``, ``e(endog)``, ``e(lags)``, ``e(df_m)``, ``e(df_r)``, ``e(nocons)`` --
and none of them were stored. ``e(depvar)`` matters most: it is what postestimation
commands read first, and for a VAR it is a whitespace-separated LIST of dependent
variables rather than one name.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _system(n: int = 300, seed: int = 1100, names=("y1", "y2")):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [np.cumsum(rng.normal(size=n)) for _ in names]
    )


@pytest.fixture(scope="module")
def two_variable_fit():
    return ws.VAR(_system(), lags=2, names=["y1", "y2"]).fit()


@pytest.fixture(scope="module")
def three_variable_fit():
    rng = np.random.default_rng(1102)
    n = 320
    data = np.column_stack([np.cumsum(rng.normal(size=n)) for _ in range(3)])
    return ws.VAR(data, lags=1, names=["a", "b", "c"]).fit()


def test_depvar_lists_every_dependent_variable(two_variable_fit) -> None:
    """For a VAR ``e(depvar)`` is the whitespace-separated variable list."""

    assert two_variable_fit.e["depvar"] == "y1 y2"


def test_depvar_follows_the_variable_count(three_variable_fit) -> None:
    assert three_variable_fit.e["depvar"] == "a b c"


def test_eqnames_matches_depvar_for_a_plain_var(two_variable_fit) -> None:
    """One equation per dependent variable, so the two macros coincide."""

    assert two_variable_fit.e["eqnames"] == two_variable_fit.e["depvar"]


def test_endog_lists_the_endogenous_variables(two_variable_fit) -> None:
    """A VAR has no exogenous variables: everything in it is endogenous."""

    assert two_variable_fit.e["endog"] == "y1 y2"
    assert two_variable_fit.e["exog"] == ""


def test_lags_lists_each_lag_number(two_variable_fit) -> None:
    """``e(lags)`` is the lag LIST, not just the maximum."""

    assert two_variable_fit.e["lags"] == "1 2"
    # And ``e(mlag)`` remains the maximum, which is the same number here.
    assert float(two_variable_fit.e["mlag"]) == 2.0


def test_a_single_lag_model_lists_one(three_variable_fit) -> None:
    """Guard for the previous test: the list must track the model, not a constant."""

    assert three_variable_fit.e["lags"] == "1"


def test_df_m_is_the_per_equation_regressor_count(two_variable_fit) -> None:
    """Two lags of two variables plus a constant: ``2*2 + 1 = 5``."""

    assert float(two_variable_fit.e["df_m"]) == 5.0


def test_df_r_completes_the_observation_count(two_variable_fit) -> None:
    """``df_m + df_r == T`` -- the two must partition the used sample."""

    e = two_variable_fit.e

    assert float(e["df_m"]) + float(e["df_r"]) == pytest.approx(
        float(e["N"]), rel=1e-12
    )


def test_df_m_grows_with_variables_not_with_the_sample(
    two_variable_fit, three_variable_fit
) -> None:
    """The regressor count depends on the system, the residual df on the data."""

    two = two_variable_fit.e
    three = three_variable_fit.e

    # Three variables with one lag: 1*3 + 1 = 4, against 5 for the two-variable fit.
    assert float(three["df_m"]) == 4.0
    assert float(two["df_m"]) != float(three["df_m"])


def test_nocons_is_zero_with_a_constant(two_variable_fit) -> None:
    """``VAR``'s default trend is a constant, so ``e(nocons)`` is not set."""

    assert float(two_variable_fit.e["nocons"]) == 0.0


def test_nocons_is_one_without_a_constant() -> None:
    """The indicator must be able to say the constant was suppressed."""

    rng = np.random.default_rng(1104)
    n = 300
    data = np.column_stack(
        [np.cumsum(rng.normal(size=n)) for _ in range(2)]
    )
    fitted = ws.VAR(data, lags=1, trend="n", names=["p", "q"]).fit()

    assert float(fitted.e["nocons"]) == 1.0


def test_the_macros_survive_a_different_sample(two_variable_fit, three_variable_fit) -> None:
    """Every new macro is present whatever the system looks like."""

    for fitted in (two_variable_fit, three_variable_fit):
        for key in ("depvar", "eqnames", "endog", "lags", "df_m", "df_r", "nocons"):
            assert key in fitted.e, key
            assert fitted.e[key] is not None, key