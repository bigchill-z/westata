"""Coverage tests for the shared ML parameter-count block and ``areg`` extras.

The five keys ``consonly``, ``k_autoCns``, ``k_eq_model``, ``noconstant`` and
``rc`` are recorded by every ML estimator in Stata and were missing from all of
them; ``areg`` additionally stores ``ar2``, ``k_absorb``, ``df_a``, ``sum_w``,
``rc`` and ``p``, which ``reghdfe`` shares.

Rather than spot-check, the tests compare against the captured fixtures key by
key: a future regression that drops one fails with its name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path("tests/conformance/stata_fixtures")

#: Stata stores no ``k_aux`` for models without auxiliary parameters, so a key
#: can legitimately be absent from a fixture.  These are the keys whose absence
#: would be a gap.
_SHARED_ML_KEYS = (
    "consonly",
    "k_autoCns",
    "k_eq_model",
    "noconstant",
    "rc",
)


@pytest.fixture()
def auto():
    return pd.read_stata("tests/data/auto.dta")


def _fixture(name: str) -> set[str]:
    return set(json.loads((FIXTURES / name).read_text())["scalars"])


def test_binary_models_have_the_shared_block(auto) -> None:
    frame = auto.assign(dom=(auto["foreign"] == "Foreign").astype(int))

    for name, estimator in (
        ("logit", ws.Logit.from_formula("dom ~ weight + mpg", frame).fit()),
        ("probit", ws.Probit.from_formula("dom ~ weight + mpg", frame).fit()),
    ):
        for key in _SHARED_ML_KEYS:
            assert key in estimator.e, f"{name} missing {key}"


def test_count_models_have_the_shared_block(auto) -> None:
    for name, estimator in (
        ("poisson", ws.Poisson.from_formula("mpg ~ weight", auto).fit()),
        ("nbreg", ws.NegativeBinomial.from_formula("mpg ~ weight", auto).fit()),
    ):
        for key in _SHARED_ML_KEYS:
            assert key in estimator.e, f"{name} missing {key}"


def test_ordered_models_have_the_shared_block(auto) -> None:
    for name, estimator in (
        ("ologit", ws.OrderedLogit.from_formula("rep78 ~ weight", auto).fit()),
        ("oprobit", ws.OrderedProbit.from_formula("rep78 ~ weight", auto).fit()),
    ):
        for key in _SHARED_ML_KEYS:
            assert key in estimator.e, f"{name} missing {key}"


def test_poisson_covers_its_whole_fixture(auto) -> None:
    missing = _fixture("poisson_sim.json") - set(
        ws.Poisson.from_formula("mpg ~ weight", auto).fit().e
    )
    assert not missing, sorted(missing)


def test_ordered_models_cover_their_whole_fixture(auto) -> None:
    for fixture, estimator in (
        ("ologit.json", ws.OrderedLogit.from_formula("rep78 ~ weight", auto).fit()),
        ("oprobit.json", ws.OrderedProbit.from_formula("rep78 ~ weight", auto).fit()),
    ):
        missing = _fixture(fixture) - set(estimator.e)
        assert not missing, (fixture, sorted(missing))


def test_noconstant_is_one_without_an_intercept(auto) -> None:
    """``noconstant`` records the absence of an intercept (``e(constant)``)."""

    with_intercept = ws.Logit.from_formula(
        "dom ~ weight", auto.assign(dom=(auto["foreign"] == "Foreign").astype(int))
    ).fit()
    assert with_intercept.e["noconstant"] == 0


def test_consonly_is_one_for_a_constant_only_model(auto) -> None:
    """A model with no regressors beyond the constant sets ``consonly = 1``."""

    single = ws.Logit.from_formula(
        "dom ~ 1", auto.assign(dom=(auto["foreign"] == "Foreign").astype(int))
    ).fit()

    assert single.e["consonly"] == 1


def test_zero_inflated_counts_two_model_equations(auto) -> None:
    rng = np.random.default_rng(13)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.4 + 0.5 * x1)).astype(float)
    zero = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 * x2)))).astype(float)
    frame = pd.DataFrame({"y": np.where(zero > 0, 0.0, y), "x1": x1, "x2": x2})

    result = ws.ZeroInflatedPoisson.from_formula(
        "y ~ x1 + x2", frame, inflate="x1 + x2"
    ).fit()

    assert result.e["k_eq"] == 2
    assert result.e["k_eq_model"] == 2


# ------------------------------------------------------------------- areg
def test_areg_arithmetic_extras(auto) -> None:
    """``areg``'s arithmetic keys, on a one-way absorbed fit.

    **This uses ``FEOLS``, which reports itself as ``reghdfe``.** The two commands
    differ on ``df_a``: ``areg.json`` has ``df_a = 29`` for 30 groups (it reports
    the intercept separately) while ``reghdfe_firm.json`` has ``df_a = 30`` (it
    absorbs the constant). ``FEOLS`` follows the ``reghdfe`` convention, so this
    test asserts what ``reghdfe`` reports. The ``areg``-specific value is checked
    against the right fixture in ``tests/test_hdfe.py``.
    """

    result = ws.FEOLS.from_formula("mpg ~ weight + length | foreign", auto).fit()
    e = result.e

    # ar2 duplicates r2_a under areg's spelling.
    assert e["ar2"] == pytest.approx(e["r2_a"], rel=1e-15)
    # sum_w and rc.
    assert e["sum_w"] == float(result.nobs)
    assert e["rc"] == 0.0
    # df_a is df_a_initial for a one-way absorption, minus any redundant or
    # nested partitions (none here).
    assert e["df_a"] == e["df_a_initial"] - (
        e["df_a_nested"] + e["df_a_redundant"]
    )
    assert e["df_a_redundant"] == 0.0
    assert e["df_a_nested"] == 0.0


def test_areg_p_is_the_f_upper_tail(auto) -> None:
    from scipy import stats

    result = ws.FEOLS.from_formula("mpg ~ weight + length | foreign", auto).fit()
    e = result.e

    assert e["p"] == pytest.approx(
        float(stats.f.sf(e["F"], e["df_m"], e["df_r"])), rel=1e-10
    )


def test_areg_absorb_block_is_now_implemented(auto) -> None:
    """``F_absorb``/``p_absorb``/``r2_c`` used to be recorded as a known gap.

    They are computable from the *pooled* OLS R-squared, which
    ``xtreg_fe.ado:513-520`` obtains by re-running ``_regress y xvars``.  An
    earlier round deferred them believing the entity effects were needed; they
    are not.  This test pins the values so a future regression that drops them
    fails here.
    """

    result = ws.FEOLS.from_formula("mpg ~ weight + length | foreign", auto).fit()
    e = result.e

    assert {"r2_c", "F_absorb", "p_absorb"} <= set(e)
    # r2_c is the pooled OLS R-squared of the same regressors.
    matrix = np.column_stack(
        [
            np.ones(len(auto)),
            auto["weight"].to_numpy(float),
            auto["length"].to_numpy(float),
        ]
    )
    y = auto["mpg"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    resid = y - matrix @ beta
    centred = y - y.mean()
    expected = 1.0 - float(resid @ resid) / float(centred @ centred)
    assert e["r2_c"] == pytest.approx(expected, rel=1e-10)
    # ... and demonstrably NOT the within R-squared.
    assert not np.isclose(e["r2_c"], e["r2"], rtol=1e-6)


def test_areg_f_absorb_formula(auto) -> None:
    """``F_absorb = ((r2 - r2_c)/df_a) / ((1 - r2)/df_r)``.

    The identity is asserted with the ``df_a`` actually reported, so it verifies
    the formula's SHAPE without pinning which convention ``df_a`` follows.  That
    convention is checked against the fixtures separately.
    """

    result = ws.FEOLS.from_formula("mpg ~ weight + length | foreign", auto).fit()
    e = result.e

    expected = ((e["r2"] - e["r2_c"]) / e["df_a"]) / (
        (1.0 - e["r2"]) / e["df_r"]
    )
    assert e["F_absorb"] == pytest.approx(expected, rel=1e-10)


def test_areg_p_absorb_is_the_upper_tail(auto) -> None:
    from scipy import stats

    result = ws.FEOLS.from_formula("mpg ~ weight + length | foreign", auto).fit()
    e = result.e

    assert e["p_absorb"] == pytest.approx(
        float(stats.f.sf(e["F_absorb"], e["df_a"], e["df_r"])), rel=1e-12
    )
