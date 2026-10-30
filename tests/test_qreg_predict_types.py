"""Quantile-regression ``predict`` types, from ``qreg_postestimation.sthlp:86-105``.

The help file gives two shapes:

    predict dtype newvar [if] [in] [, xb | stdp | residuals]        (qreg/iqreg/bsqreg)

    predict dtype newvar [if] [in], equation(eqno[,eqno]) statistic  (sqreg)

and the statistic table lists ``xb`` (default), ``stdp``, ``stddp`` and ``residuals``.

NONE of the quantile estimators was registered in ``SUPPORTED_TYPES``, so every type
raised

    UnsupportedError: predict: no prediction types registered for 'qreg'

-- including ``xb``, which needs nothing beyond the fitted design matrix the estimator
already stores. ``sqreg`` has the wider set because it fits several quantiles at once,
which is what makes ``stddp`` (the standard error of a DIFFERENCE between two linear
predictions) meaningful there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import supported_types


def _frame(n: int = 200, seed: int = 1600):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )


@pytest.fixture(scope="module")
def qreg_fit():
    return ws.QuantileRegression.from_formula("y ~ x1", _frame()).fit()


@pytest.fixture(scope="module")
def sqreg_fit():
    return ws.Sqreg.from_formula(
        "y ~ x1", _frame(), quantiles=(0.25, 0.5, 0.75)
    ).fit()


# ------------------------------------------------------------------ the registry
def test_qreg_lists_the_three_documented_types() -> None:
    assert set(supported_types("qreg")) == {"xb", "stdp", "residuals"}


def test_sqreg_lists_the_wider_set() -> None:
    """``sthlp:89-103``: sqreg adds ``stddp``."""

    assert set(supported_types("sqreg")) == {"xb", "stdp", "stddp", "residuals"}


def test_sqreg_offers_stddp_and_qreg_does_not() -> None:
    """Guard: the two sets must actually differ, or the distinction is untested."""

    assert "stddp" in supported_types("sqreg")
    assert "stddp" not in supported_types("qreg")


# --------------------------------------------------------------------- qreg
@pytest.mark.parametrize("kind", ["xb", "stdp", "residuals"])
def test_qreg_predict_works(qreg_fit, kind) -> None:
    values = np.asarray(qreg_fit.predict(kind=kind), dtype=float)

    assert values.shape == (len(_frame()),)
    assert np.isfinite(values).all()


def test_qreg_default_is_xb(qreg_fit) -> None:
    """``sthlp:101``: "``xa`` linear prediction; the default"."""

    default = np.asarray(qreg_fit.predict(), dtype=float)
    explicit = np.asarray(qreg_fit.predict(kind="xb"), dtype=float)

    assert np.allclose(default, explicit)


def test_qreg_xb_equals_the_design_times_the_coefficients(qreg_fit) -> None:
    """The linear prediction has a definition; check it rather than the shape."""

    design = np.asarray(qreg_fit.design_matrix[0], dtype=float)
    coefficients = np.asarray(qreg_fit.params, dtype=float)

    assert np.allclose(
        np.asarray(qreg_fit.predict(kind="xb"), dtype=float),
        design @ coefficients,
        rtol=1e-10,
    )


def test_qreg_residuals_are_y_minus_xb(qreg_fit) -> None:
    frame = _frame()
    predicted = np.asarray(qreg_fit.predict(kind="xb"), dtype=float)
    residuals = np.asarray(qreg_fit.predict(kind="residuals"), dtype=float)

    assert np.allclose(
        residuals, np.asarray(frame["y"], dtype=float) - predicted, rtol=1e-10
    )


# --------------------------------------------------------------------- sqreg
@pytest.mark.parametrize("kind", ["xb", "stdp", "residuals"])
def test_sqreg_predict_works(sqreg_fit, kind) -> None:
    values = np.asarray(sqreg_fit.predict(kind=kind), dtype=float)

    assert values.shape == (len(_frame()),)
    assert np.isfinite(values).all()


def test_sqreg_is_not_offered_an_unregistered_type(sqreg_fit) -> None:
    """Registration must not become "anything goes"."""

    with pytest.raises(Exception):
        sqreg_fit.predict(kind="not_a_statistic")


def test_both_estimators_predict_the_same_linear_index_at_the_median() -> None:
    """``sqreg`` at the median and ``qreg`` at the median describe the same fit.

    Not an exact identity in general, but they must agree closely on the same data --
    a gross discrepancy would mean one of them is predicting from the wrong parameters.
    """

    frame = _frame()
    median_sqreg = ws.Sqreg.from_formula(
        "y ~ x1", frame, quantiles=(0.5,)
    ).fit()
    plain = ws.QuantileRegression.from_formula("y ~ x1", frame).fit()

    assert np.allclose(
        np.asarray(median_sqreg.predict(kind="xb"), dtype=float),
        np.asarray(plain.predict(kind="xb"), dtype=float),
        rtol=1e-4,
        atol=1e-4,
    )


# -------------------------------------------------------------------- stddp
def test_stddp_is_implemented_for_sqreg(sqreg_fit) -> None:
    """``sthlp:130-132``: SE of ``xb[eq1] - xb[eq2]``.

    Registering the type without implementing it produced

        UnsupportedError: type 'stddp' is registered for 'sqreg' but has no
        implementation; this is a westata bug

    -- the internal consistency check caught it rather than letting a wrong number out.
    """

    values = np.asarray(sqreg_fit.predict(kind="stddp"), dtype=float)

    assert values.shape == (len(_frame()),)
    assert np.isfinite(values).all()
    assert np.all(values >= 0.0)


def test_stddp_matches_the_variance_difference_formula(sqreg_fit) -> None:
    """Rebuild ``sqrt(V1 + V2 - 2*Cov12)`` from the stored covariance.

    The covariance term is the whole point: with it the difference can be much SMALLER
    than either standard error, which is what distinguishes ``stddp`` from a sum of
    standard errors.
    """

    exog = np.asarray(sqreg_fit.design_matrix[0], dtype=float)
    cov = np.asarray(sqreg_fit.cov_params(), dtype=float)
    names = [str(name) for name in sqreg_fit.param_names]

    first = [i for i, name in enumerate(names) if name.startswith("q0.25:")]
    second = [i for i, name in enumerate(names) if name.startswith("q0.5:")]
    variance_first = np.einsum("ij,jk,ik->i", exog, cov[np.ix_(first, first)], exog)
    variance_second = np.einsum("ij,jk,ik->i", exog, cov[np.ix_(second, second)], exog)
    covariance = np.einsum("ij,jk,ik->i", exog, cov[np.ix_(first, second)], exog)
    expected = np.sqrt(np.maximum(variance_first + variance_second - 2 * covariance, 0))

    assert np.allclose(
        np.asarray(sqreg_fit.predict(kind="stddp"), dtype=float),
        expected,
        rtol=1e-10,
    )


def test_stddp_uses_the_equations_NOT_a_sum_of_standard_errors(sqreg_fit) -> None:
    """Guard: ignoring the covariance would give a different (larger) number."""

    exog = np.asarray(sqreg_fit.design_matrix[0], dtype=float)
    cov = np.asarray(sqreg_fit.cov_params(), dtype=float)
    names = [str(name) for name in sqreg_fit.param_names]
    first = [i for i, name in enumerate(names) if name.startswith("q0.25:")]
    second = [i for i, name in enumerate(names) if name.startswith("q0.5:")]
    variance_first = np.einsum("ij,jk,ik->i", exog, cov[np.ix_(first, first)], exog)
    variance_second = np.einsum("ij,jk,ik->i", exog, cov[np.ix_(second, second)], exog)
    without_covariance = np.sqrt(np.maximum(variance_first + variance_second, 0))

    produced = np.asarray(sqreg_fit.predict(kind="stddp"), dtype=float)

    assert not np.allclose(produced, without_covariance)


def test_stddp_needs_two_quantiles() -> None:
    """A single-quantile ``sqreg`` has no second equation to difference against."""

    single = ws.Sqreg.from_formula(
        "y ~ x1", _frame(), quantiles=(0.5,)
    ).fit()

    with pytest.raises(Exception) as caught:
        single.predict(kind="stddp")

    assert "TWO quantiles" in str(caught.value) or "2" in str(caught.value)


def test_stddp_is_not_offered_for_plain_qreg(qreg_fit) -> None:
    """``sthlp:130-131``: "allowed only after you have fit a model using sqreg"."""

    with pytest.raises(Exception):
        qreg_fit.predict(kind="stddp")