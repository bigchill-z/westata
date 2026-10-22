"""Conformance tests for the prediction-standard-error types ``stdp``/``stdf``.

Stata semantics (``[R] predict``, ``[R] regress postestimation``):

* ``stdp`` is the standard error of the *linear* prediction,
  :math:`\\sqrt{x_i' V x_i}`.
* ``stdf`` is the standard error of the *forecast*, that same quantity plus the
  residual variance: :math:`\\sqrt{x_i' V x_i + \\sigma^2}`.

Every expectation here is recomputed by hand from ``X`` and ``V`` -- never from
the implementation -- so a change in the formula is caught rather than
enshrined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.postestimation.predict import supported_types


def _linear(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _binary(n: int = 500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    return pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})


def _censored(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.65 * x1 + 0.3 * x2 + rng.normal(scale=0.35, size=n)
    return pd.DataFrame({"y": np.maximum(latent, 2.0), "x1": x1, "x2": x2})


def _xvx(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``diag(X V X')`` by hand, with an independent einsum formulation."""

    return np.einsum("ij,jk,ik->i", X, V, X)


# ------------------------------------------------------------------- stdp


def test_stdp_squared_equals_diag_xvx():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    X, _names = res.design_matrix

    expected = _xvx(np.asarray(X, dtype=float), res.cov_params())
    observed = res.predict(kind="stdp") ** 2

    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=0)


def test_stdp_matches_matrix_diagonal_product():
    """Cross-check against the naive ``diag(X @ V @ X.T)`` formulation."""

    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    X = np.asarray(res.design_matrix[0], dtype=float)
    V = res.cov_params()

    expected = np.diag(X @ V @ X.T)
    np.testing.assert_allclose(
        res.predict(kind="stdp") ** 2, expected, rtol=1e-9, atol=1e-14
    )


def test_stdp_is_positive_and_finite():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    stdp = res.predict(kind="stdp")
    assert stdp.shape == (len(res.design_matrix[0]),)
    assert np.all(np.isfinite(stdp))
    assert np.all(stdp > 0)


def test_stdp_uses_the_estimated_vce_not_the_coefficient_se():
    """A robust VCE must flow into stdp, so stdp != |x_is| * bse_i in general."""

    plain = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    robust = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit(cov_type="HC1")
    X = np.asarray(robust.design_matrix[0], dtype=float)

    expected = _xvx(X, robust.cov_params())
    np.testing.assert_allclose(
        robust.predict(kind="stdp") ** 2, expected, rtol=1e-10, atol=0
    )
    assert not np.allclose(
        robust.predict(kind="stdp"), plain.predict(kind="stdp"), rtol=1e-6
    )


# ------------------------------------------------------------------- stdf


def test_stdf_squared_minus_stdp_squared_is_rmse_squared():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    stdp = res.predict(kind="stdp")
    stdf = res.predict(kind="stdf")

    gap = stdf**2 - stdp**2
    # Constant across observations: stdf only adds the residual variance.
    np.testing.assert_allclose(gap, np.full_like(gap, gap[0]), rtol=1e-12, atol=0)
    assert gap[0] == pytest.approx(float(res.e["rmse"]) ** 2, rel=1e-12)


def test_stdf_is_larger_than_stdp():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    assert np.all(res.predict(kind="stdf") > res.predict(kind="stdp"))


def test_stdf_without_rmse_falls_back_to_rss_over_n():
    """When ``e(rmse)`` is absent, the residual variance is ``rss / N``."""

    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    del res.e["rmse"]

    resid = np.asarray(res.resid, dtype=float)
    rss_over_n = float(np.dot(resid, resid) / res.nobs)

    gap = res.predict(kind="stdf") ** 2 - res.predict(kind="stdp") ** 2
    np.testing.assert_allclose(gap, rss_over_n, rtol=1e-12, atol=0)

    # The two denominators are genuinely different here, so the assertion above
    # discriminates between the documented rss/N fallback and rss/df_resid.
    rss_over_df = float(np.dot(resid, resid) / res.df_resid)
    assert rss_over_n != pytest.approx(rss_over_df, rel=1e-6)


def test_stdf_without_residual_variance_raises_consistency_error():
    """No ``e(rmse)`` and no usable residuals must fail loudly, not guess."""

    class _Stub:
        model_name = "regress"
        param_names = ["Intercept", "x"]
        params = np.array([1.0, 2.0])
        resid = None

        def cov_params(self):
            return np.eye(2)

        class e:  # noqa: N801 - deliberately a tiny stand-in mapping
            @staticmethod
            def get(key):
                return None

    with pytest.raises(ConsistencyError, match="stdf"):
        from westata.postestimation.predict import predict

        predict(_Stub(), np.ones((3, 2)), kind="stdf")


# --------------------------------------------------------------- other models


def test_stdp_works_on_logit_but_stdf_is_refused():
    """Stata's logit offers stdp but not stdf (``r(198)``).

    Verified on StataNow 19.5: ``logit`` + ``predict, stdf`` returns "option stdf
    not allowed". Offering a number under a name Stata refuses would be exactly
    the silent non-conformance this project avoids.
    """

    frame = _binary()
    res = ws.Logit.from_formula("y ~ x1 + x2", frame).fit()

    stdp = res.predict(kind="stdp")
    assert stdp.shape == (len(frame),)
    assert np.all(stdp > 0)

    with pytest.raises(UnsupportedError):
        res.predict(kind="stdf")


def test_tobit_stdp_uses_only_the_coefficient_block():
    """Tobit's ``/sigma`` is an auxiliary parameter, not a regressor column."""

    res = ws.Tobit.from_formula("y ~ x1 + x2", _censored(), lower=2.0).fit()
    X = np.asarray(res.design_matrix[0], dtype=float)
    k = X.shape[1]
    assert len(res.params) == k + 1  # coefficients plus /sigma

    V = res.cov_params()[:k, :k]
    np.testing.assert_allclose(
        res.predict(kind="stdp") ** 2, _xvx(X, V), rtol=1e-10, atol=0
    )
    # /sigma contributes to the forecast variance, never to the linear one.
    gap = res.predict(kind="stdf") ** 2 - res.predict(kind="stdp") ** 2
    assert np.allclose(gap, gap[0], rtol=1e-12)
    assert gap[0] > 0


def test_tobit_sigma_column_is_rejected_with_a_clear_message():
    """tobit's design matrix has k columns, not k+1.

    ``/sigma`` is an auxiliary parameter, not a design column, so a caller who
    stacks a constant on top of tobit's design gets a loud error naming the
    mismatch rather than silently having the extra column ignored. That is the
    behaviour we want: silently dropping a column could mask a real mistake.
    """

    res = ws.Tobit.from_formula("y ~ x1 + x2", _censored(), lower=2.0).fit()
    X = np.asarray(res.design_matrix[0], dtype=float)
    with_sigma_column = np.column_stack([X, np.ones(len(X))])
    assert with_sigma_column.shape[1] == len(res.params)

    with pytest.raises(ValueError, match="column"):
        res.predict(exog=with_sigma_column, kind="stdp")

    # The correct k-column matrix works and is symmetric in nothing else.
    stdp = res.predict(exog=X, kind="stdp")
    assert stdp.shape == (len(X),)
    assert np.all(stdp > 0)


def _ordered_frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"g": np.tile([0, 1, 2], n // 3 + 1)[:n], "x1": rng.normal(size=n)}
    )


def test_ordered_model_stdp_drops_the_cut_points():
    """``/cut1``, ``/cut2`` are location parameters, not design columns."""

    df = _ordered_frame()
    res = ws.OrderedLogit.from_formula("g ~ x1", df).fit()
    assert res.design_matrix is None

    X = df[["x1"]].to_numpy(dtype=float)
    names = list(res.param_names)
    block = [i for i, name in enumerate(names) if not name.startswith("/")]
    assert block == [0]

    V = res.cov_params()[np.ix_(block, block)]
    np.testing.assert_allclose(
        res.predict(exog=X, kind="stdp") ** 2, _xvx(X, V), rtol=1e-10, atol=0
    )

    # One column per parameter is not a valid design matrix for these types.
    with_cuts = np.column_stack([X, np.ones(len(X)), np.ones(len(X))])
    assert with_cuts.shape[1] == len(res.params)
    with pytest.raises(ValueError, match="column"):
        res.predict(exog=with_cuts, kind="stdp")


def test_ordered_model_refuses_stdf_for_two_independent_reasons():
    """Ordered models refuse ``stdf``, and would refuse ``stdp`` too without exog.

    Two distinct guards apply, and the first one wins:

    1. ``oprobit``/``ologit`` results do not retain a design matrix, so
       ``predict(kind=...)`` without ``exog=`` raises ``ConsistencyError``.
    2. Even with ``exog=`` supplied, ``stdf`` is not offered for ordered models
       because Stata's ``predict`` has no such option there.

    The test asserts both, so a future change that removes the first guard does
    not silently start returning a number for the second.
    """

    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    latent = 0.8 * x1 + rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": pd.cut(
                latent, bins=[-np.inf, -0.7, 0.4, np.inf], labels=[1, 2, 3]
            ).astype(float),
            "x1": x1,
        }
    )
    res = ws.OrderedLogit.from_formula("y ~ x1", frame).fit()

    # Guard 1: no retained design matrix.
    assert res.design_matrix is None
    with pytest.raises(ConsistencyError):
        res.predict(kind="stdf")

    # Guard 2: even with the design supplied, the option does not exist.
    # Ordering absorbs the location into the cut points, so the design has one
    # column per slope and no intercept.
    X = frame[["x1"]].to_numpy(dtype=float)
    with pytest.raises(UnsupportedError):
        res.predict(exog=X, kind="stdf")

    # ``stdp`` is available when the design is supplied.
    stdp = res.predict(exog=X, kind="stdp")
    assert stdp.shape == (len(frame),)
    assert np.all(np.isfinite(stdp))


def test_stdp_and_stdf_are_registered_for_regress():
    kinds = supported_types("regress")
    assert "stdp" in kinds
    assert "stdf" in kinds


def test_stdp_is_registered_wherever_xb_is_and_stdf_only_for_linear_models():
    """stdp is universal; stdf belongs to the linear families only."""

    linear = ["regress", "ivregress 2sls", "reghdfe", "xtreg, fe", "tobit"]
    nonlinear = ["logit", "probit", "poisson", "nbreg", "ologit", "oprobit"]

    for model in linear:
        kinds = supported_types(model)
        assert "stdp" in kinds, model
        assert "stdf" in kinds, model

    for model in nonlinear:
        kinds = supported_types(model)
        assert "stdp" in kinds, model
        assert "stdf" not in kinds, model


def test_xb_still_equals_fittedvalues_for_ols():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    np.testing.assert_allclose(res.predict(kind="xb"), res.fittedvalues, rtol=1e-12)


def test_other_kinds_are_unaffected():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    np.testing.assert_allclose(res.predict(kind="fitted"), res.fittedvalues, rtol=1e-12)
    np.testing.assert_allclose(
        res.predict(kind="residuals"), np.asarray(res.resid), rtol=1e-12
    )


def test_unknown_kind_still_raises_unsupported():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    with pytest.raises(UnsupportedError, match="predict"):
        res.predict(kind="scores")


# ------------------------------------------------------------ validation


def test_shape_mismatch_raises_value_error():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    X = np.asarray(res.design_matrix[0], dtype=float)
    too_few = X[:, :2]  # 2 columns instead of 3

    with pytest.raises(ValueError, match="column"):
        res.predict(exog=too_few, kind="stdp")
    with pytest.raises(ValueError, match="column"):
        res.predict(exog=too_few, kind="stdf")
