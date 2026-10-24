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

from pathlib import Path

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
    # Stata codes the ordered outcome 1..J (``ologit y x`` with y = 1,2,3 names
    # the cut points cut1/cut2); westata follows that convention.
    return pd.DataFrame(
        {"g": np.tile([1, 2, 3], n // 3 + 1)[:n], "x1": rng.normal(size=n)}
    )


def test_ordered_model_stdp_drops_the_cut_points():
    """``/cut1``, ``/cut2`` are location parameters, not design columns.

    The ordered estimators now retain their design matrix (the ``discrete``
    group's repair), so ``predict`` works without ``exog=``; what must *not*
    change is that the cut points never enter ``x'Vx``.
    """

    df = _ordered_frame()
    res = ws.OrderedLogit.from_formula("g ~ x1", df).fit()
    assert res.design_matrix is not None

    X = df[["x1"]].to_numpy(dtype=float)
    names = list(res.param_names)
    block = [i for i, name in enumerate(names) if not name.startswith("/")]
    assert block == [0]

    V = res.cov_params()[np.ix_(block, block)]
    np.testing.assert_allclose(
        res.predict(exog=X, kind="stdp") ** 2, _xvx(X, V), rtol=1e-10, atol=0
    )
    # The retained design is the same k-column matrix, so omitting exog agrees.
    np.testing.assert_allclose(
        res.predict(kind="stdp"), res.predict(exog=X, kind="stdp"), rtol=1e-12
    )

    # One column per parameter is not a valid design matrix for these types.
    with_cuts = np.column_stack([X, np.ones(len(X)), np.ones(len(X))])
    assert with_cuts.shape[1] == len(res.params)
    with pytest.raises(ValueError, match="column"):
        res.predict(exog=with_cuts, kind="stdp")


def test_ordered_model_refuses_stdf_but_offers_stdp():
    """Ordered models refuse ``stdf``; ``stdp`` works from the retained design.

    Captured on StataNow 19.5 (2026-10-24), ``sysuse auto``::

        . ologit rep78 weight mpg
        . predict double s, stdp      /* accepted */
        . predict double t, stdf      ->  option stdf not allowed
                                          r(198);

    The old test also asserted that no design matrix was retained (a second,
    independent guard).  The ``discrete`` repair now stores it, so only the
    ``stdf`` refusal is left to pin -- and it is pinned with and without an
    explicit ``exog=`` so a future regression in either path is caught.
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

    # The retained design is present, so this is the ``stdf`` guard alone.
    assert res.design_matrix is not None
    with pytest.raises(UnsupportedError):
        res.predict(kind="stdf")

    # And it is still refused when the design is supplied explicitly.
    X = frame[["x1"]].to_numpy(dtype=float)
    with pytest.raises(UnsupportedError):
        res.predict(exog=X, kind="stdf")

    # ``stdp`` is available both ways.
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


# -------------------------------- Stata-captured regressions (M2 and M3)
#
# Captured from a live StataNow 19.5 (MP) session on 2026-10-24.

_AUTO = Path(__file__).parent / "data" / "auto.dta"


def _auto() -> pd.DataFrame:
    return pd.read_stata(_AUTO, convert_categoricals=False)


def test_tobit_stdf_adds_sigma_squared_not_rss_over_n():
    """``predict, stdf`` after ``tobit mpg weight foreign, ll(17)``.

    Stata: ``stdf^2 - stdp^2 = e(var_e) = 14.297844`` for every observation.
    The old fallback used ``rss/N`` of the censored response (12.127405), which
    made every prediction interval 7.7% too narrow.
    """

    frame = _auto()
    result = ws.Tobit.from_formula("mpg ~ weight + foreign", frame, lower=17).fit()

    stdp = result.predict(kind="stdp")
    stdf = result.predict(kind="stdf")
    gap = stdf**2 - stdp**2

    np.testing.assert_allclose(gap, np.full_like(gap, gap[0]), rtol=1e-12)
    assert gap[0] == pytest.approx(14.2978442, rel=1e-8)
    # The old behaviour, still computable, is genuinely different.
    resid = np.asarray(result.resid, dtype=float)
    rss_over_n = float(resid @ resid / resid.size)
    assert rss_over_n == pytest.approx(12.12740491, rel=1e-6)
    assert gap[0] != pytest.approx(rss_over_n, rel=1e-3)


def test_tobit_stdp_matches_stata():
    """``stdp``/``stdf`` after tobit match Stata's ``predict double`` output.

    Stata's first three values (``tobit mpg weight foreign, ll(17)``):
    ``stdp = [0.6076244331609947, 0.6082309540890917, 0.7242958528724734]``,
    ``stdf = [3.829758694092775, 3.829854970715876, 3.8499933357985494]``.
    """

    frame = _auto()
    result = ws.Tobit.from_formula("mpg ~ weight + foreign", frame, lower=17).fit()

    stdp = np.asarray(result.predict(kind="stdp"), dtype=float)
    stdf = np.asarray(result.predict(kind="stdf"), dtype=float)
    np.testing.assert_allclose(
        stdp[:3],
        [0.6076244331609947, 0.6082309540890917, 0.7242958528724734],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        stdf[:3],
        [3.829758694092775, 3.829854970715876, 3.8499933357985494],
        rtol=1e-7,
    )


def test_regress_stdf_is_unchanged_by_the_innovation_variance_chain():
    """``regress`` still takes the ``e(rmse)`` branch: 11.60805298."""

    frame = _auto()
    result = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit()
    gap = result.predict(kind="stdf") ** 2 - result.predict(kind="stdp") ** 2
    assert gap[0] == pytest.approx(11.60805298, rel=1e-8)
    assert gap[0] == pytest.approx(float(result.e["rmse"]) ** 2, rel=1e-12)


def test_zip_predict_n_matches_stata_instead_of_crashing():
    """``zip yzi x1, inflate(z)`` + ``predict, n`` = ``(1-pi)*mu``.

    Stata's ``e(b)`` is ``[x1, _cons, z, _cons]`` and it reports
    ``max|n - (1-pi)mu| = 4.4e-16``.  Before the repair westata raised a bare
    ``matmul`` dimension error here.
    """

    rng = np.random.default_rng(11)
    n = 400
    x1 = rng.normal(size=n)
    z = np.random.default_rng(7).normal(size=n)
    rng2 = np.random.default_rng(7)
    zero = rng2.random(n) < 1.0 / (1.0 + np.exp(-(-0.8 + 1.0 * z)))
    y = np.where(zero, 0, rng2.poisson(np.exp(0.5 + 0.5 * x1)))
    frame = pd.DataFrame({"y": y.astype(float), "x1": x1, "z": z})

    result = ws.ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="z").fit()
    values = np.asarray(result.predict(kind="n"), dtype=float)

    params = np.asarray(result.params, dtype=float)
    names = list(result.param_names)
    design = np.asarray(result.design_matrix[0], dtype=float)
    count = design @ params[[names.index("Intercept"), names.index("x1")]]
    z_design = np.column_stack([np.ones(n), z])
    # westata names the inflation equation's parameters ``inflate:<term>``.
    inflation_index = [names.index(name) for name in names if name.startswith("inflate:")]
    inflation = z_design @ params[inflation_index]
    probability = 1.0 / (1.0 + np.exp(-inflation))
    np.testing.assert_allclose(
        values, (1.0 - probability) * np.exp(count), rtol=1e-12
    )


def test_ztp_predict_n_is_the_exponential_mean_not_a_probability():
    """``ztp`` + ``predict, n`` is ``exp(xb)`` (Stata: max rel diff 0.0).

    The old dispatch pushed every non-poisson/nbreg count model through the
    probit branch, so this returned a probability in [0, 1] instead of a mean.
    """

    rng = np.random.default_rng(11)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.maximum(rng.poisson(np.exp(0.4 + 0.6 * x1 - 0.3 * x2)), 1)
    frame = pd.DataFrame({"y": y.astype(float), "x1": x1, "x2": x2})

    result = ws.TruncatedPoisson.from_formula("y ~ x1 + x2", frame).fit()
    values = np.asarray(result.predict(kind="n"), dtype=float)
    xb = np.asarray(result.predict(kind="xb"), dtype=float)

    assert np.all(values > 0.0)
    np.testing.assert_allclose(values, np.exp(xb), rtol=1e-12)
    # A probability would be bounded by 1; the mean here is not.
    assert float(np.mean(values)) > 1.0


def test_predict_delegates_to_a_model_supplied_dispatch():
    """The contract: ``_predict_dispatch`` owns the kinds it answers.

    ``_predict_types`` is the whitelist; returning ``None`` for a kind falls
    through to the shared implementation, so a model never has to reimplement
    ``xb``/``stdp``.
    """

    from westata.postestimation.predict import predict

    class Stub:
        model_name = "customcount"
        param_names = ["a", "b"]
        nobs = 4
        dist = "z"
        df_resid = 4.0
        resid = np.zeros(4)
        params = pd.Series([1.0, 2.0], index=["a", "b"])
        design_matrix = (np.ones((4, 2)), ["a", "b"])
        e = {"N": 4}
        _predict_types = ("xb", "double_mean")

        def cov_params(self):
            return np.eye(2)

        def _predict_dispatch(self, *, kind, exog, lower=None, upper=None):
            if kind == "double_mean":
                return 2.0 * (np.asarray(exog) @ np.asarray(self.params))
            return None

    stub = Stub()
    design = np.ones((4, 2))
    np.testing.assert_allclose(predict(stub, design, "xb"), np.full(4, 3.0))
    np.testing.assert_allclose(predict(stub, design, "double_mean"), np.full(4, 6.0))
    with pytest.raises(UnsupportedError, match="double_mean"):
        predict(stub, design, "n")


def test_unsupported_kind_lists_the_available_types():
    """An unknown ``kind`` must name what *does* work, never just fail."""

    frame = _auto()
    result = ws.Logit.from_formula("foreign ~ weight + mpg", frame).fit()
    with pytest.raises(UnsupportedError) as error:
        result.predict(kind="stdf")
    message = str(error.value)
    assert "xb" in message and "pr" in message and "stdp" in message
