"""Tests for ``newey`` -- regression with Newey-West standard errors.

Source of truth: ``newey.ado`` (StataNow 19.5) and ``newey.sthlp``.

``newey`` is NOT a new estimator. ``newey.ado:96`` calls ``_regress`` for the
point estimates; only the variance differs, and it is the Bartlett-kernel HAC that
``westata.vce.hac`` already implements (transcribed in rounds 27-30). What is new
here is the command surface:

* ``lag()`` is required with no default (``newey.ado:15``, ``newey.sthlp:55``);
  a negative lag is rejected (``newey.ado:24-27``);
* a time variable must be set (``newey.ado:33-37``);
* ``Checkt`` classifies spacing (``newey.ado:263-290``) and
  ``tflag == 2`` is ALWAYS an error while ``tflag == 1`` needs ``force``;
* ``e()`` is built by ``mat post``, so it does NOT inherit ``regress``'s keys --
  ``newey`` stores no ``r2``, ``rss``, ``mss``, ``rmse``, ``sigma``, ``ll``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.newey import NEWEY_TITLE, check_time_spacing


@pytest.fixture()
def frame():
    rng = np.random.default_rng(1)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "t": np.arange(n)})


# ------------------------------------------------------------- point estimates
def test_point_estimates_are_plain_ols(frame) -> None:
    """``newey.ado:96`` fits with ``_regress``; the VCE must not move ``b``."""

    new = ws.newey("y ~ x1 + x2", frame, lag=2, time=frame["t"])
    ols = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        np.asarray(new.params, dtype=float),
        np.asarray(ols.params, dtype=float),
        rtol=1e-12,
    )


def test_variance_matches_the_ols_hac_path(frame) -> None:
    """``ws.newey`` is a surface over the existing HAC engine, so they must agree."""

    new = ws.newey("y ~ x1 + x2", frame, lag=2, time=frame["t"])
    ols = ws.OLS.from_formula("y ~ x1 + x2", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )

    np.testing.assert_allclose(
        np.asarray(new.cov_params(), dtype=float),
        np.asarray(ols.cov_params(), dtype=float),
        rtol=1e-12,
    )


def test_lag_zero_matches_hc1_not_hc0(frame) -> None:
    """``newey.sthlp:113-114``: ``lag(0)`` equals ``vce(robust)``, which is HC1.

    In Stata ``vce(robust)`` carries the ``N/(N-K)`` factor, so comparing lag(0)
    against HC0 would show a spurious ratio of exactly ``N/(N-K)``.
    """

    n = len(frame)
    new = ws.newey("y ~ x1", frame, lag=0, time=frame["t"])
    hc1 = ws.OLS.from_formula("y ~ x1", frame).fit(cov_type="HC1")
    hc0 = ws.OLS.from_formula("y ~ x1", frame).fit(cov_type="HC0")

    np.testing.assert_allclose(
        np.asarray(new.cov_params(), dtype=float),
        np.asarray(hc1.cov_params(), dtype=float),
        rtol=1e-10,
    )
    # And it is NOT HC0: the ratio is sqrt(N/(N-K)).
    ratio = np.sqrt(np.diag(np.asarray(new.cov_params()))) / np.sqrt(
        np.diag(np.asarray(hc0.cov_params()))
    )
    np.testing.assert_allclose(ratio, np.sqrt(n / (n - 2)), rtol=1e-8)


def test_a_longer_lag_changes_the_variance() -> None:
    """The kernel must actually accumulate the extra autocovariance terms.

    Note what is deliberately NOT asserted: that every variance GROWS with the
    lag. An earlier version of this test claimed that and failed -- with AR(1)
    errors the intercept variance rose (0.00503 -> 0.01695) while the slope
    variance fell (0.00539 -> 0.00454). The lag-``j`` contribution is a signed
    cross-product, so its sign depends on the variable; only the fact that the
    result moves is universal.
    """

    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.7 * u[t - 1] + rng.normal()
    y = 1.0 + x + u
    data = pd.DataFrame({"y": y, "x": x, "t": np.arange(n)})

    short = np.diag(np.asarray(ws.newey("y ~ x", data, lag=0, time=data["t"]).cov_params()))
    long = np.diag(np.asarray(ws.newey("y ~ x", data, lag=5, time=data["t"]).cov_params()))

    assert not np.allclose(short, long, rtol=1e-6)
    # The intercept variance is the one that grows on this data.
    assert long[0] > short[0]


# ------------------------------------------------------------------- e() surface
def test_e_collection_is_neweys_own(frame) -> None:
    """``newey.sthlp:156-187`` lists the scalars AND the macros."""

    result = ws.newey("y ~ x1 + x2", frame, lag=2, time=frame["t"])

    scalars = {"N", "df_m", "df_r", "F", "p", "lag", "rank"}
    macros = {
        "cmd",
        "title",
        "vcetype",
        "estat_cmd",
        "predict",
        "properties",
        "asbalanced",
        "asobserved",
    }
    assert scalars <= set(result.e.keys())
    assert macros <= set(result.e.keys())
    assert "wtype" not in result.e


def test_predict_macro_names_a_real_program(frame) -> None:
    """``newey.ado:218``: ``eret local predict newey_p``.

    Round 68 found a macro naming a program that does not exist (``mvreg_p``);
    this pins the newey spelling against its ado line.
    """

    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    assert result.e["predict"] == "newey_p"
    assert result.e["estat_cmd"] == "newey_estat"
    assert result.e["cmd"] == "newey"


def test_the_macros_were_not_only_python_attributes(frame) -> None:
    """They must live in ``e()``, which is what a Stata-shaped consumer reads.

    Before this was fixed the values were reachable as Python attributes only, so
    ``result.e["predict"]`` raised KeyError even though the predict worked.
    """

    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    for key in ("predict", "estat_cmd", "cmd", "title", "vcetype"):
        assert key in result.e, key
        assert isinstance(result.e[key], str), key


def test_the_wtype_macro_appears_only_for_aweight(frame) -> None:
    plain = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])
    weighted = ws.newey(
        "y ~ x1",
        frame,
        lag=1,
        time=frame["t"],
        weights=np.linspace(0.5, 2.0, len(frame)),
        weight_type="aweight",
    )

    assert "wtype" not in plain.e
    assert weighted.e["wtype"] == "aweight"


@pytest.mark.parametrize(
    "absent", ["r2", "r2_a", "rmse", "sigma", "rss", "mss", "ll", "df_a"]
)
def test_regress_only_keys_are_absent(frame, absent) -> None:
    """``mat post`` creates a fresh ``e()``; these keys must not leak through."""

    result = ws.newey("y ~ x1 + x2", frame, lag=2, time=frame["t"])

    assert absent not in result.e


def test_lag_is_reported(frame) -> None:
    result = ws.newey("y ~ x1", frame, lag=3, time=frame["t"])

    assert float(result.e["lag"]) == 3.0
    assert result.lag == 3


def test_the_title_and_vcetype_match_the_help_file(frame) -> None:
    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    assert result.title == NEWEY_TITLE
    assert "Newey" in result.vcetype


def test_e_p_is_the_model_test_pvalue(frame) -> None:
    """``newey.ado:205``: ``e(p) = fprob(df_m, df_r, e(F))``."""

    from scipy import stats

    result = ws.newey("y ~ x1 + x2", frame, lag=1, time=frame["t"])

    expected = stats.f.sf(float(result.e["F"]), float(result.e["df_m"]), float(result.e["df_r"]))
    assert float(result.e["p"]) == pytest.approx(float(expected), rel=1e-12)


# ------------------------------------------------------------------- validation
def test_lag_is_required(frame) -> None:
    """``newey.sthlp:55``: the option has no default."""

    with pytest.raises(ValueError, match="lag"):
        ws.newey("y ~ x1", frame, lag=None, time=frame["t"])


def test_a_negative_lag_is_rejected(frame) -> None:
    """``newey.ado:24-27``."""

    with pytest.raises(ValueError, match="lag"):
        ws.newey("y ~ x1", frame, lag=-1, time=frame["t"])


def test_only_aweights_are_accepted(frame) -> None:
    """``newey.ado:15`` declares ``[aw/]`` -- no other weight type."""

    with pytest.raises(NotImplementedError, match="aweight"):
        ws.newey(
            "y ~ x1",
            frame,
            lag=1,
            time=frame["t"],
            weights=np.ones(len(frame)),
            weight_type="fweight",
        )


def test_aweight_is_supported(frame) -> None:
    weights = np.linspace(0.5, 3.0, len(frame))

    result = ws.newey(
        "y ~ x1", frame, lag=1, time=frame["t"], weights=weights, weight_type="aweight"
    )

    assert np.all(np.isfinite(np.asarray(result.cov_params())))


def test_noconstant_without_regressors_is_rejected(frame) -> None:
    """``newey.ado:55-58``."""

    with pytest.raises(ValueError, match="noconstant"):
        ws.newey("y ~ 1", frame, lag=1, time=frame["t"], noconstant=True)


def test_noconstant_drops_the_intercept(frame) -> None:
    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"], noconstant=True)

    assert "_cons" not in list(result.stata_param_names)


# --------------------------------------------------------------------- Checkt
def test_regular_unit_spacing_is_tflag_zero() -> None:
    tflag, delta = check_time_spacing(np.arange(50))

    assert tflag == 0
    assert delta == pytest.approx(1.0)


def test_regular_coarser_spacing_is_tflag_two() -> None:
    """``newey.ado:286-288``: ``r(mean) > 1 & r(min) == r(max)``."""

    tflag, delta = check_time_spacing(np.arange(50) * 4)

    assert tflag == 2
    assert delta == pytest.approx(4.0)


def test_irregular_spacing_is_tflag_one() -> None:
    tflag, delta = check_time_spacing(np.array([0, 2, 3, 7, 10], dtype=float))

    assert tflag == 1
    assert delta is None


def test_tflag_two_is_refused_even_with_force(frame) -> None:
    """``newey.ado:74-77`` rejects interval > 1 unconditionally."""

    irregular = frame.assign(t=np.arange(len(frame)) * 3)

    with pytest.raises(ValueError, match="greater than 1"):
        ws.newey("y ~ x1", irregular, lag=1, time=irregular["t"], force=True)


def test_tflag_one_requires_force() -> None:
    gap = np.cumsum(np.tile([1, 2], 40))
    rng = np.random.default_rng(2)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=gap.size),
            "x": rng.normal(size=gap.size),
            "t": gap,
        }
    )

    with pytest.raises(ValueError, match="not regularly spaced"):
        ws.newey("y ~ x", data, lag=1, time=data["t"])

    # ... and succeeds once force is given.
    result = ws.newey("y ~ x", data, lag=1, time=data["t"], force=True)
    assert np.all(np.isfinite(np.asarray(result.cov_params())))


def test_a_single_observation_has_no_spacing_to_check() -> None:
    tflag, delta = check_time_spacing(np.array([5.0]))

    assert tflag == 0
    assert delta == pytest.approx(1.0)


# ------------------------------------------------- the predict surface is narrower
@pytest.mark.parametrize("kind", ["xb", "index", "stdp", "residuals"])
def test_newey_p_types_are_available(frame, kind) -> None:
    """``newey_p.ado:20``: ``local myopts XB Index STDP REsiduals``."""

    result = ws.newey("y ~ x1 + x2", frame, lag=1, time=frame["t"])

    values = np.asarray(result.predict(kind=kind), dtype=float)

    assert values.shape[0] == len(frame)
    assert np.all(np.isfinite(values))


def test_index_is_a_synonym_for_xb(frame) -> None:
    """``newey_p.ado:54`` sends ``index`` and ``xb`` to the same branch.

    Refusing ``index`` would reject something Stata accepts.
    """

    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    np.testing.assert_allclose(
        np.asarray(result.predict(kind="index"), dtype=float),
        np.asarray(result.predict(kind="xb"), dtype=float),
    )


@pytest.mark.parametrize(
    "kind", ["cooksd", "hat", "rstudent", "dfbeta", "stdr", "covratio", "dfits", "welsch"]
)
def test_regress_influence_diagnostics_are_refused(frame, kind) -> None:
    """``regress_postestimation.sthlp:152-153``: influence diagnostics require
    ``vce(ols)``.  ``newey`` is a HAC estimator, so ``newey_p`` never lists them.
    """

    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    with pytest.raises(ValueError, match="newey_p"):
        result.predict(kind=kind)


def test_fitted_is_not_offered(frame) -> None:
    """``newey_p.ado:20`` lists ``XB`` but not ``fitted``."""

    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    with pytest.raises(ValueError, match="newey_p"):
        result.predict(kind="fitted")


# --------------------------------------------------------------- estat dispatch
def test_estat_durbinalt_refuses(frame) -> None:
    """``newey_estat.ado:14-16`` runs the Durbin alternative test and demands
    ``force``; it is not implemented for newey results."""

    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    with pytest.raises(NotImplementedError, match="durbinalt"):
        result.estat("durbinalt")


def test_estat_ic_refuses_because_newey_has_no_loglikelihood(frame) -> None:
    """``newey`` stores no ``e(ll)``, so no information criterion exists.

    This is a consequence of the trimmed ``e()`` collection, and it is the honest
    outcome: ``estat ic`` is defined as ``-2*e(ll) + 2*e(df)``.
    """

    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    assert "ll" not in result.e
    with pytest.raises(ValueError, match="log-likelihood"):
        result.estat("ic")


# ------------------------------------------------------------- rendered table
def test_the_table_reports_the_maximum_lag(frame) -> None:
    """``newey.ado:250``: "Maximum lag = <lag>" occupies the F line's left column."""

    text = str(ws.newey("y ~ x1", frame, lag=2, time=frame["t"]))

    assert "Maximum lag = 2" in text


def test_the_table_omits_regress_goodness_of_fit(frame) -> None:
    """``newey`` stores no ``r2``/``rmse``, so those rows must not appear.

    The failure mode this guards is subtle: rendering through the statsmodels
    layout instead of the Stata one prints
    ``R-squared: .`` / ``Adj R-squared: .`` -- rows Stata never shows for newey.
    """

    text = str(ws.newey("y ~ x1", frame, lag=1, time=frame["t"]))

    assert "R-squared" not in text
    assert "Adj R-squared" not in text
    assert "Root MSE" not in text


def test_the_table_shows_the_f_test(frame) -> None:
    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])
    text = str(result)

    assert "Prob > F" in text
    # Stata PADS the degrees of freedom: ``F(  1,       198)``, not ``F(1, 198)``. The
    # captured run from ``newey y s, lag(3)`` shows the field is fixed at
    # ``F(%3d, %9d)``, wide enough for any pair, so an unpadded rendering does not match.
    # Asserting the numbers rather than a spelling keeps the test about the statistic.
    assert f"F({int(result.df_model):3d}, {int(result.df_resid):9d})" in text


def test_str_renders_a_table_while_repr_stays_short(frame) -> None:
    result = ws.newey("y ~ x1", frame, lag=1, time=frame["t"])

    assert "Number of obs" in str(result)
    assert "NeweyWestResult" in repr(result)
    assert "\n" not in repr(result)
