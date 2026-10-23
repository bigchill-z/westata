"""Tests for the system estimators: ``sureg``, ``mvreg``, ``eivreg``, ``manova``.

Reference values are StataNow 19.5 ``e(b)``/``e(V)``, captured live from
``tests/conformance/stata_fixtures/p10_outputs.txt`` and re-captured directly from
Stata for the statistics the fixture does not print (``e(ll)``, ``e(r2)``,
``e(rmse)``, ``e(F)``, and the full ``e(V)`` matrices).

What each test is actually protecting
------------------------------------
``sureg``'s coefficients are per-equation OLS whenever the designs are the same,
so a coefficient-only test passes even with the cross-equation covariance wired
completely wrong.  The standard errors are the real check, and there are three
of them here:

* the printed Stata standard errors (``.0492993`` ... ``.039197``);
* the ``V`` cross-block, which must be non-zero;
* the two degenerate limits -- independent errors (SUR == OLS) and correlated
  errors (SUR standard errors != OLS standard errors).

The same reasoning drives ``mvreg`` (coefficients are OLS, only the VCE is
distinctive) and ``eivreg`` (``reliab=1`` must collapse to OLS; ``reliab=0.7``
must reproduce Stata's inflated coefficient *and* its standard error, which is
where the measurement-error correction has to enter the score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.systems import Eivreg, Manova, ManovaResult, Mvreg, Sureg

# --------------------------------------------------------------------- fixtures
def _sur(n: int = 400, seed: int = 101) -> pd.DataFrame:
    """The fixture DGP: two outcomes with strongly correlated errors (rho=0.55)."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    common = rng.normal(size=n)
    e1 = 0.7 * common + 0.7 * rng.normal(size=n)
    e2 = 0.7 * common + 0.7 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x1 - 0.5 * x2 + e1,
            "y2": -0.5 + 1.0 * x2 + 0.8 * x3 + e2,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


def _eiv(n: int = 500, seed: int = 107) -> pd.DataFrame:
    """Measurement error in ``x1`` (noise/signal = 0.7), plus a clean ``x2``."""

    rng = np.random.default_rng(seed)
    true_x = rng.normal(size=n)
    observed = true_x + rng.normal(scale=0.7, size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {"y": 1.0 + 2.0 * true_x - 0.5 * x2 + rng.normal(size=n), "x1": observed, "x2": x2}
    )


def _manova(n: int = 240, seed: int = 109) -> pd.DataFrame:
    """Three groups, two outcomes, as in the ``manova`` fixture."""

    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, size=n)
    effect = np.array([0.0, 1.0, 2.0])[group]
    return pd.DataFrame(
        {
            "y1": 1.0 + effect + rng.normal(scale=0.8, size=n),
            "y2": 2.0 + 0.5 * effect + rng.normal(scale=0.8, size=n),
            "g": group,
        }
    )


#: Stata ``sureg (y1 x1 x2) (y2 x2 x3)``: e(b) in e(b) column order.
SUR_B = {
    "y1:x1": 1.9600286877922553,
    "y1:x2": -0.6210211283235877,
    "y1:_cons": 1.0072301771532888,
    "y2:x2": 0.9412216847204872,
    "y2:x3": 0.8024322102949719,
    "y2:_cons": -0.49038155498391633,
}
#: Stata's printed standard errors (p10_outputs.txt lines 17-24).
SUR_SE = {
    "y1:x1": 0.040549,
    "y1:x2": 0.0476005,
    "y1:_cons": 0.0492993,
    "y2:x2": 0.047139,
    "y2:x3": 0.039197,
    "y2:_cons": 0.0487194,
}
#: Stata ``mvreg y1 y2 = x1 x2 x3``: e(b), every regressor in every equation.
MVREG_B = {
    "y1:x1": 1.953720110740593,
    "y1:x2": -0.6277454185263003,
    "y1:x3": -0.07326776552585014,
    "y1:_cons": 1.0029300322626655,
    "y2:x1": -0.01593844371945029,
    "y2:x2": 0.9381315813238427,
    "y2:x3": 0.7629223873499063,
    "y2:_cons": -0.4932190742518291,
}
MVREG_SE = {
    "y1:x1": 0.04886691,
    "y1:x2": 0.04794112,
    "y1:x3": 0.04781927,
    "y1:_cons": 0.04947905,
    "y2:x1": 0.04840873,
    "y2:x2": 0.04749161,
    "y2:x3": 0.04737091,
    "y2:_cons": 0.04901513,
}


# ------------------------------------------------------------------------ sureg
def test_sureg_matches_stata_coefficients():
    """Coefficients to 1e-12 -- they match Stata's e(b) to machine precision."""

    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    for name, expected in SUR_B.items():
        assert float(result.params[name]) == pytest.approx(expected, rel=1e-10), name


def test_sureg_matches_stata_standard_errors():
    """The SEs are the whole point: a wrong Sigma moves only these.

    Stata prints six significant digits, so the tolerance is relative to that
    rounding (1e-5), not to the estimator.
    """

    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    for name, expected in SUR_SE.items():
        assert float(result.bse[name]) == pytest.approx(expected, rel=1e-5), name


def test_sureg_covariance_is_full_not_block_diagonal():
    """A block-diagonal VCE would mean the equations were treated as unrelated."""

    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    covariance = np.asarray(result.cov_params(), dtype=float)

    assert covariance.shape == (6, 6)
    off_block = covariance[:3, 3:]
    assert np.any(np.abs(off_block) > 1e-12)
    # These cross-blocks are the mechanism: cov(y1:x2, y2:x2) = 1.239e-3 and
    # cov(y1:_cons, y2:_cons) = 1.330e-3, both from Stata's e(V).
    assert off_block[1, 0] == pytest.approx(0.0012390907825786347, rel=1e-9)
    assert off_block[2, 2] == pytest.approx(0.0013296350844685272, rel=1e-9)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12, atol=0.0)


def test_sureg_matches_stata_vce_elementwise():
    """The whole 6x6 VCE, from Stata's e(V), to 1e-8."""

    stata_v = np.array(
        [
            [0.0016442205814699744, -7.029699778495684e-05, 6.802394785753818e-05,
             4.247958268458501e-06, 4.4500728579458725e-05, 2.4532628441195487e-06],
            [-7.029699778495684e-05, 0.0022658043762217397, 0.00014770413677012603,
             0.0012390907825786347, -1.9025839073139601e-06, 8.238137855134031e-05],
            [6.802394785753818e-05, 0.00014770413677012603, 0.0024304218925020726,
             8.266200993965504e-05, 1.8410639512888268e-06, 0.0013296350844685272],
            [4.247958268458501e-06, 0.0012390907825786347, 8.266200993965504e-05,
             0.0022220831828331667, 0.00014666241620631305, 0.00015505582641183255],
            [4.4500728579458725e-05, -1.9025839073139601e-06, 1.8410639512888268e-06,
             0.00014666241620631305, 0.0015364050124656973, 8.469985662978559e-05],
            [2.4532628441195487e-06, 8.238137855134031e-05, 0.0013296350844685272,
             0.00015505582641183255, 8.469985662978559e-05, 0.002373576004081216],
        ]
    )
    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    np.testing.assert_allclose(np.asarray(result.cov_params()), stata_v, rtol=1e-10, atol=1e-14)


def test_sureg_parameter_names_are_equation_prefixed():
    """Both equations have a constant, so the prefix is required, not cosmetic."""

    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    assert result.param_names == ["y1:x1", "y1:x2", "y1:_cons", "y2:x2", "y2:x3", "y2:_cons"]
    assert len(set(result.param_names)) == 6


def test_sureg_constant_answers_to_both_stata_and_patsy_spellings():
    """The canonical name is Stata's ``_cons``; ``Intercept`` still resolves."""

    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    assert float(result.params["y1:_cons"]) == float(result.params["y1:Intercept"])
    assert float(result.bse["y2:_cons"]) == float(result.bse["y2:Intercept"])
    # A genuine typo must still fail rather than silently matching something.
    with pytest.raises(KeyError):
        result.params["y1:nonexistent"]


def test_sureg_coefficients_equal_per_equation_ols_when_errors_are_uncorrelated():
    """With uncorrelated errors SUR is a no-op: beta_SUR == beta_OLS exactly.

    The residuals are made *exactly* orthogonal (Gram-Schmidt) rather than
    merely sampled independently, so ``Sigma`` is exactly diagonal and the
    identity holds to machine precision.  Sampling independence gives a
    correlation of ~1/sqrt(n) and would only justify a loose tolerance.
    """

    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x1 + rng.normal(size=n),
            "y2": -0.5 + 1.0 * x2 + rng.normal(size=n),
            "x1": x1,
            "x2": x2,
        }
    )
    # Force the two per-equation residuals to be exactly orthogonal.  The raw
    # noise u1 has to be purged of its X2 component first, otherwise rewriting
    # y2 perturbs the residual that OLS would recompute and the orthogonality
    # only holds to ~1e-4.
    X1 = np.column_stack([np.ones(n), x1])
    X2 = np.column_stack([np.ones(n), x2])
    b1 = np.linalg.lstsq(X1, frame["y1"], rcond=None)[0]
    b2 = np.linalg.lstsq(X2, frame["y2"], rcond=None)[0]
    u1 = frame["y1"].to_numpy() - X1 @ b1
    u2 = frame["y2"].to_numpy() - X2 @ b2
    # A direction orthogonal to both X2 and u2, so subtracting it from u2 leaves
    # the new residual exactly orthogonal to u1's own residual.
    v = u1 - X2 @ np.linalg.lstsq(X2, u1, rcond=None)[0]
    frame["y2"] = X2 @ b2 + (u2 - v * (v @ u2) / (v @ v))

    result = Sureg.from_formula(["y1 ~ x1", "y2 ~ x2"], frame).fit()
    first = ws.OLS.from_formula("y1 ~ x1", frame).fit()
    second = ws.OLS.from_formula("y2 ~ x2", frame).fit()

    correlation = float(np.asarray(result.e["corr"])[0, 1])
    assert abs(correlation) < 1e-12, "the constructed residuals are not orthogonal"

    assert float(result.params["y1:x1"]) == pytest.approx(float(first.params["x1"]), abs=1e-10)
    assert float(result.params["y1:_cons"]) == pytest.approx(
        float(first.params["Intercept"]), abs=1e-10
    )
    assert float(result.params["y2:x2"]) == pytest.approx(float(second.params["x2"]), abs=1e-10)
    assert float(result.params["y2:_cons"]) == pytest.approx(
        float(second.params["Intercept"]), abs=1e-10
    )


def test_sureg_standard_errors_differ_from_ols_when_errors_are_correlated():
    """The gain from SUR: correlated errors reduce the standard errors.

    This is the counterpart of the previous test and the reason ``sureg``
    exists.  Stata: y1:x1 .040549 under SUR versus .048887 under OLS.
    """

    frame = _sur()
    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], frame).fit()
    first = ws.OLS.from_formula("y1 ~ x1 + x2", frame).fit()
    second = ws.OLS.from_formula("y2 ~ x2 + x3", frame).fit()

    correlation = float(np.asarray(result.e["corr"])[0, 1])
    assert 0.5 < correlation < 0.6

    # SUR exploits the correlation, so the SEs are strictly smaller here.
    assert float(result.bse["y1:x1"]) < float(first.bse["x1"])
    assert float(result.bse["y1:x2"]) < float(first.bse["x2"])
    # y2:x3 is the exception in this DGP (.039197 SUR vs .047257 OLS); the gap
    # is what makes the comparison non-trivial.
    assert float(result.bse["y2:x3"]) == pytest.approx(0.039197, rel=1e-5)
    assert float(result.bse["y2:x3"]) < float(second.bse["x3"])
    # And the point estimates do move slightly too, because the designs differ.
    assert float(result.params["y1:x1"]) != pytest.approx(float(first.params["x1"]), rel=1e-6)


def test_sureg_rejects_iterated_estimation():
    """``iterate=True`` is Stata's isure, which is out of scope and says so."""

    model = Sureg.from_formula(["y1 ~ x1", "y2 ~ x2"], _sur())
    with pytest.raises(UnsupportedError, match="iterate"):
        model.fit(iterate=True)


def test_sureg_reports_stata_fit_statistics():
    """e(ll), e(r2), e(rmse) and e(chi2) per equation, from Stata."""

    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    assert float(result.e["ll"]) == pytest.approx(-1042.83530993779, rel=1e-10)
    assert float(result.e["r2_y1"]) == pytest.approx(0.8122297261145611, rel=1e-10)
    assert float(result.e["r2_y2"]) == pytest.approx(0.599515039818118, rel=1e-10)
    # RMSE divides the SUR residual sum of squares by n, not n - k.
    assert float(result.e["rmse_y1"]) == pytest.approx(0.9834311830146563, rel=1e-10)
    assert float(result.e["rmse_y2"]) == pytest.approx(0.9723055408248386, rel=1e-10)
    assert float(result.e["chi2_1"]) == pytest.approx(2464.039068310004, rel=1e-10)
    assert float(result.e["chi2_2"]) == pytest.approx(757.6555036767809, rel=1e-10)
    assert float(result.e["N"]) == 400.0
    assert float(result.e["k"]) == 6.0


def test_sureg_rejects_equations_with_different_samples():
    frame = _sur()
    frame.loc[:4, "x3"] = np.nan
    with pytest.raises(ConsistencyError, match="same observations"):
        Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], frame)


def test_sureg_rejects_a_robust_vce_rather_than_returning_the_wrong_one():
    model = Sureg.from_formula(["y1 ~ x1", "y2 ~ x2"], _sur())
    with pytest.raises(UnsupportedError, match="cov_type"):
        model.fit(cov_type="HC1")


# ------------------------------------------------------------------------ mvreg
def test_mvreg_matches_stata_coefficients():
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", _sur()).fit()
    for name, expected in MVREG_B.items():
        assert float(result.params[name]) == pytest.approx(expected, rel=1e-10), name


def test_mvreg_matches_stata_standard_errors():
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", _sur()).fit()
    for name, expected in MVREG_SE.items():
        assert float(result.bse[name]) == pytest.approx(expected, rel=1e-5), name


def test_mvreg_coefficients_equal_per_equation_ols_but_the_vce_differs():
    """The coefficients are per-equation OLS to 1e-10, but the VCE is not OLS's.

    ``mvreg``'s VCE is ``Sigma (x) (X'X)^-1`` with a *shared* ``Sigma`` from all
    equations.  When every equation has the same design that gives the *same*
    standard errors as per-equation OLS -- Stata shows this too: ``mvreg``'s y1
    errors (.0488669/.0479411/.0478193/.0494791) are ``regress``'s, because
    ``u_i'u_i/(n-k)`` is exactly OLS's ``s^2``.  The difference lives in the
    off-diagonal blocks of e(V), which no per-equation ``regress`` produces.
    Asserting a difference in the standard errors here would be wrong.
    """

    frame = _sur()
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", frame).fit()
    first = ws.OLS.from_formula("y1 ~ x1 + x2 + x3", frame).fit()
    second = ws.OLS.from_formula("y2 ~ x1 + x2 + x3", frame).fit()

    # Coefficients are identical to per-equation OLS.
    for name in ("x1", "x2", "x3"):
        assert float(result.params[f"y1:{name}"]) == pytest.approx(
            float(first.params[name]), abs=1e-10
        )
        assert float(result.params[f"y2:{name}"]) == pytest.approx(
            float(second.params[name]), abs=1e-10
        )
    assert float(result.params["y1:_cons"]) == pytest.approx(
        float(first.params["Intercept"]), abs=1e-10
    )

    # The VCE is *not* the block-diagonal stack of the two OLS VCEs: it carries
    # the cross-equation covariance, which OLS cannot produce at all.
    covariance = np.asarray(result.cov_params(), dtype=float)
    assert covariance.shape == (8, 8)
    assert np.abs(covariance[:4, 4:]).max() > 1e-6
    assert covariance[0, 4] == pytest.approx(0.0013157331665067924, rel=1e-9)
    assert covariance[2, 7] == pytest.approx(6.6767769811343e-05, rel=1e-9)

    # ... and that cross-covariance is not a rounding artefact: it is 55% of the
    # geometric mean of the two variances, i.e. the error correlation of 0.556.
    ratio = covariance[0, 4] / np.sqrt(covariance[0, 0] * covariance[4, 4])
    assert ratio == pytest.approx(float(np.asarray(result.e["corr"])[0, 1]), rel=1e-9)
    assert 0.5 < ratio < 0.6

    # Stata's own numbers confirm the diagonal agreement, so this is a property
    # of the estimator rather than of westata's implementation.
    assert float(result.bse["y1:x1"]) == pytest.approx(float(first.bse["x1"]), rel=1e-9)
    assert float(result.bse["y2:x2"]) == pytest.approx(float(second.bse["x2"]), rel=1e-9)


def test_mvreg_shares_one_design_so_its_names_carry_the_outcome_prefix():
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", _sur()).fit()
    assert result.param_names == [
        "y1:x1", "y1:x2", "y1:x3", "y1:_cons",
        "y2:x1", "y2:x2", "y2:x3", "y2:_cons",
    ]


def test_mvreg_and_sureg_differ_by_specification_not_by_bug():
    """mvreg puts every regressor in every equation; sureg does not.

    The coefficients therefore differ on identical data.  This test asserts the
    difference explicitly so it cannot later be "fixed" into a bug.
    """

    frame = _sur()
    sureg = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], frame).fit()
    mvreg = Mvreg.from_formula("y1 y2 = x1 x2 x3", frame).fit()

    # y1 under sureg omits x3; under mvreg it includes it, so the estimates move.
    assert float(sureg.params["y1:x1"]) != pytest.approx(
        float(mvreg.params["y1:x1"]), rel=1e-6
    )
    assert float(sureg.params["y1:x1"]) == pytest.approx(1.960029, rel=1e-5)
    assert float(mvreg.params["y1:x1"]) == pytest.approx(1.95372, rel=1e-5)


def test_mvreg_reports_per_equation_fit_statistics():
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", _sur()).fit()
    assert float(result.e["F_y1"]) == pytest.approx(575.2300094917224, rel=1e-9)
    assert float(result.e["F_y2"]) == pytest.approx(198.29447572298326, rel=1e-9)
    assert float(result.e["r2_y1"]) == pytest.approx(0.8133563363708692, rel=1e-10)
    assert float(result.e["rmse_y1"]) == pytest.approx(0.9854159399833834, rel=1e-10)
    assert float(result.e["df_r"]) == 396.0


def test_mvreg_rejects_a_missing_equals_sign():
    with pytest.raises(ConsistencyError, match="="):
        Mvreg.from_formula("y1 y2 ~ x1 x2", _sur())


# ----------------------------------------------------------------------- eivreg
def test_eivreg_matches_stata_coefficients():
    """``reliab=0.7`` must undo the attenuation: x1 2.073214, not OLS's 1.451249."""

    result = Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"x1": 0.7}).fit()
    assert float(result.params["x1"]) == pytest.approx(2.0732139575219293, rel=1e-10)
    assert float(result.params["x2"]) == pytest.approx(-0.4678675417372797, rel=1e-10)
    assert float(result.params["_cons"]) == pytest.approx(0.8649386902679352, rel=1e-10)


def test_eivreg_matches_stata_standard_errors():
    """The measurement-error correction has to enter the score, not just beta."""

    result = Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"x1": 0.7}).fit()
    assert float(result.bse["x1"]) == pytest.approx(0.0757645, rel=1e-5)
    assert float(result.bse["x2"]) == pytest.approx(0.0730309, rel=1e-5)
    assert float(result.bse["_cons"]) == pytest.approx(0.0703323, rel=1e-5)


def test_eivreg_matches_the_full_stata_vce():
    stata_v = np.array(
        [
            [0.005740258452504991, 0.0003535405623976796, 0.0001984852773324549],
            [0.0003535405623976796, 0.005333509269688202, 0.00019616770101510504],
            [0.0001984852773324549, 0.00019616770101510504, 0.0049466256851059326],
        ]
    )
    result = Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"x1": 0.7}).fit()
    np.testing.assert_allclose(np.asarray(result.cov_params()), stata_v, rtol=1e-9, atol=1e-13)


def test_eivreg_with_perfect_reliability_reduces_to_ols():
    """``reliab=1.0`` zeroes the correction, so the estimator *is* OLS."""

    frame = _eiv()
    result = Eivreg.from_formula("y ~ x1 + x2", frame, reliab={"x1": 1.0}).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    assert float(result.params["x1"]) == pytest.approx(float(plain.params["x1"]), abs=1e-10)
    assert float(result.params["x2"]) == pytest.approx(float(plain.params["x2"]), abs=1e-10)
    assert float(result.params["_cons"]) == pytest.approx(
        float(plain.params["Intercept"]), abs=1e-10
    )
    assert float(result.params["x1"]) == pytest.approx(1.451249, rel=1e-6)


def test_eivreg_with_perfect_reliability_keeps_statas_smaller_vce():
    """At reliab=1 the coefficients are OLS but the SEs are *not*: Stata .0530382.

    Stata's eivreg variance is ``A^-1 (Psi'Psi/n) A^-1 / n``; with ``S=0`` the
    score term vanishes and what remains is the nonrobust sandwich, which is
    ``s^2 (X'X)^-1`` times ``(n - k)/n`` rather than exactly OLS's VCE.  Pinning
    the value here records that deliberate agreement with Stata.
    """

    frame = _eiv()
    result = Eivreg.from_formula("y ~ x1 + x2", frame, reliab={"x1": 1.0}).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    assert float(result.bse["x1"]) == pytest.approx(0.0530382, rel=1e-5)
    assert float(result.bse["x1"]) != pytest.approx(float(plain.bse["x1"]), rel=1e-5)


def test_eivreg_reports_stata_fit_statistics():
    result = Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"x1": 0.7}).fit()
    assert float(result.e["rmse"]) == pytest.approx(0.8357141596188606, rel=1e-10)
    assert float(result.e["r2"]) == pytest.approx(0.8639837035435246, rel=1e-10)
    assert float(result.e["F"]) == pytest.approx(407.77933077539274, rel=1e-9)
    assert float(result.e["df_r"]) == 497.0
    assert float(result.e["df_m"]) == 2.0


def test_eivreg_matches_stata_with_two_noisy_regressors():
    """A second reference design, to prove the formula is not tuned to one seed.

    Captured from Stata: ``eivreg y o1 o2 x3, reliab(o1 0.7 o2 0.8)`` on n=300.
    """

    rng = np.random.default_rng(211)
    n = 300
    t1 = rng.normal(size=n)
    t2 = rng.normal(size=n)
    o1 = t1 + rng.normal(scale=0.5, size=n)
    o2 = t2 + rng.normal(scale=0.4, size=n)
    x3 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * t1 - 0.5 * t2 + 0.3 * x3 + rng.normal(size=n),
            "o1": o1,
            "o2": o2,
            "x3": x3,
        }
    )

    result = Eivreg.from_formula("y ~ o1 + o2 + x3", frame, reliab={"o1": 0.7, "o2": 0.8}).fit()

    assert float(result.params["o1"]) == pytest.approx(2.220854, rel=1e-6)
    assert float(result.params["o2"]) == pytest.approx(-0.591144, rel=1e-6)
    assert float(result.params["x3"]) == pytest.approx(0.3609197, rel=1e-6)
    assert float(result.params["_cons"]) == pytest.approx(0.8373598, rel=1e-6)
    assert float(result.bse["o1"]) == pytest.approx(0.0974175, rel=1e-5)
    assert float(result.bse["o2"]) == pytest.approx(0.1000914, rel=1e-5)
    assert float(result.bse["x3"]) == pytest.approx(0.0884091, rel=1e-5)


def test_eivreg_requires_reliab():
    with pytest.raises(ConsistencyError, match="reliab is required"):
        Eivreg.from_formula("y ~ x1 + x2", _eiv())


def test_eivreg_rejects_a_reliability_for_a_non_regressor():
    """Stata r(111) "x1 not independent variable" -- a typo must not be ignored."""

    with pytest.raises(ConsistencyError, match="not regressors"):
        Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"nonexistent": 0.9})


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_eivreg_rejects_an_out_of_range_reliability(value):
    with pytest.raises(ConsistencyError, match="reliability"):
        Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"x1": value})


# ------------------------------------------------------------------------ manova
def test_manova_wilks_lambda_matches_stata():
    outcome = Manova.from_formula("y1 y2 = g", _manova()).fit()
    # Stata: W 0.4290, df 2, F(4.0, 472.0) 62.15, p 0.0000
    assert float(outcome.wilks_lambda) == pytest.approx(0.4290, abs=5e-5)
    assert float(outcome.wilks_lambda) == pytest.approx(0.4290353821408352, rel=1e-10)


def test_manova_rao_f_matches_stata():
    """Wilks' lambda alone would pass with a broken F; the F is checked too."""

    outcome = Manova.from_formula("y1 y2 = g", _manova()).fit()
    assert float(outcome.df) == 2.0
    assert float(outcome.df1) == 4.0
    assert float(outcome.df2) == 472.0
    assert float(outcome.F) == pytest.approx(62.15, abs=5e-3)
    assert float(outcome.F) == pytest.approx(62.15049200342134, rel=1e-9)
    assert float(outcome.p) < 1e-10
    assert isinstance(outcome, ManovaResult)


def test_manova_rao_f_matches_stata_on_a_second_design():
    """p=3 outcomes and 4 groups: the Rao scaling and df2 are non-trivial here.

    Captured from Stata: W 0.1495, F(9.0, 472.3) 62.10.
    """

    rng = np.random.default_rng(304)
    n = 200
    g = rng.integers(0, 4, size=n)
    effect = np.arange(4, dtype=float)[g]
    frame = pd.DataFrame(
        {
            **{
                f"y{j + 1}": 1.0 + (j + 1) * 0.4 * effect + rng.normal(scale=0.8, size=n)
                for j in range(3)
            },
            "g": g,
        }
    )
    outcome = Manova.from_formula("y1 y2 y3 = g", frame).fit()

    assert float(outcome.wilks_lambda) == pytest.approx(0.149514, abs=5e-6)
    assert float(outcome.df) == 3.0
    assert float(outcome.df1) == 9.0
    assert float(outcome.df2) == pytest.approx(472.3, abs=0.05)
    assert float(outcome.F) == pytest.approx(62.10, abs=5e-3)


def test_manova_reduces_to_the_univariate_f_for_one_response():
    """With p=1 Wilks' lambda is an exact transform of the one-way ANOVA F.

    Stata: ``manova y1 = g`` on this frame gives W 0.5381, F(2, 197) = 84.56836.
    The same F must come out of a plain one-way ANOVA, which is the invariant
    that makes this a real check rather than a second hard-coded fixture.
    """

    rng = np.random.default_rng(401)
    n = 200
    g = rng.integers(0, 3, size=n)
    y = 1.0 + np.array([0.0, 1.0, 2.0])[g] + rng.normal(size=n)
    frame = pd.DataFrame({"y1": y, "g": g})
    outcome = Manova.from_formula("y1 = g", frame).fit()

    assert float(outcome.wilks_lambda) == pytest.approx(0.53805, abs=5e-5)
    assert float(outcome.df1) == 2.0
    assert float(outcome.df2) == 197.0
    assert float(outcome.F) == pytest.approx(84.56836, abs=5e-4)

    # Independent check: the classic one-way ANOVA F.
    groups = [y[g == level] for level in np.unique(g)]
    grand = y.mean()
    between = sum(len(block) * (block.mean() - grand) ** 2 for block in groups)
    within = sum(((block - block.mean()) ** 2).sum() for block in groups)
    anova_f = (between / (len(groups) - 1)) / (within / (n - len(groups)))
    assert float(outcome.F) == pytest.approx(anova_f, rel=1e-9)


def test_manova_rejects_multiple_factors():
    frame = _manova()
    frame["h"] = np.random.default_rng(1).integers(0, 2, size=len(frame))
    with pytest.raises(UnsupportedError, match="one factor"):
        Manova.from_formula("y1 y2 = g + h", frame)


def test_manova_rejects_a_non_integer_treatment():
    """Stata r(452): "factor variables may not contain noninteger values"."""

    frame = _manova()
    frame["g"] = frame["g"] + 0.5
    with pytest.raises(ConsistencyError, match="integer"):
        Manova.from_formula("y1 y2 = g", frame)


def test_manova_rejects_a_single_level_factor():
    frame = _manova()
    frame["g"] = 0
    with pytest.raises(ConsistencyError, match="single level"):
        Manova.from_formula("y1 y2 = g", frame).fit()
