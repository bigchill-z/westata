"""Tests for ``r(mineig)`` -- the Cragg-Donald minimum eigenvalue.

``FirstStage`` returns ``r(mineig)`` (``ivregress_estat.ado:777-778``, computed by
the Mata builtin ``_iv_craggdonald_wrk``). It is the minimum eigenvalue of the
canonical-correlation matrix between the endogenous regressors and the
residualised excluded instruments.

**Why a number is reportable here although the Mata source is unreadable.** With
ONE endogenous regressor the matrix is 1x1, so its only eigenvalue *is* the
squared canonical correlation -- the first stage's partial R-squared. And the
partial-F identity ties that to the printed F:

    partial_r2 = F / (F + N - k)

So at (1 endog, 1 instrument) three independently computed quantities must agree.
That is what these tests pin. Beyond one endogenous regressor the eigenvalue
problem is genuinely larger and no self-check exists; that part is recorded as
unverified rather than claimed.

A bug this anchor caught: when a model has no included exogenous regressors,
``_ols`` returns a zero fit for an empty design, so ``_residualise`` left the
columns uncentred. The eigenvalue then came out 0.21434 against a partial R2 of
0.21110 — a 1.5% gap that only the identity could reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n=400, seed=1, n_instr=1, n_endog=1, include_exog=False):
    rng = np.random.default_rng(seed)
    instruments = [rng.normal(size=n) for _ in range(n_instr)]
    u = rng.normal(size=n)
    endog = [
        0.6 * instruments[0] + 0.4 * u + rng.normal(size=n) for _ in range(n_endog)
    ]
    y = 1.0 + 2.0 * endog[0] + u
    data = {"y": y, "u": u}
    for j, column in enumerate(instruments):
        data[f"z{j + 1}"] = column
    for i, column in enumerate(endog):
        data[f"x{i + 1}"] = column
    if include_exog:
        data["w"] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_mineig_equals_the_partial_r2_with_one_endogenous_regressor() -> None:
    """The canonical matrix is 1x1, so its eigenvalue IS the partial R-squared."""

    frame = _frame()
    first_stage = ws.IV2SLS.from_formula("y ~ [x1 ~ z1]", frame).fit().estat_firststage()

    assert first_stage.mineig == pytest.approx(
        float(first_stage.partial_r2[0]), rel=1e-12
    )


def test_mineig_satisfies_the_partial_f_identity() -> None:
    """``partial_r2 = F / (F + df_denom)`` -- a second, independent path.

    The denominator is Stata's ``N - rank - df_a - nendog`` (``ivregress_estat.ado:581``),
    NOT ``N - k`` for the auxiliary regression.  The two differ, so using the auxiliary
    ``k`` broke this identity when the first-stage denominator was corrected.
    """

    frame = _frame()
    first_stage = ws.IV2SLS.from_formula("y ~ [x1 ~ z1]", frame).fit().estat_firststage()

    f_stat = float(first_stage.f[0])
    implied = f_stat / (f_stat + float(first_stage.df_denom[0]))

    assert first_stage.mineig == pytest.approx(implied, rel=1e-12)


def test_the_three_quantities_agree_to_machine_precision() -> None:
    frame = _frame(seed=7)
    first_stage = ws.IV2SLS.from_formula("y ~ [x1 ~ z1]", frame).fit().estat_firststage()

    mineig = first_stage.mineig
    partial = float(first_stage.partial_r2[0])
    f_stat = float(first_stage.f[0])
    # Stata's denominator, not the auxiliary regression's ``N - k``.
    from_f = f_stat / (f_stat + float(first_stage.df_denom[0]))

    assert mineig == pytest.approx(partial, rel=1e-12)
    assert mineig == pytest.approx(from_f, rel=1e-12)


def test_the_identity_holds_with_included_exogenous_regressors() -> None:
    """The residualisation then runs on a non-empty design; same conclusion."""

    frame = _frame(n_instr=1, n_endog=1, include_exog=True)
    first_stage = (
        ws.IV2SLS.from_formula("y ~ w + [x1 ~ z1]", frame).fit().estat_firststage()
    )

    assert first_stage.mineig == pytest.approx(
        float(first_stage.partial_r2[0]), rel=1e-12
    )


def test_mineig_is_between_zero_and_one() -> None:
    """It is a squared canonical correlation, so it is bounded by [0, 1]."""

    for seed in (1, 2, 3):
        frame = _frame(seed=seed)
        first_stage = (
            ws.IV2SLS.from_formula("y ~ [x1 ~ z1]", frame).fit().estat_firststage()
        )
        assert 0.0 <= first_stage.mineig <= 1.0, seed


def test_mineig_matches_a_hand_built_eigenvalue_with_two_endogenous() -> None:
    """Beyond one regressor the matrix is 2x2; compare against a direct build.

    This checks the generalised eigenvalue construction, not Stata's value: no
    Stata reference for this shape exists in the repository, so the assertion is
    against the linear algebra rather than against Stata.  The construction is the
    same one the 1x1 case validates against two independent identities.
    """

    frame = _frame(n_instr=2, n_endog=2, seed=11)
    first_stage = (
        ws.IV2SLS.from_formula("y ~ [x1 + x2 ~ z1 + z2]",
                               frame).fit().estat_firststage()
    )

    n = len(frame)
    endog = np.column_stack([frame["x1"], frame["x2"]])
    instruments = np.column_stack([frame["z1"], frame["z2"]])

    def demean(column):
        return column - column.mean()

    endog_r = np.column_stack([demean(endog[:, i]) for i in range(2)])
    instr_r = np.column_stack([demean(instruments[:, j]) for j in range(2)])

    s_ee = endog_r.T @ endog_r
    s_zz = instr_r.T @ instr_r
    s_ez = endog_r.T @ instr_r
    middle = np.linalg.solve(s_ee, s_ez) @ np.linalg.solve(s_zz, s_ez.T)
    expected = float(np.min(np.real(np.linalg.eigvals(middle))))

    assert first_stage.mineig == pytest.approx(expected, rel=1e-10)


def test_mineig_differs_from_any_partial_r2_with_two_endogenous() -> None:
    """Recording the limitation: the identity that validates the 1x1 case does
    NOT extend here, so no independent check exists for this shape."""

    frame = _frame(n_instr=2, n_endog=2, seed=11)
    first_stage = (
        ws.IV2SLS.from_formula("y ~ [x1 + x2 ~ z1 + z2]",
                               frame).fit().estat_firststage()
    )

    assert not np.any(
        np.isclose(first_stage.mineig, first_stage.partial_r2, rtol=1e-6)
    )


def test_demeaning_fix_is_regression_guarded() -> None:
    """``_residualise`` must demean an empty design.

    Without it the columns stayed uncentred when a model had no included
    exogenous regressors, and the eigenvalue disagreed with the partial R-squared
    by 1.5% -- invisible without the identity check.
    """

    from westata.postestimation.firststage import _residualise

    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    residual = _residualise(values, np.empty((values.size, 0)))

    assert residual.mean() == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(residual, values - values.mean())


# ------------------- the demeaning fix moved partial_r2 to the CORRECT value
def _rss(values, design):
    augmented = np.column_stack([design, np.ones(values.size)])
    beta, *_ = np.linalg.lstsq(augmented, values, rcond=None)
    residual = values - augmented @ beta
    return float(residual @ residual)


def _partial_r2_textbook(target, included, instruments) -> float:
    """``1 - RSS_full/RSS_restricted`` -- Stata's restriction form.

    The full regression puts the endogenous regressor on included exog PLUS the
    excluded instruments; the restricted one drops the instruments.  This path
    never calls ``_residualise``, so it is independent of how that helper demeans.
    """

    full = np.column_stack([included, instruments])
    return 1.0 - _rss(target, full) / _rss(target, included)


@pytest.mark.parametrize(
    ("formula", "with_exog", "n_instr", "n_endog"),
    [
        ("y ~ [x1 ~ z1]", False, 1, 1),
        ("y ~ w + [x1 ~ z1]", True, 1, 1),
        ("y ~ [x1 ~ z1 + z2]", False, 2, 1),
        ("y ~ [x1 + x2 ~ z1 + z2]", False, 2, 2),
    ],
)
def test_partial_r2_matches_the_restriction_definition(
    formula, with_exog, n_instr, n_endog
) -> None:
    """Round 50 changed ``_residualise``; this confirms the new values are RIGHT.

    A fix that merely changes a number is not a fix. Comparing against the
    restriction form -- which is algebraically equivalent but takes a different
    computational path -- shows the demeaning change moved ``partial_r2`` to the
    correct value rather than to a different one.
    """

    frame = _frame(n_instr=n_instr, n_endog=n_endog, include_exog=with_exog)
    first_stage = ws.IV2SLS.from_formula(formula, frame).fit().estat_firststage()

    included = (
        np.column_stack([frame["w"]]) if with_exog else np.empty((len(frame), 0))
    )
    instruments = np.column_stack([frame[f"z{j + 1}"] for j in range(n_instr)])
    endog = np.column_stack([frame[f"x{i + 1}"] for i in range(n_endog)])

    for i in range(n_endog):
        expected = _partial_r2_textbook(endog[:, i], included, instruments)
        assert float(first_stage.partial_r2[i]) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    ("formula", "with_exog", "n_instr", "n_endog"),
    [
        ("y ~ [x1 ~ z1]", False, 1, 1),
        ("y ~ w + [x1 ~ z1]", True, 1, 1),
        ("y ~ [x1 ~ z1 + z2]", False, 2, 1),
    ],
)
def test_the_partial_f_matches_its_definition(
    formula, with_exog, n_instr, n_endog
) -> None:
    """``F = (partial_r2/df_num) / ((1 - partial_r2)/df_denom)``."""

    frame = _frame(n_instr=n_instr, n_endog=n_endog, include_exog=with_exog)
    first_stage = ws.IV2SLS.from_formula(formula, frame).fit().estat_firststage()

    df_num = float(n_instr)
    df_denom = float(first_stage.df_denom[0])
    for i in range(n_endog):
        p = float(first_stage.partial_r2[i])
        expected = (p / df_num) / ((1.0 - p) / df_denom)
        assert float(first_stage.f[i]) == pytest.approx(expected, rel=1e-10)


def test_r2_and_partial_r2_coincide_only_without_included_exogenous() -> None:
    """With no included regressors the two are the same regression.

    Recording this guards the distinction: a future refactor that aliased one to
    the other would break the case WITH a relevant included regressor, where they
    differ. The included regressor here is deliberately correlated with the
    endogenous one -- an unrelated regressor leaves both quantities almost equal
    and would make the distinction untestable.
    """

    without = _frame(n_instr=1, n_endog=1, include_exog=False)
    plain = ws.IV2SLS.from_formula("y ~ [x1 ~ z1]", without).fit().estat_firststage()
    assert float(plain.r2[0]) == pytest.approx(float(plain.partial_r2[0]), rel=1e-12)

    # Build an included regressor that carries real explanatory power for x1.
    rng = np.random.default_rng(5)
    n = 400
    z1 = rng.normal(size=n)
    u = rng.normal(size=n)
    w = rng.normal(size=n)
    x1 = 0.5 * z1 + 0.8 * w + 0.4 * u + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + 0.3 * w + u
    frame = pd.DataFrame({"y": y, "x1": x1, "z1": z1, "w": w})

    rich = ws.IV2SLS.from_formula("y ~ w + [x1 ~ z1]", frame).fit().estat_firststage()

    assert not np.isclose(float(rich.r2[0]), float(rich.partial_r2[0]), rtol=1e-2)
