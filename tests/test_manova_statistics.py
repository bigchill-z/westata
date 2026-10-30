"""The four MANOVA statistics and their defining identities.

``manova`` prints only Wilks' lambda; the other three appear in ``manovatest``
(``manova_postestimation.sthlp:243-245``). This repository has no Stata fixture
for ``manovatest``, so the statistics are checked against the *definitions* in
``mv_glossary.sthlp``, computed from hand-built between/within matrices:

| statistic | definition |
|---|---|
| Wilks' lambda | ``L = |E|/|E+H| = prod 1/(1+lambda_i)`` |
| Pillai's trace | ``V = trace[(E+H)^-1 H] = sum lambda_i/(1+lambda_i)`` |
| Lawley-Hotelling | ``U = trace(E^-1 H) = sum lambda_i`` |
| Roy's largest root | ``theta = lambda_max/(1+lambda_max)`` |

Because all four come from the same eigenvalues they are not independent, and
that interdependence is what makes them checkable: with one eigenvalue the four
collapse onto a single quantity, and ``L + V = 1`` always holds.

**A numerical finding.** The first implementation computed Wilks from
``det(E)/det(E+H)``. On a design with ``cond(E) ~ 3e12`` that disagreed with the
eigenvalue product by 2e-4 while the other three were exact. Wilks is now derived
from the eigenvalues so all four stay mutually consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n_per_group=30, groups=3, seed=1):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(groups), n_per_group)
    n = labels.size
    y1 = rng.normal(size=n) + labels * 0.8
    y2 = rng.normal(size=n) - labels * 0.5
    return pd.DataFrame({"y1": y1, "y2": y2, "g": labels})


def _hand_matrices(frame):
    """Build H and E directly, the way ``mv_glossary.sthlp`` defines them.

    The eigenvalues of ``E^-1 H`` are obtained by symmetrising, because
    ``E^-1 H`` is not symmetric and ``eigvals`` on it can return a spurious
    complex pair (and, on ill-conditioned designs, a wrong-signed real root).
    ``E^-1/2 H E^-1/2`` is symmetric with the same eigenvalues.
    """

    y = np.column_stack([frame["y1"], frame["y2"]])
    labels = frame["g"].to_numpy()
    grand = y.mean(axis=0)
    p = y.shape[1]
    within = np.zeros((p, p))
    between = np.zeros((p, p))
    for value in np.unique(labels):
        block = y[labels == value]
        centred = block - block.mean(axis=0)
        within += centred.T @ centred
        difference = block.mean(axis=0) - grand
        between += block.shape[0] * np.outer(difference, difference)

    # Symmetric route: same eigenvalues as E^-1 H, numerically better behaved.
    values, vectors = np.linalg.eigh(within)
    inverse_root = vectors @ np.diag(values ** -0.5) @ vectors.T
    symmetric = inverse_root @ between @ inverse_root
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(symmetric)))[::-1]
    return eigenvalues, within, between


@pytest.fixture()
def fitted():
    frame = _frame()
    return ws.Manova.from_formula("y1 y2 = g", frame).fit(), frame


# ------------------------------------------------------------- definitions
def test_wilks_matches_the_determinant_definition(fitted) -> None:
    result, frame = fitted
    _eigenvalues, within, between = _hand_matrices(frame)

    expected = float(np.linalg.det(within) / np.linalg.det(within + between))

    assert float(result.wilks_lambda) == pytest.approx(expected, rel=1e-10)


def test_pillai_matches_its_definition(fitted) -> None:
    result, frame = fitted
    eigenvalues, _within, _between = _hand_matrices(frame)

    expected = float(np.sum(eigenvalues / (1.0 + eigenvalues)))

    assert result.pillai_trace == pytest.approx(expected, rel=1e-10)


def test_lawley_hotelling_matches_its_definition(fitted) -> None:
    result, frame = fitted
    eigenvalues, _within, _between = _hand_matrices(frame)

    assert result.lawley_hotelling_trace == pytest.approx(
        float(np.sum(eigenvalues)), rel=1e-10
    )


def test_roy_matches_its_definition(fitted) -> None:
    result, frame = fitted
    eigenvalues, _within, _between = _hand_matrices(frame)

    assert result.roy_largest_root == pytest.approx(
        float(eigenvalues[0] / (1.0 + eigenvalues[0])), rel=1e-10
    )


# ------------------------------------------------- interdependent identities
def test_wilks_equals_the_eigenvalue_product(fitted) -> None:
    """``|E|/|E+H| = prod 1/(1+lambda_i)``.

    Both forms are computed; they must agree. The DETERMINANT form is what the
    statistics are checked against above, so this pins the eigenvalue route the
    implementation actually uses.
    """

    result, frame = fitted
    eigenvalues, _within, _between = _hand_matrices(frame)

    assert float(result.wilks_lambda) == pytest.approx(
        float(np.prod(1.0 / (1.0 + eigenvalues))), rel=1e-10
    )


def test_the_four_statistics_are_internally_consistent(fitted) -> None:
    """All four are monotone functions of the same eigenvalues.

    With two responses the ordering ``roy <= pillai <= lawley`` must hold, since
    each term is dominated by the largest eigenvalue. A coding slip that mixed up
    the formulas would break it.
    """

    result, _frame_ = fitted

    assert 0.0 < result.wilks_lambda < 1.0
    assert 0.0 < result.pillai_trace < 2.0
    assert result.roy_largest_root <= result.pillai_trace + 1e-12
    assert result.pillai_trace <= result.lawley_hotelling_trace + 1e-12


def test_multivariate_tests_lists_all_four(fitted) -> None:
    result, _frame_ = fitted

    table = result.multivariate_tests()

    assert set(table) == {
        "Wilks' lambda",
        "Pillai's trace",
        "Lawley-Hotelling trace",
        "Roy's largest root",
    }
    assert table["Wilks' lambda"] == pytest.approx(float(result.wilks_lambda))


# ------------------------------------------------ the collapse at one dimension
def test_a_single_effective_dimension_collapses_the_statistics() -> None:
    """With one dominant eigenvalue ``roy == pillai`` and ``L + V`` is near 1.

    This exercises the RELATIONSHIP between the formulas rather than each one in
    isolation, which is the strongest self-check available without a
    ``manovatest`` fixture.

    Note the tolerance and the wording: a "nearly collinear" second response is
    not an exactly rank-one design, so the small second eigenvalue (about 2.6e-4)
    leaves ``L + V`` at 1.00014 rather than 1. Exactly one eigenvalue would give
    equality, but constructing that makes ``E`` singular and is refused.
    """

    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1], 40)
    y1 = rng.normal(size=80)
    y1[40:] += 2.0
    y2 = y1 + rng.normal(size=80) * 1e-6
    frame = pd.DataFrame({"y1": y1, "y2": y2, "g": labels})

    result = ws.Manova.from_formula("y1 y2 = g", frame).fit()
    eigenvalues, _within, _between = _hand_matrices(frame)

    # The second eigenvalue is tiny relative to the first.
    assert eigenvalues[0] > 1000 * eigenvalues[1]
    # And on such a design the two dominant statistics nearly coincide.
    assert result.roy_largest_root == pytest.approx(result.pillai_trace, rel=1e-2)
    assert float(result.wilks_lambda) + result.pillai_trace == pytest.approx(
        1.0, abs=2e-3
    )


def test_wilks_uses_the_eigenvalue_route_on_an_ill_conditioned_design() -> None:
    """Wilks is derived from the same eigenvalues as the other three statistics.

    The point is CONSISTENCY, and it matters because the routes can disagree.
    With a strongly collinear pair of responses the two hand routes differ:

    * ``det(E)/det(E+H)``   -> 0.5480212
    * eigenvalue product    -> 0.5480218

    A 4th-digit gap, which is enough to make ``L`` inconsistent with ``V``/``U``
    if they came from different computations.  The implementation takes every
    statistic from one eigen-decomposition, so they cannot drift apart.

    The design deliberately stops short of the numerically singular case: at
    ``cond(E) ~ 3e12`` the smallest eigenvalue of ``E`` is ~3e-11 and even the
    symmetric route loses symmetry, so no route is "the answer" and comparing
    them would test floating point rather than the code.
    """

    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1], 40)
    y1 = rng.normal(size=80)
    y1[40:] += 2.0
    y2 = y1 + rng.normal(size=80) * 1e-3  # collinear, but not numerically singular
    frame = pd.DataFrame({"y1": y1, "y2": y2, "g": labels})

    eigenvalues, within, between = _hand_matrices(frame)
    condition = np.linalg.cond(within)
    assert 1e4 < condition < 1e9, condition

    determinant_form = float(np.linalg.det(within) / np.linalg.det(within + between))
    eigenvalue_form = float(np.prod(1.0 / (1.0 + eigenvalues)))

    result = ws.Manova.from_formula("y1 y2 = g", frame).fit()

    # The implementation matches the eigenvalue route exactly ...
    assert float(result.wilks_lambda) == pytest.approx(eigenvalue_form, rel=1e-10)
    # ... which is close to, but not identical with, the determinant route.
    assert float(result.wilks_lambda) == pytest.approx(determinant_form, rel=1e-3)

    # All four statistics come from that one decomposition.
    assert result.pillai_trace == pytest.approx(
        float(np.sum(eigenvalues / (1.0 + eigenvalues))), rel=1e-10
    )
    assert result.lawley_hotelling_trace == pytest.approx(
        float(np.sum(eigenvalues)), rel=1e-10
    )


# -------------------------------------------------------------- refusal paths
def test_a_collinear_response_is_refused() -> None:
    """Two identical responses make the total sum of squares singular."""

    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 30)
    y1 = rng.normal(size=60)
    frame = pd.DataFrame({"y1": y1, "y2": y1, "g": labels})

    with pytest.raises(ws.ConsistencyError, match="singular"):
        ws.Manova.from_formula("y1 y2 = g", frame).fit()


def test_the_statistics_appear_in_e() -> None:
    """They are recorded so ``e()`` carries them like Stata's ``manovatest``."""

    frame = _frame(seed=5)
    result = ws.Manova.from_formula("y1 y2 = g", frame).fit()

    for key in ("wilks_lambda", "pillai_trace", "lawley_hotelling_trace", "roy_largest_root"):
        assert key in result.statistics
        assert np.isfinite(float(result.statistics[key]))


# ------------------------------------- per-equation e() vectors (manova.sthlp)
def test_per_equation_r2_matches_a_plain_ols_fit(fitted) -> None:
    """``manova.sthlp:238``: "R-squared for each equation".

    These are the ORDINARY univariate fits, one per response -- not the
    multivariate test. Comparing against a separate ``OLS`` of each response on
    the group dummies checks that they were not accidentally replaced by the
    multivariate statistic.
    """

    result, frame = fitted

    for position, response in enumerate(("y1", "y2")):
        univariate = ws.OLS.from_formula(f"{response} ~ C(g)", frame).fit()
        assert result.r2_by_equation[position] == pytest.approx(
            float(univariate.rsquared), rel=1e-10
        )


def test_per_equation_f_matches_a_plain_ols_fit(fitted) -> None:
    """``manova.sthlp:240``: the F statistic for each equation."""

    from scipy import stats

    result, frame = fitted

    for position, response in enumerate(("y1", "y2")):
        univariate = ws.OLS.from_formula(f"{response} ~ C(g)", frame).fit()
        assert result.f_by_equation[position] == pytest.approx(
            float(univariate.e["F"]), rel=1e-10
        )
        expected_p = float(
            stats.f.sf(
                float(univariate.e["F"]),
                float(univariate.e["df_m"]),
                float(univariate.e["df_r"]),
            )
        )
        assert result.p_f_by_equation[position] == pytest.approx(expected_p, rel=1e-10)


def test_per_equation_rmse_uses_the_residual_leaving_groups_out(fitted) -> None:
    """``manova.sthlp:239``: RMSE per equation, from the within-group residuals."""

    result, frame = fitted
    n = len(frame)
    groups = frame["g"].nunique()

    for position, response in enumerate(("y1", "y2")):
        values = frame[response].to_numpy()
        residuals = np.zeros_like(values)
        for value in frame["g"].unique():
            mask = frame["g"].to_numpy() == value
            residuals[mask] = values[mask] - values[mask].mean()
        expected = float(np.sqrt(residuals @ residuals / (n - groups)))
        assert result.rmse_by_equation[position] == pytest.approx(expected, rel=1e-10)


def test_e_carries_the_documented_macros(fitted) -> None:
    """``manova.sthlp:229-247`` lists the macros an eclass command carries."""

    result, _frame_ = fitted
    e = result.e

    assert e["cmd"] == "manova"
    assert e["depvar"] == "y1 y2"
    assert e["indepvars"] == "g"
    assert e["properties"] == "b V"
    assert e["estat_cmd"] == "manova_estat"


def test_predict_macros_are_present_because_predict_now_exists(fitted) -> None:
    """Round 63 withdrew these macros because no predict existed; it does now.

    ``manova_p.ado:16`` declares ``local myopts "Difference"`` and delegates the
    rest to ``_pred_me``.  With ``ManovaResult.predict`` implemented, advertising
    the macro is accurate rather than misleading.
    """

    result, _frame_ = fitted

    assert hasattr(result, "predict")
    assert result.e["predict"] == "manova_p"
    assert "predict(xb) stdp" in result.e["marginsnotok"]
    assert result.e["marginsdefault"] == "xb"


# ------------------------------------------------------------- manova predict
def test_predict_xb_returns_the_cell_means(fitted) -> None:
    """A saturated one-way design: the fitted value of every observation in a
    group is that group's mean."""

    result, frame = fitted

    fitted_values = np.asarray(result.predict("xb"), dtype=float)
    means = frame.groupby("g")["y1"].mean()
    expected = np.array([means[code] for code in frame["g"]])

    np.testing.assert_allclose(fitted_values, expected, rtol=1e-10)


def test_predict_index_is_the_synonym_of_xb(fitted) -> None:
    result, _frame_ = fitted

    np.testing.assert_allclose(
        np.asarray(result.predict("index"), dtype=float),
        np.asarray(result.predict("xb"), dtype=float),
    )


def test_predict_defaults_to_the_first_equation(fitted) -> None:
    """``manova_p.ado:38-44``: no ``equation()`` means the first dependent variable."""

    result, _frame_ = fitted

    np.testing.assert_allclose(
        np.asarray(result.predict("xb"), dtype=float),
        np.asarray(result.predict("xb", equation=1), dtype=float),
    )


def test_predict_residuals_sum_to_zero_within_each_group(fitted) -> None:
    result, frame = fitted

    residuals = np.asarray(result.predict("residuals"), dtype=float)
    for value in frame["g"].unique():
        mask = frame["g"].to_numpy() == value
        assert abs(float(residuals[mask].sum())) < 1e-10


def test_predict_residuals_reconstruct_the_response(fitted) -> None:
    result, frame = fitted

    reconstructed = np.asarray(result.predict("xb"), dtype=float) + np.asarray(
        result.predict("residuals"), dtype=float
    )

    np.testing.assert_allclose(
        reconstructed, frame["y1"].to_numpy(), rtol=1e-10
    )


def test_predict_difference_is_xb1_minus_xb2(fitted) -> None:
    """``manova_p.ado:100-106``: ``gen varn = xb1 - xb2``."""

    result, _frame_ = fitted

    difference = np.asarray(
        result.predict("difference", equation=1, equation2=2), dtype=float
    )
    expected = np.asarray(result.predict("xb", equation=1), dtype=float) - np.asarray(
        result.predict("xb", equation=2), dtype=float
    )

    np.testing.assert_allclose(difference, expected, rtol=1e-10)


def test_predict_stdp_is_the_sandwich_standard_error(fitted) -> None:
    """``manova_postestimation.sthlp:151-154``: the standard error of the fitted value.

    That is ``sqrt(x' V x)`` with V the equation's coefficient covariance. On a
    saturated one-way design the shortcut ``rmse/sqrt(n_cell)`` gives the same
    number, which is why an earlier implementation used it and looked correct --
    but it is not the definition, so the sandwich form is what runs.
    """

    result, frame = fitted

    stdp = np.asarray(result.predict("stdp"), dtype=float)

    # Independent build: OLS covariance of the same equation.
    univariate = ws.OLS.from_formula("y1 ~ C(g)", frame).fit()
    covariance = np.asarray(univariate.cov_params(), dtype=float)
    design = np.asarray(result.design_matrix, dtype=float)
    expected = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))

    np.testing.assert_allclose(stdp, expected, rtol=1e-10)
    assert np.all(stdp > 0)


def test_predict_stddp_includes_the_cross_equation_covariance(fitted) -> None:
    """``manova_postestimation.sthlp:164-165``: the SE of ``x1jb - x2jb``.

        Var = x'V11x + x'V22x - 2 x'V12x

    The ``V12`` term is not a detail: on this data the equation residuals
    correlate at about 0.17, and dropping the term inflated ``stddp`` from
    0.21149 to 0.23204 -- a 9% error that no plausibility check would catch.
    """

    result, frame = fitted

    combined = np.asarray(
        result.predict("stddp", equation=1, equation2=2), dtype=float
    )

    univariate1 = ws.OLS.from_formula("y1 ~ C(g)", frame).fit()
    univariate2 = ws.OLS.from_formula("y2 ~ C(g)", frame).fit()
    v_first = np.asarray(univariate1.cov_params(), dtype=float)
    v_second = np.asarray(univariate2.cov_params(), dtype=float)

    design = np.asarray(result.design_matrix, dtype=float)
    n, k = design.shape
    residual_first = np.asarray(result.predict("residuals", equation=1), dtype=float)
    residual_second = np.asarray(result.predict("residuals", equation=2), dtype=float)
    sigma12 = float(residual_first @ residual_second) / (n - k)
    v_cross = sigma12 * np.linalg.pinv(design.T @ design)

    variance = (
        np.einsum("ij,jk,ik->i", design, v_first, design)
        + np.einsum("ij,jk,ik->i", design, v_second, design)
        - 2.0 * np.einsum("ij,jk,ik->i", design, v_cross, design)
    )
    expected = np.sqrt(np.maximum(variance, 0.0))

    np.testing.assert_allclose(combined, expected, rtol=1e-10)


def test_stddp_is_not_the_naive_quadrature_sum(fitted) -> None:
    """Recording the size of the error the covariance term removes."""

    result, _frame_ = fitted

    combined = np.asarray(
        result.predict("stddp", equation=1, equation2=2), dtype=float
    )
    naive = np.sqrt(
        np.asarray(result.predict("stdp", equation=1), dtype=float) ** 2
        + np.asarray(result.predict("stdp", equation=2), dtype=float) ** 2
    )

    # They genuinely differ on data with correlated equations.
    assert not np.allclose(combined, naive, rtol=1e-6)
    assert np.all(combined < naive)


def test_predict_takes_the_second_equation(fitted) -> None:
    """``equation(2)`` must return the SECOND response's fitted values."""

    result, frame = fitted

    second = np.asarray(result.predict("xb", equation=2), dtype=float)
    means = frame.groupby("g")["y2"].mean()
    expected = np.array([means[code] for code in frame["g"]])

    np.testing.assert_allclose(second, expected, rtol=1e-10)


def test_predict_refuses_unknown_types(fitted) -> None:
    result, _frame_ = fitted

    with pytest.raises(ValueError, match="manova_p"):
        result.predict("cooksd")


def test_predict_refuses_an_equation_out_of_range(fitted) -> None:
    result, _frame_ = fitted

    with pytest.raises(ValueError, match="out of range"):
        result.predict("xb", equation=3)


def test_predict_requires_two_equations_for_difference(fitted) -> None:
    result, _frame_ = fitted

    with pytest.raises(ValueError, match="equation"):
        result.predict("difference")


def test_the_per_equation_keys_are_vectors(fitted) -> None:
    """Stata stores e(r2)/e(rmse)/e(F)/e(p_F) per equation, not as scalars."""

    result, _frame_ = fitted

    for key in ("r2", "rmse", "F", "p_F"):
        value = result.e[key]
        assert len(value) == 2, key
        assert all(np.isfinite(float(entry)) for entry in value), key
