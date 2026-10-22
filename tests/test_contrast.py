"""Conformance tests for ``contrast`` and ``pwcompare``.

Every reported quantity is checked against an independent hand computation from
``[R] contrast`` / ``[R] pwcompare`` -- a weight vector ``w`` built from the
level structure, with estimate ``w'b`` and standard error ``sqrt(w' V w)`` --
never against the implementation under test.

The factor here is ``C(g, base='a')`` on a 3-level ``g``, so the design matrix
is ``[Intercept, C(g, Treatment(reference='a'))[T.b],
C(g, Treatment(reference='a'))[T.c], x]`` and the level coefficients are
``b_a = 0`` (the reference, by construction), ``b_b``, ``b_c``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.errors import UnsupportedError
from westata.postestimation.contrast import contrast, pwcompare
from westata.postestimation.lincom import LincomResult

LEVELS = ["a", "b", "c"]


def _linear(n: int = 120, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.array(LEVELS * (n // len(LEVELS)))
    x = rng.normal(size=g.size)
    effect = {"a": 0.0, "b": 2.0, "c": -1.0}
    y = (
        1.0
        + np.array([effect[level] for level in g])
        + 0.5 * x
        + rng.normal(scale=0.7, size=g.size)
    )
    return pd.DataFrame({"y": y, "x": x, "g": g})


def _binary(n: int = 600, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.array(LEVELS * (n // len(LEVELS)))
    x = rng.normal(size=g.size)
    effect = {"a": 0.0, "b": 1.1, "c": -0.7}
    latent = (
        -0.3
        + np.array([effect[level] for level in g])
        + 0.8 * x
        + rng.logistic(size=g.size)
    )
    return pd.DataFrame({"y": (latent > 0).astype(float), "x": x, "g": g})


@pytest.fixture(scope="module")
def ols_result():
    return ws.OLS.from_formula("y ~ x + C(g, base='a')", _linear()).fit()


@pytest.fixture(scope="module")
def logit_result():
    return ws.Logit.from_formula("y ~ x + C(g, base='a')", _binary()).fit()


# ------------------------------------------------------------------ helpers
def _level_coefficients(result) -> dict[str, float]:
    """Map each level to its coefficient, with the reference level set to 0."""

    coefficients = {"a": 0.0}
    for name in result.param_names:
        if "[T." not in name:
            continue
        level = name.split("[T.", 1)[1].rstrip("]")
        coefficients[level] = float(result.params[name])
    return coefficients


def _level_weights(result, level: str) -> np.ndarray:
    """Design-column weight vector selecting one level (reference -> zeros)."""

    weights = np.zeros(len(result.param_names))
    for position, name in enumerate(result.param_names):
        if "[T." in name and name.split("[T.", 1)[1].rstrip("]") == level:
            weights[position] = 1.0
    return weights


def _manual(result, weights: np.ndarray) -> tuple[float, float]:
    """Hand-computed ``w'b`` and ``sqrt(w' V w)``."""

    beta = result.params.to_numpy()
    cov = result.cov_params()
    return float(weights @ beta), float(np.sqrt(weights @ cov @ weights))


# --------------------------------------------------------------- reference
def test_reference_contrast_has_one_row_per_non_reference_level(ols_result):
    rows = contrast(ols_result, "C(g)", "r")

    assert len(rows) == len(LEVELS) - 1
    assert all(isinstance(row, LincomResult) for row in rows)


def test_reference_contrast_estimate_is_level_minus_reference(ols_result):
    rows = contrast(ols_result, "C(g)", "r")
    beta = _level_coefficients(ols_result)
    reference_column = "C(g, Treatment(reference='a'))[T.{}]"

    for row, level in zip(rows, ["b", "c"]):
        # The reference coefficient is zero by construction, so this is the
        # level's own coefficient -- exactly Stata's `contrast r.` output.
        assert beta["a"] == 0.0
        assert row.estimate == pytest.approx(beta[level] - beta["a"], rel=1e-12)
        assert row.estimate == pytest.approx(
            float(ols_result.params[reference_column.format(level)]), rel=1e-12
        )

    # Reference contrast rows follow level order: b first, then c.
    assert rows[0].label.startswith("b vs")
    assert rows[1].label.startswith("c vs")


def test_reference_contrast_row_order_is_b_then_c(ols_result):
    rows = contrast(ols_result, "C(g)", "r")
    beta = _level_coefficients(ols_result)

    assert rows[0].estimate == pytest.approx(beta["b"], rel=1e-12)
    assert rows[1].estimate == pytest.approx(beta["c"], rel=1e-12)
    assert beta["b"] > beta["c"]


# ---------------------------------------------------------------- adjacent
def test_adjacent_contrast_has_one_row_per_consecutive_pair(ols_result):
    """Stata's ``contrast a.`` reports ``low vs high``, i.e. the earlier level
    minus the later one. Verified on StataNow 19.5: for a 3-level factor,
    ``contrast a.g`` prints "(a vs b)" with the sign opposite to
    ``contrast r.g``'s "(b vs a)"."""

    rows = contrast(ols_result, "C(g)", "a")
    beta = _level_coefficients(ols_result)

    assert len(rows) == len(LEVELS) - 1
    assert rows[0].estimate == pytest.approx(beta["a"] - beta["b"], rel=1e-12)
    assert rows[1].estimate == pytest.approx(beta["b"] - beta["c"], rel=1e-12)


def test_adjacent_contrast_standard_errors_match_hand_computation(ols_result):
    rows = contrast(ols_result, "C(g)", "a")

    for row, (low, high) in zip(rows, [("a", "b"), ("b", "c")]):
        # low minus high, matching Stata's adjacent direction.
        weights = _level_weights(ols_result, low) - _level_weights(ols_result, high)
        estimate, std_err = _manual(ols_result, weights)
        assert row.estimate == pytest.approx(estimate, rel=1e-12)
        assert row.std_err == pytest.approx(std_err, rel=1e-12)


# -------------------------------------------------------------- grand mean
def test_grand_mean_contrast_has_one_row_per_level(ols_result):
    rows = contrast(ols_result, "C(g)", "g")
    beta = _level_coefficients(ols_result)

    assert len(rows) == len(LEVELS)
    grand_mean = float(np.mean([beta[level] for level in LEVELS]))
    for row, level in zip(rows, LEVELS):
        assert row.estimate == pytest.approx(beta[level] - grand_mean, rel=1e-12)


def test_grand_mean_contrast_uses_unweighted_average_of_levels(ols_result):
    """The grand mean is over levels, not over observations."""

    rows = contrast(ols_result, "C(g)", "g")
    beta = _level_coefficients(ols_result)

    unweighted = (beta["a"] + beta["b"] + beta["c"]) / 3.0
    # Rows sum to zero only when the average is unweighted over levels.
    assert sum(row.estimate for row in rows) == pytest.approx(0.0, abs=1e-12)
    assert rows[1].estimate == pytest.approx(beta["b"] - unweighted, rel=1e-12)


def test_grand_mean_standard_error_matches_hand_computation(ols_result):
    rows = contrast(ols_result, "C(g)", "g")

    # w_b = e_b - (e_a + e_b + e_c)/3, with e_a = 0.
    weights = (
        _level_weights(ols_result, "b")
        - sum(_level_weights(ols_result, level) for level in LEVELS) / 3.0
    )
    estimate, std_err = _manual(ols_result, weights)
    assert rows[1].estimate == pytest.approx(estimate, rel=1e-12)
    assert rows[1].std_err == pytest.approx(std_err, rel=1e-12)
    # Sanity: the grand-mean SE differs from the plain coefficient SE.
    assert rows[1].std_err != pytest.approx(
        float(ols_result.bse["C(g, Treatment(reference='a'))[T.b]"]), rel=1e-6
    )


# --------------------------------------------------------------- pwcompare
def test_pwcompare_emits_one_row_per_pair(ols_result):
    rows = pwcompare(ols_result, "C(g)")

    assert len(rows) == len(LEVELS) * (len(LEVELS) - 1) // 2 == 3


def test_pwcompare_b_vs_a_equals_difference_of_coefficients(ols_result):
    rows = pwcompare(ols_result, "C(g)")
    beta = _level_coefficients(ols_result)

    pairs = [("b", "a"), ("c", "a"), ("c", "b")]
    for row, (high, low) in zip(rows, pairs):
        assert row.estimate == pytest.approx(beta[high] - beta[low], rel=1e-12)

    # b - a is the first pair (j > i in level order a, b, c).
    assert rows[0].estimate == pytest.approx(beta["b"] - beta["a"], rel=1e-12)


def test_pwcompare_standard_errors_match_hand_computation(ols_result):
    rows = pwcompare(ols_result, "C(g)")

    for row, (high, low) in zip(rows, [("b", "a"), ("c", "a"), ("c", "b")]):
        weights = _level_weights(ols_result, high) - _level_weights(ols_result, low)
        estimate, std_err = _manual(ols_result, weights)
        assert row.estimate == pytest.approx(estimate, rel=1e-12)
        assert row.std_err == pytest.approx(std_err, rel=1e-12)


def test_pwcompare_is_antisymmetric_in_pairs(ols_result):
    """Contrasting a pair twice with the opposite sign flips the estimate."""

    beta = _level_coefficients(ols_result)
    rows = pwcompare(ols_result, "C(g)")
    for row, (high, low) in zip(rows, [("b", "a"), ("c", "a"), ("c", "b")]):
        reversed_weights = _level_weights(ols_result, low) - _level_weights(
            ols_result, high
        )
        reversed_estimate, _ = _manual(ols_result, reversed_weights)
        assert row.estimate == pytest.approx(-reversed_estimate, rel=1e-12)
        assert row.estimate == pytest.approx(beta[high] - beta[low], rel=1e-12)


# ------------------------------------------------------- inference metadata
def test_ols_contrast_uses_t_distribution_and_model_degrees_of_freedom(ols_result):
    row = contrast(ols_result, "C(g)", "r")[0]

    assert row.distribution == "t"
    assert row.df == pytest.approx(ols_result.df_resid)
    critical = stats.t.ppf(0.975, ols_result.df_resid)
    assert row.ci_lower == pytest.approx(
        row.estimate - critical * row.std_err, rel=1e-12
    )
    assert row.ci_upper == pytest.approx(
        row.estimate + critical * row.std_err, rel=1e-12
    )
    assert row.statistic == pytest.approx(row.estimate / row.std_err, rel=1e-12)
    assert row.pvalue == pytest.approx(
        2.0 * stats.t.sf(abs(row.statistic), ols_result.df_resid), rel=1e-12
    )


def test_contrast_honours_the_requested_level(ols_result):
    row = contrast(ols_result, "C(g)", "r", level=0.90)[0]

    critical = stats.t.ppf(0.95, ols_result.df_resid)
    assert row.ci_lower == pytest.approx(
        row.estimate - critical * row.std_err, rel=1e-12
    )
    assert row.ci_upper == pytest.approx(
        row.estimate + critical * row.std_err, rel=1e-12
    )


def test_contrast_propagates_a_robust_vce():
    """A robust VCE must change the contrast standard error, not the estimate."""

    data = _linear()
    plain = ws.OLS.from_formula("y ~ x + C(g, base='a')", data).fit()
    robust = ws.OLS.from_formula("y ~ x + C(g, base='a')", data).fit(cov_type="HC1")

    plain_row = plain_contrast = contrast(plain, "C(g)", "r")[0]
    robust_row = contrast(robust, "C(g)", "r")[0]

    assert plain_row.estimate == pytest.approx(robust_row.estimate, rel=1e-12)
    assert plain_row.std_err != pytest.approx(robust_row.std_err, rel=1e-6)
    assert plain_contrast.std_err > 0.0


# --------------------------------------------------------------- label/term
def test_contrast_accepts_the_bare_variable_name(ols_result):
    by_full = contrast(ols_result, "C(g)", "r")
    by_short = contrast(ols_result, "g", "r")

    assert [row.estimate for row in by_short] == pytest.approx(
        [row.estimate for row in by_full], rel=1e-12
    )


def test_contrast_labels_each_comparison(ols_result):
    reference_rows = contrast(ols_result, "C(g)", "r")
    assert [row.label for row in reference_rows] == ["b vs a", "c vs a"]

    grand_rows = contrast(ols_result, "C(g)", "g")
    assert [row.label for row in grand_rows] == [
        "a vs grand mean",
        "b vs grand mean",
        "c vs grand mean",
    ]

    pairwise_rows = pwcompare(ols_result, "C(g)")
    assert [row.label for row in pairwise_rows] == ["b vs a", "c vs a", "c vs b"]


def test_contrast_accepts_the_explicit_reference_term(ols_result):
    """``C(g, base='a')`` and ``C(g)`` must resolve the same factor."""

    by_base = contrast(ols_result, "C(g, base='a')", "r")
    by_short = contrast(ols_result, "C(g)", "r")

    assert [row.estimate for row in by_base] == pytest.approx(
        [row.estimate for row in by_short], rel=1e-12
    )


def test_expression_does_not_claim_a_column_for_the_reference_level(ols_result):
    """The reference level has no coefficient, so it must not look like one."""

    row = contrast(ols_result, "C(g)", "r")[0]

    # "b" is a real design column; "a" is the omitted reference level.
    assert "C(g, Treatment(reference='a'))[T.b]" in row.expression
    assert "[T.a]" not in row.expression
    assert "base(g[a])" in row.expression


def test_contrast_unknown_term_raises_keyerror(ols_result):
    with pytest.raises(KeyError, match="does not match any factor column"):
        contrast(ols_result, "C(not_a_variable)", "r")


def test_contrast_of_a_plain_numeric_term_raises_keyerror(ols_result):
    # ``x`` is a continuous regressor: it has no levels to contrast.
    with pytest.raises(KeyError, match="does not match any factor column"):
        contrast(ols_result, "x", "r")


# ------------------------------------------------------- unsupported inputs
def test_contrast_unknown_method_raises_unsupported(ols_result):
    with pytest.raises(UnsupportedError, match="method"):
        contrast(ols_result, "C(g)", "nope")


@pytest.mark.parametrize("method", ["h", "m", "h.", "m."])
def test_contrast_helmert_and_multiple_comparison_raise(ols_result, method):
    with pytest.raises(UnsupportedError, match="not implemented"):
        contrast(ols_result, "C(g)", method)


def test_pwcompare_bonferroni_raises_unsupported(ols_result):
    with pytest.raises(UnsupportedError, match="adjustment is not implemented"):
        pwcompare(ols_result, "C(g)", adjust="bonferroni")


@pytest.mark.parametrize("adjust", ["sidak", "tukey", "scheffe", "dunnett", "none", ""])
def test_pwcompare_rejects_every_other_adjust(ols_result, adjust):
    with pytest.raises(UnsupportedError):
        pwcompare(ols_result, "C(g)", adjust=adjust)


# ------------------------------------------------------------- logit (z) path
def test_logit_contrast_uses_z_distribution(logit_result):
    assert logit_result.dist == "z"
    rows = contrast(logit_result, "C(g)", "r")

    assert len(rows) == len(LEVELS) - 1
    for row in rows:
        assert row.distribution == "z"
        assert row.df is None
        assert row.pvalue == pytest.approx(
            2.0 * stats.norm.sf(abs(row.statistic)), rel=1e-12
        )


def test_logit_contrast_estimate_is_level_minus_reference(logit_result):
    rows = contrast(logit_result, "C(g)", "r")
    beta = _level_coefficients(logit_result)
    assert beta["a"] == 0.0

    for row, level in zip(rows, ["b", "c"]):
        assert row.estimate == pytest.approx(beta[level], rel=1e-12)


def test_logit_pwcompare_standard_errors_match_hand_computation(logit_result):
    rows = pwcompare(logit_result, "C(g)")
    assert len(rows) == 3

    for row, (high, low) in zip(rows, [("b", "a"), ("c", "a"), ("c", "b")]):
        weights = _level_weights(logit_result, high) - _level_weights(
            logit_result, low
        )
        estimate, std_err = _manual(logit_result, weights)
        assert row.estimate == pytest.approx(estimate, rel=1e-12)
        assert row.std_err == pytest.approx(std_err, rel=1e-12)


def test_logit_grand_mean_contrast_matches_hand_computation(logit_result):
    rows = contrast(logit_result, "C(g)", "g")
    beta = _level_coefficients(logit_result)

    assert len(rows) == len(LEVELS)
    grand_mean = float(np.mean([beta[level] for level in LEVELS]))
    for row, level in zip(rows, LEVELS):
        assert row.estimate == pytest.approx(beta[level] - grand_mean, rel=1e-12)


# ------------------------------------------------------------ small factor
def test_two_level_factor_contrast_and_pwcompare():
    rng = np.random.default_rng(31)
    g = np.array(["a", "b"] * 60)
    x = rng.normal(size=g.size)
    y = 1.0 + np.where(g == "b", 1.5, 0.0) + x + rng.normal(size=g.size)
    data = pd.DataFrame({"y": y, "x": x, "g": g})
    res = ws.OLS.from_formula("y ~ x + C(g, base='a')", data).fit()

    reference_rows = contrast(res, "C(g)", "r")
    adjacent_rows = contrast(res, "C(g)", "a")
    grand_rows = contrast(res, "C(g)", "g")
    pairwise_rows = pwcompare(res, "C(g)")

    assert len(reference_rows) == 1
    assert len(adjacent_rows) == 1
    assert len(grand_rows) == 2
    assert len(pairwise_rows) == 1
    assert pairwise_rows[0].estimate == pytest.approx(
        reference_rows[0].estimate, rel=1e-12
    )


def test_single_level_factor_cannot_be_contrasted():
    rng = np.random.default_rng(37)
    data = pd.DataFrame(
        {"y": rng.normal(size=40), "x": rng.normal(size=40), "g": ["a"] * 40}
    )
    res = ws.OLS.from_formula("y ~ x + C(g)", data).fit()

    # One level, no column at all: there is nothing to contrast.
    with pytest.raises(KeyError, match="does not match any factor column"):
        contrast(res, "C(g)", "r")
    with pytest.raises(KeyError, match="does not match any factor column"):
        pwcompare(res, "C(g)")


# -------------------------------------------------- factor coding variations
def test_numeric_factor_levels_resolve_in_sorted_order():
    rng = np.random.default_rng(41)
    g = np.array([1, 2, 3] * 30)
    y = g + 0.1 * rng.normal(size=g.size)
    res = ws.OLS.from_formula("y ~ C(g)", pd.DataFrame({"y": y, "g": g})).fit()

    # patsy writes numeric levels as "2.0"/"3.0"; the report labels use the
    # observed level values, which is Stata's `contrast r.` row order.
    rows = contrast(res, "C(g)", "r")
    assert [row.label for row in rows] == ["2 vs 1", "3 vs 1"]
    assert len(pwcompare(res, "C(g)")) == 3


def test_categorical_factor_preserves_declared_level_order():
    rng = np.random.default_rng(43)
    order = ["b", "a", "c"]
    g = pd.Categorical(np.array(order * 30), categories=order)
    y = rng.normal(size=len(g))
    res = ws.OLS.from_formula(
        "y ~ C(g)", pd.DataFrame({"y": y, "g": g})
    ).fit()

    # The declared categories fix the order, so "b" is the reference level and
    # is reported first -- not "a" as a plain sort would produce.
    assert [row.label for row in contrast(res, "C(g)", "r")] == ["a vs b", "c vs b"]
    assert [row.label for row in contrast(res, "C(g)", "g")] == [
        "b vs grand mean",
        "a vs grand mean",
        "c vs grand mean",
    ]


def test_explicit_reference_term_is_honoured():
    rng = np.random.default_rng(47)
    g = np.array(["a", "b", "c"] * 30)
    res = ws.OLS.from_formula(
        "y ~ C(g, base='b')",
        pd.DataFrame({"y": rng.normal(size=g.size), "g": g}),
    ).fit()

    rows = contrast(res, "C(g)", "r")
    assert [row.label for row in rows] == ["a vs b", "c vs b"]
    for row, level in zip(rows, ["a", "c"]):
        assert row.estimate == pytest.approx(
            float(res.params[f"C(g, Treatment(reference='b'))[T.{level}]"]), rel=1e-12
        )


def test_interaction_only_term_raises_unsupported():
    rng = np.random.default_rng(53)
    g = np.array(["a", "b", "c"] * 30)
    data = pd.DataFrame(
        {"y": rng.normal(size=g.size), "g": g, "x": rng.normal(size=g.size)}
    )
    res = ws.OLS.from_formula("y ~ x + C(g):x", data).fit()

    with pytest.raises(UnsupportedError, match="interaction"):
        contrast(res, "C(g)", "r")
    with pytest.raises(UnsupportedError, match="interaction"):
        pwcompare(res, "C(g)")


def test_reference_level_recovered_without_the_estimation_sample():
    rng = np.random.default_rng(59)
    g = np.array(["a", "b", "c"] * 30)
    res = ws.OLS.from_formula(
        "y ~ C(g, base='c')", pd.DataFrame({"y": rng.normal(size=g.size), "g": g})
    ).fit()
    res.model_data = None  # force the patsy-name fallback

    rows = contrast(res, "C(g)", "r")
    assert [row.label for row in rows] == ["a vs c", "b vs c"]
    assert len(pwcompare(res, "C(g)")) == 3


def test_reference_level_unknown_without_the_estimation_sample_raises():
    rng = np.random.default_rng(61)
    g = np.array(["a", "b", "c"] * 30)
    res = ws.OLS.from_formula(
        "y ~ C(g)", pd.DataFrame({"y": rng.normal(size=g.size), "g": g})
    ).fit()
    res.model_data = None  # patsy names carry no reference literal here

    # The omitted level is unknowable, so every contrast refuses rather than
    # silently reporting a table over an incomplete level set (which would drop
    # one of the three pairwise comparisons and bias a grand mean).
    with pytest.raises(UnsupportedError, match="complete level set"):
        contrast(res, "C(g)", "r")
    with pytest.raises(UnsupportedError, match="complete level set"):
        contrast(res, "C(g)", "a")
    with pytest.raises(UnsupportedError, match="complete level set"):
        pwcompare(res, "C(g)")

    # With an explicit reference the level set *is* recoverable, so the same
    # result is contrastable without the estimation sample.
    res_explicit = ws.OLS.from_formula(
        "y ~ C(g, base='b')", pd.DataFrame({"y": rng.normal(size=g.size), "g": g})
    ).fit()
    res_explicit.model_data = None
    assert len(contrast(res_explicit, "C(g)", "a")) == 2
    assert len(pwcompare(res_explicit, "C(g)")) == 3


# ------------------------------------------------------- cell-means coding
def test_cell_means_coding_supports_pwcompare_and_grand_mean():
    """``y ~ 0 + C(g)`` estimates each level mean directly, so contrasts work."""

    rng = np.random.default_rng(67)
    m = 40
    g = np.array(["a", "b", "c"] * m)
    effect = {"a": 0.5, "b": 2.5, "c": -1.5}
    y = np.array([effect[level] for level in g]) + rng.normal(scale=0.25, size=g.size)
    res = ws.OLS.from_formula(
        "y ~ 0 + C(g)", pd.DataFrame({"y": y, "g": g})
    ).fit()

    # Every level has its own column, so the estimates are level means.
    beta = {
        "a": float(res.params["C(g)[a]"]),
        "b": float(res.params["C(g)[b]"]),
        "c": float(res.params["C(g)[c]"]),
    }
    for level, value in effect.items():
        assert beta[level] == pytest.approx(value, abs=0.15)

    rows = pwcompare(res, "C(g)")
    assert [row.label for row in rows] == ["b vs a", "c vs a", "c vs b"]
    assert rows[0].estimate == pytest.approx(beta["b"] - beta["a"], rel=1e-12)
    assert rows[2].estimate == pytest.approx(beta["c"] - beta["b"], rel=1e-12)

    grand_rows = contrast(res, "C(g)", "g")
    grand_mean = float(np.mean(list(beta.values())))
    for row, level in zip(grand_rows, LEVELS):
        assert row.estimate == pytest.approx(beta[level] - grand_mean, rel=1e-12)

    # No level is omitted, so a "which one is the reference?" contrast is
    # undefined -- and says so instead of inventing a reference.
    with pytest.raises(UnsupportedError, match="reference level"):
        contrast(res, "C(g)", "r")


def test_adjacent_contrast_on_cell_means_uses_level_order():
    rng = np.random.default_rng(71)
    g = np.array(["a", "b", "c"] * 40)
    y = np.array([{"a": 0.0, "b": 1.0, "c": 4.0}[level] for level in g])
    y = y + 0.05 * rng.normal(size=g.size)
    res = ws.OLS.from_formula("y ~ 0 + C(g)", pd.DataFrame({"y": y, "g": g})).fit()

    rows = contrast(res, "C(g)", "a")
    # Stata's adjacent order is "low vs high".
    assert [row.label for row in rows] == ["a vs b", "b vs c"]
    # Cell means rise 0 -> 1 -> 4, so low-minus-high is negative.
    assert rows[0].estimate == pytest.approx(-1.0, abs=0.05)
    assert rows[1].estimate == pytest.approx(-3.0, abs=0.05)


@pytest.mark.parametrize(
    "formula, coding",
    [
        ("y ~ x + C(g, Helmert)", "H"),
        ("y ~ 0 + C(g, Sum)", "Sum"),
    ],
)
def test_non_treatment_codings_raise_unsupported(formula, coding):
    """Sum/Helmert columns are already contrasts, not level means."""

    rng = np.random.default_rng(73)
    g = np.array(["a", "b", "c"] * 40)
    data = pd.DataFrame(
        {"y": rng.normal(size=g.size), "g": g, "x": rng.normal(size=g.size)}
    )
    res = ws.OLS.from_formula(formula, data).fit()

    for call in (
        lambda: contrast(res, "C(g)", "r"),
        lambda: contrast(res, "C(g)", "a"),
        lambda: contrast(res, "C(g)", "g"),
        lambda: pwcompare(res, "C(g)"),
    ):
        with pytest.raises(UnsupportedError, match="non-treatment contrast"):
            call()
